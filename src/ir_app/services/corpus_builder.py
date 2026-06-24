"""Build a unified searchable corpus from crawler project outputs."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.ir_app.services.data_contract import (
    compute_dedup_hash,
    normalize_tags,
    validate_article,
)


BLOCKED_TITLES = {"sorry, you have been blocked"}


@dataclass
class CorpusBuildStats:
    """Corpus build counters and source-level diagnostics.

    Complexity:
        Time: O(1)
        Space: O(s) where s is source count
    """

    scanned: int = 0
    written: int = 0
    invalid: int = 0
    duplicates: int = 0
    input_files: list[str] = field(default_factory=list)
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)
    invalid_reasons: dict[str, int] = field(default_factory=dict)

    def source_bucket(self, source: str) -> dict[str, int]:
        """Return mutable counters for one source.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return self.by_source.setdefault(
            source or "unknown",
            {"scanned": 0, "written": 0, "invalid": 0, "duplicates": 0},
        )

    def add_invalid_reason(self, code: str) -> None:
        """Track an invalid record reason.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        self.invalid_reasons[code] = self.invalid_reasons.get(code, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert counters to JSON-serializable data.

        Complexity:
            Time: O(s)
            Space: O(s)
        """
        return {
            "scanned": self.scanned,
            "written": self.written,
            "invalid": self.invalid,
            "duplicates": self.duplicates,
            "input_files": self.input_files,
            "by_source": self.by_source,
            "invalid_reasons": self.invalid_reasons,
        }


class CorpusBuilder:
    """Normalize multi-project data into one JSONL corpus.

    Complexity:
        Time: O(n) for n source records
        Space: O(d) for d duplicate hashes
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.stats = CorpusBuildStats()
        self._seen_hashes: set[str] = set()

    def build(
        self,
        input_paths: Iterable[Path],
        output_path: Path,
        report_path: Path | None = None,
        max_docs: int | None = None,
        max_docs_per_source: int | None = None,
        max_docs_per_input: int | None = None,
        max_docs_per_raw_input: int | None = None,
    ) -> CorpusBuildStats:
        """Build a unified JSONL corpus from JSONL/JSON/SQLite sources.

        Complexity:
            Time: O(n)
            Space: O(d)
        """
        self.stats = CorpusBuildStats()
        self._seen_hashes = set()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as output:
            for path in input_paths:
                if max_docs is not None and self.stats.written >= max_docs:
                    break
                path = path.expanduser()
                if not path.exists():
                    continue
                self.stats.input_files.append(str(path))
                input_written = 0
                input_limit = max_docs_per_input
                if input_limit is None and self._is_raw_input(path):
                    input_limit = max_docs_per_raw_input
                for raw in self._iter_records(path):
                    if max_docs is not None and self.stats.written >= max_docs:
                        break
                    if (
                        input_limit is not None
                        and input_written >= input_limit
                    ):
                        break
                    normalized = self.normalize_record(raw, path)
                    if normalized is None:
                        continue
                    source = str(normalized.get("source") or "unknown")
                    source_stats = self.stats.source_bucket(source)
                    if (
                        max_docs_per_source is not None
                        and source_stats["written"] >= max_docs_per_source
                    ):
                        continue
                    output.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    self.stats.written += 1
                    input_written += 1
                    source_stats["written"] += 1

        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(self.stats.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return self.stats

    def normalize_record(self, raw: dict[str, Any], origin_path: Path) -> dict[str, Any] | None:
        """Normalize one article or forum post into the application schema.

        Complexity:
            Time: O(t) where t is tag/topic count
            Space: O(t)
        """
        self.stats.scanned += 1

        source = str(raw.get("source") or raw.get("crawl_source") or self._infer_source(origin_path))
        bucket = self.stats.source_bucket(source)
        bucket["scanned"] += 1

        title = str(raw.get("title") or raw.get("title_clean") or "").strip()
        content = str(
            raw.get("content")
            or raw.get("content_clean")
            or raw.get("text")
            or raw.get("body")
            or raw.get("excerpt")
            or ""
        ).strip()

        if title.lower() in BLOCKED_TITLES:
            self._record_invalid(bucket, "BLOCKED_PAGE")
            return None

        tags = normalize_tags(raw.get("tags") or self._topics_to_tags(raw.get("topics")))
        url = raw.get("url")
        dedup_hash = raw.get("dedup_hash") or compute_dedup_hash(title, str(url or ""))

        candidate = {
            "article_id": raw.get("article_id") or raw.get("post_id") or dedup_hash,
            "url": url,
            "source": source,
            "source_name": raw.get("source_name") or raw.get("forum_name") or source,
            "title": title,
            "content": content,
            "author": raw.get("author") or raw.get("school") or "",
            "published_date": (
                raw.get("published_date")
                or raw.get("publish_date")
                or raw.get("created_at")
                or raw.get("date")
            ),
            "category": raw.get("category") or raw.get("forum_alias") or "unknown",
            "category_name": raw.get("category_name") or raw.get("forum_name") or "",
            "tags": tags,
            "image_url": raw.get("image_url"),
            "crawled_at": raw.get("crawled_at"),
            "content_type": "forum_post" if source.lower() == "dcard" else "news_article",
            "origin_path": str(origin_path),
            "dedup_hash": dedup_hash,
        }

        issues = validate_article(candidate, self.stats.scanned - 1)
        if issues:
            for issue in issues[1:]:
                self.stats.add_invalid_reason(issue.code)
            self._record_invalid(bucket, issues[0].code)
            return None

        if dedup_hash in self._seen_hashes:
            self.stats.duplicates += 1
            bucket["duplicates"] += 1
            return None
        self._seen_hashes.add(dedup_hash)

        return candidate

    def _iter_records(self, path: Path) -> Iterable[dict[str, Any]]:
        """Yield dictionaries from JSONL, JSON, or supported SQLite files.

        Complexity:
            Time: O(n)
            Space: O(1) for JSONL/SQLite, O(n) for JSON arrays
        """
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    yield json.loads(line)
            return

        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(data, dict):
                yield data
            return

        if suffix == ".db":
            yield from self._iter_dcard_db(path)

    def _iter_dcard_db(self, path: Path) -> Iterable[dict[str, Any]]:
        """Yield Dcard posts from the crawler SQLite database.

        Complexity:
            Time: O(n)
            Space: O(1)
        """
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        try:
            tables = {
                row["name"]
                for row in connection.execute(
                    "select name from sqlite_master where type = 'table'"
                )
            }
            if "posts" not in tables:
                return
            for row in connection.execute("select * from posts"):
                item = dict(row)
                item["source"] = "dcard"
                yield item
        finally:
            connection.close()

    def _infer_source(self, path: Path) -> str:
        """Infer a source code from a known crawler file name.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        name = path.name.lower()
        if name.startswith("yahoo_"):
            return "yahoo"
        if name.endswith("_14days.jsonl"):
            return name.split("_", 1)[0]
        if "dcard" in str(path).lower():
            return "dcard"
        return path.stem.split("_", 1)[0]

    def _is_raw_input(self, path: Path) -> bool:
        """Return whether a path is from a raw data directory.

        Complexity:
            Time: O(p) where p is path depth
            Space: O(1)
        """
        return "raw" in path.parts

    def _topics_to_tags(self, topics: Any) -> list[str]:
        """Convert Dcard topic objects to tags.

        Complexity:
            Time: O(t)
            Space: O(t)
        """
        if not isinstance(topics, list):
            return []
        tags: list[str] = []
        for topic in topics:
            if isinstance(topic, dict):
                value = topic.get("name") or topic.get("title") or topic.get("alias")
                if value:
                    tags.append(str(value))
            elif topic:
                tags.append(str(topic))
        return tags

    def _record_invalid(self, bucket: dict[str, int], code: str) -> None:
        """Increment invalid counters.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        self.stats.invalid += 1
        bucket["invalid"] += 1
        self.stats.add_invalid_reason(code)


def default_input_paths(project_root: Path, include_yahoo: bool, include_dcard: bool) -> list[Path]:
    """Return stable default corpus inputs.

    Complexity:
        Time: O(k)
        Space: O(k)
    """
    paths = [
        project_root / "data" / "preprocessed" / "merged_14days_preprocessed.jsonl",
        project_root / "data" / "processed" / "cna_mvp_cleaned.jsonl",
    ]
    if include_yahoo:
        paths.extend(sorted((project_root / "data" / "raw").glob("yahoo_*_14days.jsonl")))
    if include_dcard:
        dcard_root = project_root.parent / "dcard-trending-crawler"
        paths.append(dcard_root / "data" / "dcard_crawler.db")
        paths.extend(sorted((dcard_root / "data" / "raw").glob("post_*.json")))
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/processed/unified_news_corpus.jsonl")
    parser.add_argument("--report", default="data/stats/unified_corpus_report.json")
    parser.add_argument("--input", action="append", dest="inputs", default=[])
    parser.add_argument("--include-yahoo", action="store_true")
    parser.add_argument("--include-dcard", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    parser.add_argument("--max-docs-per-input", type=int, default=None)
    parser.add_argument("--max-docs-per-raw-input", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for corpus construction.

    Complexity:
        Time: O(n)
        Space: O(d)
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    inputs = [Path(value) for value in args.inputs]
    if not inputs:
        inputs = default_input_paths(project_root, args.include_yahoo, args.include_dcard)
    else:
        default_extras = default_input_paths(project_root, args.include_yahoo, args.include_dcard)[2:]
        existing = {str(path) for path in inputs}
        inputs.extend(path for path in default_extras if str(path) not in existing)

    builder = CorpusBuilder(project_root)
    stats = builder.build(
        input_paths=inputs,
        output_path=project_root / args.output,
        report_path=project_root / args.report,
        max_docs=args.max_docs,
        max_docs_per_source=args.max_docs_per_source,
        max_docs_per_input=args.max_docs_per_input,
        max_docs_per_raw_input=args.max_docs_per_raw_input,
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
