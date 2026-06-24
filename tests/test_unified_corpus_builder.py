"""Tests for unified corpus construction."""

import json

from src.ir_app.services.corpus_builder import CorpusBuilder


def write_jsonl(path, rows):
    """Write rows as JSONL for tests.

    Complexity:
        Time: O(n)
        Space: O(1)
    """
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path):
    """Read a JSONL file into dictionaries.

    Complexity:
        Time: O(n)
        Space: O(n)
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_corpus_builder_merges_and_deduplicates_sources(tmp_path):
    """Builder writes validated records and skips duplicate title/url pairs."""
    source_a = tmp_path / "merged_14days_preprocessed.jsonl"
    source_b = tmp_path / "cna_mvp_cleaned.jsonl"
    output = tmp_path / "unified.jsonl"
    report = tmp_path / "report.json"

    write_jsonl(
        source_a,
        [
            {
                "article_id": "A1",
                "url": "https://example.com/a1",
                "source": "LTN",
                "source_name": "自由時報",
                "title": "台灣經濟成長動能延續",
                "content": "台灣經濟與半導體出口持續成長，市場關注後續投資動能。",
                "published_date": "2026-01-01",
                "category": "finance",
            },
            {
                "article_id": "A1_DUP",
                "url": "https://example.com/a1",
                "source": "LTN",
                "title": "台灣經濟成長動能延續",
                "content": "這筆資料應該被 dedup 掉，因為 title 與 url 相同。",
            },
        ],
    )
    write_jsonl(
        source_b,
        [
            {
                "article_id": "CNA1",
                "url": "https://www.cna.com.tw/news/aipl/1.aspx",
                "source": "cna",
                "source_name": "中央社",
                "title": "人工智慧應用擴大",
                "content": "人工智慧技術在醫療、教育與新聞分析場景快速擴大應用。",
                "tags": ["AI", "醫療"],
            }
        ],
    )

    stats = CorpusBuilder(tmp_path).build([source_a, source_b], output, report)
    rows = read_jsonl(output)

    assert stats.scanned == 3
    assert stats.written == 2
    assert stats.duplicates == 1
    assert len(rows) == 2
    assert rows[0]["content_type"] == "news_article"
    assert rows[0]["dedup_hash"]
    assert rows[1]["source"] == "cna"
    assert json.loads(report.read_text(encoding="utf-8"))["written"] == 2


def test_corpus_builder_rejects_blocked_dcard_pages(tmp_path):
    """Anti-bot block pages from Dcard are counted but not written."""
    dcard_json = tmp_path / "post_1.json"
    output = tmp_path / "unified.jsonl"

    dcard_json.write_text(
        json.dumps(
            {
                "url": "https://www.dcard.tw/f/trending/p/1",
                "title": "Sorry, you have been blocked",
                "content": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    stats = CorpusBuilder(tmp_path).build([dcard_json], output)

    assert stats.scanned == 1
    assert stats.written == 0
    assert stats.invalid == 1
    assert stats.invalid_reasons["BLOCKED_PAGE"] == 1
    assert output.read_text(encoding="utf-8") == ""


def test_corpus_builder_limits_each_input_independently(tmp_path):
    """Per-input caps keep large sources balanced in the generated corpus."""
    source_a = tmp_path / "source_a.jsonl"
    source_b = tmp_path / "source_b.jsonl"
    output = tmp_path / "unified.jsonl"

    def rows(prefix):
        return [
            {
                "article_id": f"{prefix}{index}",
                "url": f"https://example.com/{prefix}/{index}",
                "source": prefix,
                "title": f"{prefix} 新聞標題 {index}",
                "content": "這是一段足夠長的新聞內容，用來測試每個輸入檔案的上限控制。",
            }
            for index in range(3)
        ]

    write_jsonl(source_a, rows("A"))
    write_jsonl(source_b, rows("B"))

    stats = CorpusBuilder(tmp_path).build(
        [source_a, source_b],
        output,
        max_docs_per_input=2,
    )
    built_rows = read_jsonl(output)

    assert stats.written == 4
    assert [row["article_id"] for row in built_rows] == ["A0", "A1", "B0", "B1"]


def test_corpus_builder_limits_raw_inputs_only(tmp_path):
    """Raw caps leave curated inputs uncapped while sampling raw files."""
    curated = tmp_path / "data" / "preprocessed" / "merged.jsonl"
    raw = tmp_path / "data" / "raw" / "yahoo.jsonl"
    output = tmp_path / "unified.jsonl"
    curated.parent.mkdir(parents=True)
    raw.parent.mkdir(parents=True)

    def rows(prefix, count):
        return [
            {
                "article_id": f"{prefix}{index}",
                "url": f"https://example.com/{prefix}/{index}",
                "source": prefix,
                "title": f"{prefix} 新聞標題 {index}",
                "content": "這是一段足夠長的新聞內容，用來測試 raw input 的取樣上限。",
            }
            for index in range(count)
        ]

    write_jsonl(curated, rows("curated", 3))
    write_jsonl(raw, rows("raw", 3))

    stats = CorpusBuilder(tmp_path).build(
        [curated, raw],
        output,
        max_docs_per_raw_input=1,
    )
    built_rows = read_jsonl(output)

    assert stats.written == 4
    assert [row["article_id"] for row in built_rows] == [
        "curated0",
        "curated1",
        "curated2",
        "raw0",
    ]
