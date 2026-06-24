"""Source and topic taxonomy helpers for the news corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOPIC_LABELS = {
    "politics": "政治 Politics",
    "world": "國際 World",
    "business": "財經 Business",
    "tech": "科技 Tech",
    "society": "社會 Society",
    "lifestyle": "生活 Lifestyle",
    "entertainment": "娛樂 Entertainment",
    "sports": "體育 Sports",
    "health": "健康 Health",
    "local": "地方 Local",
    "other": "其他 Other",
}

SOURCE_LABELS = {
    "cna": "中央社 CNA",
    "LTN": "自由時報 LTN",
    "NextApple": "壹蘋新聞網 NextApple",
    "SETN": "三立新聞 SETN",
    "UDN": "聯合新聞網 UDN",
    "PTS": "公視 PTS",
    "Yahoo": "Yahoo 新聞",
    "dcard": "Dcard",
}

YAHOO_FILE_TOPICS = {
    "yahoo_entertainment": ("entertainment", "entertainment"),
    "yahoo_finance": ("business", "finance"),
    "yahoo_health": ("health", "health"),
    "yahoo_lifestyle": ("lifestyle", "lifestyle"),
    "yahoo_politics": ("politics", "politics"),
    "yahoo_sports": ("sports", "sports"),
    "yahoo_tech": ("tech", "tech"),
    "yahoo_world": ("world", "world"),
}

CATEGORY_ALIASES = {
    "aipl": ("politics", "politics"),
    "政治": ("politics", "politics"),
    "politics": ("politics", "politics"),
    "國際": ("world", "world"),
    "全球": ("world", "world"),
    "world": ("world", "world"),
    "財經": ("business", "finance"),
    "產經": ("business", "finance"),
    "finance": ("business", "finance"),
    "business": ("business", "business"),
    "AI科技": ("tech", "ai"),
    "3C": ("tech", "gadget"),
    "gadget": ("tech", "gadget"),
    "aitech": ("tech", "ai"),
    "tech": ("tech", "tech"),
    "社會": ("society", "society"),
    "society": ("society", "society"),
    "生活": ("lifestyle", "life"),
    "life": ("lifestyle", "life"),
    "娛樂": ("entertainment", "entertainment"),
    "entertainment": ("entertainment", "entertainment"),
    "體育": ("sports", "sports"),
    "sports": ("sports", "sports"),
    "健康": ("health", "health"),
    "health": ("health", "health"),
    "地方": ("local", "local"),
    "local": ("local", "local"),
    "兩岸": ("politics", "cross_strait"),
    "房地產": ("business", "property"),
    "property": ("business", "property"),
    "其他": ("other", "other"),
    "other": ("other", "other"),
    "unknown": ("other", "unknown"),
    "未分類": ("other", "unknown"),
    "": ("other", "unknown"),
}


@dataclass(frozen=True)
class TaxonomyInfo:
    """Normalized taxonomy fields for a searchable record.

    Complexity:
        Time: O(1)
        Space: O(1)
    """

    source: str
    source_name: str
    source_label: str
    taxonomy_topic: str
    taxonomy_label: str
    taxonomy_path: str

    def to_dict(self) -> dict[str, str]:
        """Convert taxonomy info to a dictionary.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return {
            "source": self.source,
            "source_name": self.source_name,
            "source_label": self.source_label,
            "taxonomy_topic": self.taxonomy_topic,
            "taxonomy_label": self.taxonomy_label,
            "taxonomy_path": self.taxonomy_path,
        }


def normalize_source(source: Any, source_name: Any = None) -> tuple[str, str, str]:
    """Normalize source code, source name, and source label.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    source_code = str(source or "").strip() or "unknown"
    if source_code.lower() == "yahoo":
        source_code = "Yahoo"
    if source_code.lower() == "cna":
        source_code = "cna"
    resolved_name = str(source_name or "").strip() or SOURCE_LABELS.get(source_code, source_code)
    return source_code, resolved_name, SOURCE_LABELS.get(source_code, resolved_name)


def classify_category(
    category: Any,
    category_name: Any = None,
    origin_path: Any = None,
) -> tuple[str, str]:
    """Map raw category metadata to a top-level topic and path segment.

    Complexity:
        Time: O(k) where k is known Yahoo filename prefix count
        Space: O(1)
    """
    path_text = str(origin_path or "")
    path_name = Path(path_text).name.lower() if path_text else ""
    for prefix, mapped in YAHOO_FILE_TOPICS.items():
        if path_name.startswith(prefix):
            return mapped

    candidates = [category, category_name]
    for value in candidates:
        key = str(value or "").strip()
        if key in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[key]
        lower_key = key.lower()
        if lower_key in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[lower_key]
    return CATEGORY_ALIASES["unknown"]


def normalize_taxonomy(raw: dict[str, Any], origin_path: Any = None) -> TaxonomyInfo:
    """Build normalized taxonomy info from a raw record.

    Complexity:
        Time: O(k)
        Space: O(1)
    """
    source, source_name, source_label = normalize_source(
        raw.get("source") or raw.get("crawl_source"),
        raw.get("source_name") or raw.get("forum_name"),
    )
    topic, leaf = classify_category(
        raw.get("category") or raw.get("forum_alias"),
        raw.get("category_name") or raw.get("forum_name"),
        origin_path or raw.get("origin_path"),
    )
    return TaxonomyInfo(
        source=source,
        source_name=source_name,
        source_label=source_label,
        taxonomy_topic=topic,
        taxonomy_label=TOPIC_LABELS.get(topic, TOPIC_LABELS["other"]),
        taxonomy_path=f"news/{topic}/{leaf}",
    )


def facet_label(field: str, value: Any) -> str:
    """Return a display label for one facet field/value pair.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    text = str(value or "")
    if field == "taxonomy_topic":
        return TOPIC_LABELS.get(text, text)
    if field == "source":
        return SOURCE_LABELS.get(text, text)
    if field == "content_type":
        return {"news_article": "新聞 News", "forum_post": "論壇 Forum"}.get(text, text)
    return text
