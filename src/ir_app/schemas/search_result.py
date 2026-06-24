"""Search result schema used by the Flask application layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """Normalized search result for API responses.

    Complexity:
        Time: O(1)
        Space: O(m) where m is metadata/explanation size
    """

    doc_id: int | str
    article_id: str | None
    title: str
    snippet: str
    highlighted_snippet: str
    score: float | None
    model: str
    url: str | None = None
    published_date: str | None = None
    category: str | None = None
    category_name: str | None = None
    source: str | None = None
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    rank: int = 0
    explanation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary.

        Complexity:
            Time: O(t) where t is number of tags
            Space: O(t)
        """
        return {
            "doc_id": self.doc_id,
            "article_id": self.article_id,
            "title": self.title,
            "snippet": self.snippet,
            "highlighted_snippet": self.highlighted_snippet,
            "score": self.score,
            "model": self.model,
            "url": self.url,
            "published_date": self.published_date,
            "pub_date": self.published_date,
            "category": self.category,
            "category_name": self.category_name,
            "source": self.source,
            "author": self.author,
            "tags": self.tags,
            "rank": self.rank,
            "explanation": self.explanation,
            "metadata": {
                "article_id": self.article_id,
                "url": self.url,
                "published_date": self.published_date,
                "date": self.published_date,
                "category": self.category,
                "category_name": self.category_name,
                "source": self.source,
                "author": self.author,
                "tags": self.tags,
            },
        }
