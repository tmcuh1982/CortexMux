"""Schemas returned by deterministic web-page extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WebTable(BaseModel):
    """One bounded HTML table."""

    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)

    def to_records(self) -> list[dict[str, str]]:
        """Convert table rows into dictionaries keyed by normalized headers."""
        return [
            {
                header: row[index] if index < len(row) else ""
                for index, header in enumerate(self.headers)
            }
            for row in self.rows
        ]


class WebPage(BaseModel):
    """Visible content and provenance extracted from one web response."""

    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    title: str | None = None
    text: str
    tables: list[WebTable] = Field(default_factory=list)
    bytes_received: int
    text_truncated: bool = False

    def text_record(self) -> dict[str, str | bool]:
        """Return a data-analysis compatible record for the page text."""
        return {
            "source_url": self.final_url,
            "title": self.title or "",
            "content": self.text,
            "text_truncated": self.text_truncated,
        }
