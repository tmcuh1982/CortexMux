"""Small dependency-free extractor for visible HTML text and tables."""

from __future__ import annotations

import re
from html.parser import HTMLParser

from cortexmux.core.exceptions import WebFetchError
from cortexmux.web.schemas import WebTable

_IGNORED_TAGS = {"script", "style", "noscript", "svg", "template"}
_BREAK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "p",
    "section",
    "table",
    "td",
    "th",
    "tr",
}


class _DocumentParser(HTMLParser):
    def __init__(
        self, *, max_tables: int, max_table_rows: int, max_table_columns: int, max_table_cells: int
    ) -> None:
        super().__init__(convert_charrefs=True)
        self.max_tables = max_tables
        self.max_table_rows = max_table_rows
        self.max_table_columns = max_table_columns
        self.max_table_cells = max_table_cells
        self.cells_used = 0
        self.current_table_cells = 0
        self.ignored_depth = 0
        self.in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.tables: list[WebTable] = []
        self.table_rows: list[tuple[list[str], bool]] | None = None
        self.current_row: list[str] | None = None
        self.current_row_has_header = False
        self.current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in _IGNORED_TAGS:
            self.ignored_depth += 1
            return
        if self.ignored_depth:
            return
        if tag == "title":
            self.in_title = True
        if tag in _BREAK_TAGS:
            self.text_parts.append("\n")
        if tag == "table" and self.table_rows is None and len(self.tables) < self.max_tables:
            self.table_rows = []
            self.current_table_cells = 0
        elif (
            tag == "tr"
            and self.table_rows is not None
            and len(self.table_rows) < self.max_table_rows + 1
        ):
            self.current_row = []
            self.current_row_has_header = False
        elif tag in {"th", "td"} and self.current_row is not None:
            self.current_cell = []
            self.current_row_has_header = self.current_row_has_header or tag == "th"

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _IGNORED_TAGS:
            self.ignored_depth = max(0, self.ignored_depth - 1)
            return
        if self.ignored_depth:
            return
        if tag == "title":
            self.in_title = False
        elif tag in {"th", "td"} and self.current_cell is not None:
            if self.current_row is not None:
                if len(self.current_row) >= self.max_table_columns:
                    raise WebFetchError("Web table exceeded the column limit.")
                if self.cells_used + self.current_table_cells >= self.max_table_cells:
                    raise WebFetchError("Web tables exceeded the cell limit.")
                self.current_row.append(_clean_inline(" ".join(self.current_cell)))
                self.current_table_cells += 1
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            if (
                self.table_rows is not None
                and any(self.current_row)
                and len(self.table_rows) < self.max_table_rows + 1
            ):
                self.table_rows.append((self.current_row, self.current_row_has_header))
            self.current_row = None
            self.current_row_has_header = False
        elif tag == "table" and self.table_rows is not None:
            cells = 0
            if self.table_rows:
                width = max(len(row) for row, _is_header in self.table_rows)
                first_is_header = self.table_rows[0][1]
                data_rows = len(self.table_rows) - int(first_is_header)
                cells = width * (1 + min(data_rows, self.max_table_rows))
                if self.cells_used + cells > self.max_table_cells:
                    raise WebFetchError("Web tables exceeded the cell limit.")
            table = _build_table(self.table_rows, max_rows=self.max_table_rows)
            if table is not None:
                self.cells_used += cells
                self.tables.append(table)
            self.table_rows = None
        if tag in _BREAK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.ignored_depth:
            return
        if self.in_title:
            self.title_parts.append(data)
            return
        self.text_parts.append(data)
        if self.current_cell is not None:
            self.current_cell.append(data)


def extract_html(
    html: str,
    *,
    max_tables: int,
    max_table_rows: int,
    max_table_columns: int = 256,
    max_table_cells: int = 100_000,
) -> tuple[str | None, str, list[WebTable]]:
    """Extract normalized visible text, document title, and simple tables."""
    parser = _DocumentParser(
        max_tables=max_tables,
        max_table_rows=max_table_rows,
        max_table_columns=max_table_columns,
        max_table_cells=max_table_cells,
    )
    parser.feed(html)
    parser.close()
    title = _clean_inline(" ".join(parser.title_parts)) or None
    text = _clean_text("".join(parser.text_parts))
    return title, text, parser.tables


def normalize_plain_text(text: str) -> str:
    """Normalize whitespace in a plain-text web response."""
    return _clean_text(text)


def _build_table(
    rows: list[tuple[list[str], bool]],
    *,
    max_rows: int,
) -> WebTable | None:
    if not rows:
        return None
    width = max(len(row) for row, _is_header in rows)
    first_row, first_is_header = rows[0]
    if first_is_header:
        headers = _unique_headers(first_row, width)
        data_rows = rows[1:]
    else:
        headers = [f"column_{index + 1}" for index in range(width)]
        data_rows = rows
    normalized_rows = [row + [""] * (width - len(row)) for row, _is_header in data_rows[:max_rows]]
    return WebTable(headers=headers, rows=normalized_rows)


def _unique_headers(values: list[str], width: int) -> list[str]:
    headers: list[str] = []
    counts: dict[str, int] = {}
    for index in range(width):
        base = values[index] if index < len(values) and values[index] else f"column_{index + 1}"
        count = counts.get(base, 0) + 1
        counts[base] = count
        headers.append(base if count == 1 else f"{base}_{count}")
    return headers


def _clean_inline(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clean_text(value: str) -> str:
    lines = [_clean_inline(line) for line in value.splitlines()]
    return "\n".join(line for line in lines if line)
