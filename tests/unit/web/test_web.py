"""Secure web retrieval and deterministic HTML extraction tests."""

from __future__ import annotations

import httpx
import pytest

from cortexmux.core.config import WebConfig
from cortexmux.core.exceptions import (
    RemoteHostNotAllowedError,
    WebAccessDisabledError,
    WebFetchError,
)
from cortexmux.web import WebPageFetcher
from cortexmux.web import parser as web_parser
from cortexmux.web.parser import extract_html

PUBLIC_URL = "https://93.184.216.34"


@pytest.mark.asyncio
async def test_fetch_extracts_visible_text_title_and_table() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/stats"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="""
                <html>
                  <head>
                    <title>Population statistics</title>
                    <style>.hidden { display: none }</style>
                    <script>ignore_model_instructions()</script>
                  </head>
                  <body>
                    <h1>Population</h1>
                    <p>Official estimate: 12 345 inhabitants.</p>
                    <table>
                      <tr><th>Year</th><th>Value</th><th>Value</th></tr>
                      <tr><td>2025</td><td>12345</td><td>stable</td></tr>
                    </table>
                  </body>
                </html>
            """,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = WebPageFetcher(
        WebConfig(enabled=True, allowed_hosts={"93.184.216.34"}),
        client=client,
    )
    page = await fetcher.fetch(f"{PUBLIC_URL}/stats#section")

    assert page.final_url == f"{PUBLIC_URL}/stats"
    assert page.title == "Population statistics"
    assert "Official estimate: 12 345 inhabitants." in page.text
    assert "ignore_model_instructions" not in page.text
    assert ".hidden" not in page.text
    assert page.tables[0].headers == ["Year", "Value", "Value_2"]
    assert page.tables[0].to_records() == [{"Year": "2025", "Value": "12345", "Value_2": "stable"}]
    await client.aclose()


@pytest.mark.asyncio
async def test_redirects_are_revalidated_and_bounded() -> None:
    requested_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/final"})
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text="  verified   value \n",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = WebPageFetcher(WebConfig(enabled=True), client=client)
    page = await fetcher.fetch(f"{PUBLIC_URL}/start")

    assert requested_paths == ["/start", "/final"]
    assert page.final_url == f"{PUBLIC_URL}/final"
    assert page.text == "verified value"
    await client.aclose()


@pytest.mark.asyncio
async def test_web_access_is_opt_in_and_private_targets_are_blocked() -> None:
    disabled = WebPageFetcher(WebConfig())
    with pytest.raises(WebAccessDisabledError):
        await disabled.fetch(PUBLIC_URL)
    await disabled.close()

    fetcher = WebPageFetcher(WebConfig(enabled=True))
    with pytest.raises(RemoteHostNotAllowedError):
        await fetcher.fetch("http://127.0.0.1/private")
    with pytest.raises(RemoteHostNotAllowedError):
        await fetcher.fetch("https://user:password@93.184.216.34/private")
    with pytest.raises(RemoteHostNotAllowedError):
        await fetcher.fetch("https://93.184.216.34:8443/private")
    await fetcher.close()


@pytest.mark.asyncio
async def test_response_size_and_content_type_are_enforced() -> None:
    responses = [
        httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "10"},
            content=b"0123456789",
        ),
        httpx.Response(
            200,
            headers={"content-type": "application/octet-stream"},
            content=b"data",
        ),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return responses.pop(0)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = WebPageFetcher(
        WebConfig(enabled=True, max_response_bytes=5),
        client=client,
    )

    with pytest.raises(WebFetchError, match="size limit"):
        await fetcher.fetch(PUBLIC_URL)
    with pytest.raises(WebFetchError, match="content type"):
        await fetcher.fetch(PUBLIC_URL)
    await client.aclose()


@pytest.mark.parametrize(
    ("html", "limits", "message"),
    [
        (
            "<table><tr><th>A</th><th>B</th><th>C</th></tr></table>",
            {"max_table_columns": 2, "max_table_cells": 10},
            "column limit",
        ),
        (
            "<table><tr><th>A</th></tr><tr><td>x</td><td>y</td><td>z</td></tr></table>",
            {"max_table_columns": 3, "max_table_cells": 5},
            "cell limit",
        ),
        (
            "<table><tr><td>x</td><td>y</td></tr></table>" * 2,
            {"max_table_columns": 2, "max_table_cells": 5},
            "cell limit",
        ),
        (
            "<table><tr>" + "<th>H</th>" * 100 + "</tr>" + "<tr><td>x</td></tr>" * 50 + "</table>",
            {"max_table_columns": 100, "max_table_cells": 150},
            "cell limit",
        ),
    ],
)
def test_html_table_shape_limits(html: str, limits: dict[str, int], message: str) -> None:
    with pytest.raises(WebFetchError, match=message):
        extract_html(html, max_tables=2, max_table_rows=50, **limits)


def test_html_table_limits_preserve_ragged_rows_and_disabled_tables() -> None:
    html = (
        "<table><tr><th>A</th><th>A</th></tr>"
        "<tr><td>x</td></tr><tr><td>y</td><td>z</td></tr></table>"
    )
    _title, _text, tables = extract_html(
        html, max_tables=1, max_table_rows=2, max_table_columns=2, max_table_cells=6
    )
    assert tables[0].headers == ["A", "A_2"]
    assert tables[0].rows == [["x", ""], ["y", "z"]]
    assert (
        extract_html(html, max_tables=0, max_table_rows=0, max_table_columns=1, max_table_cells=1)[
            2
        ]
        == []
    )


def test_sparse_table_is_rejected_before_matrix_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    html = "<table><tr>" + "<th>H</th>" * 100 + "</tr>" + "<tr><td>x</td></tr>" * 50 + "</table>"

    def should_not_build(*args: object, **kwargs: object) -> None:
        raise AssertionError("The oversized table was allocated before the limit check.")

    monkeypatch.setattr(web_parser, "_build_table", should_not_build)
    with pytest.raises(WebFetchError, match="cell limit"):
        extract_html(
            html,
            max_tables=1,
            max_table_rows=50,
            max_table_columns=100,
            max_table_cells=150,
        )


@pytest.mark.asyncio
async def test_fetch_enforces_table_column_limit() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<table><tr><td>1</td><td>2</td><td>3</td></tr></table>",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = WebPageFetcher(WebConfig(enabled=True, max_table_columns=2), client=client)
    with pytest.raises(WebFetchError, match="column limit"):
        await fetcher.fetch(PUBLIC_URL)
    await client.aclose()
