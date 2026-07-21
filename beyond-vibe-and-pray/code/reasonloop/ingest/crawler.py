"""Deterministic website ingestion via a real browser engine.

Target sources such as the FFIEC BSA/AML Examination Manual sit behind a
Cloudflare managed challenge: plain HTTP clients (requests/httpx/scrapy) receive
a 403 challenge page because they do not execute the challenge script. A
headless Chromium driven by Playwright, with a stealth patch, renders the
challenge and returns the real content. This module wraps that into a small
fetch API that yields clean markdown plus provenance for each page.

The output per page is three files under the destination directory:

    <slug>.html   the extracted main-content HTML fragment
    <slug>.md     markdown (heading structure preserved) with a provenance header
    <slug>.json   provenance sidecar (source url, title, fetch time, byte sizes)

The markdown feeds the knowlytix document pipeline (convert -> build_rag_store)
unchanged; the json sidecar carries the source pointer back to the live URL.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify

# Content-container selectors, most specific first. The FFIEC manual wraps the
# section body in `#manualcontent`; the others are conservative fallbacks.
_CONTENT_SELECTORS = ("#manualcontent", "#content", "main", "article", "body")

# Chrome-like UA; the version tracks the bundled Chromium closely enough.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)

_CHALLENGE_MARKERS = ("CAPTCHA Error", "Just a moment", "cf-mitigated")


@dataclass
class CrawlResult:
    """Outcome of fetching one page."""

    url: str
    final_url: str
    title: str
    status: int
    fetched_at: str
    cleared_challenge: bool
    html_path: str
    markdown_path: str
    provenance_path: str
    n_chars_markdown: int


def _slug(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1] or "index"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", tail)


def _looks_like_challenge(title: str, html: str) -> bool:
    hay = f"{title}\n{html[:4000]}"
    return any(m in hay for m in _CHALLENGE_MARKERS)


def _extract_main(html: str) -> tuple[str, str]:
    """Return (main_content_html, page_title) from a rendered page."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(strip=True) if soup.title else "").strip()
    for sel in _CONTENT_SELECTORS:
        node = soup.select_one(sel)
        if node and len(node.get_text(strip=True)) > 500:
            # Drop non-content chrome that can live inside the container.
            for junk in node.select("script, style, nav, form, button"):
                junk.decompose()
            return str(node), title
    return str(soup.body or soup), title


def _to_markdown(main_html: str, *, title: str, source_url: str, fetched_at: str) -> str:
    body = markdownify(main_html, heading_style="ATX", strip=["a"]).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)  # collapse runaway blank lines
    header = (
        f"<!-- source: {source_url} -->\n"
        f"<!-- fetched: {fetched_at} -->\n\n"
        f"# {title}\n\n"
        f"> Source: {source_url}\n\n"
    )
    return header + body + "\n"


def _render(page, url: str, *, settle_secs: float, max_wait_secs: float) -> tuple[str, str, int, bool]:
    """Navigate and wait out any Cloudflare challenge. Returns html, title, status, cleared."""
    resp = page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    status = resp.status if resp else 0
    deadline = time.monotonic() + max_wait_secs
    html, title, cleared = page.content(), page.title(), False
    while time.monotonic() < deadline:
        html, title = page.content(), page.title()
        if not _looks_like_challenge(title, html) and len(html) > 3_000:
            cleared = True
            break
        time.sleep(settle_secs)
    return html, title, status, cleared


def fetch_pages(
    urls: list[str],
    dest_dir: str | Path,
    *,
    rate_limit_secs: float = 3.0,
    settle_secs: float = 5.0,
    max_wait_secs: float = 40.0,
) -> list[CrawlResult]:
    """Fetch several pages in one browser session, politely rate-limited.

    Uses one browser/context for the whole batch so the Cloudflare clearance
    cookie is reused across pages. Writes html/md/json per page under dest_dir.
    """
    from playwright.sync_api import sync_playwright

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    results: list[CrawlResult] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
        )
        ctx = browser.new_context(
            user_agent=_USER_AGENT,
            locale="en-US",
            viewport={"width": 1366, "height": 900},
        )
        try:
            from playwright_stealth import Stealth

            _stealth = Stealth()
        except Exception:
            _stealth = None

        for i, url in enumerate(urls):
            if i:
                time.sleep(rate_limit_secs)
            page = ctx.new_page()
            if _stealth is not None:
                _stealth.apply_stealth_sync(page)
            html, title, status, cleared = _render(
                page, url, settle_secs=settle_secs, max_wait_secs=max_wait_secs
            )
            final_url = page.url
            page.close()

            fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            main_html, extracted_title = _extract_main(html)
            title = extracted_title or title
            md = _to_markdown(main_html, title=title, source_url=url, fetched_at=fetched_at)

            slug = _slug(url)
            html_path = dest / f"{slug}.html"
            md_path = dest / f"{slug}.md"
            prov_path = dest / f"{slug}.json"
            html_path.write_text(main_html, encoding="utf-8")
            md_path.write_text(md, encoding="utf-8")

            result = CrawlResult(
                url=url,
                final_url=final_url,
                title=title,
                status=status,
                fetched_at=fetched_at,
                cleared_challenge=cleared,
                html_path=str(html_path),
                markdown_path=str(md_path),
                provenance_path=str(prov_path),
                n_chars_markdown=len(md),
            )
            prov_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
            results.append(result)

        browser.close()
    return results


def fetch_page(url: str, dest_dir: str | Path, **kwargs) -> CrawlResult:
    """Fetch a single page (convenience wrapper over `fetch_pages`)."""
    return fetch_pages([url], dest_dir, **kwargs)[0]
