"""Ingestion: source content -> markdown + provenance, ready for a GMS store.

The document path (PDF/TeX/MD) is reused from knowlytix; the piece built here is
the deterministic website path, since the target sources (e.g. the FFIEC
BSA/AML manual) sit behind a Cloudflare challenge that plain HTTP crawlers
cannot clear. `crawler` drives a real browser engine to render the page, then
extracts and converts the main content.
"""

from reasonloop.ingest.crawler import CrawlResult, fetch_page, fetch_pages

__all__ = ["CrawlResult", "fetch_page", "fetch_pages"]
