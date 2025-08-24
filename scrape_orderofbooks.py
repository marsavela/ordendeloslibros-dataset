#!/usr/bin/env python3
"""
Enhanced OrderOfBooks Scraper with Multiple Orderings Support
=============================================================

A comprehensive scraper that extracts book series data from orderofbooks.com with support
for multiple ordering types (publication, chronological, companion series, etc.).

Key Features:
- Scrapes all series and author pages from orderofbooks.com
- Captures multiple orderings per series (publication vs chronological)
- Concurrent processing for high performance
- Supports resume functionality to continue interrupted scrapes
- Outputs structured JSON data compatible with series matching tools

Output Structure:
- `index.json`: Master index of all series and authors
- `data/series/`: Individual JSON files for each series with multiple orderings
- Each series file contains: name, slug, source URL, image, authors, and orderings

Usage Examples:
```bash
# Scrape all series with multiple orderings
python scrape_orderofbooks.py --only-series

# Scrape specific series
python scrape_orderofbooks.py --only-series --include "harry-potter,narnia"

# Resume interrupted scrape
python scrape_orderofbooks.py --resume --concurrency 30
```
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup
from slugify import slugify
from tqdm import tqdm
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ---------------- Defaults ----------------
CHAR_INDEX = "https://www.orderofbooks.com/characters/"
AUTH_INDEX = "https://www.orderofbooks.com/authors/"
CHAR_PAGES_DEFAULT = 55
AUTH_PAGES_DEFAULT = 101

OUT_BASE_DEFAULT = Path("data")
CONSOLIDATED_INDEX_DEFAULT = Path("index.json")
SERIES_LIST_DEFAULT = Path("series.json")
AUTHORS_LIST_DEFAULT = Path("authors.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ---------------- CLI ----------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scrape OrderOfBooks indexes (JS pager) and details (books order) into Hugo‑friendly JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Phases
    ph = p.add_argument_group("Phases")
    ph.add_argument("--skip-index", action="store_true", help="Skip index discovery (read from --index-path or authors/series lists)")
    ph.add_argument("--skip-details", action="store_true", help="Skip detail crawling (only write index files)")

    # Inputs/outputs
    io = p.add_argument_group("Inputs/Outputs")
    io.add_argument("--index-path", type=Path, default=CONSOLIDATED_INDEX_DEFAULT, help="Path to consolidated index.json to read/write")
    io.add_argument("--authors-list", type=Path, default=AUTHORS_LIST_DEFAULT, help="Path to authors.json (raw list)")
    io.add_argument("--series-list", type=Path, default=SERIES_LIST_DEFAULT, help="Path to series.json (raw list)")
    io.add_argument("--out-dir", type=Path, default=OUT_BASE_DEFAULT, help="Base output directory for per‑item JSONs (data/")
    io.add_argument("--save-debug-lists", action="store_true", help="Write authors.json and series.json after index phase")

    # Selection filters
    sel = p.add_argument_group("Selection")
    sel.add_argument("--only-authors", action="store_true", help="Process authors only")
    sel.add_argument("--only-series", action="store_true", help="Process series only")
    sel.add_argument("--include", type=str, default="", help="Comma‑separated slugs or names to include (others skipped)")
    sel.add_argument("--exclude", type=str, default="", help="Comma‑separated slugs or names to exclude")
    sel.add_argument("--limit-authors", type=int, default=0, help="Limit number of authors to process (0 = no limit)")
    sel.add_argument("--limit-series", type=int, default=0, help="Limit number of series to process (0 = no limit)")
    sel.add_argument("--resume", action="store_true", help="Skip items whose output file already exists")

    # Performance / behavior
    perf = p.add_argument_group("Performance")
    perf.add_argument("--concurrency", type=int, default=24, help="Concurrent detail fetchers")
    perf.add_argument("--request-timeout", type=int, default=30, help="Per‑request timeout (seconds)")
    perf.add_argument("--headed", action="store_true", help="Run Playwright in headed mode (visible Chromium)")
    perf.add_argument("--char-pages", type=int, default=CHAR_PAGES_DEFAULT, help="Expected Characters/Series pager count")
    perf.add_argument("--auth-pages", type=int, default=AUTH_PAGES_DEFAULT, help="Expected Authors pager count")
    perf.add_argument("--debug-concurrency", action="store_true", help="Print active detail task counts (diagnostic)")
    # Hidden expansion always on; legacy flag removed.

    # Output / logging
    outg = p.add_argument_group("Output")
    outg.add_argument("--quiet", action="store_true", help="Reduce output (suppress index logs & progress bars, only errors and final summary)")

    return p

# ---------------- Helpers ----------------

def now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def extract_year(text: str) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", text)
    return int(m.group(0)) if m else None


def absolute_url(base: str, maybe: str) -> str:
    try:
        return urljoin(base, maybe)
    except Exception:
        return maybe


def normalize_series_name(heading_text: str) -> str:
    t = clean_text(heading_text)
    t = re.sub(r"(?i)\b(books?|novels?|series)\b\s*in\s*order.*$", "", t)
    t = re.sub(r"(?i)\breading\s*order.*$", "", t)
    t = t.strip(" :–-—")
    return t or heading_text.strip()


def determine_ordering_type(heading_text: str) -> str:
    """Determine the ordering type from heading text.
    
    OrderOfBooks typically has headings like:
    - "Publication Order of The Chronicles Of Narnia Books"
    - "Chronological Order of The Chronicles Of Narnia Books"
    """
    text_lower = heading_text.lower()
    if "chronological" in text_lower or "chronology" in text_lower:
        return "chronological"
    elif "publication" in text_lower:
        return "publication"
    elif "companion" in text_lower:
        return "companion"
    elif "world of" in text_lower:
        return "world"
    else:
        return "publication"  # Default assumption for orderofbooks.com


def cleaned_index_name(link_text: str) -> str:
    name = re.sub(r"(?i)^order\s+of\s+", "", link_text.strip())
    name = re.sub(r"(?i)\s+books?$", "", name)
    return name.strip(" –-—")


def extract_series_authors(soup: BeautifulSoup) -> List[str]:
    """Heuristically extract author name(s) for a series page.

    Strategy (lightweight, no external requests):
      1. Look for the first <p> inside #content mentioning 'by <Name>' after the main H1/H2.
      2. Fallback: parse the first heading's text for ' by ' pattern.
      3. Fallback: derive from hero image alt/src if it contains a capitalized name phrase.
    Returns a list of distinct author names (may be multi‑author collaborations).
    """
    content = soup.find(id="content") or soup
    authors: List[str] = []

    def add(name: str):
        n = clean_text(name)
        if not n:
            return
        # Basic sanity: must contain a space and at least one lowercase letter after first char.
        if len(n.split()) > 1 and n not in authors and len(n) <= 80:
            authors.append(n)

    # 1) Paragraph with ' by ' pattern
    for p in content.find_all("p", limit=4):  # just early intro paragraphs
        txt = clean_text(p.get_text(" ", strip=True))
        m = re.search(r"\bby\s+([A-Z][A-Za-z'’.\-]+(?:\s+[A-Z][A-Za-z'’.\-]+){0,4})", txt)
        if m:
            add(m.group(1))
            break

    # 2) Heading pattern
    if not authors:
        h1 = content.find(["h1", "h2"]) or soup.find(["h1", "h2"])  # main heading
        if h1:
            htxt = clean_text(h1.get_text(" ", strip=True))
            m = re.search(r"\bby\s+([A-Z][A-Za-z'’.\-]+(?:\s+[A-Z][A-Za-z'’.\-]+){0,4})", htxt)
            if m:
                add(m.group(1))

    # 3) Image alt or filename hint
    if not authors:
        img = content.find("img", alt=True) or content.find("img")
        if img:
            alt = clean_text(img.get("alt") or "")
            m = re.search(r"by\s+([A-Z][A-Za-z'’.\-]+(?:\s+[A-Z][A-Za-z'’.\-]+){0,4})", alt)
            if m:
                add(m.group(1))
            else:
                # Try from filename segments (e.g. ...-by-Ellery-Adams-500x200.jpg)
                src = img.get("src", "")
                fm = re.search(r"-by-([A-Za-z-]+)-\d", src)
                if fm:
                    parts = [w for w in fm.group(1).split('-') if w]
                    if parts:
                        guess = " ".join(p.capitalize() for p in parts)
                        add(guess)

    return authors

# ---------------- Playwright index scraper ----------------

async def extract_items_from_index_html(html: str, section_root: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find(id="content") or soup
    out: List[Dict] = []
    for a in content.find_all("a", href=True):
        text = (a.get_text() or "").strip()
        href = a["href"].strip()
        if not href.startswith(section_root):
            continue
        if not re.match(r"(?i)^order\s+of\b", text):
            continue
        out.append({
            "title": text,
            "name": cleaned_index_name(text),
            "link": href if href.endswith("/") else href + "/",
        })
    return out


async def scrape_index_section(context, index_url: str, expected_pages: int, label: str, headed: bool, quiet: bool) -> List[Dict]:
    section_root = index_url.rstrip("/") + "/"
    page = await context.new_page()
    page.set_default_timeout(30000)

    if not quiet:
        print(f"[INDEX] {label}: open {index_url}", file=sys.stderr)
    await page.goto(index_url)

    async def content_html() -> str:
        return await page.locator("#content").inner_html()

    def top_pager_btn(page_no: int):
        return page.locator('.show_links_pagination').first.locator(f'a[data-page="{page_no}"]').first

    items_by_url: Dict[str, Dict] = {}

    # Page 1
    html0 = await content_html()
    for it in await extract_items_from_index_html(html0, section_root):
        items_by_url[it["link"]] = it
    if not quiet:
        print(f"[INDEX] {label}: page 1 -> {len(items_by_url)} items", file=sys.stderr)

    # Subsequent pages
    for pno in range(2, expected_pages + 1):
        if await top_pager_btn(pno).count() == 0:
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(120)
            if await top_pager_btn(pno).count() == 0:
                alt = page.locator('.show_links_pagination').nth(1).locator(f'a[data-page="{pno}"]').first
                if await alt.count() == 0:
                    if not quiet:
                        print(f"[INDEX] {label}: skip p{pno} (button not found)", file=sys.stderr)
                    continue
                btn = alt
            else:
                btn = top_pager_btn(pno)
        else:
            btn = top_pager_btn(pno)

        try:
            await btn.scroll_into_view_if_needed()
            await page.evaluate("(el) => el.click()", await btn.element_handle())
            prev = html0
            await page.wait_for_function(
                "(prev) => { const el = document.querySelector('#content'); return el && el.innerHTML !== prev; }",
                arg=prev,
                timeout=20000,
            )
        except PWTimeout:
            if not quiet:
                print(f"[INDEX] {label}: timeout on p{pno}; continuing", file=sys.stderr)
        html0 = await content_html()
        batch = await extract_items_from_index_html(html0, section_root)
        for it in batch:
            items_by_url[it["link"]] = it
        if not quiet:
            print(f"[INDEX] {label}: page {pno} -> +{len(batch)} (total {len(items_by_url)})", file=sys.stderr)

    await page.close()
    return sorted(items_by_url.values(), key=lambda x: x["name"].lower())


async def run_index_phase(headed: bool, char_pages: int, auth_pages: int,
                          authors_list_path: Path, series_list_path: Path,
                          save_debug_lists: bool, quiet: bool) -> Tuple[List[Dict], List[Dict]]:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not headed)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        series = await scrape_index_section(context, CHAR_INDEX, char_pages, "Characters/Series", headed, quiet)
        authors = await scrape_index_section(context, AUTH_INDEX, auth_pages, "Authors", headed, quiet)
        await context.close()
        await browser.close()
    if save_debug_lists:
        series_list_path.write_text(json.dumps(series, ensure_ascii=False, indent=2), encoding="utf-8")
        authors_list_path.write_text(json.dumps(authors, ensure_ascii=False, indent=2), encoding="utf-8")
    return series, authors

# ---------------- Detail parsing helpers ----------------

def book_from_text(txt: str) -> Optional[Dict]:
    """Create a book dict from raw list/text entry.

    Returns None for known non‑book placeholder rows such as
    "+ Show All Books in this Series" that appear on some pages.
    """
    txt = clean_text(txt)
    # Skip JS expansion placeholder lines (we don't have the dynamic expansion content anyway)
    if re.search(r"(?i)show all books in this series", txt):
        return None
    # Remove leading index markers like "1." or "12)" etc.
    txt = re.sub(r"^\s*\d+[\.!\)]\s*", "", txt)
    year = extract_year(txt)
    title = re.sub(r"\((19|20)\d{2}\)", "", txt).strip(" -–—")
    return {"title": title, "year": year, "links": {"amazon_es": ""}}


def parse_table_block(tbl) -> List[Dict]:
    # Parse a ribbon/list table layout.
    # Hidden rows (class="hiderow") are appended after the visible ones to
    # reflect the order after user expands the section. Skip the control row
    # containing the "+ Show All Books in this Series" trigger and any text
    # variations thereof.
    visibles: List[Dict] = []
    hiddens: List[Dict] = []
    for tr in tbl.select("tr"):
        if tr.find(class_="showall"):
            continue
        classes = tr.get("class", []) or []
        tt = tr.select_one("td.booktitle")
        yy = tr.select_one("td.bookyear")
        if not tt:
            tds = tr.find_all("td")
            if not tds:
                continue
            if len(tds) == 1 and re.search(r"(?i)show all books in this series", tds[0].get_text(" ", strip=True)):
                continue
            tt = tds[0]
            yy = tds[1] if len(tds) > 1 else None
        raw_title = clean_text(tt.get_text(" ", strip=True))
        # Remove any trailing expansion hint accidentally captured inside title cell
        raw_title = re.sub(r"\s*\+\s*Show All Books in this Series\s*$", "", raw_title, flags=re.IGNORECASE)
        if re.search(r"(?i)show all books in this series", raw_title):
            continue
        if not raw_title:
            continue
        year = extract_year(yy.get_text(" ", strip=True)) if yy else None
        book = {"title": raw_title, "year": year, "links": {"amazon_es": ""}}
        (hiddens if any(c == "hiderow" for c in classes) else visibles).append(book)
    return visibles + hiddens


def find_ribbon_blocks(soup: BeautifulSoup) -> List[Tuple[Optional[str], List[Dict]]]:
    out: List[Tuple[Optional[str], List[Dict]]] = []
    for ribbon in soup.select("div.ribbon"):
        series_name = None
        cand = ribbon.find_previous(lambda t: t.name in ("h2", "h3", "h4"))
        while cand:
            raw_txt = cand.get_text(" ", strip=True)
            norm = normalize_series_name(raw_txt)
            if not re.match(r"^\(with\b", norm, re.IGNORECASE) and not norm.startswith("("):
                series_name = norm
                break
            # Try earlier heading (e.g., skip the byline heading)
            cand = cand.find_previous(lambda t: t.name in ("h2", "h3", "h4"))
        # Fallback: if all we saw were byline style headings, keep the last norm
        if not series_name and cand:
            series_name = norm
        lst = ribbon.find_next_sibling("div", class_="list") or ribbon.find_next("div", class_="list")
        if not lst:
            continue
        tbl = lst.find("table")
        if not tbl:
            ulol = lst.find(lambda t: t.name in ("ul", "ol"))
            raw_books = [book_from_text(li.get_text(" ", strip=True)) for li in ulol.find_all("li", recursive=False)] if ulol else []
            books = [b for b in raw_books if b]
        else:
            books = parse_table_block(tbl)
        if books:
            out.append((series_name, books))
    return out


def find_heading_list_blocks(soup: BeautifulSoup) -> List[Tuple[str, List[Dict]]]:
    results: List[Tuple[str, List[Dict]]] = []
    for h in soup.select("h2, h3, h4"):
        htxt = clean_text(h.get_text(" ", strip=True))
        if not re.search(r"(?i)(book series in order|series in order|books in order|reading order)", htxt):
            continue
        name = normalize_series_name(htxt)
        lst = h.find_next(lambda tag: tag.name in ("ol", "ul"))
        if not lst:
            continue
        raw_books = [book_from_text(li.get_text(" ", strip=True)) for li in lst.find_all("li", recursive=False)]
        books = [b for b in raw_books if b]
        if books:
            results.append((name, books))
    return results


def find_representative_image(soup: BeautifulSoup, page_url: str) -> Optional[str]:
    og = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
    if og and og.get("content"):
        return absolute_url(page_url, og["content"])
    content = soup.find(id="content") or soup
    for img in content.find_all("img", src=True):
        src = img.get("src", "").strip()
        w = (img.get("width") or "").strip()
        h = (img.get("height") or "").strip()
        try:
            wv = int(w) if w else 0
            hv = int(h) if h else 0
        except ValueError:
            wv = hv = 0
        if wv < 120 and hv < 120:
            continue
        return absolute_url(page_url, src)
    return None

# ---------------- Detail fetch ----------------

async def fetch_html(session: aiohttp.ClientSession, url: str, timeout_s: int) -> Optional[str]:
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with session.get(url, headers=HEADERS, timeout=timeout) as r:
            if r.status != 200:
                return None
            return await r.text()
    except Exception:
        return None


def base_payload(kind: str, name: str, slug: str, source: str, image: Optional[str]) -> Dict:
    return {
        "type": kind,
        "name": name,
        "slug": slugify(slug),
        "source": source,
        "image": image,
    }


def finalize_series_payload(name: str, link: str, soup: BeautifulSoup) -> Dict:
    image = find_representative_image(soup, link)
    payload = base_payload("series", name, name, link, image)
    # Attempt to extract author name(s) from early page content (heading, intro paragraphs, image filename)
    authors = extract_series_authors(soup)
    if authors:
        payload["authors"] = [{"name": a, "slug": slugify(a)} for a in authors]
    
    # Capture ALL orderings, not just the first one
    rib = find_ribbon_blocks(soup)
    orderings = {}
    
    if rib:
        # Process all ribbon blocks (multiple orderings)
        for series_name, books in rib:
            if not books:
                continue
            # Determine ordering type from the heading text
            ordering_type = determine_ordering_type(series_name or "")
            # Ensure unique ordering keys
            base_key = ordering_type
            counter = 1
            key = base_key
            while key in orderings:
                counter += 1
                key = f"{base_key}_{counter}"
            
            # Add index to books
            for i, b in enumerate(books, 1):
                b["index"] = i
            orderings[key] = {
                "heading": series_name,
                "books": books
            }
    else:
        # Fallback to older heading + list layout
        head = find_heading_list_blocks(soup)
        if head:
            for series_name, books in head:
                if not books:
                    continue
                ordering_type = determine_ordering_type(series_name or "")
                base_key = ordering_type
                counter = 1
                key = base_key
                while key in orderings:
                    counter += 1
                    key = f"{base_key}_{counter}"
                
                for i, b in enumerate(books, 1):
                    b["index"] = i
                orderings[key] = {
                    "heading": series_name,
                    "books": books
                }
    
    # If we found orderings, use them; otherwise create a default empty one
    if orderings:
        payload["orderings"] = orderings
    else:
        # Fallback to old format for compatibility
        payload["books"] = []
    
    return payload


def finalize_author_payload(name: str, link: str, soup: BeautifulSoup) -> Dict:
    image = find_representative_image(soup, link)
    payload = base_payload("author", name, name, link, image)
    series_out: List[Dict] = []
    rib = find_ribbon_blocks(soup)
    if rib:
        for idx, (series_name, books) in enumerate(rib, 1):
            sname = series_name or f"Series {idx}"
            for i, b in enumerate(books, 1):
                b["index"] = i
            series_out.append({
                "name": sname,
                "slug": slugify(sname),
                "order_note": "Order shown on OrderOfBooks",
                "books": books,
            })
    else:
        head = find_heading_list_blocks(soup)
        for (series_name, books) in head:
            for i, b in enumerate(books, 1):
                b["index"] = i
            series_out.append({
                "name": series_name,
                "slug": slugify(series_name),
                "order_note": "Order shown on OrderOfBooks",
                "books": books,
            })
    # Reclassify any single-book series as standalone titles
    multi_series: List[Dict] = []
    standalone_books: List[Dict] = []
    for s in series_out:
        books = s.get("books", [])
        if len(books) == 1:
            # Take the lone book and treat as standalone. Remove its per-series index later.
            book = books[0].copy()
            # A single-book "series" usually has index 1; we'll recompute standalone indices.
            book.pop("index", None)
            standalone_books.append(book)
        else:
            multi_series.append(s)
    # Assign indices for standalone list (in discovered order)
    for i, b in enumerate(standalone_books, 1):
        b["index"] = i
    payload["series"] = multi_series
    payload["standalone"] = standalone_books
    return payload


_detail_pw_ctx: Dict[str, Optional[object]] = {"playwright": None, "browser": None, "context": None, "lock": asyncio.Lock()}


async def _ensure_detail_browser(headless: bool = True):
    if _detail_pw_ctx["context"]:
        return _detail_pw_ctx["context"]
    async with _detail_pw_ctx["lock"]:
        if _detail_pw_ctx["context"]:
            return _detail_pw_ctx["context"]
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context()
        _detail_pw_ctx.update({"playwright": pw, "browser": browser, "context": context})
        return context


# JS expansion removed: static HTML already includes hidden rows via <tr class="hiderow">.


async def process_one(session: aiohttp.ClientSession, item: Dict, kind: str, outdir: Path, timeout_s: int, resume: bool) -> Optional[Tuple[str, Dict]]:
    name, link = item["name"], item["link"]
    slug = slugify(name)
    path = outdir / f"{slug}.json"
    if resume and path.exists():
        return slug, {"name": name, "slug": slug, "link": link}

    html = await fetch_html(session, link, timeout_s)
    if not html:
        return None
    # No JS expansion necessary.
    soup = BeautifulSoup(html, "html.parser")
    if kind == "series":
        payload = finalize_series_payload(name, link, soup)
    else:
        payload = finalize_author_payload(name, link, soup)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return slug, {"name": name, "slug": slug, "link": link}


async def crawl_all(kind: str, inputs: List[Dict], outdir: Path, concurrency: int, timeout_s: int, resume: bool, debug_conc: bool = False, quiet: bool = False) -> List[Dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    results: List[Dict] = []
    active = 0
    peak = 0
    lock = asyncio.Lock()
    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(inputs), desc=f"{kind}", unit="page", disable=quiet)

        async def run_one(it):
            async with semaphore:
                nonlocal active, peak
                if debug_conc and not quiet:
                    async with lock:
                        active += 1
                        if active > peak:
                            peak = active
                            print(f"[DEBUG] {kind} concurrency peak now {peak}")
                        else:
                            # Occasionally report current active every 25 processed items.
                            if (pbar.n % 25) == 0:
                                print(f"[DEBUG] {kind} active={active} peak={peak}")
                try:
                    res = await process_one(session, it, kind, outdir, timeout_s, resume)
                finally:
                    if debug_conc and not quiet:
                        async with lock:
                            active -= 1
                pbar.update(1)
                if res:
                    _slug, idx = res
                    results.append(idx)

        tasks = [asyncio.create_task(run_one(it)) for it in inputs]
        await asyncio.gather(*tasks)
        pbar.close()
    return results

# ---------------- Filtering ----------------

def to_slug_set(csv: str) -> set[str]:
    items = [s.strip() for s in csv.split(",") if s.strip()]
    return {slugify(s) for s in items}


def filter_items(items: List[Dict], include_csv: str, exclude_csv: str, limit: int) -> List[Dict]:
    inc = to_slug_set(include_csv)
    exc = to_slug_set(exclude_csv)
    out = []
    for it in items:
        s = slugify(it["name"])
        if inc and s not in inc and it["name"].lower() not in {x.replace('-', ' ') for x in inc}:
            continue
        if exc and (s in exc or it["name"].lower() in {x.replace('-', ' ') for x in exc}):
            continue
        out.append(it)
        if limit and len(out) >= limit:
            break
    return out

# ---------------- Main ----------------

async def main():
    args = build_arg_parser().parse_args()

    # No expansion runtime flag needed.

    out_base = args.out_dir
    out_series_dir = out_base / "series"
    out_authors_dir = out_base / "authors"

    # Phase 1: Index discovery (or load from disk)
    if not args.skip_index:
        series, authors = await run_index_phase(
            headed=args.headed,
            char_pages=args.char_pages,
            auth_pages=args.auth_pages,
            authors_list_path=args.authors_list,
            series_list_path=args.series_list,
            save_debug_lists=args.save_debug_lists,
            quiet=args.quiet,
        )
        consolidated = {
            "series": sorted(
                [{"name": s["name"], "slug": slugify(s["name"]), "link": s["link"]} for s in series],
                key=lambda x: x["name"].lower(),
            ),
            "authors": sorted(
                [{"name": a["name"], "slug": slugify(a["name"]), "link": a["link"]} for a in authors],
                key=lambda x: x["name"].lower(),
            ),
        }
        args.index_path.write_text(json.dumps(consolidated, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        # Load from --index-path or authors/series list files
        if args.index_path.exists():
            data = json.loads(args.index_path.read_text(encoding="utf-8"))
            series = data.get("series", [])
            authors = data.get("authors", [])
        elif args.series_list.exists() and args.authors_list.exists():
            series = json.loads(args.series_list.read_text(encoding="utf-8"))
            authors = json.loads(args.authors_list.read_text(encoding="utf-8"))
        else:
            print("ERROR: --skip-index was set but no usable index/authors/series files were found.", file=sys.stderr)
            sys.exit(1)

    # Phase 2: Detail crawl (unless skipped)
    if args.skip_details:
        print("Index phase complete; skipping details as requested.")
        return

    # Section selection
    do_series = True
    do_authors = True
    if args.only_series:
        do_authors = False
    if args.only_authors:
        do_series = False

    # De‑dup by link
    seen = set()
    series = [s for s in series if not (s["link"] in seen or seen.add(s["link"]))]
    seen.clear()
    authors = [a for a in authors if not (a["link"] in seen or seen.add(a["link"]))]

    # Filters & limits
    if do_series:
        series = filter_items(series, args.include, args.exclude, args.limit_series)
    else:
        series = []
    if do_authors:
        authors = filter_items(authors, args.include, args.exclude, args.limit_authors)
    else:
        authors = []

    # Crawl
    if series:
        await crawl_all("series", series, out_series_dir, args.concurrency, args.request_timeout, args.resume, args.debug_concurrency, args.quiet)
    if authors:
        await crawl_all("author", authors, out_authors_dir, args.concurrency, args.request_timeout, args.resume, args.debug_concurrency, args.quiet)

    if not args.quiet:
        print("Done. Per‑item JSONs under data/, consolidated index at", args.index_path)
    else:
        # Minimal newline to ensure CI step ends cleanly
        print("done", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())