#!/usr/bin/env python3
"""
scrape_shl.py:- scraper for SHL assessments
#run command:- python scrape_shl_assessments.py --output catalog_clean.json --concurrency 6 --delay 0.25        
"""

import asyncio
import json
import re
from urllib.parse import urljoin, urlencode, urlparse
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from tqdm.asyncio import tqdm_asyncio

# ---------- Constants ----------
LISTING_BASE = "https://www.shl.com/products/product-catalog/"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 SHL-AsyncFast/1.0"

LISTING_TABLE_SELECTOR = "div.custom__table-wrapper table"
PRODUCT_READY_SELECTOR = "h1"
LISTING_WAIT_TIMEOUT = 8000
PRODUCT_WAIT_TIMEOUT = 8000
NAV_TIMEOUT = 45000
SHORT_WAIT = 0.08

_re_minutes = re.compile(r"(\d{1,3})\s*(?:minutes|mins|min)\b", re.I)
_re_eqnum = re.compile(r"=\s*(\d{1,3})")

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}

# ---------- Utilities ----------
def safe_text(node):
    return node.get_text(" ", strip=True) if node else ""

def parse_minutes_from_text(text):
    if not text:
        return None
    m = _re_minutes.search(text)
    if m:
        return int(m.group(1))
    m = _re_eqnum.search(text)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d{1,3})", text)
    if m2:
        return int(m2.group(1))
    return None

def slug_from_string(s):
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"https?://", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:100]

def _clean_test_letters_from_spans(spans):
    out = []
    for ks in spans:
        txt = ks.get_text(" ", strip=True)
        if not txt:
            continue
        letters = re.findall(r"[A-Za-z]", txt)
        for L in letters:
            L = L.upper()
            if len(L) == 1 and L not in out:
                out.append(L)
    return out

def _string_to_list_field(s):
    if not s:
        return []
    parts = re.split(r"[;,]\s*|\n", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"\s*,\s*$", "", p)
        if p not in out:
            out.append(p)
    return out

# ---------- Listing HTML Parsing ----------
def extract_rows_from_listing_html(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    wrappers = soup.find_all("div", class_=re.compile(r"custom__table-wrapper|js-target-table-wrapper|custom__table-responsive", re.I))
    rows_out = []
    for w in wrappers:
        table = w.find("table")
        if not table:
            continue
        header_ths = table.find_all("th")
        header_texts = " ".join([th.get_text(" ", strip=True) for th in header_ths])
        if not re.search(r"Individual\s+Test\s+Solutions", header_texts, flags=re.I):
            continue
        trows = table.find_all("tr", attrs={"data-entity-id": True})
        for tr in trows:
            a = tr.find("a", href=True)
            if not a:
                continue
            href = a["href"].strip()
            url = urljoin(base_url, href.split("#")[0])
            tds = tr.find_all("td")
            remote = "No"
            adaptive = "No"
            try:
                if len(tds) >= 2 and tds[1].find("span", class_=re.compile(r"catalogue__circle.*-yes")):
                    remote = "Yes"
                if len(tds) >= 3 and tds[2].find("span", class_=re.compile(r"catalogue__circle.*-yes")):
                    adaptive = "Yes"
            except Exception:
                pass

            test_types = []
            try:
                if len(tds) >= 4:
                    test_td = tds[3]
                    spans = test_td.find_all("span", class_=re.compile(r"product-catalogue_key|product-catalogue__key", re.I))
                    test_types = _clean_test_letters_from_spans(spans)
                if not test_types:
                    label_el = tr.find(lambda tag: tag.name in ("p","div","span") and "test type" in tag.get_text(" ", strip=True).lower())
                    if label_el:
                        spans = label_el.find_all("span", class_=re.compile(r"product-catalogue_key|product-catalogue__key", re.I))
                        if not spans:
                            nxt = label_el.find_next_sibling()
                            if nxt:
                                spans = nxt.find_all("span", class_=re.compile(r"product-catalogue_key|product-catalogue__key", re.I))
                        test_types = _clean_test_letters_from_spans(spans)
            except Exception:
                test_types = []

            rows_out.append({"url": url, "remote_support": remote, "adaptive_support": adaptive, "test_type": test_types})
        if rows_out:
            return rows_out

    soup2 = BeautifulSoup(html, "html.parser")
    for a in soup2.find_all("a", href=True):
        href = a["href"].strip()
        if "/product-catalog/view" in href or ("/product-catalog" in href and "/view/" in href):
            rows_out.append({"url": urljoin(base_url, href), "remote_support": "Unknown", "adaptive_support": "Unknown", "test_type": []})
    return rows_out

# ---------- Product Page Parsing ----------
def parse_product_page_fields(html):
    soup = BeautifulSoup(html, "html.parser")
    name = ""
    h1 = soup.find("h1")
    if h1:
        name = safe_text(h1)
    elif soup.title:
        name = soup.title.string.strip()

    description = ""
    job_levels = ""
    languages = ""
    duration = None

    blocks = soup.find_all("div", class_=re.compile(r"product-catalogue-training-calendar__row|product-catalogue-training-calendar_row|product-catalogue-training-calendar__row", re.I))
    for block in blocks:
        h4 = block.find("h4")
        if not h4:
            continue
        key = h4.get_text(" ", strip=True).lower()
        p = block.find("p")
        val = safe_text(p) if p else ""
        if "description" in key and val:
            description = val
        elif "job levels" in key and val:
            job_levels = val
        elif "languages" in key and val:
            languages = val
        elif "assessment length" in key or "assessment duration" in key or "assessment time" in key:
            dur = parse_minutes_from_text(val)
            if dur:
                duration = dur

    if not (description or job_levels or languages or duration):
        all_h4 = soup.find_all("h4")
        for h4 in all_h4:
            key = h4.get_text(" ", strip=True).lower()
            p = h4.find_next_sibling("p")
            val = safe_text(p) if p else ""
            if "description" in key and val and not description:
                description = val
            elif "job levels" in key and val and not job_levels:
                job_levels = val
            elif "languages" in key and val and not languages:
                languages = val
            elif ("assessment length" in key or "assessment" in key) and val and not duration:
                d = parse_minutes_from_text(val)
                if d:
                    duration = d

    page_test_types = []
    try:
        test_label_el = soup.find(lambda tag: tag.name in ("p", "div", "span") and "test type" in tag.get_text(" ", strip=True).lower())
        if test_label_el:
            spans = test_label_el.find_all("span", class_=re.compile(r"product-catalogue_key|product-catalogue__key", re.I))
            if not spans:
                nxt = test_label_el.find_next_sibling()
                if nxt:
                    spans = nxt.find_all("span", class_=re.compile(r"product-catalogue_key|product-catalogue__key", re.I))
            page_test_types = _clean_test_letters_from_spans(spans)
        else:
            spans = soup.find_all("span", class_=re.compile(r"product-catalogue_key|product-catalogue__key", re.I))
            page_test_types = _clean_test_letters_from_spans(spans)
    except Exception:
        page_test_types = []

    adaptive_page = "No"
    remote_page = "No"
    for sp in soup.find_all("span", class_=re.compile(r"catalogue__circle", re.I)):
        cls = " ".join(sp.get("class") or [])
        parent_text = safe_text(sp.find_parent())
        if re.search(r"remote", parent_text, flags=re.I) and re.search(r"-yes", cls):
            remote_page = "Yes"
        if re.search(r"adaptive|irt", parent_text, flags=re.I) and re.search(r"-yes", cls):
            adaptive_page = "Yes"

    return {
        "name": name,
        "description": description or "",
        "job_levels": job_levels or "",
        "languages": languages or "",
        "duration": duration,
        "page_test_types": page_test_types,
        "adaptive_page": adaptive_page,
        "remote_page": remote_page
    }

# ---------- Playwright Route Handler ----------
async def route_handler(route):
    req = route.request
    typ = req.resource_type
    if typ in ("image", "media", "font", "stylesheet"):
        await route.abort()
        return
    url = req.url
    if any(domain in url for domain in ("google-analytics", "doubleclick", "googlesyndication", "vimeo", "facebook", "quantserve")):
        await route.abort()
        return
    await route.continue_()

# ---------- Fetch Listing Rows ----------
async def fetch_listing_rows(context, offset, delay):
    page = await context.new_page()
    params = {"type": "1", "start": str(offset)}
    url = LISTING_BASE + "?" + urlencode(params)
    try:
        await page.goto(url, timeout=NAV_TIMEOUT)
        try:
            await page.wait_for_selector(LISTING_TABLE_SELECTOR, timeout=LISTING_WAIT_TIMEOUT)
        except PWTimeout:
            pass
        await asyncio.sleep(SHORT_WAIT)
        html = await page.content()
    finally:
        await page.close()
    rows = extract_rows_from_listing_html(html, url)
    await asyncio.sleep(delay)
    return rows

# ---------- Fetch Product Item ----------
async def fetch_product_item(context, info):
    url = info["url"]
    page = await context.new_page()
    try:
        await page.goto(url, timeout=NAV_TIMEOUT)
        try:
            await page.wait_for_selector(PRODUCT_READY_SELECTOR, timeout=PRODUCT_WAIT_TIMEOUT)
        except PWTimeout:
            pass
        await asyncio.sleep(SHORT_WAIT)
        html = await page.content()
    except Exception as e:
        await page.close()
        return {"error": str(e), "url": url}
    await page.close()
    parsed = parse_product_page_fields(html)

    page_types = parsed.get("page_test_types") or []
    if page_types:
        test_letters = page_types
    else:
        raw_listing_types = info.get("test_type") or []
        clean_listing = []
        for v in raw_listing_types:
            letters = re.findall(r"[A-Za-z]", v)
            for L in letters:
                L = L.upper()
                if len(L) == 1 and L not in clean_listing:
                    clean_listing.append(L)
        test_letters = clean_listing

    test_type_names = [TEST_TYPE_MAP.get(c, c) for c in test_letters]

    job_levels_list = _string_to_list_field(parsed.get("job_levels", ""))
    languages_list = _string_to_list_field(parsed.get("languages", ""))

    duration = parsed.get("duration")
    if duration is not None:
        try:
            duration = int(duration)
        except Exception:
            duration = None

    adaptive = info.get("adaptive_support") if info.get("adaptive_support") in ("Yes", "No") else parsed.get("adaptive_page", "No")
    remote = info.get("remote_support") if info.get("remote_support") in ("Yes", "No") else parsed.get("remote_page", "No")
    adaptive = "Yes" if str(adaptive).strip().lower() == "yes" else "No"
    remote = "Yes" if str(remote).strip().lower() == "yes" else "No"

    pid = ""
    try:
        parsed_url = urlparse(url)
        seg = parsed_url.path.rstrip("/").split("/")[-1]
        if seg:
            pid = slug_from_string(seg)
    except Exception:
        pid = ""
    if not pid:
        pid = slug_from_string(parsed.get("name") or url)

    search_parts = [
        parsed.get("name", ""),
        parsed.get("description", ""),
        " ".join(job_levels_list),
        " ".join(languages_list),
        " ".join(test_letters)
    ]
    search_text = " ".join([p for p in search_parts if p]).strip().lower()

    item = {
        "product_id": pid,
        "url": url,
        "name": parsed.get("name") or "",
        "description": parsed.get("description") or "",
        "job_levels": job_levels_list,
        "languages": languages_list,
        "duration": duration,
        "adaptive_support": adaptive,
        "remote_support": remote,
        "test_type": test_letters,
        "test_type_names": test_type_names,
        "search_text": search_text
    }
    return item

# ---------- Main Scrape Flow ----------
async def main(output="catalog_clean.json", delay=0.25, concurrency=6):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = await browser.new_context(user_agent=USER_AGENT, locale="en-US")
        await context.route("**/*", route_handler)

        collected = []
        offset = 0
        per_page = 12
        while True:
            rows = await fetch_listing_rows(context, offset, delay)
            if not rows:
                break
            new = 0
            for r in rows:
                if all(existing["url"] != r["url"] for existing in collected):
                    collected.append(r)
                    new += 1
            print(f"[+] Listing offset {offset}: found {len(rows)} rows, {new} new")
            offset += per_page
            if len(rows) < per_page:
                break
        print(f"[+] Collected {len(collected)} product rows.")

        semaphore = asyncio.Semaphore(concurrency)
        async def worker(info):
            async with semaphore:
                return await fetch_product_item(context, info)
        tasks = [asyncio.create_task(worker(info)) for info in collected]
        results = []
        errors = []
        for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Fetching products"):
            res = await fut
            if res is None:
                continue
            if res.get("error"):
                errors.append(res)
            else:
                results.append(res)

        await context.close()
        await browser.close()

        unique = {}
        for it in results:
            key = it.get("url")
            if not key:
                continue
            unique[key] = it
        final = list(unique.values())

        cleaned = []
        for p in final:
            if not p.get("product_id") or not p.get("name"):
                continue
            cleaned.append(p)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)

        err_path = output.replace(".json", "_errors.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)

        print(f"[+] Saved {len(cleaned)} items to {output}; errors: {len(errors)}")
        return cleaned, errors

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="catalog_clean.json")
    parser.add_argument("--delay", type=float, default=0.25, help="delay between listing pages (seconds)")
    parser.add_argument("--concurrency", type=int, default=6, help="number of concurrent product fetches")
    args = parser.parse_args()
    asyncio.run(main(output=args.output, delay=args.delay, concurrency=args.concurrency))
