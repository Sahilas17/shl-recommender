"""
scraper.py
Scrapes the SHL product catalog (Individual Test Solutions only).
Run this to refresh shl_catalog.json if the catalog changes.

Usage:
    pip install requests beautifulsoup4
    python scraper.py

Note: The pre-built shl_catalog.json is already committed to the repo.
Only run this if you need to refresh catalog data.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# SHL uses these type codes
TEST_TYPE_MAP = {
    "A": "Ability/Aptitude",
    "B": "Biodata",
    "K": "Knowledge/Skills",
    "P": "Personality",
    "S": "Simulation",
}


def scrape_catalog_page(start: int = 0) -> list[dict]:
    """Scrape one page of the catalog listing."""
    params = {
        "start": start,
        "type": 1,   # 1 = Individual Test Solutions (not pre-packaged job solutions)
        "action_doFilteringForm": "Search",
    }
    try:
        resp = requests.get(
            CATALOG_URL, params=params, headers=HEADERS, timeout=30
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch catalog page (start={start}): {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    items = []

    # Products are in a <table> with rows; each row has a link and type indicators
    for row in soup.select("table.custom__table tbody tr, table tbody tr"):
        link_tag = row.find("a", href=lambda h: h and "/product-catalog/view/" in h)
        if not link_tag:
            continue

        name = link_tag.text.strip()
        url = BASE_URL + link_tag["href"]
        if not url.endswith("/"):
            url += "/"

        # Extract test type circles/badges
        test_types = []
        for badge in row.select(
            ".catalogue__circle, .product-catalogue__circle, span[class*='circle']"
        ):
            code = badge.text.strip()
            if code in TEST_TYPE_MAP:
                test_types.append(code)

        items.append({"name": name, "url": url, "test_types": test_types})

    logger.info(f"  Page start={start}: found {len(items)} products")
    return items


def scrape_product_detail(url: str) -> dict:
    """Scrape the detail page for a single product."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"  Failed to fetch {url}: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    detail = {}

    # Description — look for common selectors
    for sel in [
        ".product-catalogue-training-card__description p",
        ".product-catalogue-training-card__description",
        ".product__description",
        "meta[name='description']",
    ]:
        el = soup.select_one(sel)
        if el:
            detail["description"] = el.get("content", el.text).strip()
            break

    # Key–value pairs (job levels, languages, duration, etc.)
    for item in soup.select(
        ".product-catalogue-training-card__list-item, .product__detail-item"
    ):
        label_el = item.select_one(
            ".product-catalogue-training-card__label, .product__detail-label"
        )
        value_el = item.select_one(
            ".product-catalogue-training-card__value, .product__detail-value"
        )
        if label_el and value_el:
            key = label_el.text.strip().lower().replace(" ", "_")
            detail[key] = value_el.text.strip()

    # Normalise common fields
    if "job_levels" in detail:
        detail["job_levels"] = [
            lvl.strip()
            for lvl in detail["job_levels"].split(",")
            if lvl.strip()
        ]
    if "assessment_length" in detail:
        detail["duration"] = detail.pop("assessment_length")
    if "languages" not in detail:
        detail["languages"] = "English"

    return detail


def scrape_all(output_path: str = "shl_catalog.json") -> None:
    """Scrape the full catalog and save to JSON."""
    all_products = []
    start = 0
    page_size = 12   # SHL uses 12 items per page

    logger.info("Starting catalog scrape (Individual Test Solutions only)...")

    while True:
        logger.info(f"Fetching page start={start}...")
        page_products = scrape_catalog_page(start)
        if not page_products:
            logger.info("No more products found. Stopping.")
            break

        for p in page_products:
            logger.info(f"  Detail: {p['name']}")
            detail = scrape_product_detail(p["url"])
            p.update(detail)
            all_products.append(p)
            time.sleep(0.8)   # Polite delay

        start += page_size
        time.sleep(1.0)

    logger.info(f"\nTotal products scraped: {len(all_products)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_products, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    scrape_all()
