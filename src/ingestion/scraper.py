"""
GSU ECTS Bologna Page Scraper
Fetches the official programme page, strips non-content HTML,
and saves clean HTML to data/raw_docs/ for downstream LLM chunking.
"""
import os
import logging
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_URL = "https://ects.gsu.edu.tr/tr/program/programmedetails/12"
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "raw_docs")
OUTPUT_FILE = "ects_gsu.html"


def scrape_and_clean(url: str) -> str:
    """Fetch the page and strip everything that isn't content."""
    logger.info(f"Fetching: {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    soup = BeautifulSoup(resp.text, "html.parser")

    # 1. Remove noise tags entirely
    for tag in soup.find_all(["script", "style", "nav", "footer", "iframe", "noscript", "link", "meta"]):
        tag.decompose()

    # 2. Remove the top navigation menu and non-content elements
    for selector in [".navbar", ".dm-social", "#header", ".breadcrumb", ".cookie-banner"]:
        for el in soup.select(selector):
            el.decompose()

    # 3. Remove images (logos, flags etc.)
    for img in soup.find_all("img"):
        img.decompose()

    # 4. Get everything remaining in body
    body = soup.body
    if not body:
        raise RuntimeError("Could not locate body content on the page.")

    clean_html = str(body)
    logger.info(f"Extracted {len(clean_html)} characters of clean HTML.")
    return clean_html


def save(content: str, output_dir: str, filename: str):
    """Save content to the raw_docs directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Saved to: {path}")


if __name__ == "__main__":
    html = scrape_and_clean(TARGET_URL)
    save(html, OUTPUT_DIR, OUTPUT_FILE)
    logger.info("Done. Run chunker next to process this file.")