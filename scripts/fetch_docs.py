"""
Fetch AWS documentation pages and save as cleaned Markdown.

Uses LangChain's WebBaseLoader (the standard, polite approach used in AWS's
own RAG tutorials). Reads URLs from src/ingestion/urls.yaml.

Usage:
    python scripts/fetch_docs.py --test          # Fetch 5 pages only
    python scripts/fetch_docs.py                 # Fetch all pages
    python scripts/fetch_docs.py --service s3    # Fetch one service only
"""

import argparse
import time
import re
from pathlib import Path
from urllib.parse import urlparse

import yaml
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm

# Polite scraping settings
USER_AGENT = "AskAWS-Portfolio-Project/1.0 (Educational; LangChain WebBaseLoader pattern)"
DELAY_BETWEEN_REQUESTS = 1.0   # 1 second between requests
TIMEOUT = 30
MAX_RETRIES = 3

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
URLS_FILE = PROJECT_ROOT / "src" / "ingestion" / "urls.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_urls(service_filter=None):
    """Load URLs from YAML config. Returns dict: {service: [urls]}."""
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        all_urls = yaml.safe_load(f)
    if service_filter:
        return {service_filter: all_urls.get(service_filter, [])}
    return all_urls


def url_to_filename(url):
    """Convert URL to a safe filename. Example:
       https://docs.aws.amazon.com/.../Welcome.html -> Welcome.md
    """
    path = urlparse(url).path
    name = path.rsplit("/", 1)[-1].replace(".html", ".md")
    # Sanitize for Windows
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name


def extract_main_content(html):
    """Pull only the main documentation content from AWS docs HTML.
    Skips navigation, sidebars, footers, ads."""
    soup = BeautifulSoup(html, "lxml")

    # AWS docs wrap main content in <main id="main"> or <div id="main-col-body">
    main = (
        soup.find("main", id="main")
        or soup.find("div", id="main-col-body")
        or soup.find("main")
        or soup.find("article")
    )

    if not main:
        # Fallback: take body but strip nav/footer/aside
        main = soup.find("body") or soup
        for tag in main.find_all(["nav", "footer", "aside", "script", "style"]):
            tag.decompose()

    # Strip unwanted nested elements
    for tag in main.find_all(["script", "style", "nav", "aside", "footer"]):
        tag.decompose()

    # Get title
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    # Convert to text, preserving some structure
    text = main.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return title, text


def fetch_url(url, session, retries=MAX_RETRIES):
    """Fetch a single URL with retry logic."""
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{retries} after {wait}s ({e})")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} retries: {e}")
                return None


def save_doc(service, url, title, content):
    """Save fetched doc as a Markdown file with metadata header."""
    service_dir = OUTPUT_DIR / service
    service_dir.mkdir(parents=True, exist_ok=True)

    filename = url_to_filename(url)
    filepath = service_dir / filename

    # Markdown with simple frontmatter for traceability
    markdown = f"""---
source_url: {url}
service: {service}
title: {title}
---

# {title}

{content}
"""
    filepath.write_text(markdown, encoding="utf-8")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Fetch AWS documentation")
    parser.add_argument("--test", action="store_true", help="Fetch only 5 pages")
    parser.add_argument("--service", help="Fetch only one service (s3, lambda, ec2, iam, sagemaker)")
    args = parser.parse_args()

    urls_by_service = load_urls(service_filter=args.service)

    # Test mode: only 5 URLs total
    if args.test:
        test_urls = {}
        for service, urls in urls_by_service.items():
            if urls:
                test_urls[service] = urls[:1]  # 1 per service => 5 total
        urls_by_service = test_urls

    total = sum(len(urls) for urls in urls_by_service.values())
    print(f"Fetching {total} URLs across {len(urls_by_service)} services")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Delay between requests: {DELAY_BETWEEN_REQUESTS}s")
    print("-" * 60)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    success_count = 0
    fail_count = 0
    skipped_count = 0

    for service, urls in urls_by_service.items():
        print(f"\n[{service.upper()}] {len(urls)} URLs")
        for url in tqdm(urls, desc=service):
            # Skip if already downloaded (resume capability)
            filename = url_to_filename(url)
            filepath = OUTPUT_DIR / service / filename
            if filepath.exists() and not args.test:
                skipped_count += 1
                continue

            html = fetch_url(url, session)
            if html is None:
                fail_count += 1
                continue

            try:
                title, content = extract_main_content(html)
                if len(content) < 200:
                    print(f"  WARN: very short content ({len(content)} chars) for {url}")
                save_doc(service, url, title, content)
                success_count += 1
            except Exception as e:
                print(f"  ERROR parsing {url}: {e}")
                fail_count += 1

            time.sleep(DELAY_BETWEEN_REQUESTS)

    print("\n" + "=" * 60)
    print(f"DONE")
    print(f"  Success:  {success_count}")
    print(f"  Failed:   {fail_count}")
    print(f"  Skipped:  {skipped_count} (already downloaded)")
    print(f"  Output:   {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()