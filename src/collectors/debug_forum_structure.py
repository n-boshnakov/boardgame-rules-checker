"""
Debug script to inspect BGG forum HTML structure
"""

import requests
from bs4 import BeautifulSoup
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STALKER_RULES_FORUM = "https://boardgamegeek.com/boardgame/381246/stalker-the-board-game/forums/66"

logger.info("Fetching forum page to inspect HTML structure...")

# Fetch the page
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
})

response = session.get(STALKER_RULES_FORUM, timeout=10)
html = response.text

# Save to file for inspection
output_file = Path("data/raw/forum_page_debug.html")
output_file.parent.mkdir(parents=True, exist_ok=True)
output_file.write_text(html, encoding='utf-8')
logger.info(f"Saved raw HTML to: {output_file}")

# Parse and analyze
soup = BeautifulSoup(html, 'lxml')

logger.info("\n" + "="*70)
logger.info("HTML STRUCTURE ANALYSIS")
logger.info("="*70)

# Find all links
all_links = soup.find_all('a', href=True)
logger.info(f"\nTotal links found: {len(all_links)}")

# Find links with 'thread' in href
thread_links = [a for a in all_links if 'thread' in a.get('href', '').lower()]
logger.info(f"Links with 'thread' in href: {len(thread_links)}")

if thread_links:
    logger.info("\nFirst 10 thread links found:")
    for i, link in enumerate(thread_links[:10], 1):
        href = link.get('href', '')
        text = link.get_text(strip=True)
        logger.info(f"{i}. href: {href}")
        logger.info(f"   text: {text}")
        logger.info(f"   class: {link.get('class', 'no class')}")
        logger.info("")
else:
    logger.warning("No thread links found!")
    logger.info("\nSearching for any structure that might contain threads...")
    
    # Look for common forum structures
    tables = soup.find_all('table')
    logger.info(f"\nTables found: {len(tables)}")
    
    articles = soup.find_all('article')
    logger.info(f"Articles found: {len(articles)}")
    
    divs_with_thread = soup.find_all('div', class_=lambda x: x and 'thread' in x.lower() if x else False)
    logger.info(f"Divs with 'thread' in class: {len(divs_with_thread)}")
    
    # Look for any structure containing thread IDs
    logger.info("\nSearching HTML for thread ID patterns...")
    import re
    thread_pattern = re.compile(r'thread[/\-_](\d+)', re.IGNORECASE)
    matches = thread_pattern.findall(html)
    if matches:
        logger.info(f"Found {len(set(matches))} unique thread IDs in HTML:")
        for thread_id in list(set(matches))[:10]:
            logger.info(f"  - Thread ID: {thread_id}")
    
    # Show page title
    title = soup.find('title')
    if title:
        logger.info(f"\nPage title: {title.get_text(strip=True)}")
    
    # Check if we're being redirected or blocked
    if 'login' in html.lower() or 'sign in' in html.lower():
        logger.warning("\n⚠ Page might require login!")
    
    if 'blocked' in html.lower() or 'captcha' in html.lower():
        logger.warning("\n⚠ Page might be blocking automated access!")

logger.info("\n" + "="*70)
logger.info(f"Check {output_file} to see the full HTML structure")
logger.info("="*70)
