"""
Test script to manually visit a thread with Selenium and save the fully-rendered HTML
"""

import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_THREAD_URL = "https://boardgamegeek.com/thread/3649302/traps"

logger.info(f"Testing Selenium with thread: {TEST_THREAD_URL}")

# Setup Chrome with headless mode OFF so we can see what's happening
chrome_options = Options()
# chrome_options.add_argument('--headless')  # Comment out to see the browser
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=chrome_options)

try:
    logger.info("Loading page...")
    driver.get(TEST_THREAD_URL)
    
    # Wait longer and try multiple selectors
    logger.info("Waiting for content to load...")
    time.sleep(5)  # Initial wait
    
    # Try to find specific BGG thread elements
    selectors_to_try = [
        "article",
        ".forum-post",
        "[class*='post']",
        "[class*='thread']",
        "[class*='comment']",
        "main"
    ]
    
    for selector in selectors_to_try:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                logger.info(f"Found {len(elements)} elements with selector: {selector}")
        except Exception as e:
            logger.warning(f"Selector '{selector}' failed: {e}")
    
    # Get the page source
    html = driver.page_source
    
    # Save to file
    output_file = Path("data/raw/selenium_test_thread.html")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding='utf-8')
    
    logger.info(f"Saved HTML to: {output_file}")
    logger.info(f"HTML size: {len(html)} bytes")
    
    # Check for keywords
    keywords = ["trap", "enemy", "moving", "movement"]
    found_keywords = [kw for kw in keywords if kw.lower() in html.lower()]
    logger.info(f"Found keywords in HTML: {found_keywords}")
    
    # Count occurrences of common post-related words
    if "Traps" in html:
        logger.info("✓ Title 'Traps' found in HTML")
    else:
        logger.warning("✗ Title 'Traps' NOT found in HTML - page may not be fully loaded")
    
    # Wait a bit longer to let user see the page
    logger.info("Waiting 10 seconds before closing browser... (check what you see)")
    time.sleep(10)
    
finally:
    driver.quit()
    logger.info("Browser closed")

logger.info("\nNext steps:")
logger.info("1. Check data/raw/selenium_test_thread.html to see what was captured")
logger.info("2. If content is missing, we may need to use BGG API or adjust wait strategy")
