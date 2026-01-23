"""
BGG Forum Web Scraper
Scrapes BoardGameGeek forum threads for Q&A extraction.
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import re
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Selenium not available. Install with: pip install selenium")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BGGForumScraper:
    """Scraper for BoardGameGeek forum threads."""
    
    BASE_URL = "https://boardgamegeek.com"
    
    def __init__(self, cache_dir: str = "data/raw/forum_cache", 
                 output_dir: str = "data/raw/forum_threads",
                 delay: float = 2.5):
        """
        Initialize the scraper.
        
        Args:
            cache_dir: Directory to cache raw HTML
            output_dir: Directory to save parsed JSON
            delay: Delay between requests in seconds (be polite!)
        """
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.delay = delay
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Ensure we don't make requests too quickly."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            sleep_time = self.delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _fetch_page(self, url: str, max_retries: int = 3) -> Optional[str]:
        """
        Fetch a page with retry logic.
        
        Args:
            url: URL to fetch
            max_retries: Maximum number of retry attempts
            
        Returns:
            HTML content or None if failed
        """
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                logger.info(f"Fetching: {url}")
                
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                return response.text
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * self.delay  # Exponential backoff
                    logger.info(f"Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                    return None
    
    def _save_cache(self, thread_id: str, html: str):
        """Save raw HTML to cache."""
        cache_file = self.cache_dir / f"{thread_id}.html"
        cache_file.write_text(html, encoding='utf-8')
        logger.debug(f"Cached HTML for thread {thread_id}")
    
    def _load_cache(self, thread_id: str) -> Optional[str]:
        """Load HTML from cache if available."""
        cache_file = self.cache_dir / f"{thread_id}.html"
        if cache_file.exists():
            logger.debug(f"Loading cached HTML for thread {thread_id}")
            return cache_file.read_text(encoding='utf-8')
        return None
    
    def _init_selenium_driver(self):
        """Initialize Selenium WebDriver for JavaScript rendering."""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is required for scraping JavaScript-rendered pages. Install with: pip install selenium")
        
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in background
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument(f'user-agent={self.session.headers["User-Agent"]}')
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e:
            logger.error(f"Failed to initialize Chrome driver: {e}")
            logger.info("Make sure Chrome/Chromium and chromedriver are installed")
            raise
    
    def _get_threads_via_selenium(self, forum_url: str, max_pages: int = 5) -> List[Dict]:
        """Get threads using Selenium to handle JavaScript rendering."""
        logger.info("Using Selenium to scrape JavaScript-rendered forum...")
        
        driver = self._init_selenium_driver()
        threads = []
        seen_ids = set()
        
        try:
            for page in range(1, max_pages + 1):
                page_url = forum_url if page == 1 else f"{forum_url}?pageid={page}"
                logger.info(f"Loading page {page} with Selenium: {page_url}")
                
                driver.get(page_url)
                time.sleep(3)  # Wait for JavaScript to load
                
                # Try to wait for forum content
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "a"))
                    )
                except:
                    logger.warning("Timeout waiting for page load")
                
                # Get page HTML after JavaScript execution
                html = driver.page_source
                soup = BeautifulSoup(html, 'lxml')
                
                # Find thread links
                thread_links = soup.find_all('a', href=re.compile(r'/thread/\d+/'))
                
                page_thread_count = 0
                for link in thread_links:
                    href = link.get('href', '')
                    match = re.search(r'/thread/(\d+)/', href)
                    if not match or match.group(1) in seen_ids:
                        continue
                    
                    thread_id = match.group(1)
                    seen_ids.add(thread_id)
                    page_thread_count += 1
                    
                    title = link.get_text(strip=True)
                    full_url = self.BASE_URL + href if href.startswith('/') else href
                    
                    threads.append({
                        'thread_id': thread_id,
                        'title': title or f"Thread {thread_id}",
                        'url': full_url,
                    })
                
                logger.info(f"Found {page_thread_count} new threads on page {page} (total: {len(threads)})")
                
                if page_thread_count == 0:
                    break
                    
        finally:
            driver.quit()
        
        return threads
    
    def get_forum_thread_list(self, forum_url: str, max_pages: int = 5, use_selenium: bool = True) -> List[Dict]:
        """
        Get list of threads from the STALKER Rules Forum.
        
        Args:
            forum_url: URL of the forum page (e.g., .../forums/66)
            max_pages: Maximum number of pages to scrape
            use_selenium: Whether to use Selenium for JavaScript rendering
            
        Returns:
            List of thread info dictionaries with thread_id, title, and url
        """
        # Try Selenium first for JavaScript-rendered pages
        if use_selenium and SELENIUM_AVAILABLE:
            try:
                return self._get_threads_via_selenium(forum_url, max_pages)
            except Exception as e:
                logger.warning(f"Selenium scraping failed: {e}")
                logger.info("Falling back to regular requests...")
        
        # Fallback to regular requests
        threads = []
        seen_ids = set()  # Track unique thread IDs
        
        for page in range(1, max_pages + 1):
            # Construct paginated URL for BGG forums
            if page == 1:
                page_url = forum_url
            else:
                # BGG uses pageid parameter for pagination
                page_url = f"{forum_url}?pageid={page}"
            
            logger.info(f"Fetching forum page {page}: {page_url}")
            html = self._fetch_page(page_url)
            
            if not html:
                logger.warning(f"Failed to fetch page {page}, stopping")
                break
            
            soup = BeautifulSoup(html, 'lxml')
            
            # BGG forum threads are in links with pattern /thread/{id}/{slug}
            # Look for all thread links
            thread_links = soup.find_all('a', href=re.compile(r'/thread/\d+/'))
            
            if not thread_links:
                logger.warning(f"No threads found on page {page} with primary selector")
                # Try backup strategy: find any link with 'thread' in href
                thread_links = [a for a in soup.find_all('a', href=True) 
                               if '/thread/' in a.get('href', '')]
            
            page_thread_count = 0
            for link in thread_links:
                href = link.get('href', '')
                
                # Extract thread ID from URL like /thread/3649302/traps
                match = re.search(r'/thread/(\d+)/', href)
                if not match:
                    continue
                
                thread_id = match.group(1)
                
                # Skip duplicates (same thread can appear multiple times on page)
                if thread_id in seen_ids:
                    continue
                
                seen_ids.add(thread_id)
                page_thread_count += 1
                
                # Get thread title from link text
                title = link.get_text(strip=True)
                if not title or title in ['[no subject]', '']:
                    title = f"Thread {thread_id}"
                
                # Construct full URL
                full_url = self.BASE_URL + href if href.startswith('/') else href
                
                thread_info = {
                    'thread_id': thread_id,
                    'title': title,
                    'url': full_url,
                }
                
                threads.append(thread_info)
            
            logger.info(f"Found {page_thread_count} new threads on page {page} (total: {len(threads)})")
            
            # Check if there are more pages
            # Look for "next" link or page numbers
            if page_thread_count == 0:
                logger.info("No new threads found on this page, assuming end of forum")
                break
            
            # BGG typically shows "›" or "next" for next page
            next_link = soup.find('a', text=re.compile(r'›|next', re.I))
            if not next_link and page >= max_pages:
                logger.info(f"Reached max pages ({max_pages})")
                break
        
        logger.info(f"Total unique threads found: {len(threads)}")
        return threads
    
    def scrape_thread(self, thread_id: str, thread_url: str, 
                     use_cache: bool = True, use_selenium: bool = True) -> Optional[Dict]:
        """
        Scrape a single thread.
        
        Args:
            thread_id: BGG thread ID
            thread_url: Full URL to the thread
            use_cache: Whether to use cached HTML if available
            use_selenium: Whether to use Selenium for JavaScript rendering
            
        Returns:
            Dictionary with thread data or None if failed
        """
        # Check cache first
        html = None
        if use_cache:
            html = self._load_cache(thread_id)
        
        # Fetch if not cached
        if not html:
            if use_selenium and SELENIUM_AVAILABLE:
                # Use Selenium for JavaScript-rendered pages
                logger.debug(f"Using Selenium to scrape thread {thread_id}")
                driver = self._init_selenium_driver()
                try:
                    driver.get(thread_url)
                    time.sleep(5)  # Wait longer for content to load
                    
                    # Wait for posts to appear
                    try:
                        WebDriverWait(driver, 15).until(
                            EC.presence_of_element_located((By.TAG_NAME, "article"))
                        )
                    except:
                        logger.warning(f"Timeout waiting for thread content: {thread_id}")
                    
                    # Additional wait for full render
                    time.sleep(2)
                    
                    html = driver.page_source
                finally:
                    driver.quit()
            else:
                html = self._fetch_page(thread_url)
            
            if not html:
                return None
            self._save_cache(thread_id, html)
        
        # Parse the thread
        try:
            return self._parse_thread(thread_id, thread_url, html)
        except Exception as e:
            logger.error(f"Failed to parse thread {thread_id}: {e}", exc_info=True)
            return None
    
    def _parse_thread(self, thread_id: str, thread_url: str, html: str) -> Dict:
        """
        Parse thread HTML into structured data.
        
        Args:
            thread_id: Thread ID
            thread_url: Thread URL
            html: Raw HTML content
            
        Returns:
            Structured thread data
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # Extract thread title
        title = self._extract_title(soup)
        
        # Extract all posts
        posts = self._extract_posts(soup)
        
        thread_data = {
            'thread_id': thread_id,
            'url': thread_url,
            'title': title,
            'scraped_date': datetime.now().isoformat(),
            'post_count': len(posts),
            'posts': posts
        }
        
        logger.info(f"Parsed thread {thread_id}: '{title}' ({len(posts)} posts)")
        return thread_data
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract thread title from soup."""
        # Try the <title> tag first and extract the first part
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            # Split by | and take the first part (thread title)
            if '|' in title_text:
                title = title_text.split('|')[0].strip()
                if title and title.lower() != 'boardgame geek':
                    return title
        
        # Try other selectors
        selectors = [
            'h1',
            'h2',
            '[class*="title"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if title and len(title) > 3:
                    return title
        
        return "Unknown Title"
    
    def _extract_posts(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all posts from thread."""
        posts = []
        
        # BGG uses <article> tags for posts
        post_containers = soup.find_all('article')
        
        if not post_containers:
            # Fallback: Try other selectors
            post_containers = soup.find_all('div', class_=re.compile(r'post|comment|article'))
            logger.debug(f"Fallback: found {len(post_containers)} posts with div selector")
        
        logger.debug(f"Found {len(post_containers)} post containers")
        
        for idx, post_elem in enumerate(post_containers):
            try:
                post_data = self._extract_post(post_elem, idx)
                if post_data:
                    posts.append(post_data)
            except Exception as e:
                logger.warning(f"Failed to extract post {idx}: {e}")
                continue
        
        return posts
    
    def _extract_post(self, post_elem, index: int) -> Optional[Dict]:
        """Extract data from a single post element."""
        # Extract content
        content_elem = post_elem.find('div', class_=re.compile(r'post_content|content|body'))
        if not content_elem:
            content_elem = post_elem
        
        content = content_elem.get_text(strip=True)
        if not content or len(content) < 10:
            return None
        
        # Extract author
        author_elem = post_elem.find(['span', 'div', 'a'], class_=re.compile(r'author|username|user'))
        author = author_elem.get_text(strip=True) if author_elem else f"User_{index}"
        
        # Extract date - look for time elements with datetime attribute
        date_str = ""
        time_elem = post_elem.find('time')
        if time_elem:
            date_str = time_elem.get('datetime', '') or time_elem.get('title', '') or time_elem.get_text(strip=True)
        
        # Fallback: look for date in other elements
        if not date_str:
            date_elem = post_elem.find(['span', 'div'], class_=re.compile(r'date|time|posted|timestamp'))
            if date_elem:
                date_str = date_elem.get('datetime', '') or date_elem.get('data-time', '') or date_elem.get_text(strip=True)
        
        # Extract post ID if available
        post_id = post_elem.get('id', f"post_{index}")
        
        return {
            'post_id': post_id,
            'author': author,
            'date': date_str,
            'content': content,
            'position': index
        }
    
    def save_thread(self, thread_data: Dict):
        """Save parsed thread data to JSON."""
        thread_id = thread_data['thread_id']
        output_file = self.output_dir / f"{thread_id}.json"
        
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(thread_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved thread {thread_id} to {output_file}")
    
    def scrape_forum(self, forum_url: str, max_threads: int = 50, 
                     max_pages: int = 3) -> List[Dict]:
        """
        Scrape entire forum.
        
        Args:
            forum_url: URL of the forum
            max_threads: Maximum number of threads to scrape
            max_pages: Maximum forum pages to check for threads
            
        Returns:
            List of scraped thread data
        """
        logger.info(f"Starting forum scrape: {forum_url}")
        logger.info(f"Max threads: {max_threads}, Max pages: {max_pages}")
        
        # Get thread list
        thread_list = self.get_forum_thread_list(forum_url, max_pages)
        thread_list = thread_list[:max_threads]
        
        logger.info(f"Will scrape {len(thread_list)} threads")
        
        # Scrape each thread
        scraped_threads = []
        failed_threads = []
        
        for i, thread_info in enumerate(thread_list, 1):
            logger.info(f"Progress: {i}/{len(thread_list)} - {thread_info['title']}")
            
            thread_data = self.scrape_thread(
                thread_info['thread_id'],
                thread_info['url'],
                use_cache=True
            )
            
            if thread_data:
                self.save_thread(thread_data)
                scraped_threads.append(thread_data)
            else:
                failed_threads.append(thread_info)
            
            # Progress update every 10 threads
            if i % 10 == 0:
                logger.info(f"Checkpoint: {i} threads processed, {len(scraped_threads)} successful, {len(failed_threads)} failed")
        
        # Save manifest
        manifest = {
            'scrape_date': datetime.now().isoformat(),
            'forum_url': forum_url,
            'total_threads': len(thread_list),
            'successful': len(scraped_threads),
            'failed': len(failed_threads),
            'threads': [
                {
                    'thread_id': t['thread_id'],
                    'title': t['title'],
                    'post_count': t['post_count']
                }
                for t in scraped_threads
            ],
            'failed_threads': failed_threads
        }
        
        manifest_file = self.output_dir / 'manifest.json'
        with manifest_file.open('w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Scraping complete! Manifest saved to {manifest_file}")
        logger.info(f"Successful: {len(scraped_threads)}, Failed: {len(failed_threads)}")
        
        return scraped_threads


def main():
    """Scrape STALKER Rules Forum."""
    # STALKER Rules Forum URL - this is the only forum we need
    STALKER_RULES_FORUM = "https://boardgamegeek.com/boardgame/381246/stalker-the-board-game/forums/66"
    
    scraper = BGGForumScraper(
        cache_dir="data/raw/forum_cache",
        output_dir="data/raw/forum_threads",
        delay=2.5  # 2.5 seconds between requests - be polite!
    )
    
    logger.info(f"Scraping STALKER Rules Forum: {STALKER_RULES_FORUM}")
    logger.info("This will extract all threads from the forum page")
    
    # Scrape all threads from the rules forum
    # Adjust max_threads and max_pages based on forum size
    threads = scraper.scrape_forum(
        forum_url=STALKER_RULES_FORUM,
        max_threads=200,  # Increase if forum has more threads
        max_pages=10      # Number of forum pages to check
    )
    
    logger.info(f"Scraping complete! Successfully scraped {len(threads)} threads")
    logger.info(f"Results saved to: {scraper.output_dir}")
    logger.info(f"Manifest file: {scraper.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
