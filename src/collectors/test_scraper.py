"""
Test script for BGG forum scraper - scrapes only the first thread
"""

import sys
from pathlib import Path
import json
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.bgg_scraper import BGGForumScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_thread():
    """Test scraping a single thread from the forum."""
    
    STALKER_RULES_FORUM = "https://boardgamegeek.com/boardgame/381246/stalker-the-board-game/forums/66"
    
    logger.info("="*70)
    logger.info("BGG Forum Scraper Test - Single Thread")
    logger.info("="*70)
    
    # Initialize scraper
    scraper = BGGForumScraper(
        cache_dir="data/raw/forum_cache",
        output_dir="data/raw/forum_threads",
        delay=2.0  # Faster for testing
    )
    
    # Step 1: Get thread list from first page only
    logger.info("\n[Step 1] Fetching thread list from forum...")
    thread_list = scraper.get_forum_thread_list(
        forum_url=STALKER_RULES_FORUM,
        max_pages=1  # Only first page
    )
    
    if not thread_list:
        logger.error("No threads found! Check if the forum URL is correct.")
        return
    
    logger.info(f"Found {len(thread_list)} threads on first page")
    
    # Step 2: Show first few threads
    logger.info("\n[Step 2] First 5 threads found:")
    for i, thread in enumerate(thread_list[:5], 1):
        logger.info(f"  {i}. [{thread['thread_id']}] {thread['title']}")
        logger.info(f"     URL: {thread['url']}")
    
    # Step 3: Scrape only the first thread
    first_thread = thread_list[0]
    logger.info(f"\n[Step 3] Scraping first thread: {first_thread['title']}")
    logger.info(f"Thread ID: {first_thread['thread_id']}")
    logger.info(f"URL: {first_thread['url']}")
    
    thread_data = scraper.scrape_thread(
        thread_id=first_thread['thread_id'],
        thread_url=first_thread['url'],
        use_cache=True
    )
    
    if not thread_data:
        logger.error("Failed to scrape thread!")
        return
    
    # Step 4: Display results
    logger.info("\n[Step 4] Scraping Results:")
    logger.info("="*70)
    logger.info(f"Thread Title: {thread_data['title']}")
    logger.info(f"Thread ID: {thread_data['thread_id']}")
    logger.info(f"Thread URL: {thread_data['url']}")
    logger.info(f"Total Posts: {thread_data['post_count']}")
    logger.info(f"Scraped Date: {thread_data['scraped_date']}")
    
    logger.info(f"\n[Posts Content]")
    logger.info("-"*70)
    
    for i, post in enumerate(thread_data['posts'], 1):
        logger.info(f"\nPost #{i} (Position: {post['position']})")
        logger.info(f"Author: {post['author']}")
        logger.info(f"Date: {post['date']}")
        logger.info(f"Content Preview: {post['content'][:200]}...")
        logger.info(f"Full Content Length: {len(post['content'])} characters")
        if i >= 5:  # Show only first 5 posts
            logger.info(f"\n... and {len(thread_data['posts']) - 5} more posts")
            break
    
    # Step 5: Save to file
    logger.info("\n[Step 5] Saving thread data...")
    scraper.save_thread(thread_data)
    
    output_file = scraper.output_dir / f"{thread_data['thread_id']}.json"
    logger.info(f"Saved to: {output_file}")
    
    # Show file contents
    logger.info("\n[Step 6] Verifying saved file...")
    with open(output_file, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    logger.info(f"File contains {len(saved_data['posts'])} posts")
    logger.info(f"File size: {output_file.stat().st_size} bytes")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST COMPLETE ✓")
    logger.info("="*70)
    logger.info(f"Successfully scraped: {thread_data['title']}")
    logger.info(f"Posts extracted: {thread_data['post_count']}")
    logger.info(f"Output file: {output_file}")
    logger.info("\nIf you see post content above, the scraper is working!")
    logger.info("Next step: Run src/collectors/bgg_scraper.py to scrape all threads")
    logger.info("="*70)


if __name__ == "__main__":
    try:
        test_single_thread()
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)
