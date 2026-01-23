"""
Forum Q&A Pair Extractor
Extracts question-answer pairs from scraped forum threads.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerRanker:
    """Ranks forum answers by usefulness."""
    
    # Patterns for acknowledgments and non-useful replies
    ACKNOWLEDGMENT_PATTERNS = [
        r'^(thanks?|thank you|thx|ty)[\s!.]*$',
        r'^(ok|okay)[\s!.]*$',
        r'^(got it|understood)[\s!.]*$',
        r'^(nice|cool|great)[\s!.]*$',
        r'^(\+1|agree|same here)[\s!.]*$',
    ]
    
    # Patterns indicating useful content
    USEFUL_PATTERNS = [
        r'(?:page|p\.|pg\.)\s*\d+',  # Page references
        r'(?:rulebook|rules?|manual)',  # Rule references
        r'(?:according to|the rules? state|in the)',  # Authority references
        r'(?:you (?:can|should|must|need)|it (?:says|states))',  # Instructions
    ]
    
    def __init__(self):
        self.acknowledgment_regex = [re.compile(p, re.IGNORECASE) for p in self.ACKNOWLEDGMENT_PATTERNS]
        self.useful_regex = [re.compile(p, re.IGNORECASE) for p in self.USEFUL_PATTERNS]
    
    def score_answer(self, post: Dict, is_original_poster: bool = False) -> int:
        """Score an answer based on multiple factors."""
        content = post.get('content', '').strip()
        score = 0
        
        # Filter out very short posts
        if len(content) < 15:
            return 0
        
        # Check if it's just an acknowledgment
        for pattern in self.acknowledgment_regex:
            if pattern.match(content):
                return 0
        
        # Length scoring
        if len(content) > 50:
            score += 2
        if len(content) > 150:
            score += 2
        if len(content) > 300:
            score += 1
        
        # Useful content indicators
        for pattern in self.useful_regex:
            if pattern.search(content):
                score += 3
                break
        
        # Check for specific information (numbers, specifics)
        if re.search(r'\b\d+\b', content):  # Contains numbers
            score += 1
        
        # Penalize if original poster (usually follow-ups, not answers)
        if is_original_poster:
            score -= 2
        
        # Check for question marks (probably asking follow-up)
        question_marks = content.count('?')
        if question_marks > 1:
            score -= 2
        elif question_marks == 1:
            score -= 1
        
        return max(0, score)  # Don't return negative scores
    
    def rank_answers(self, posts: List[Dict], original_author: str) -> List[Tuple[Dict, int]]:
        """Rank all posts as potential answers."""
        ranked = []
        
        for post in posts:
            author = post.get('author', '')
            is_op = author == original_author
            score = self.score_answer(post, is_op)
            
            if score > 0:  # Only include posts with positive scores
                ranked.append((post, score))
        
        # Sort by score (descending)
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked


class QuestionExtractor:
    """Extracts and normalizes questions from thread data."""
    
    # Patterns to remove from questions
    GREETING_PATTERNS = [
        r'^(hi|hello|hey|greetings)[\s!,]*',
        r'(thanks?|thank you|thx|ty)\s*(in advance|!)*$',
        r'^(sorry|excuse me)[\s!,]*',
    ]
    
    # Question word indicators
    QUESTION_WORDS = ['how', 'what', 'when', 'where', 'why', 'who', 'which', 'can', 'should', 'does', 'is', 'are']
    
    def __init__(self):
        self.greeting_regex = [re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS]
    
    def is_question(self, text: str) -> bool:
        """Check if text appears to be a question."""
        text_lower = text.lower()
        
        # Has question mark
        if '?' in text:
            return True
        
        # Starts with question word
        first_word = text_lower.split()[0] if text_lower.split() else ''
        if first_word in self.QUESTION_WORDS:
            return True
        
        return False
    
    def clean_question(self, text: str) -> str:
        """Remove greetings and unnecessary text from question."""
        # Remove greetings
        for pattern in self.greeting_regex:
            text = pattern.sub('', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text.strip()
    
    def extract_question(self, thread: Dict) -> Optional[str]:
        """Extract question from thread title and first post."""
        title = thread.get('title', '').strip()
        
        # Clean the title
        question = self.clean_question(title)
        
        # If title is a good question, use it
        if self.is_question(question) and len(question) > 10:
            return question
        
        # Otherwise, try to use first post content
        posts = thread.get('posts', [])
        if posts:
            first_post = posts[0]
            content = first_post.get('content', '').strip()
            
            # Extract first question from content
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences:
                sentence = self.clean_question(sentence)
                if self.is_question(sentence) and len(sentence) > 10:
                    return sentence
            
            # If no question found, combine title with first sentence
            if len(sentences) > 0:
                first_sentence = self.clean_question(sentences[0])
                if len(first_sentence) > 10:
                    combined = f"{question}: {first_sentence}"
                    return combined
        
        # Fallback to cleaned title
        return question if question else None


class ForumQAExtractor:
    """Main class for extracting Q&A pairs from forum data."""
    
    def __init__(self, input_dir: str = "data/raw/forum_threads",
                 output_dir: str = "data/processed/forum_qa"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.question_extractor = QuestionExtractor()
        self.answer_ranker = AnswerRanker()
    
    def load_thread(self, thread_file: Path) -> Optional[Dict]:
        """Load a thread JSON file."""
        try:
            with thread_file.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {thread_file}: {e}")
            return None
    
    def extract_qa_pair(self, thread: Dict) -> Optional[Dict]:
        """Extract Q&A pair from a thread."""
        thread_id = thread.get('thread_id')
        thread_url = thread.get('url')
        language = thread.get('language', 'unknown')
        
        # Skip non-English threads
        if language != 'en':
            logger.debug(f"Skipping non-English thread {thread_id} (language: {language})")
            return None
        
        # Extract question
        question = self.question_extractor.extract_question(thread)
        if not question:
            logger.debug(f"No valid question found in thread {thread_id}")
            return None
        
        # Get posts (skip first post as it's the question)
        posts = thread.get('posts', [])
        if len(posts) < 2:
            logger.debug(f"Thread {thread_id} has no replies")
            return None
        
        original_author = posts[0].get('author', '') if posts else ''
        reply_posts = posts[1:]  # Skip the original question post
        
        # Rank answers
        ranked_answers = self.answer_ranker.rank_answers(reply_posts, original_author)
        
        if not ranked_answers:
            logger.debug(f"No useful answers found in thread {thread_id}")
            return None
        
        # Take top answer (could take top 3 and combine)
        top_answer, top_score = ranked_answers[0]
        answer_text = top_answer.get('content', '').strip()
        
        # Build Q&A pair
        qa_pair = {
            'id': f"forum_{thread_id}",
            'source': 'forum',
            'thread_id': thread_id,
            'thread_url': thread_url,
            'question': question,
            'answer': answer_text,
            'raw_question': {
                'title': thread.get('title', ''),
                'content': posts[0].get('content', '') if posts else '',
                'author': original_author,
                'date': posts[0].get('date', '') if posts else ''
            },
            'raw_answers': [
                {
                    'content': ans[0].get('content', ''),
                    'author': ans[0].get('author', ''),
                    'date': ans[0].get('date', ''),
                    'score': ans[1],
                    'rank': i + 1
                }
                for i, ans in enumerate(ranked_answers[:3])  # Keep top 3
            ],
            'metadata': {
                'language': language,
                'scraped_date': thread.get('scraped_date', ''),
                'processed_date': datetime.now().isoformat(),
                'answer_count': len(reply_posts),
                'useful_answer_count': len(ranked_answers)
            }
        }
        
        return qa_pair
    
    def process_all_threads(self) -> List[Dict]:
        """Process all threads and extract Q&A pairs."""
        logger.info(f"Processing threads from {self.input_dir}")
        
        thread_files = list(self.input_dir.glob('*.json'))
        # Exclude manifest.json
        thread_files = [f for f in thread_files if f.name != 'manifest.json']
        
        logger.info(f"Found {len(thread_files)} thread files")
        
        qa_pairs = []
        skipped_count = 0
        
        for i, thread_file in enumerate(thread_files, 1):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(thread_files)} threads processed, {len(qa_pairs)} Q&A pairs extracted")
            
            thread = self.load_thread(thread_file)
            if not thread:
                skipped_count += 1
                continue
            
            qa_pair = self.extract_qa_pair(thread)
            if qa_pair:
                qa_pairs.append(qa_pair)
            else:
                skipped_count += 1
        
        logger.info(f"Extraction complete!")
        logger.info(f"Total threads: {len(thread_files)}")
        logger.info(f"Q&A pairs extracted: {len(qa_pairs)}")
        logger.info(f"Skipped: {skipped_count}")
        
        return qa_pairs
    
    def save_qa_pairs(self, qa_pairs: List[Dict], filename: str = 'forum_qa_pairs.json'):
        """Save Q&A pairs to JSON file."""
        output_file = self.output_dir / filename
        
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {output_file}")
        
        # Also save a summary
        summary = {
            'total_pairs': len(qa_pairs),
            'generated_date': datetime.now().isoformat(),
            'source': 'BoardGameGeek STALKER Rules Forum',
            'average_answer_score': sum(qa['raw_answers'][0]['score'] for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0,
            'threads_with_multiple_answers': sum(1 for qa in qa_pairs if len(qa['raw_answers']) > 1)
        }
        
        summary_file = self.output_dir / 'qa_summary.json'
        with summary_file.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {summary_file}")


def main():
    """Extract Q&A pairs from forum threads."""
    extractor = ForumQAExtractor(
        input_dir="data/raw/forum_threads",
        output_dir="data/processed/forum_qa"
    )
    
    # Process all threads
    qa_pairs = extractor.process_all_threads()
    
    # Save results
    extractor.save_qa_pairs(qa_pairs)
    
    # Show some examples
    logger.info("\n" + "="*70)
    logger.info("Example Q&A Pairs:")
    logger.info("="*70)
    
    for i, qa in enumerate(qa_pairs[:5], 1):
        logger.info(f"\n[{i}] Thread: {qa['thread_id']}")
        logger.info(f"Q: {qa['question']}")
        logger.info(f"A: {qa['answer'][:200]}...")
        logger.info(f"Score: {qa['raw_answers'][0]['score']}")


if __name__ == "__main__":
    main()
