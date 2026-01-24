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

class QuoteDetector:
    """Detects and removes quoted text in answers."""
    
    def __init__(self):
        """Initialize QuoteDetector with AnswerCleaner instance."""
        self.answer_cleaner = AnswerCleaner()
    
    def detect_concatenated_quotes(self, text: str) -> str:
        """
        Detect when multiple users' responses are concatenated.
        Extract only the final (most direct) response.
        
        Pattern: User1@username1...User2@username2...actual answer
        """
        # Find all @mentions positions
        mentions = list(re.finditer(r'@\w+', text))
        
        if len(mentions) >= 2:
            # Multiple mentions = likely chain of quotes
            # Take text AFTER the last mention
            last_mention_end = mentions[-1].end()
            
            # Extract text after last mention
            direct_answer = text[last_mention_end:].strip()
            
            # If it's substantial, use it; otherwise use original
            if len(direct_answer) >= 50:
                return direct_answer
        
        return text
    
    def detect_and_score_quotes(self, answer: str, question: str) -> Tuple[str, int]:
        """
        Remove quoted text and return bonus score for direct responses.
        
        Args:
            answer: Answer text potentially containing quotes
            question: Original question text
            
        Returns:
            (cleaned_answer, quote_bonus_score)
        """
        bonus = 0
        cleaned = answer
        
        # 1. Detect BGG user mention format (anywhere in text)
        # Pattern: FirstName LastName@username
        cleaned = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+@\w+\s*', '', cleaned)
        
        # 2. Check if answer starts by repeating the question
        question_normalized = self._normalize_for_matching(question)
        answer_start = self._normalize_for_matching(answer[:min(len(question) + 100, len(answer))])
        
        similarity = self._text_similarity(question_normalized, answer_start)
        if similarity > 0.6:  # 60% similar - likely quoted
            cleaned = self._remove_leading_quote(cleaned, question)
            bonus += 3  # High bonus for directly addressing question
        
        # 3. Detect inline Q&A format
        if re.search(r'(Question:|Q:)\s*.+?(Answer:|A:)\s*', cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(r'(Question:|Q:)\s*.+?(Answer:|A:)\s*', '', cleaned, flags=re.IGNORECASE)
            bonus += 2
        
        return cleaned.strip(), bonus
    
    def clean_answer_comprehensive(self, text: str, question: str = '') -> str:
        """
        Comprehensive cleaning pipeline for answer text.
        
        Steps:
        1. Remove concatenated quotes
        2. Remove @mentions
        3. Detect and remove leading question quotes
        4. Clean conversational elements
        5. Truncate if too long
        """
        # 1. Remove concatenated quotes
        text = self.detect_concatenated_quotes(text)
        
        # 2. Remove @mentions
        text = self.answer_cleaner.remove_mentions(text)
        
        # 3. Detect and remove leading question quotes
        if question:
            text, _ = self.detect_and_score_quotes(text, question)
        
        # 4. Clean conversational elements
        text = self.answer_cleaner.clean_answer(text)
        
        # 5. Truncate if too long
        text = self.answer_cleaner.truncate_long_answer(text)
        
        return text
    
    def _normalize_for_matching(self, text: str) -> str:
        """Normalize text for similarity comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts (0-1)."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _remove_leading_quote(self, answer: str, question: str) -> str:
        """Remove question quote from start of answer."""
        # Split into sentences
        sentences = re.split(r'([.!?]\s+)', answer)
        
        # Rejoin sentence pairs (text + delimiter)
        sentence_list = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence_list.append(sentences[i] + sentences[i+1])
            else:
                sentence_list.append(sentences[i])
        
        # Find first sentence that doesn't match question
        result = []
        skip_first = False
        
        if sentence_list:
            first_sent_normalized = self._normalize_for_matching(sentence_list[0])
            question_normalized = self._normalize_for_matching(question)
            
            # If first sentence is very similar to question, skip it
            if self._text_similarity(first_sent_normalized, question_normalized) > 0.5:
                skip_first = True
                result = sentence_list[1:]
            else:
                result = sentence_list
        
        return ''.join(result).strip()


class AnswerCleaner:
    """Cleans conversational and filler text from answers."""
    
    # Conversational prefixes to remove
    CONVERSATIONAL_PREFIXES = [
        r'^(Hmm,?|Well,?|So,?|Actually,?|Basically,?)\s+',
        r'^(I think|I believe|In my opinion|IMHO|IMO)\s+',
        r'^(Good question[.,!]\s+)',
        r'^(Let me see[.,!]\s+)',
        r"^(Can't remember [^,]+,\s+but\s+)",
        r"^(I don't have [^,]+,\s+but\s+)",
    ]
    
    # Trailing conversational elements
    CONVERSATIONAL_SUFFIXES = [
        r'\s+(Hope (that )?helps?|Cheers|Thanks)[.,!]*$',
        r'\s+(Let me know if you have questions?)[.,!]*$',
    ]
    
    # Patterns for @mention removal (BGG forum format)
    MENTION_PATTERNS = [
        # BGG format: FirstName LastName@username (flexible case)
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]*@\w+\s*',
        # Lowercase variants: jim tullis@jimtullis
        r'\b[a-z]+\s+[a-z]+@\w+\s*',
        # Standalone @username at start or after punctuation
        r'(?:^|[.!?]\s+)@\w+\s*',
        # Quote format: @username says/wrote/mentioned
        r'@\w+\s+(?:says?|wrote|mentioned):\s*',
    ]
    
    MAX_ANSWER_LENGTH = 500  # Maximum characters for answers
    MIN_USEFUL_LENGTH = 50   # Minimum useful content
    
    def __init__(self):
        self.prefix_regex = [re.compile(p, re.IGNORECASE) for p in self.CONVERSATIONAL_PREFIXES]
        self.suffix_regex = [re.compile(p, re.IGNORECASE) for p in self.CONVERSATIONAL_SUFFIXES]
        self.mention_regex = [re.compile(p) for p in self.MENTION_PATTERNS]
    
    def remove_mentions(self, text: str) -> str:
        """Remove all @username mentions and quote attributions."""
        cleaned = text
        
        for pattern in self.mention_regex:
            cleaned = pattern.sub('', cleaned)
        
        # Clean up extra whitespace after removal
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        
        return cleaned.strip()
    
    def truncate_long_answer(self, text: str, max_length: int = None) -> str:
        """
        Intelligently truncate overly long answers to most useful part.
        
        Strategy:
        1. If under max_length, return as-is
        2. Split into sentences
        3. Take first N sentences that fit within max_length
        4. Ensure ends with complete sentence
        """
        if max_length is None:
            max_length = self.MAX_ANSWER_LENGTH
        
        if len(text) <= max_length:
            return text
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Build truncated version
        result = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) <= max_length:
                result.append(sentence)
                current_length += len(sentence) + 1  # +1 for space
            else:
                break
        
        # If we got at least MIN_USEFUL_LENGTH, return it
        truncated = ' '.join(result)
        if len(truncated) >= self.MIN_USEFUL_LENGTH:
            return truncated
        
        # Otherwise, hard truncate at max_length with ellipsis
        return text[:max_length-3] + '...'
    
    def clean_answer(self, answer_text: str) -> str:
        """Clean answer text while preserving factual content."""
        cleaned = answer_text
        
        # Remove conversational prefixes
        for pattern in self.prefix_regex:
            cleaned = pattern.sub('', cleaned)
        
        # Remove conversational suffixes
        for pattern in self.suffix_regex:
            cleaned = pattern.sub('', cleaned)
        
        # Clean up extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Capitalize first letter if needed
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned.strip()


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
    
    # Phase 3: Speculative/uncertain answer patterns
    SPECULATIVE_PATTERNS = [
        r"I don't know",
        r"I'm not sure",
        r"I think maybe",
        r"\bprobably\b",
        r"I guess",
        r"not certain",
        r"can't remember",
        r"unsure",
        r"unclear",
    ]
    
    # Phase 3: Definitive language patterns (indicates confident, factual answers)
    DEFINITIVE_PATTERNS = [
        r'\byes\b', r'\bno\b',
        r'\bcorrect\b', r'\bincorrect\b',
        r'\bmust\b', r'\bcannot\b',
        r'\balways\b', r'\bnever\b',
        r'\bexactly\b', r'\bdefinitely\b',
    ]
    
    def __init__(self):
        self.acknowledgment_regex = [re.compile(p, re.IGNORECASE) for p in self.ACKNOWLEDGMENT_PATTERNS]
        self.useful_regex = [re.compile(p, re.IGNORECASE) for p in self.USEFUL_PATTERNS]
        # Phase 3: Compile speculative and definitive patterns
        self.speculative_regex = [re.compile(p, re.IGNORECASE) for p in self.SPECULATIVE_PATTERNS]
        self.definitive_regex = [re.compile(p, re.IGNORECASE) for p in self.DEFINITIVE_PATTERNS]
        self.quote_detector = QuoteDetector()
        self.answer_cleaner = AnswerCleaner()
    
    def score_answer(self, post: Dict, is_original_poster: bool = False, question_text: str = '') -> Tuple[int, str]:
        """Score an answer and return cleaned content with Phase 3 improvements."""
        content = post.get('content', '').strip()
        score = 0
        
        # Filter out very short posts
        if len(content) < 15:
            return 0, content
        
        # Check if it's just an acknowledgment
        for pattern in self.acknowledgment_regex:
            if pattern.match(content):
                return 0, content
        
        # Use comprehensive cleaning pipeline (Phase 1 improvement)
        if question_text:
            content = self.quote_detector.clean_answer_comprehensive(content, question_text)
            # Award bonus for successful cleaning/direct response
            score += 2
        else:
            # Fallback to basic cleaning
            content = self.quote_detector.detect_concatenated_quotes(content)
            content = self.answer_cleaner.remove_mentions(content)
            content = self.answer_cleaner.clean_answer(content)
            content = self.answer_cleaner.truncate_long_answer(content)
        
        # Recheck length after cleaning
        if len(content) < 15:
            return max(0, score), content
        
        # === PHASE 3: IMPROVED LENGTH SCORING ===
        length = len(content)
        
        if 50 <= length <= 200:
            # Sweet spot: concise and complete
            score += 4
        elif 200 < length <= 400:
            # Good length
            score += 3
        elif 400 < length <= 600:
            # Getting long, but still acceptable
            score += 1
        elif length > 600:
            # Too long, likely concatenated or off-topic
            score -= 2
        
        # === PHASE 3: SPECIFICITY SCORING ===
        
        # Useful content indicators (page refs, rulebook mentions)
        for pattern in self.useful_regex:
            if pattern.search(content):
                score += 3
                break
        
        # Check for specific information (numbers, specifics)
        if re.search(r'\b\d+\b', content):  # Contains numbers
            score += 1
        
        # Phase 3: Definitive language bonus
        has_definitive = False
        for pattern in self.definitive_regex:
            if pattern.search(content):
                score += 2
                has_definitive = True
                break
        
        # === PHASE 3: UNCERTAINTY PENALTIES ===
        
        # Speculative language detection
        for pattern in self.speculative_regex:
            if pattern.search(content):
                score -= 3
                break
        
        # Multiple question marks (uncertainty/follow-up questions)
        question_marks = content.count('?')
        if question_marks > 2:
            score -= 3
        elif question_marks > 0:
            score -= 1
        
        # === OTHER PENALTIES ===
        
        # Penalize if original poster (usually follow-ups, not answers)
        if is_original_poster:
            score -= 2
        
        # Starts with uncertain phrases
        if re.match(r'^(I think|Maybe|Perhaps)', content, re.IGNORECASE):
            score -= 1
        
        # Check for question marks (probably asking follow-up)
        question_marks = content.count('?')
        if question_marks > 1:
            score -= 2
        elif question_marks == 1:
            score -= 1
        
        return max(0, score), content  # Return score and cleaned content
    
    def rank_answers(self, posts: List[Dict], original_author: str, question_text: str = '') -> List[Tuple[Dict, int, str]]:
        """Rank all posts as potential answers."""
        ranked = []
        
        for post in posts:
            author = post.get('author', '')
            is_op = author == original_author
            score, cleaned_content = self.score_answer(post, is_op, question_text)
            
            if score > 0:  # Only include posts with positive scores
                ranked.append((post, score, cleaned_content))
        
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
    
    # Phase 2: Maximum reasonable question length
    MAX_QUESTION_LENGTH = 150
    
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
    
    def ensure_question_mark(self, text: str) -> str:
        """Ensure question ends with '?' if it's clearly a question (Phase 2)."""
        text = text.strip()
        
        # Already has "?"
        if text.endswith('?'):
            return text
        
        # Check if it's clearly a question (starts with question word)
        first_word = text.split()[0].lower() if text.split() else ''
        if first_word in self.QUESTION_WORDS:
            return text + '?'
        
        # Has question words in middle (like "can you", "should I")
        if any(f' {word} ' in text.lower() for word in ['can', 'should', 'could', 'would', 'does', 'do']):
            return text + '?'
        
        return text
    
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
    
    def extract_core_question(self, long_title: str) -> Optional[str]:
        """
        Extract core question from overly long title (Phase 2).
        
        Strategy:
        - Find first sentence with "?"
        - Or find first clause starting with question word
        - Or take first N words (smart truncation)
        """
        # Find first sentence with "?"
        sentences = re.split(r'(?<=[.!?])\s+', long_title)
        for sentence in sentences:
            if '?' in sentence and len(sentence) < self.MAX_QUESTION_LENGTH:
                return sentence
        
        # Find first clause with question word
        clauses = long_title.split(',')
        for clause in clauses:
            clause_clean = clause.strip()
            first_word = clause_clean.split()[0].lower() if clause_clean.split() else ''
            if first_word in self.QUESTION_WORDS and len(clause_clean) < self.MAX_QUESTION_LENGTH:
                # Ensure it ends with "?"
                return self.ensure_question_mark(clause_clean)
        
        # Smart truncation: Take first MAX_QUESTION_LENGTH chars at word boundary
        if len(long_title) > self.MAX_QUESTION_LENGTH:
            truncated = long_title[:self.MAX_QUESTION_LENGTH]
            # Find last space
            last_space = truncated.rfind(' ')
            if last_space > 50:  # Ensure we keep meaningful content
                truncated = truncated[:last_space]
            
            # Add "?" if missing
            return self.ensure_question_mark(truncated)
        
        return long_title
    
    def split_multiple_questions(self, text: str) -> List[str]:
        """Split text into separate questions if multiple exist."""
        # Pattern: question mark followed by capital letter, Also, And, or Additionally
        # This catches: "Question 1? Question 2" or "Question 1?Also, question 2"
        pattern = r'\?\s*(?=(?:[A-Z]|Also|And\s|Additionally))'
        
        # Split on the pattern
        parts = re.split(pattern, text)
        
        # Clean and filter
        questions = []
        for part in parts:
            part = part.strip()
            if part and not part.endswith('?'):
                part = part + '?'
            
            cleaned = self.clean_question(part)
            # Only keep if it's a valid question and long enough
            if self.is_question(cleaned) and len(cleaned) > 15:
                questions.append(cleaned)
        
        return questions if questions else [text]
    
    def extract_question(self, thread: Dict) -> Optional[str]:
        """Extract question from thread title and first post with Phase 2 improvements."""
        title = thread.get('title', '').strip()
        posts = thread.get('posts', [])
        content = posts[0].get('content', '').strip() if posts else ''
        
        # Clean the title
        cleaned_title = self.clean_question(title)
        
        # Phase 2: Strategy 1 - Short, complete title with "?" - use as-is
        if '?' in cleaned_title and len(cleaned_title) < self.MAX_QUESTION_LENGTH:
            return cleaned_title
        
        # Phase 2: Strategy 2 - Long title without "?" - extract core question
        if len(cleaned_title) > self.MAX_QUESTION_LENGTH:
            core_question = self.extract_core_question(cleaned_title)
            if core_question and len(core_question) > 10:
                return core_question
        
        # Phase 2: Strategy 3 - Title is topic, content has actual question
        if content and ':' in cleaned_title:
            # Title is likely "Topic: Description" format
            # Look for question in content
            content_questions = self.split_multiple_questions(content)
            if content_questions:
                # Merge topic with first question if short
                topic = cleaned_title.split(':')[0].strip()
                first_q = content_questions[0]
                
                if len(topic) < 30 and len(first_q) < 100:
                    return f"{topic}: {first_q}"
                else:
                    return first_q
        
        # Strategy 4: Title is complete and good - use it
        title_is_complete = (
            '?' in cleaned_title or 
            len(cleaned_title) > 80 or
            (len(cleaned_title) > 40 and not content)
        )
        
        if title_is_complete and len(cleaned_title) > 10:
            # Phase 2: Ensure question mark if it looks like a question
            return self.ensure_question_mark(cleaned_title)
        
        # Strategy 5: Title is incomplete - merge with content
        if not title_is_complete and content:
            # Check if content starts with similar text to title (avoid duplication)
            content_lower = content.lower()[:100]
            title_lower = cleaned_title.lower()
            
            # If content starts with title text, just use content
            if title_lower and len(title_lower) > 10 and content_lower.startswith(title_lower[:20]):
                # Content includes title, extract question from content
                questions = self.split_multiple_questions(content)
                if questions:
                    return self.ensure_question_mark(questions[0])
            else:
                # Title is a fragment - merge with content
                # Look for first question in content to append
                sentences = re.split(r'[.!?]+', content)
                for sentence in sentences:
                    cleaned_sent = self.clean_question(sentence)
                    if self.is_question(cleaned_sent) and len(cleaned_sent) > 10:
                        # Merge: "Title: Question from content?"
                        merged = f"{cleaned_title}: {cleaned_sent}"
                        merged = self.ensure_question_mark(merged)
                        
                        # Phase 2: Truncate if too long
                        if len(merged) > self.MAX_QUESTION_LENGTH:
                            merged = self.extract_core_question(merged) or merged[:self.MAX_QUESTION_LENGTH] + '?'
                        
                        return merged
                
                # No clear question in content, combine title with first sentence
                if sentences and len(sentences) > 0:
                    first_sentence = self.clean_question(sentences[0])
                    if len(first_sentence) > 10:
                        merged = f"{cleaned_title}: {first_sentence}"
                        merged = self.ensure_question_mark(merged)
                        
                        # Phase 2: Truncate if too long
                        if len(merged) > self.MAX_QUESTION_LENGTH:
                            merged = self.extract_core_question(merged) or merged[:self.MAX_QUESTION_LENGTH] + '?'
                        
                        return merged
        
        # Strategy 6: Try to find question in content only
        if content:
            questions = self.split_multiple_questions(content)
            if questions:
                return self.ensure_question_mark(questions[0])
        
        # Fallback to cleaned title with question mark
        if cleaned_title:
            return self.ensure_question_mark(cleaned_title)
        
        return None
    
    def extract_all_questions(self, thread: Dict) -> List[str]:
        """Extract all questions from thread (for multi-question posts)."""
        posts = thread.get('posts', [])
        if not posts:
            return []
        
        content = posts[0].get('content', '').strip()
        if not content:
            title = thread.get('title', '').strip()
            return [self.clean_question(title)] if title else []
        
        # Split content into multiple questions
        questions = self.split_multiple_questions(content)
        
        # If only one question and title exists, try merging
        if len(questions) == 1:
            title = thread.get('title', '').strip()
            if title:
                cleaned_title = self.clean_question(title)
                # Check if question seems incomplete (doesn't start with question word)
                first_word = questions[0].split()[0].lower() if questions[0].split() else ''
                if first_word not in self.QUESTION_WORDS and len(cleaned_title) < 80:
                    # Merge title with question
                    questions[0] = f"{cleaned_title}: {questions[0]}"
        
        return questions


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
        
        # Extract primary question (use single question extraction for now)
        # Multi-question support is available via extract_qa_pairs if needed
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
        
        # Rank answers (pass question for quote detection)
        ranked_answers = self.answer_ranker.rank_answers(reply_posts, original_author, question)
        
        if not ranked_answers:
            logger.debug(f"No useful answers found in thread {thread_id}")
            return None
        
        # Take top answer's cleaned content
        # ranked_answers is List[Tuple[post, score, cleaned_content]]
        top_post, top_score, top_cleaned = ranked_answers[0]
        answer_text = top_cleaned  # Use cleaned content
        
        # Build Q&A pair
        qa_pair = {
            'id': f"forum_{thread_id}",
            'source': 'forum',
            'thread_id': thread_id,
            'thread_url': thread_url,
            'question': question,
            'answer': answer_text,  # Cleaned answer
            'raw_question': {
                'title': thread.get('title', ''),
                'content': posts[0].get('content', '') if posts else '',
                'author': original_author,
                'date': posts[0].get('date', '') if posts else ''
            },
            'raw_answers': [
                {
                    'content': ans[0].get('content', ''),  # Original content
                    'cleaned_content': ans[2],  # Cleaned content
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
    
    def extract_qa_pairs(self, thread: Dict) -> List[Dict]:
        """Extract multiple Q&A pairs if thread contains multiple questions."""
        thread_id = thread.get('thread_id')
        thread_url = thread.get('url')
        language = thread.get('language', 'unknown')
        
        # Skip non-English threads
        if language != 'en':
            return []
        
        # Extract all questions
        questions = self.question_extractor.extract_all_questions(thread)
        if not questions:
            return []
        
        # Get posts
        posts = thread.get('posts', [])
        if len(posts) < 2:
            return []
        
        original_author = posts[0].get('author', '') if posts else ''
        reply_posts = posts[1:]
        
        # If only one question, use standard extraction
        if len(questions) == 1:
            pair = self.extract_qa_pair(thread)
            return [pair] if pair else []
        
        # Multiple questions - create separate pairs
        qa_pairs = []
        for idx, question in enumerate(questions):
            # Rank answers for this specific question
            ranked_answers = self.answer_ranker.rank_answers(reply_posts, original_author, question)
            
            if not ranked_answers:
                continue
            
            # Take top answer's cleaned content
            top_post, top_score, top_cleaned = ranked_answers[0]
            answer_text = top_cleaned
            
            # Build Q&A pair
            qa_pair = {
                'id': f"forum_{thread_id}_q{idx+1}",
                'source': 'forum',
                'thread_id': thread_id,
                'thread_url': thread_url,
                'question': question,
                'answer': answer_text,
                'raw_question': {
                    'title': thread.get('title', ''),
                    'content': posts[0].get('content', '') if posts else '',
                    'author': original_author,
                    'date': posts[0].get('date', '') if posts else '',
                    'question_index': idx + 1,
                    'total_questions': len(questions)
                },
                'raw_answers': [
                    {
                        'content': ans[0].get('content', ''),
                        'cleaned_content': ans[2],
                        'author': ans[0].get('author', ''),
                        'date': ans[0].get('date', ''),
                        'score': ans[1],
                        'rank': i + 1
                    }
                    for i, ans in enumerate(ranked_answers[:3])
                ],
                'metadata': {
                    'language': language,
                    'scraped_date': thread.get('scraped_date', ''),
                    'processed_date': datetime.now().isoformat(),
                    'answer_count': len(reply_posts),
                    'useful_answer_count': len(ranked_answers),
                    'is_multi_question': True
                }
            }
            
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
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
    
    def save_qa_pairs_archive(self, qa_pairs: List[Dict]):
        """Save an archival copy of Q&A pairs with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        archive_file = self.output_dir / f'forum_qa_pairs_{timestamp}.json'
        
        with archive_file.open('w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved archival copy to {archive_file}")
        return archive_file
    
    def save_qa_pairs(self, qa_pairs: List[Dict], filename: str = 'forum_qa_pairs.json'):
        """Save Q&A pairs to JSON file."""
        # Save archival copy first
        self.save_qa_pairs_archive(qa_pairs)
        
        # Save main file
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
