"""
Forum Q&A Pair Validator
========================
Validates and quality checks extracted forum Q&A pairs.

Features:
- Field validation (presence, types, non-empty)
- URL validation
- Language verification
- Duplicate detection
- Quality scoring
- Manual review sampling
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from urllib.parse import urlparse
import logging
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForumQAValidator:
    """Validates and quality checks forum Q&A pairs."""
    
    # Required fields for each Q&A pair
    REQUIRED_FIELDS = [
        'id', 'source', 'thread_id', 'thread_url', 
        'question', 'answer', 'raw_question', 'raw_answers', 'metadata'
    ]
    
    # Required fields in metadata
    REQUIRED_METADATA = [
        'language', 'scraped_date', 'processed_date', 
        'answer_count', 'useful_answer_count'
    ]
    
    # Quality thresholds
    MIN_QUESTION_LENGTH = 10  # Minimum characters for valid question
    MIN_ANSWER_LENGTH = 10    # Minimum characters for valid answer
    MIN_ANSWER_SCORE = 2      # Minimum score for acceptable answer
    
    def __init__(self):
        self.validation_errors = []
        self.warnings = []
        self.quality_issues = []
        
    def validate_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a list of Q&A pairs.
        
        Returns validation report with errors, warnings, and quality metrics.
        """
        logger.info(f"Validating {len(qa_pairs)} Q&A pairs...")
        
        self.validation_errors = []
        self.warnings = []
        self.quality_issues = []
        
        valid_pairs = []
        duplicate_tracker = {}  # Track duplicates by question
        
        for idx, pair in enumerate(qa_pairs):
            pair_id = pair.get('id', f'unknown_{idx}')
            
            # Field validation
            field_errors = self._validate_fields(pair, pair_id)
            if field_errors:
                self.validation_errors.extend(field_errors)
                continue
            
            # URL validation
            url_errors = self._validate_url(pair['thread_url'], pair_id)
            if url_errors:
                self.validation_errors.extend(url_errors)
            
            # Language validation
            if pair['metadata']['language'] != 'en':
                self.warnings.append(f"{pair_id}: Non-English language detected: {pair['metadata']['language']}")
            
            # Content validation
            content_issues = self._validate_content(pair, pair_id)
            if content_issues:
                self.quality_issues.extend(content_issues)
            
            # Duplicate detection
            question_normalized = self._normalize_text(pair['question'])
            if question_normalized in duplicate_tracker:
                self.warnings.append(
                    f"{pair_id}: Potential duplicate of {duplicate_tracker[question_normalized]} "
                    f"(question: '{pair['question'][:50]}...')"
                )
            else:
                duplicate_tracker[question_normalized] = pair_id
            
            valid_pairs.append(pair)
        
        # Generate quality metrics
        quality_metrics = self._calculate_quality_metrics(valid_pairs)
        
        # Create validation report
        report = {
            'validation_date': datetime.now().isoformat(),
            'total_pairs': len(qa_pairs),
            'valid_pairs': len(valid_pairs),
            'invalid_pairs': len(qa_pairs) - len(valid_pairs),
            'errors': len(self.validation_errors),
            'warnings': len(self.warnings),
            'quality_issues': len(self.quality_issues),
            'quality_metrics': quality_metrics,
            'error_details': self.validation_errors[:20],  # First 20 errors
            'warning_details': self.warnings[:20],  # First 20 warnings
            'quality_issue_details': self.quality_issues[:20]  # First 20 issues
        }
        
        logger.info(f"Validation complete: {len(valid_pairs)}/{len(qa_pairs)} valid pairs")
        logger.info(f"Errors: {len(self.validation_errors)}, Warnings: {len(self.warnings)}, "
                   f"Quality Issues: {len(self.quality_issues)}")
        
        return report
    
    def _validate_fields(self, pair: Dict[str, Any], pair_id: str) -> List[str]:
        """Validate that all required fields are present and non-empty."""
        errors = []
        
        # Check required top-level fields
        for field in self.REQUIRED_FIELDS:
            if field not in pair:
                errors.append(f"{pair_id}: Missing required field '{field}'")
            elif not pair[field]:
                errors.append(f"{pair_id}: Empty required field '{field}'")
        
        # Check metadata fields
        if 'metadata' in pair and isinstance(pair['metadata'], dict):
            for field in self.REQUIRED_METADATA:
                if field not in pair['metadata']:
                    errors.append(f"{pair_id}: Missing metadata field '{field}'")
        
        return errors
    
    def _validate_url(self, url: str, pair_id: str) -> List[str]:
        """Validate URL format and structure."""
        errors = []
        
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                errors.append(f"{pair_id}: Invalid URL scheme: {parsed.scheme}")
            
            # Check domain
            if 'boardgamegeek.com' not in parsed.netloc:
                errors.append(f"{pair_id}: Unexpected domain: {parsed.netloc}")
            
            # Check path structure
            if '/thread/' not in parsed.path:
                errors.append(f"{pair_id}: Invalid thread URL path: {parsed.path}")
                
        except Exception as e:
            errors.append(f"{pair_id}: URL parsing error: {str(e)}")
        
        return errors
    
    def _validate_content(self, pair: Dict[str, Any], pair_id: str) -> List[str]:
        """Validate content quality."""
        issues = []
        
        # Check question length
        question = pair.get('question', '')
        if len(question) < self.MIN_QUESTION_LENGTH:
            issues.append(
                f"{pair_id}: Question too short ({len(question)} chars): '{question}'"
            )
        
        # Check answer length
        answer = pair.get('answer', '')
        if len(answer) < self.MIN_ANSWER_LENGTH:
            issues.append(
                f"{pair_id}: Answer too short ({len(answer)} chars): '{answer}'"
            )
        
        # Check answer score (from raw_answers)
        raw_answers = pair.get('raw_answers', [])
        if raw_answers and len(raw_answers) > 0:
            top_score = raw_answers[0].get('score', 0)
            if top_score < self.MIN_ANSWER_SCORE:
                issues.append(
                    f"{pair_id}: Low answer score ({top_score}), may be low quality"
                )
        
        # Check for question marks in questions
        if '?' not in question and pair.get('raw_question', {}).get('content', ''):
            raw_content = pair['raw_question']['content']
            if '?' not in raw_content:
                issues.append(
                    f"{pair_id}: Question lacks question mark: '{question[:50]}...'"
                )
        
        return issues
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for duplicate detection."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _calculate_quality_metrics(self, valid_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality metrics for valid pairs."""
        if not valid_pairs:
            return {}
        
        # Question length stats
        question_lengths = [len(pair['question']) for pair in valid_pairs]
        
        # Answer length stats
        answer_lengths = [len(pair['answer']) for pair in valid_pairs]
        
        # Answer score stats
        answer_scores = []
        for pair in valid_pairs:
            raw_answers = pair.get('raw_answers', [])
            if raw_answers:
                answer_scores.append(raw_answers[0].get('score', 0))
        
        # Answer count stats
        answer_counts = [pair['metadata']['useful_answer_count'] for pair in valid_pairs]
        
        # Calculate stats
        metrics = {
            'question_length': {
                'min': min(question_lengths),
                'max': max(question_lengths),
                'avg': sum(question_lengths) / len(question_lengths),
                'median': sorted(question_lengths)[len(question_lengths) // 2]
            },
            'answer_length': {
                'min': min(answer_lengths),
                'max': max(answer_lengths),
                'avg': sum(answer_lengths) / len(answer_lengths),
                'median': sorted(answer_lengths)[len(answer_lengths) // 2]
            },
            'answer_score': {
                'min': min(answer_scores) if answer_scores else 0,
                'max': max(answer_scores) if answer_scores else 0,
                'avg': sum(answer_scores) / len(answer_scores) if answer_scores else 0,
                'median': sorted(answer_scores)[len(answer_scores) // 2] if answer_scores else 0
            },
            'answer_count': {
                'min': min(answer_counts),
                'max': max(answer_counts),
                'avg': sum(answer_counts) / len(answer_counts),
                'median': sorted(answer_counts)[len(answer_counts) // 2]
            },
            'pairs_with_single_answer': sum(1 for c in answer_counts if c == 1),
            'pairs_with_multiple_answers': sum(1 for c in answer_counts if c > 1),
            'high_quality_answers': sum(1 for s in answer_scores if s >= 7),
            'low_quality_answers': sum(1 for s in answer_scores if s < 4)
        }
        
        return metrics
    
    def select_manual_review_sample(
        self, 
        qa_pairs: List[Dict[str, Any]], 
        sample_size: int = 33
    ) -> List[Dict[str, Any]]:
        """
        Select a representative sample for manual review (10% of 330 = 33).
        
        Sampling strategy:
        - 10 high-score pairs (score >= 8)
        - 10 medium-score pairs (4 <= score < 8)
        - 10 low-score pairs (score < 4)
        - 3 edge cases (very short questions/answers, etc.)
        """
        logger.info(f"Selecting {sample_size} pairs for manual review...")
        
        # Categorize by score
        high_score = []
        medium_score = []
        low_score = []
        edge_cases = []
        
        for pair in qa_pairs:
            raw_answers = pair.get('raw_answers', [])
            score = raw_answers[0].get('score', 0) if raw_answers else 0
            
            # Check for edge cases
            question_len = len(pair.get('question', ''))
            answer_len = len(pair.get('answer', ''))
            
            if question_len < 20 or answer_len < 30 or '?' not in pair.get('question', ''):
                edge_cases.append(pair)
            elif score >= 8:
                high_score.append(pair)
            elif score >= 4:
                medium_score.append(pair)
            else:
                low_score.append(pair)
        
        # Sample from each category
        import random
        random.seed(42)  # Reproducible sampling
        
        sample = []
        sample.extend(random.sample(high_score, min(10, len(high_score))))
        sample.extend(random.sample(medium_score, min(10, len(medium_score))))
        sample.extend(random.sample(low_score, min(10, len(low_score))))
        sample.extend(random.sample(edge_cases, min(3, len(edge_cases))))
        
        logger.info(f"Selected {len(sample)} pairs for manual review")
        return sample


def main():
    """Run validation on forum Q&A pairs."""
    
    # Load Q&A pairs
    qa_file = Path('data/processed/forum_qa/forum_qa_pairs.json')
    
    if not qa_file.exists():
        logger.error(f"Q&A file not found: {qa_file}")
        return
    
    logger.info(f"Loading Q&A pairs from {qa_file}...")
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs")
    
    # Create validator
    validator = ForumQAValidator()
    
    # Run validation
    report = validator.validate_qa_pairs(qa_pairs)
    
    # Save validation report
    output_dir = Path('data/processed/forum_qa')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / 'validation_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Validation report saved to {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total pairs: {report['total_pairs']}")
    print(f"Valid pairs: {report['valid_pairs']}")
    print(f"Invalid pairs: {report['invalid_pairs']}")
    print(f"Errors: {report['errors']}")
    print(f"Warnings: {report['warnings']}")
    print(f"Quality issues: {report['quality_issues']}")
    print("\nQuality Metrics:")
    metrics = report['quality_metrics']
    print(f"  Question length (avg): {metrics['question_length']['avg']:.1f} chars")
    print(f"  Answer length (avg): {metrics['answer_length']['avg']:.1f} chars")
    print(f"  Answer score (avg): {metrics['answer_score']['avg']:.2f}")
    print(f"  High quality answers (score >= 7): {metrics['high_quality_answers']}")
    print(f"  Low quality answers (score < 4): {metrics['low_quality_answers']}")
    print(f"  Pairs with multiple answers: {metrics['pairs_with_multiple_answers']}")
    
    # Select manual review sample
    sample = validator.select_manual_review_sample(qa_pairs, sample_size=33)
    
    # Save sample for manual review
    sample_file = output_dir / 'manual_review_sample.json'
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Manual review sample (33 pairs) saved to {sample_file}")
    
    # Show first few errors/warnings if any
    if report['error_details']:
        print("\nSample Errors:")
        for error in report['error_details'][:5]:
            print(f"  - {error}")
    
    if report['warning_details']:
        print("\nSample Warnings:")
        for warning in report['warning_details'][:5]:
            print(f"  - {warning}")
    
    if report['quality_issue_details']:
        print("\nSample Quality Issues:")
        for issue in report['quality_issue_details'][:5]:
            print(f"  - {issue}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
