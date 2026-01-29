"""
Multi-dimensional answer scoring system.

Evaluates answers across multiple dimensions:
1. Relevance - How well the answer addresses the question
2. Completeness - Whether key facts/information are present
3. Accuracy - Factual correctness against ground truth
4. Conciseness - Appropriate length without excessive verbosity

Replaces single-score evaluation with weighted multi-dimensional approach.
"""

from sentence_transformers import CrossEncoder, SentenceTransformer
from typing import Dict, List, Optional, Tuple
import re
from difflib import SequenceMatcher
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MultiDimensionalScorer:
    """Evaluates answers across multiple quality dimensions."""
    
    def __init__(self, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the multi-dimensional scorer.
        
        Args:
            cross_encoder_model: Model for relevance scoring
        """
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        # Add SentenceTransformer for semantic embeddings
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Default weights for each dimension (adjusted for semantic understanding)
        # Heavily prioritize accuracy since it uses robust semantic embeddings
        self.weights = {
            'relevance': 0.30,      # How relevant to the question
            'completeness': 0.25,   # Contains key information
            'accuracy': 0.40,       # Factually correct (increased - most important!)
            'conciseness': 0.05     # Appropriate length (reduced - less critical)
        }
        
        # Question type keywords for completeness checking
        self.question_keywords = {
            'procedural': ['how', 'steps', 'process', 'perform', 'do'],
            'definitional': ['what', 'define', 'means', 'is'],
            'quantitative': ['how many', 'how much', 'number'],
            'conditional': ['when', 'if', 'during', 'while'],
            'permission': ['can', 'may', 'allowed', 'able to'],
            'locational': ['where', 'which space', 'location']
        }
    
    def score_answer(
        self,
        question: str,
        answer: str,
        ground_truth: Optional[str] = None,
        chunks: Optional[List[Dict]] = None
    ) -> Dict[str, float]:
        """
        Compute multi-dimensional score for an answer.
        
        Args:
            question: The question being answered
            answer: The generated answer
            ground_truth: Optional ground truth for accuracy comparison
            chunks: Optional source chunks for context
            
        Returns:
            Dictionary with scores for each dimension and overall score
        """
        scores = {}
        
        # 1. Relevance Score (CrossEncoder Q→A)
        scores['relevance'] = self._score_relevance(question, answer)
        
        # 2. Completeness Score (key information present)
        scores['completeness'] = self._score_completeness(question, answer)
        
        # 3. Accuracy Score (factual correctness)
        if ground_truth:
            scores['accuracy'] = self._score_accuracy(answer, ground_truth)
        else:
            # Without GT, use chunk consistency as proxy
            scores['accuracy'] = self._score_chunk_consistency(answer, chunks) if chunks else 0.5
        
        # 4. Conciseness Score (appropriate length)
        scores['conciseness'] = self._score_conciseness(question, answer)
        
        # Calculate weighted overall score
        overall = sum(scores[dim] * self.weights[dim] for dim in self.weights.keys())
        scores['overall'] = overall
        
        # Add metadata
        scores['answer_length'] = len(answer)
        scores['question_length'] = len(question)
        
        return scores
    
    def _score_relevance(self, question: str, answer: str) -> float:
        """
        Score how relevant the answer is to the question.
        Uses CrossEncoder for semantic similarity.
        
        Returns: Normalized score 0-1
        """
        try:
            # CrossEncoder gives scores typically in range [-10, 10]
            # Normalize to [0, 1] using sigmoid-like transformation
            raw_score = self.cross_encoder.predict([(question, answer)])[0]
            
            # Sigmoid normalization: score in [-10, 10] → [0, 1]
            normalized = 1 / (1 + np.exp(-raw_score / 3.0))
            
            return float(normalized)
        except Exception:
            return 0.5  # Neutral score on error
    
    def _score_completeness(self, question: str, answer: str) -> float:
        """
        Score whether the answer contains key information expected based on question type.
        
        Returns: Score 0-1
        """
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Detect question type
        question_type = self._detect_question_type(question_lower)
        
        score = 0.5  # Base score
        
        # Check for question type-specific completeness markers
        if question_type == 'procedural':
            # Should have action verbs, sequence markers
            has_actions = any(word in answer_lower for word in ['move', 'place', 'perform', 'draw', 'roll', 'must', 'should'])
            has_sequence = any(word in answer_lower for word in ['first', 'then', 'after', 'before', 'when', 'next'])
            score += 0.25 if has_actions else 0
            score += 0.25 if has_sequence else 0
            
        elif question_type == 'definitional':
            # Should have defining language
            has_definition = any(word in answer_lower for word in ['is', 'are', 'means', 'refers to', 'represents'])
            has_examples = any(word in answer_lower for word in ['such as', 'example', 'including', 'like', 'e.g.'])
            score += 0.3 if has_definition else 0
            score += 0.2 if has_examples else 0
            
        elif question_type == 'quantitative':
            # Should have numbers
            has_numbers = bool(re.search(r'\d+', answer))
            has_quantity_words = any(word in answer_lower for word in ['each', 'per', 'total', 'maximum', 'minimum'])
            score += 0.3 if has_numbers else 0
            score += 0.2 if has_quantity_words else 0
            
        elif question_type == 'permission':
            # Should have clear yes/no indication
            has_permission_marker = any(word in answer_lower for word in ['can', 'may', 'allowed', 'must', "can't", 'cannot', 'forbidden', 'not allowed'])
            has_conditional = any(word in answer_lower for word in ['if', 'when', 'unless', 'except', 'only'])
            score += 0.3 if has_permission_marker else 0
            score += 0.2 if has_conditional else 0
        
        # Extract key entities from question and check if they appear in answer
        question_words = set(re.findall(r'\b[a-z]{4,}\b', question_lower))
        answer_words = set(re.findall(r'\b[a-z]{4,}\b', answer_lower))
        
        # Remove common words
        common_words = {'this', 'that', 'what', 'when', 'where', 'which', 'while', 'with', 'from', 'have', 'does', 'their', 'they', 'your', 'about'}
        question_words -= common_words
        
        if question_words:
            key_word_coverage = len(question_words & answer_words) / len(question_words)
            # Weight entity coverage (up to 0.3 bonus)
            score += key_word_coverage * 0.3
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _score_accuracy(self, answer: str, ground_truth: str) -> float:
        """
        Score factual accuracy by comparing with ground truth.
        Uses multiple semantic and lexical measures.
        
        Returns: Score 0-1
        """
        # Multiple accuracy measures
        
        # 1. Semantic similarity using sentence embeddings (most robust)
        try:
            answer_embedding = self.sentence_transformer.encode([answer])
            gt_embedding = self.sentence_transformer.encode([ground_truth])
            embedding_similarity = cosine_similarity(answer_embedding, gt_embedding)[0][0]
            # Normalize to 0-1 range (cosine similarity is already in [-1, 1], typically [0, 1] for similar texts)
            embedding_similarity = max(0.0, float(embedding_similarity))
        except Exception:
            embedding_similarity = 0.5
        
        # 2. CrossEncoder semantic similarity (second measure)
        try:
            cross_score = self.cross_encoder.predict([(ground_truth, answer)])[0]
            cross_normalized = 1 / (1 + np.exp(-cross_score / 3.0))
        except Exception:
            cross_normalized = 0.5
        
        # 3. Word overlap (Jaccard similarity for key terms)
        answer_words = set(re.findall(r'\b[a-z]{4,}\b', answer.lower()))
        gt_words = set(re.findall(r'\b[a-z]{4,}\b', ground_truth.lower()))
        
        if gt_words:
            word_overlap = len(answer_words & gt_words) / len(gt_words)
        else:
            word_overlap = 0.5
        
        # 4. Sequence matching (for exact phrases)
        sequence_ratio = SequenceMatcher(None, answer.lower(), ground_truth.lower()).ratio()
        
        # Combine measures with adaptive weights based on semantic confidence
        # If embedding similarity is high (>0.7), trust it more and reduce strictness
        if embedding_similarity > 0.7:
            # High semantic similarity - answer is likely correct, boost the score
            accuracy = (
                0.60 * embedding_similarity +  # Trust embeddings heavily
                0.25 * cross_normalized +      # CrossEncoder validation
                0.10 * word_overlap +          # Term coverage (reduced weight)
                0.05 * sequence_ratio          # Exact matching (minimal weight)
            )
        else:
            # Lower semantic similarity - use balanced approach
            accuracy = (
                0.45 * embedding_similarity +  # Sentence embeddings
                0.25 * cross_normalized +      # CrossEncoder validation
                0.20 * word_overlap +          # Term coverage
                0.10 * sequence_ratio          # Exact phrase matching
            )
        
        # Tiered paraphrase boosting - trust high semantic similarity
        if embedding_similarity > 0.8:
            # Very high semantic match - definitely correct
            accuracy = max(accuracy, 0.90)
        elif embedding_similarity > 0.7:
            # High semantic match - likely correct
            accuracy = max(accuracy, 0.85)
        elif embedding_similarity > 0.6:
            # Moderate semantic match - possibly correct
            accuracy = max(accuracy, 0.75)
        
        return float(accuracy)
    
    def _score_chunk_consistency(self, answer: str, chunks: List[Dict]) -> float:
        """
        Score how consistent the answer is with source chunks (proxy for accuracy).
        
        Returns: Score 0-1
        """
        if not chunks:
            return 0.5
        
        answer_lower = answer.lower()
        
        # Check how much of the answer comes from high-scoring chunks
        total_overlap = 0
        chunk_texts = [chunk.get('text', '').lower() for chunk in chunks[:5]]
        
        # Extract sentences from answer
        answer_sentences = re.split(r'[.!?]+', answer_lower)
        answer_sentences = [s.strip() for s in answer_sentences if len(s.strip()) > 10]
        
        if not answer_sentences:
            return 0.5
        
        # Check each sentence against chunks
        for sentence in answer_sentences:
            sentence_words = set(sentence.split())
            best_overlap = 0
            
            for chunk_text in chunk_texts:
                chunk_words = set(chunk_text.split())
                if chunk_words:
                    overlap = len(sentence_words & chunk_words) / len(sentence_words)
                    best_overlap = max(best_overlap, overlap)
            
            total_overlap += best_overlap
        
        avg_overlap = total_overlap / len(answer_sentences)
        return float(avg_overlap)
    
    def _score_conciseness(self, question: str, answer: str) -> float:
        """
        Score whether the answer is appropriately concise.
        Penalizes both excessive verbosity and insufficient detail.
        
        Returns: Score 0-1
        """
        question_length = len(question)
        answer_length = len(answer)
        
        # Determine optimal answer length based on question type
        question_type = self._detect_question_type(question.lower())
        
        optimal_ranges = {
            'procedural': (400, 800),      # Need steps and details
            'definitional': (200, 500),    # Concise but complete
            'quantitative': (100, 300),    # Short, specific answer
            'permission': (150, 400),      # Yes/no with explanation
            'locational': (100, 300),      # Specific location info
            'conditional': (200, 500),     # Conditions and outcomes
            'general': (200, 600)          # Default range
        }
        
        min_len, max_len = optimal_ranges.get(question_type, (200, 600))
        
        # Score based on how close to optimal range
        if answer_length < min_len:
            # Too short - linear penalty
            score = answer_length / min_len
        elif answer_length > max_len:
            # Too long - diminishing penalty
            excess = answer_length - max_len
            penalty = excess / (max_len * 2)  # Penalty grows slower
            score = max(0, 1 - penalty)
        else:
            # In optimal range - perfect score
            score = 1.0
        
        # Bonus for being near the middle of range
        if min_len <= answer_length <= max_len:
            middle = (min_len + max_len) / 2
            distance_from_middle = abs(answer_length - middle) / (max_len - min_len)
            middle_bonus = 0.1 * (1 - distance_from_middle)
            score = min(score + middle_bonus, 1.0)
        
        return float(score)
    
    def _detect_question_type(self, question_lower: str) -> str:
        """Detect the type of question being asked."""
        for qtype, keywords in self.question_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return qtype
        return 'general'
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Update the scoring weights.
        
        Args:
            weights: Dictionary with keys: relevance, completeness, accuracy, conciseness
        """
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self.weights.update(weights)
    
    def explain_score(self, scores: Dict[str, float]) -> str:
        """
        Generate human-readable explanation of scores.
        
        Args:
            scores: Score dictionary from score_answer()
            
        Returns:
            Formatted explanation string
        """
        explanation = [
            f"Overall Score: {scores['overall']:.3f}",
            "",
            "Breakdown:",
            f"  Relevance:    {scores['relevance']:.3f} (weight: {self.weights['relevance']:.0%})",
            f"  Completeness: {scores['completeness']:.3f} (weight: {self.weights['completeness']:.0%})",
            f"  Accuracy:     {scores['accuracy']:.3f} (weight: {self.weights['accuracy']:.0%})",
            f"  Conciseness:  {scores['conciseness']:.3f} (weight: {self.weights['conciseness']:.0%})",
            "",
            f"Answer length: {scores['answer_length']} characters"
        ]
        
        # Add interpretation
        overall = scores['overall']
        if overall >= 0.8:
            interpretation = "Excellent - High quality answer"
        elif overall >= 0.6:
            interpretation = "Good - Satisfactory answer"
        elif overall >= 0.4:
            interpretation = "Fair - Some improvement needed"
        else:
            interpretation = "Poor - Significant issues"
        
        explanation.append(f"\nInterpretation: {interpretation}")
        
        return "\n".join(explanation)


# Convenience function for single-call scoring
def score_answer(
    question: str,
    answer: str,
    ground_truth: Optional[str] = None,
    chunks: Optional[List[Dict]] = None,
    return_explanation: bool = False
) -> Tuple[float, Dict[str, float], Optional[str]]:
    """
    Score an answer using multi-dimensional evaluation.
    
    Args:
        question: The question
        answer: The generated answer
        ground_truth: Optional ground truth for accuracy
        chunks: Optional source chunks
        return_explanation: Whether to return explanation text
        
    Returns:
        Tuple of (overall_score, detailed_scores, explanation_text)
    """
    scorer = MultiDimensionalScorer()
    scores = scorer.score_answer(question, answer, ground_truth, chunks)
    
    explanation = scorer.explain_score(scores) if return_explanation else None
    
    return scores['overall'], scores, explanation


if __name__ == "__main__":
    # Test the scorer
    scorer = MultiDimensionalScorer()
    
    test_question = "How do I perform an attack action?"
    test_answer = "To perform an attack, choose an Attack Action and a valid Target within range and line of sight. Roll the required dice and compare results to the target's body part thresholds."
    test_gt = "Choose an Attack Action and a valid Target. The Target must be within the Weapon's Max. Range and in LoS. Roll dice and resolve the results."
    
    scores = scorer.score_answer(test_question, test_answer, test_gt)
    
    print("="*60)
    print("MULTI-DIMENSIONAL SCORING TEST")
    print("="*60)
    print(f"\nQuestion: {test_question}")
    print(f"\nAnswer: {test_answer}")
    print(f"\nGround Truth: {test_gt}")
    print("\n" + "="*60)
    print(scorer.explain_score(scores))
    print("="*60)
