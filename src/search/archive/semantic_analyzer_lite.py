"""
Lightweight Semantic Question Analyzer - Minimal dependencies.
Uses basic NLP techniques: regex patterns, word matching, and term expansion.
No external NLP library required, but less sophisticated than NLTK or spaCy versions.
"""

import re
from typing import List, Dict, Set, Optional
from collections import defaultdict

class SemanticQuestionAnalyzer:
    """Lightweight semantic analyzer using regex and pattern matching."""
    
    def __init__(self):
        """Initialize with game-specific vocabulary and patterns."""
        # Game-specific terminology for word sense expansion
        self.domain_vocabulary = {
            # Actions/Verbs
            'move': ['relocate', 'traverse', 'travel', 'go'],
            'attack': ['combat', 'fight', 'engage', 'strike'],
            'draw': ['take', 'pick up', 'obtain'],
            'discard': ['remove', 'dispose', 'eliminate'],
            'place': ['put', 'position', 'set'],
            'flip': ['turn over', 'reverse'],
            'reveal': ['show', 'uncover'],
            'resolve': ['execute', 'complete'],
            
            # Game Concepts
            'round': ['turn', 'cycle', 'phase'],
            'space': ['tile', 'location', 'position'],
            'token': ['marker', 'counter'],
            'damage': ['injury', 'wound'],
            'range': ['distance', 'reach'],
            'anomaly': ['hazard'],
            'artifact': ['item'],
            'mission': ['scenario', 'quest'],
            'stalker': ['character'],
            'enemy': ['hostile', 'foe'],
            
            # Modifiers
            'adjacent': ['next to', 'neighboring'],
        }
        
        # Question type patterns
        self.question_types = {
            'how_to': r'\b(how (do|does|to|can|should)|what (is|are) the (steps|process|way))',
            'definition': r'\b(what (is|are|does|means?)|define)',
            'quantity': r'\b(how many|how much|number of|amount of)',
            'location': r'\b(where|which space|which tile)',
            'timing': r'\b(when|at what point|timing|order)',
            'condition': r'\b(can you|is it possible|are you allowed|may you)',
            'comparison': r'\b(difference between|versus|vs|compared)',
            'reason': r'\b(why|reason|purpose)',
        }
        
        # Key action verbs for game mechanics
        self.action_verbs = {
            'move', 'attack', 'draw', 'discard', 'place', 'flip', 'reveal',
            'resolve', 'use', 'take', 'gain', 'lose', 'damage', 'heal',
            'roll', 'play', 'activate', 'trigger', 'setup', 'remove'
        }
        
        # Game concepts
        self.game_concepts = {
            'stalker', 'enemy', 'anomaly', 'artifact', 'mission', 'token',
            'card', 'space', 'tile', 'round', 'turn', 'phase', 'range',
            'damage', 'line of sight', 'los', 'hp', 'action', 'movement'
        }
    
    def analyze(self, question: str) -> Dict:
        """Perform lightweight semantic analysis on a question."""
        question_lower = question.lower()
        
        return {
            'original': question,
            'question_type': self._identify_question_type(question),
            'key_terms': self._extract_key_terms(question_lower),
            'action_verbs': self._extract_action_verbs(question_lower),
            'game_concepts': self._identify_game_concepts(question_lower),
            'query_expansion': self._expand_query_terms(question_lower),
        }
    
    def _identify_question_type(self, question: str) -> str:
        """Identify question type using regex patterns."""
        question_lower = question.lower()
        for qtype, pattern in self.question_types.items():
            if re.search(pattern, question_lower):
                return qtype
        return 'general'
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract important words (simplified without POS tagging)."""
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'may', 'might', 'must', 'can', 'of', 'in', 'on', 'at', 'to',
            'for', 'with', 'from', 'by', 'as', 'or', 'and', 'but', 'if', 'then',
            'this', 'that', 'these', 'those', 'my', 'your', 'their', 'i', 'you'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        return list(set(key_terms))
    
    def _extract_action_verbs(self, text: str) -> List[str]:
        """Extract action verbs from question."""
        words = re.findall(r'\b\w+\b', text)
        verbs = [w for w in words if w in self.action_verbs]
        return list(set(verbs))
    
    def _identify_game_concepts(self, text: str) -> List[str]:
        """Identify game-specific concepts in the text."""
        concepts = []
        for concept in self.game_concepts:
            if concept in text:
                concepts.append(concept)
        return concepts
    
    def _expand_query_terms(self, text: str) -> Dict[str, List[str]]:
        """Expand terms with synonyms from domain vocabulary."""
        expansions = {}
        for term, synonyms in self.domain_vocabulary.items():
            if term in text:
                expansions[term] = synonyms
        return expansions
    
    def enhance_query(self, question: str) -> str:
        """Generate enhanced query with additional terms."""
        analysis = self.analyze(question)
        
        enhanced_parts = [question]
        
        # Add game concepts
        if analysis['game_concepts']:
            enhanced_parts.append(' '.join(analysis['game_concepts']))
        
        # Add action verbs
        if analysis['action_verbs']:
            enhanced_parts.append(' '.join(analysis['action_verbs']))
        
        # Add synonyms (limit to avoid over-expansion)
        for term, synonyms in analysis['query_expansion'].items():
            if synonyms:
                enhanced_parts.append(' '.join(synonyms[:2]))
        
        return ' '.join(enhanced_parts)
    
    def extract_search_terms(self, question: str) -> List[str]:
        """Extract ranked search terms."""
        analysis = self.analyze(question)
        
        search_terms = []
        search_terms.extend(analysis['game_concepts'])
        search_terms.extend(analysis['action_verbs'])
        search_terms.extend([t for t in analysis['key_terms'] 
                            if t not in search_terms][:10])
        
        return search_terms
    
    def get_question_intent(self, question: str) -> Dict:
        """Determine question intent."""
        analysis = self.analyze(question)
        
        intent = {
            'type': analysis['question_type'],
            'primary_action': analysis['action_verbs'][0] if analysis['action_verbs'] else None,
            'game_concepts': analysis['game_concepts'],
            'needs_procedural': analysis['question_type'] in ['how_to', 'timing'],
            'needs_definition': analysis['question_type'] == 'definition',
            'needs_quantitative': analysis['question_type'] == 'quantity',
            'needs_conditional': analysis['question_type'] == 'condition',
        }
        
        return intent
