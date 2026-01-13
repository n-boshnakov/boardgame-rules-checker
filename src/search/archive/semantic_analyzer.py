"""
Semantic Question Analyzer - Advanced NLP using spaCy.

Note: Requires Pydantic v1 which is incompatible with Python 3.14+.
Use semantic_analyzer_nltk.py for newer Python versions.

Implements:
1. Morphological Analysis - Understanding word forms and structures
2. POS Tagging - Part-of-speech identification 
3. Word Sense Disambiguation - Resolving word meanings in context
4. Query Enhancement - Expanding queries with synonyms and related terms
"""

import spacy
from typing import List, Dict, Set, Tuple, Optional
import re
from collections import defaultdict

class SemanticQuestionAnalyzer:
    """Analyzes questions using advanced NLP to improve retrieval and answer generation."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize with spaCy model for NLP processing.
        
        Args:
            model_name: spaCy model to use (en_core_web_sm, en_core_web_md, en_core_web_lg)
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=False)
            self.nlp = spacy.load(model_name)
        
        # Game-specific terminology for word sense disambiguation
        self.domain_vocabulary = {
            # Actions/Verbs
            'move': ['relocate', 'traverse', 'travel', 'go'],
            'attack': ['combat', 'fight', 'engage', 'strike'],
            'draw': ['take', 'pick up', 'obtain'],
            'discard': ['remove', 'dispose', 'eliminate'],
            'place': ['put', 'position', 'set'],
            'flip': ['turn over', 'reverse', 'invert'],
            'reveal': ['show', 'uncover', 'expose'],
            'resolve': ['execute', 'complete', 'finish'],
            
            # Game Concepts
            'round': ['turn', 'cycle', 'phase'],
            'space': ['tile', 'location', 'position', 'square'],
            'token': ['marker', 'counter', 'piece'],
            'card': ['deck element'],
            'damage': ['injury', 'harm', 'wound', 'hurt'],
            'range': ['distance', 'reach'],
            'line of sight': ['LoS', 'visibility', 'view'],
            'anomaly': ['hazard', 'danger zone'],
            'artifact': ['special item', 'relic'],
            'mission': ['scenario', 'quest', 'objective'],
            'stalker': ['player character', 'character'],
            'enemy': ['opponent', 'hostile', 'foe'],
            'setup': ['preparation', 'initialization'],
            
            # Modifiers
            'adjacent': ['next to', 'neighboring', 'beside'],
            'occupied': ['full', 'containing entity'],
            'empty': ['vacant', 'unoccupied'],
        }
        
        # Question type patterns for better understanding
        self.question_types = {
            'how_to': r'\b(how (do|does|to|can|should)|what (is|are) the (steps|process|way|method))',
            'definition': r'\b(what (is|are|does|means?)|define|definition)',
            'quantity': r'\b(how many|how much|number of|amount of)',
            'location': r'\b(where|which space|which tile|location)',
            'timing': r'\b(when|at what point|timing|order|sequence)',
            'condition': r'\b(can you|is it possible|are you allowed|may you|under what condition)',
            'comparison': r'\b(difference between|versus|vs|compared to|rather than)',
            'reason': r'\b(why|reason|purpose|what for)',
        }
    
    def analyze(self, question: str) -> Dict:
        """Perform comprehensive semantic analysis on a question.
        
        Args:
            question: The input question string
            
        Returns:
            Dictionary containing:
                - original: Original question
                - doc: spaCy Doc object
                - tokens: List of token analysis
                - entities: Named entities found
                - key_terms: Important terms extracted
                - question_type: Identified question type
                - query_expansion: Synonyms and related terms
                - focus: Main focus of the question
                - action_verbs: Key action verbs
                - game_concepts: Game-specific concepts identified
        """
        doc = self.nlp(question.lower())
        
        analysis = {
            'original': question,
            'doc': doc,
            'tokens': self._analyze_tokens(doc),
            'entities': self._extract_entities(doc),
            'key_terms': self._extract_key_terms(doc),
            'question_type': self._identify_question_type(question),
            'query_expansion': self._expand_query(doc),
            'focus': self._identify_focus(doc),
            'action_verbs': self._extract_action_verbs(doc),
            'game_concepts': self._identify_game_concepts(doc),
            'dependencies': self._analyze_dependencies(doc)
        }
        
        return analysis
    
    def _analyze_tokens(self, doc) -> List[Dict]:
        """Morphological analysis and POS tagging for each token.
        
        Returns:
            List of token analyses with morphological features
        """
        tokens = []
        for token in doc:
            # Skip punctuation and stop words for key analysis
            if token.is_punct or token.is_space:
                continue
            
            token_info = {
                'text': token.text,
                'lemma': token.lemma_,  # Base form
                'pos': token.pos_,  # Part of speech
                'tag': token.tag_,  # Detailed POS tag
                'dep': token.dep_,  # Dependency relation
                'is_stop': token.is_stop,
                'is_alpha': token.is_alpha,
                'morph': str(token.morph),  # Morphological features
                'head': token.head.text if token.head != token else None,
            }
            tokens.append(token_info)
        
        return tokens
    
    def _extract_entities(self, doc) -> List[Dict]:
        """Extract named entities from the question."""
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities
    
    def _extract_key_terms(self, doc) -> List[str]:
        """Extract key terms that are likely important for retrieval.
        
        Focus on nouns, proper nouns, verbs, and adjectives.
        """
        key_terms = []
        for token in doc:
            # Include nouns, proper nouns, verbs (non-auxiliary), and adjectives
            if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not token.is_stop:
                key_terms.append(token.lemma_)
            # Also include game-specific terms even if they're stop words
            elif token.text.lower() in self.domain_vocabulary:
                key_terms.append(token.text.lower())
        
        return list(set(key_terms))  # Remove duplicates
    
    def _identify_question_type(self, question: str) -> str:
        """Identify the type of question being asked."""
        question_lower = question.lower()
        
        for qtype, pattern in self.question_types.items():
            if re.search(pattern, question_lower):
                return qtype
        
        return 'general'
    
    def _expand_query(self, doc) -> Dict[str, List[str]]:
        """Expand query with synonyms and related terms for better retrieval.
        
        Returns:
            Dictionary mapping original terms to their expansions
        """
        expansions = {}
        
        for token in doc:
            lemma = token.lemma_.lower()
            
            # Check if this term has domain-specific synonyms
            if lemma in self.domain_vocabulary:
                expansions[lemma] = self.domain_vocabulary[lemma]
            
            # Use spaCy's similarity for related terms (if available)
            if hasattr(token, 'similarity'):
                # Find related terms in our vocabulary
                related = []
                for domain_term in self.domain_vocabulary.keys():
                    if domain_term != lemma:
                        # Check similarity
                        domain_doc = self.nlp(domain_term)
                        if len(domain_doc) > 0:
                            sim = token.similarity(domain_doc[0])
                            if sim > 0.6:  # High similarity threshold
                                related.extend(self.domain_vocabulary[domain_term][:2])
                
                if related and lemma not in expansions:
                    expansions[lemma] = related[:3]  # Limit to top 3
        
        return expansions
    
    def _identify_focus(self, doc) -> Optional[str]:
        """Identify the main focus/subject of the question.
        
        Returns:
            The primary noun phrase or subject
        """
        # Look for the root of the dependency tree
        root = [token for token in doc if token.dep_ == 'ROOT']
        if not root:
            return None
        
        root_token = root[0]
        
        # Find the subject or object
        focus_candidates = []
        
        for token in doc:
            # Look for subjects, objects, and noun phrases
            if token.dep_ in ['nsubj', 'dobj', 'pobj', 'attr']:
                # Get the full noun phrase
                focus_candidates.append(' '.join([t.text for t in token.subtree]))
        
        # Return the first significant focus, or the root
        if focus_candidates:
            return focus_candidates[0]
        
        return root_token.text
    
    def _extract_action_verbs(self, doc) -> List[str]:
        """Extract action verbs from the question.
        
        These are often key to understanding what the question is about.
        """
        verbs = []
        for token in doc:
            if token.pos_ == 'VERB' and not token.is_stop:
                # Get the lemma (base form) of the verb
                verbs.append(token.lemma_)
        
        return verbs
    
    def _identify_game_concepts(self, doc) -> List[str]:
        """Identify game-specific concepts mentioned in the question."""
        concepts = []
        text_lower = doc.text.lower()
        
        # Check for multi-word concepts first
        for concept in self.domain_vocabulary.keys():
            if ' ' in concept:  # Multi-word concept
                if concept in text_lower:
                    concepts.append(concept)
        
        # Then check for single-word concepts
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in self.domain_vocabulary and lemma not in concepts:
                concepts.append(lemma)
        
        return concepts
    
    def _analyze_dependencies(self, doc) -> List[Dict]:
        """Analyze syntactic dependencies to understand relationships.
        
        Returns:
            List of important dependency relationships
        """
        dependencies = []
        
        for token in doc:
            if token.dep_ not in ['punct', 'det', 'aux']:
                dependencies.append({
                    'token': token.text,
                    'dep': token.dep_,
                    'head': token.head.text,
                    'children': [child.text for child in token.children]
                })
        
        return dependencies
    
    def enhance_query(self, question: str) -> str:
        """Generate an enhanced query string with synonyms and related terms.
        
        Args:
            question: Original question
            
        Returns:
            Enhanced query with added related terms
        """
        analysis = self.analyze(question)
        
        # Start with the original question
        enhanced_parts = [question]
        
        # Add key game concepts if found
        if analysis['game_concepts']:
            enhanced_parts.append(' '.join(analysis['game_concepts']))
        
        # Add important action verbs
        if analysis['action_verbs']:
            enhanced_parts.append(' '.join(analysis['action_verbs']))
        
        # Add synonyms for key terms (limit to avoid over-expansion)
        for term, synonyms in analysis['query_expansion'].items():
            if synonyms:
                # Add top 2 synonyms for each term
                enhanced_parts.append(' '.join(synonyms[:2]))
        
        return ' '.join(enhanced_parts)
    
    def extract_search_terms(self, question: str) -> List[str]:
        """Extract the most important terms for search.
        
        Args:
            question: Original question
            
        Returns:
            List of terms ranked by importance
        """
        analysis = self.analyze(question)
        
        # Build ranked list of search terms
        search_terms = []
        
        # Priority 1: Game concepts (highest importance)
        search_terms.extend(analysis['game_concepts'])
        
        # Priority 2: Action verbs
        search_terms.extend(analysis['action_verbs'])
        
        # Priority 3: Key terms (nouns, adjectives)
        search_terms.extend([t for t in analysis['key_terms'] 
                            if t not in search_terms])
        
        # Priority 4: Focus/subject
        if analysis['focus'] and analysis['focus'] not in ' '.join(search_terms):
            search_terms.append(analysis['focus'])
        
        return search_terms
    
    def get_question_intent(self, question: str) -> Dict:
        """Determine the intent and key information needs of the question.
        
        Returns:
            Dictionary with intent classification and information needs
        """
        analysis = self.analyze(question)
        
        intent = {
            'type': analysis['question_type'],
            'primary_action': analysis['action_verbs'][0] if analysis['action_verbs'] else None,
            'main_subject': analysis['focus'],
            'game_concepts': analysis['game_concepts'],
            'needs_procedural': analysis['question_type'] in ['how_to', 'timing'],
            'needs_definition': analysis['question_type'] == 'definition',
            'needs_quantitative': analysis['question_type'] == 'quantity',
            'needs_conditional': analysis['question_type'] == 'condition',
        }
        
        return intent
