"""
NLTK-based semantic analyzer for board game QA system.
Provides full NLP capabilities (POS tagging, lemmatization, morphological analysis)
while maintaining compatibility with Python 3.14+ (unlike spaCy which requires Pydantic v1).
"""
import re
from typing import Dict, List, Set
import nltk
from collections import defaultdict

class SemanticAnalyzerNLTK:
    """Enhanced semantic analysis using NLTK."""
    
    def __init__(self):
        """Initialize NLTK components."""
        try:
            # Download required NLTK data
            import os
            nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
            
            # Only download if not already present
            required_packages = [
                'averaged_perceptron_tagger', 
                'averaged_perceptron_tagger_eng',
                'punkt', 
                'punkt_tab', 
                'wordnet', 
                'omw-1.4'
            ]
            for package in required_packages:
                try:
                    if package in ['punkt', 'punkt_tab']:
                        nltk.data.find(f'tokenizers/{package}')
                    elif package in ['wordnet', 'omw-1.4']:
                        nltk.data.find(f'corpora/{package}')
                    else:
                        nltk.data.find(f'taggers/{package}')
                except LookupError:
                    nltk.download(package, quiet=True)
            
            from nltk import pos_tag, word_tokenize
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import wordnet
            
            self.lemmatizer = WordNetLemmatizer()
            self.pos_tag = pos_tag
            self.word_tokenize = word_tokenize
            self.wordnet = wordnet
            
        except Exception:
            self.lemmatizer = None
        
        # Domain vocabulary for board game concepts
        # Each concept includes itself plus synonyms/related terms
        self.game_vocabulary = {
            'action': ['action', 'perform', 'execute', 'take', 'do', 'make', 'conduct'],
            'movement': ['movement', 'move', 'travel', 'relocate', 'position', 'place'],
            'combat': ['combat', 'fight', 'attack', 'battle', 'engage', 'shoot', 'damage', 'wound'],
            'resource': ['resource', 'gain', 'spend', 'pay', 'receive', 'acquire', 'collect'],
            'card': ['card', 'draw', 'discard', 'play', 'reveal', 'hand'],
            'turn': ['turn', 'round', 'phase', 'step', 'sequence', 'order'],
            'player': ['player', 'stalker', 'character', 'hero', 'agent'],
            'equipment': ['equipment', 'gear', 'item', 'weapon', 'tool', 'artifact'],
            'location': ['location', 'zone', 'area', 'space', 'territory', 'region', 'base'],
            'condition': ['condition', 'status', 'effect', 'modifier', 'state'],
            'test': ['test', 'check', 'roll', 'skill test', 'dice', 'attribute test'],
            'enemy': ['enemy', 'mutant', 'monster', 'creature', 'threat', 'anomaly'],
            'mission': ['mission', 'quest', 'task', 'objective', 'goal'],
            'win': ['win', 'victory', 'succeed', 'complete', 'achieve'],
            'lose': ['lose', 'defeat', 'fail', 'eliminated'],
        }
        
        # Reverse mapping for concept identification
        self.term_to_concept = {}
        for concept, terms in self.game_vocabulary.items():
            for term in terms:
                self.term_to_concept[term.lower()] = concept
        
        # Question type patterns
        self.question_patterns = {
            'how_many': r'\b(how many|how much|what number)\b',
            'how_to': r'\b(how (do|does|can|to)|what (is the (way|process|method)|are the steps))\b',
            'what_is': r'\b(what is|what are|what does|define)\b',
            'when': r'\b(when|at what point|during which)\b',
            'where': r'\b(where|in which location)\b',
            'who': r'\b(who|which player)\b',
            'why': r'\b(why|for what reason)\b',
            'can': r'\b(can|may|is it (possible|allowed))\b',
        }
    
    def get_wordnet_pos(self, treebank_tag):
        """Convert Penn Treebank POS tag to WordNet POS tag."""
        if treebank_tag.startswith('J'):
            return self.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return self.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return self.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return self.wordnet.ADV
        else:
            return self.wordnet.NOUN
    
    def analyze(self, text: str) -> Dict:
        """
        Perform semantic analysis on text.
        
        Returns:
            Dictionary with:
            - lemmas: List of lemmatized words
            - pos_tags: List of (word, POS tag) tuples
            - game_concepts: List of identified game concepts
            - action_verbs: List of action verbs
            - key_nouns: List of important nouns
        """
        result = {
            'lemmas': [],
            'pos_tags': [],
            'game_concepts': [],
            'action_verbs': [],
            'key_nouns': [],
            'entities': [],
        }
        
        if not self.lemmatizer:
            return result
        
        try:
            # Tokenize and POS tag
            tokens = self.word_tokenize(text.lower())
            pos_tags = self.pos_tag(tokens)
            result['pos_tags'] = pos_tags
            
            # Lemmatize with POS awareness
            for word, tag in pos_tags:
                if word.isalpha():
                    wn_tag = self.get_wordnet_pos(tag)
                    lemma = self.lemmatizer.lemmatize(word, pos=wn_tag)
                    result['lemmas'].append(lemma)
                    
                    # Identify action verbs
                    if tag.startswith('VB'):
                        result['action_verbs'].append(lemma)
                    
                    # Identify key nouns
                    if tag.startswith('NN'):
                        result['key_nouns'].append(lemma)
                    
                    # Identify game concepts
                    if lemma in self.term_to_concept:
                        concept = self.term_to_concept[lemma]
                        if concept not in result['game_concepts']:
                            result['game_concepts'].append(concept)
            
            # Look for multi-word game concepts
            text_lower = text.lower()
            for concept, terms in self.game_vocabulary.items():
                for term in terms:
                    if len(term.split()) > 1 and term in text_lower:
                        if concept not in result['game_concepts']:
                            result['game_concepts'].append(concept)
            
            # Extract named entities (capitalized sequences)
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            result['entities'] = list(set(entities))
            
        except Exception:
            pass
        
        return result
    
    def get_question_intent(self, question: str) -> Dict:
        """
        Classify question intent.
        
        Returns:
            Dictionary with intent flags:
            - question_type: Primary question type
            - needs_procedural: Requires step-by-step explanation
            - needs_definition: Requires conceptual definition
            - needs_quantitative: Requires numeric answer
        """
        intent = {
            'question_type': 'general',
            'needs_procedural': False,
            'needs_definition': False,
            'needs_quantitative': False,
            'needs_temporal': False,
            'needs_locational': False,
            'needs_permission': False,
        }
        
        question_lower = question.lower()
        
        # Detect question type
        for qtype, pattern in self.question_patterns.items():
            if re.search(pattern, question_lower):
                intent['question_type'] = qtype
                break
        
        # Set intent flags
        if intent['question_type'] in ['how_to']:
            intent['needs_procedural'] = True
        elif intent['question_type'] in ['what_is']:
            intent['needs_definition'] = True
        elif intent['question_type'] == 'how_many':
            intent['needs_quantitative'] = True
        elif intent['question_type'] == 'when':
            intent['needs_temporal'] = True
        elif intent['question_type'] == 'where':
            intent['needs_locational'] = True
        elif intent['question_type'] == 'can':
            intent['needs_permission'] = True
        
        return intent
    
    def extract_search_terms(self, text: str) -> List[str]:
        """
        Extract key search terms from text.
        
        Returns:
            List of important terms for search.
        """
        analysis = self.analyze(text)
        
        # Combine different term types
        search_terms = []
        
        # Add key nouns (most important)
        search_terms.extend(analysis['key_nouns'][:3])
        
        # Add action verbs
        search_terms.extend(analysis['action_verbs'][:2])
        
        # Add entities
        search_terms.extend([e.lower() for e in analysis['entities'][:2]])
        
        # Add game concepts
        search_terms.extend(analysis['game_concepts'][:2])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            if term not in seen and len(term) > 2:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms
    
    def enhance_query(self, query: str, max_additions: int = 1) -> str:
        """
        Enhance a query with semantic expansion - very conservative.
        
        Args:
            query: Original query
            max_additions: Maximum number of terms to add (default: 1)
        
        Returns:
            Enhanced query string.
        """
        analysis = self.analyze(query)
        intent = self.get_question_intent(query)
        
        # Start with original query
        enhanced = query
        synonym_additions = []  # Prioritize synonyms
        concept_additions = []  # Secondary: concept words
        
        # Dynamic synonym expansion: For any concept word in query, add its most relevant synonym
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for concept, synonyms in self.game_vocabulary.items():
            # Check if any term from this concept appears as a whole word in query
            matching_terms = [term for term in synonyms if term in query_words]
            if matching_terms:
                # Find a synonym NOT already in the query to add
                for synonym in synonyms:
                    if synonym not in query_words and synonym not in matching_terms:
                        synonym_additions.append(synonym)
                        break  # Only add one synonym per concept
        
        # Only add the single most relevant term based on question type
        if intent['needs_procedural'] and analysis['action_verbs']:
            # For procedural, add primary action verb
            concept_additions.append(analysis['action_verbs'][0])
        elif intent['needs_definition'] and analysis['key_nouns']:
            # For definition, add primary noun
            concept_additions.append(analysis['key_nouns'][0])
        elif intent['needs_quantitative'] and analysis['key_nouns']:
            # For quantitative, focus on the thing being counted
            concept_additions.append(analysis['key_nouns'][0])
        elif analysis['game_concepts']:
            # For general questions, add only top game concept
            concept_additions.append(analysis['game_concepts'][0])
        
        # Prioritize synonyms, then concept terms, up to max_additions
        all_additions = synonym_additions + concept_additions
        if all_additions:
            enhanced += " " + " ".join(all_additions[:max_additions])
        
        return enhanced.strip()

