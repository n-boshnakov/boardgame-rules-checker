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
        # Order matters: more specific/useful synonyms should come before generic concept names
        self.game_vocabulary = {
            'action': ['action', 'perform', 'execute', 'take', 'do', 'make', 'conduct'],
            'movement': ['move', 'travel', 'relocate', 'movement', 'position', 'place'],
            'combat': ['attack', 'fight', 'shoot', 'battle', 'engage', 'combat', 'damage', 'wound'],
            'resource': ['gain', 'spend', 'pay', 'receive', 'acquire', 'collect', 'resource'],
            'card': ['draw', 'discard', 'play', 'reveal', 'card', 'hand'],
            'turn': ['round', 'phase', 'step', 'turn', 'sequence', 'order'],
            'player': ['stalker', 'character', 'hero', 'agent', 'player'],
            'equipment': ['weapon', 'gear', 'item', 'tool', 'artifact', 'equipment'],
            'location': ['zone', 'area', 'space', 'territory', 'region', 'base', 'location'],
            'condition': ['status', 'effect', 'modifier', 'state', 'condition'],
            'test': ['check', 'skill test', 'attribute test', 'roll', 'test', 'dice'],
            'enemy': ['mutant', 'monster', 'creature', 'threat', 'anomaly', 'enemy'],
            'mission': ['quest', 'task', 'objective', 'goal', 'mission'],
            'win': ['victory', 'succeed', 'complete', 'achieve', 'win'],
            'lose': ['defeat', 'fail', 'eliminated', 'lose'],
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
    
    def extract_domain_keywords(self, text: str) -> List[str]:
        """
        Extract domain-specific keywords that should be present in the answer.
        These are critical terms that define the question's focus.
        
        Returns:
            List of critical keywords (nouns, game-specific terms)
        """
        keywords = []
        
        # Get analysis
        analysis = self.analyze(text)
        text_lower = text.lower()
        
        # Priority 1: Domain-specific nouns (game mechanics, components)
        # Common question words to exclude
        stop_words = {'action', 'move', 'cost', 'through', 'can', 'does', 'what', 'how', 'when', 'where', 'who', 'why', 'do'}
        
        # Extract important nouns (but filter common question structure words)
        for noun in analysis['key_nouns']:
            if noun not in stop_words and len(noun) > 2:
                keywords.append(noun)
        
        # Priority 2: Game-specific terms from vocabulary
        domain_terms = ['water', 'reload', 'window', 'anomaly', 'artifact', 'enemy', 'stalker', 
                       'radiation', 'loot', 'weapon', 'card', 'mission', 'combat', 'damage',
                       'heal', 'rest', 'search', 'trade', 'wound', 'status', 'equipment', 'psionic', 'mutant']
        
        for term in domain_terms:
            if term in text_lower and term not in keywords:
                keywords.append(term)
        
        # Priority 3: Multi-word game concepts
        multi_word_terms = ['line of sight', 'action point', 'skill test', 'zone of control', 'free action']
        for term in multi_word_terms:
            if term in text_lower:
                keywords.append(term)
        
        return keywords[:5]  # Return top 5 most critical keywords
    
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
        # Remove punctuation and create set of clean words
        import string
        query_words = set(word.strip(string.punctuation) for word in query_lower.split())
        
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
        # Make sure we don't add terms already in the query
        if intent['needs_procedural'] and analysis['action_verbs']:
            # For procedural, add primary action verb NOT already in query
            for verb in analysis['action_verbs']:
                if verb not in query_words:
                    concept_additions.append(verb)
                    break
        elif intent['needs_definition'] and analysis['key_nouns']:
            # For definition, add primary noun NOT already in query
            for noun in analysis['key_nouns']:
                if noun not in query_words:
                    concept_additions.append(noun)
                    break
        elif intent['needs_quantitative'] and analysis['key_nouns']:
            # For quantitative, focus on the thing being counted NOT already in query
            for noun in analysis['key_nouns']:
                if noun not in query_words:
                    concept_additions.append(noun)
                    break
        elif analysis['game_concepts']:
            # For general questions, add only top game concept NOT already in query
            for concept in analysis['game_concepts']:
                if concept not in query_words:
                    concept_additions.append(concept)
                    break
        
        # Prioritize synonyms, then concept terms, up to max_additions
        # Filter out generic/structural words like "action", "do", "make" when we have more specific options
        generic_words = {'action', 'do', 'make', 'perform', 'take', 'execute', 'conduct'}
        
        # Separate specific vs generic synonyms
        specific_synonyms = [s for s in synonym_additions if s not in generic_words]
        generic_synonyms = [s for s in synonym_additions if s in generic_words]
        
        # Prefer specific over generic
        prioritized_additions = specific_synonyms + generic_synonyms + concept_additions
        
        if prioritized_additions:
            enhanced += " " + " ".join(prioritized_additions[:max_additions])
        
        return enhanced.strip()

