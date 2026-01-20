# Archived Semantic Analyzer Implementations

This directory contains fallback implementations of semantic analyzers that are no longer actively used but kept for reference.

## Active Implementation

**semantic_analyzer_nltk.py** (in parent directory)
- **Status**: ACTIVE - Used in production
- **Technology**: NLTK (Natural Language Toolkit)
- **Python Compatibility**: 3.11+ including 3.14+
- **Features**: Full NLP with POS tagging, lemmatization, WordNet, game vocabulary mapping
- **Reason for use**: Python 3.14 compatible, no Pydantic v1 dependency

## Archived Implementations

### semantic_analyzer.py
- **Status**: ARCHIVED - Fallback only
- **Technology**: spaCy
- **Python Compatibility**: <3.14 (requires Pydantic v1)
- **Features**: Advanced NLP with dependency parsing, named entity recognition
- **Reason for archival**: Incompatible with Python 3.14+ due to Pydantic v1 requirement
- **Use case**: Can be used as fallback for Python <3.14 environments

### semantic_analyzer_lite.py
- **Status**: ARCHIVED - Fallback only
- **Technology**: Regex patterns (no external NLP library)
- **Python Compatibility**: All versions
- **Features**: Basic pattern matching and term expansion
- **Reason for archival**: Less sophisticated than NLTK implementation
- **Use case**: Minimal dependency fallback if both NLTK and spaCy fail

## Retriever Fallback Order

The `RulebookRetriever` tries to load analyzers in this order:

1. **NLTK** (semantic_analyzer_nltk.py) - Primary
2. **spaCy** (archive/semantic_analyzer.py) - Fallback #1
3. **Lightweight** (archive/semantic_analyzer_lite.py) - Fallback #2

If all fail, semantic analysis is disabled but basic retrieval still works.

## Restoration

To restore a fallback analyzer:
```bash
# Move back to parent directory
mv archive/semantic_analyzer.py ../
mv archive/semantic_analyzer_lite.py ../
```

Note: For spaCy version, also need:
```bash
pip install spacy pydantic==1.10.13
python -m spacy download en_core_web_sm
```
