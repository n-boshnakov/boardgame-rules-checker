# Search Module

Hybrid search and retrieval system for board game rulebooks using Elasticsearch and semantic embeddings.

## Components

### Core Search

- **retriever.py** - Main retrieval interface with hybrid search (vector + keyword)
- **embedder.py** - Generate embeddings for text chunks
- **indexer.py** - Index chunks and embeddings into Elasticsearch

### Semantic Analyzers

Three implementations with different trade-offs:

#### semantic_analyzer_nltk.py (Recommended)
- **Best choice for Python 3.14+**
- Full NLP capabilities: POS tagging, lemmatization, morphological analysis
- Game-specific concept detection
- Question intent classification (8 types)
- Compatible with modern Python versions

#### semantic_analyzer.py (Legacy)
- Uses spaCy for advanced NLP
- **Requires Pydantic v1** (incompatible with Python 3.14+)
- Most sophisticated analysis when compatible
- Use only with Python 3.12 or earlier

#### semantic_analyzer_lite.py (Fallback)
- No external NLP dependencies
- Regex-based pattern matching
- Lightweight but less accurate
- Use when NLTK/spaCy unavailable

## Usage

```python
from src.search.retriever import RulebookRetriever

# Initialize with semantic analysis
retriever = RulebookRetriever(use_semantic_analysis=True)

# Search for relevant chunks
results = retriever.search("How do I move my character?", top_k=10)

# Generate answer from chunks
answer = retriever.generate_answer("How do I move my character?", results)
```

## Semantic Enhancement

The semantic analyzer enhances queries by:
- Adding 1 highly relevant term (conservative approach)
- Detecting question intent (procedural, definitional, quantitative, etc.)
- Identifying game-specific concepts
- Lemmatizing terms for better matching

**Performance**: 43.56% mean score vs 43.15% baseline (0.41% improvement)

## Configuration

Set `use_semantic_analysis=False` to disable semantic enhancement and use raw queries only.
