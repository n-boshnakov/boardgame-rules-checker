# Archived Search/Indexing Scripts

## Semantic Analyzer Implementations (Original Archive Content)

This section contains fallback implementations of semantic analyzers that are no longer actively used but kept for reference.

### Active Implementation

**semantic_analyzer_nltk.py** (in parent directory)
- **Status**: ACTIVE - Used in production
- **Technology**: NLTK (Natural Language Toolkit)
- **Python Compatibility**: 3.11+ including 3.14+
- **Features**: Full NLP with POS tagging, lemmatization, WordNet, game vocabulary mapping
- **Reason for use**: Python 3.14 compatible, no Pydantic v1 dependency

### Archived Implementations

#### semantic_analyzer.py
- **Status**: ARCHIVED - Fallback only
- **Technology**: spaCy
- **Python Compatibility**: <3.14 (requires Pydantic v1)
- **Features**: Advanced NLP with dependency parsing, named entity recognition
- **Reason for archival**: Incompatible with Python 3.14+ due to Pydantic v1 requirement
- **Use case**: Can be used as fallback for Python <3.14 environments

#### semantic_analyzer_lite.py
- **Status**: ARCHIVED - Fallback only
- **Technology**: Regex patterns (no external NLP library)
- **Python Compatibility**: All versions
- **Features**: Basic pattern matching and term expansion
- **Reason for archival**: Less sophisticated than NLTK implementation
- **Use case**: Minimal dependency fallback if both NLTK and spaCy fail

### Retriever Fallback Order

The `RulebookRetriever` tries to load analyzers in this order:

1. **NLTK** (semantic_analyzer_nltk.py) - Primary
2. **spaCy** (archive/semantic_analyzer.py) - Fallback #1
3. **Lightweight** (archive/semantic_analyzer_lite.py) - Fallback #2

If all fail, semantic analysis is disabled but basic retrieval still works.

### Restoration

To restore a fallback analyzer:
```bash
# Move back to parent directory
mv archive/semantic_analyzer.py ../
mv archive/semantic_analyzer_lite.py ../
```

---

## Deprecated Indexing Scripts (Archived February 7, 2026)

These indexing/merging scripts have been superseded by universal versions.

### indexer.py
**Status:** ⚠️ Deprecated  
**Replaced by:** `index_chunks.py`  
**Reason:** Basic implementation without quality fields. The universal indexer auto-detects chunk format and includes comprehensive field mappings.

### indexer_unified.py
**Status:** ⚠️ Deprecated  
**Replaced by:** `index_chunks.py`  
**Reason:** Hardcoded file paths and limited to specific chunk format. The universal indexer is more flexible and handles any chunk format.

### forum_indexer.py
**Status:** ⚠️ Deprecated  
**Replaced by:** `index_chunks.py` (auto-detects forum chunks)  
**Reason:** Specialized for forum Q&A only. The universal indexer handles forum chunks automatically when they're present in the data.

### reindex_forum_qa.py
**Status:** ⚠️ Deprecated  
**Replaced by:** Standard pipeline with `index_chunks.py`  
**Reason:** No longer needed. Forum reindexing is handled by the standard indexing pipeline.

### Migration Guide for Indexing Scripts

See [documentation/SCRIPT_CONSOLIDATION_2026-02-07.md](../../documentation/SCRIPT_CONSOLIDATION_2026-02-07.md) for complete migration instructions.

### Do Not Use These Scripts

These scripts are no longer maintained and may not work correctly with the current codebase. Use the universal indexer instead:

```powershell
# Instead of the old scripts, use:
python src/search/index_chunks.py <embeddings.parquet> [--index name] [--chunks file.pkl]

# Examples:
python src/search/index_chunks.py data/processed/chunks_embeddings.parquet
python src/search/index_chunks.py data/processed/chunks_embeddings.parquet --index custom_index
python src/search/index_chunks.py data/processed/chunks_embeddings.parquet --chunks data/processed/chunks.pkl
```

### What to Use Instead

| Old Script | New Approach |
|------------|--------------|
| `indexer.py embeddings.parquet` | `index_chunks.py embeddings.parquet` |
| `indexer_unified.py` | `index_chunks.py embeddings.parquet` |
| `forum_indexer.py` | `merge_chunks_universal.py --forum X` + `index_chunks.py` |
| `reindex_forum_qa.py` | Use standard pipeline |

Note: For spaCy version, also need:
```bash
pip install spacy pydantic==1.10.13
python -m spacy download en_core_web_sm
```
