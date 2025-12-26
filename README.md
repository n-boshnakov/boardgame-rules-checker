# Boardgame Rules Checker

A question-answering system for board game rulebooks using hybrid search (semantic + keyword) with Elasticsearch and sentence transformers.

## Project Structure

```
boardgame-rules-checker/
├── config/                    # Configuration files
│   ├── rag_config.json       # RAG system configuration
│   └── settings.yaml         # General settings
├── data/                      # Data files
│   ├── processed/            # Processed QA results
│   │   ├── qa_results.csv    # Latest test results
│   │   └── archive/          # Historical test runs
│   └── raw/                  # Raw rulebook PDFs
├── notebooks/                 # Jupyter notebooks for analysis
│   └── qa_success_analysis.ipynb
├── src/                       # Source code
│   ├── parsers/              # PDF parsing and text extraction
│   │   ├── pdf_parser.py     # Main PDF parsing (1000-char chunks)
│   │   └── spellcheck_utils.py
│   ├── qa/                   # Question answering logic
│   │   ├── testing/          # Testing utilities (see testing/README.md)
│   │   └── run_qa_batch.py   # Main Q&A batch testing script
│   ├── search/               # Search and retrieval
│   │   ├── embedder.py       # Text embedding generation
│   │   ├── indexer.py        # Elasticsearch indexing
│   │   └── retriever.py      # Hybrid search retrieval
│   └── ui/                   # User interface (if applicable)

## Quick Start

### Prerequisites
- Python 3.8+
- Elasticsearch 7/8 running on localhost:9200
- Install dependencies: `pip install -r requirements.txt`

### Run Q&A Batch Test
```bash
python src/qa/run_qa_batch.py --max_questions 40
```

### Check Results
```bash
python src/qa/testing/compare_results.py
```

### Analyze Performance
- Open `notebooks/qa_success_analysis.ipynb` in Jupyter/VS Code
- Or use testing scripts: `python src/qa/testing/analyze_mismatches.py`

## Current Performance (December 26, 2025)

**Optimal Configuration (Validated through Experimentation):**
- **Speed:** ~19-21 seconds for 40 questions (~2.1 it/s)
- **Quality:** 46.92% mean similarity
- **Success Rate:** 4/40 questions (10.0%) pass 80% threshold
- **Answer Length:** ~800 characters average

**Recent Improvements:**
- ✓ Code cleanup completed across all major scripts (Dec 26, 2025)
- ✓ Comprehensive documentation added to all modules
- ✓ Chunking experiments completed (1000-char confirmed optimal)
- ✓ Testing workflow documented in `src/qa/testing/README.md`

## System Configuration

### Optimal Settings (Empirically Validated)
- **Chunk Size:** 1000 characters with 150-char overlap
- **Embedding Model:** sentence-transformers/all-mpnet-base-v2 (768 dimensions)
- **Retrieval:** Hybrid search (85% semantic, 15% BM25 keyword)
- **Reranking:** CrossEncoder ms-marco-MiniLM-L-6-v2 (top 10 chunks)
- **Answer Generation:** Extractive concatenation with 800-char max
- **Search Parameters:** top_k=25, hybrid_weight=0.85

### Configuration Files
- `config/rag_config.json` - RAG system parameters
- `config/settings.yaml` - General application settings

## Recent Updates

### December 26, 2025 - Code Quality & Experimentation

- **Code Cleanup:** Improved readability across 6 major scripts
  - `run_qa_batch.py` - Removed duplicates, added comprehensive docstrings
  - `embedder.py` - Complete documentation overhaul
  - `indexer.py` - Enhanced error handling and docs
  - Testing scripts - Better structure and formatting
  - `testing/README.md` - Comprehensive testing guide created
- **Chunking Experiments:** Tested smaller chunks (700-char) and sentence-level extraction
  - Result: 1000-char chunks confirmed optimal (baseline: 48.15%)
  - 700-char chunks: 46.92% (-1.23%)
  - Sentence-level: 43.47% (-4.68%)
- **Documentation:** Created `CODE_CLEANUP_SUMMARY.md` and `CHUNKING_EXPERIMENTS_2025-12-26.md`

## Documentation

See the `documentation/` folder for detailed documentation:

- **CODE_CLEANUP_SUMMARY.md** - Recent code quality improvements
- **CHUNKING_EXPERIMENTS_2025-12-26.md** - Chunking strategy experiments and findings
- **IMPLEMENTATION_PLAN.md** - System architecture and implementation details
- **RAG_IMPLEMENTATION.md** - RAG system design and configuration
- **testing/README.md** - Comprehensive guide to testing scripts and workflows

## Testing

See `src/qa/testing/README.md` for comprehensive testing documentation including:

- Quick Start guide
- Script descriptions (Analysis, Diagnostic, Verification, Comparison)
- Common workflows (Quick Analysis, Deep Debugging, Performance Comparison)
- File structure and configuration notes
