# Boardgame Rules Checker

A question-answering system for board game rulebooks using hybrid search (semantic + keyword) with Elasticsearch and sentence transformers. Now enhanced with **optional semantic analysis** using NLTK for improved query understanding and multi-dimensional answer evaluation.

## Features

- **Hybrid Search**: Combines semantic embeddings (BAAI/bge-m3) with BM25 keyword matching
- **Multi-Dimensional Scoring**: Evaluates answers across 4 dimensions (relevance, completeness, accuracy, conciseness)
- **Optional Semantic Analysis**: NLTK-based query enhancement with game-specific vocabulary mapping (foundation for NLP course)
- **Comprehensive Testing**: Full comparison suite for baseline vs semantic search evaluation
- **Jupyter Notebooks**: Interactive analysis and visualization tools

## Prerequisites

- Python 11+ (tested on 3.11.3)
- Elasticsearch 8.x running on localhost:9200
- Install dependencies: `pip install -r requirements.txt`
- See [DEPENDENCIES.md](DEPENDENCIES.md) for detailed installation guide

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd boardgame-rules-checker
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/macOS
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Elasticsearch**
   ```bash
   # Ensure Elasticsearch is running on localhost:9200
   ```

### Run Q&A Batch Test

**Baseline (without semantic search - January 2026)

### Baseline Performance (No Semantic Analysis)
- **Speed:** ~37 seconds for 40 questions (1.45 it/s)
- **Overall Score:** 61.96%
- **Pass Rate:** 7/40 questions (17.5%) ≥80% threshold
- **Dimension Breakdown:**
  - Relevance: 51.19%
  - Completeness: 91.13%
  - Accuracy: 31.29% ⚠️ (bottleneck)
  - Conciseness: 88.78%

### With Semantic Analysis (Optional)
- **Speed:** ~480 seconds for 40 questions (0.085 it/s, 13x slower)
- **Overall Score:** 65.48% (+5.7% improvement)
- **Pass Rate:** 12/40 questions (30%, +71% increase)
- **Dimension Breakdown:**
  - Relevance: 57.56% (+12.4%)
  - Completeness: 91.32% (+0.2%)
  - Accuracy: 34.64% (+10.7%)
  - Conciseness: 92.83% (+4.6%)

**Note:** Semantic analysis is **optional** and serves as a foundation for NLP course concepts. It demonstrates:
- Query enhancement with game-specific vocabulary
- Dynamic synonym expansion
- Question intent classification
- Multi-dimensional answer evaluation

### Analyze Performance
- Open `notebooks/qa_success_analysis.ipynb` in Jupyter/VS Code for multi-dimensional visualizations
- Install dependencies: `pip install -r requirements.txt`

### Run Q&A Batch Test
```bash
python src/qa/run_qa_batch.py --max_questions 40
```
