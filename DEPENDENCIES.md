# Project Dependencies and Installation Guide

## System Requirements

- **Python**: 3.11.3 (tested) or 3.11+ (recommended)
  - ⚠️ **Note**: Python 3.14+ has compatibility issues with spaCy (requires Pydantic v1)
  - The project uses NLTK as an alternative for semantic analysis when spaCy is unavailable

## Environment Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Core Dependencies

### Machine Learning & NLP

| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | 5.2.0 | Embedding generation (BAAI/bge-m3) and reranking (ms-marco-MiniLM-L-6-v2) |
| `torch` | 2.9.1 | PyTorch backend for transformers |
| `transformers` | 4.57.3 | Hugging Face transformers library |
| `nltk` | 3.9.2 | Natural language processing (POS tagging, lemmatization, WordNet) |
| `spacy` | 3.8.11 | Advanced NLP (optional, for Python <3.14) |
| `scikit-learn` | 1.8.0 | ML utilities and metrics |

**NLTK Data Required:**
- `punkt` / `punkt_tab` - Tokenization
- `averaged_perceptron_tagger` / `averaged_perceptron_tagger_eng` - POS tagging
- `wordnet` - Lexical database
- `omw-1.4` - Open Multilingual WordNet

**Download NLTK data (automatically handled by code):**
```python
import nltk
nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 
               'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4'])
```

### Search & Database

| Package | Version | Purpose |
|---------|---------|---------|
| `elasticsearch` | 8.12.1 | Document indexing and hybrid search |
| `elastic-transport` | 8.17.1 | Elasticsearch transport layer |

**Elasticsearch Server:**
- Version: 8.x (compatible)
- Required for indexing and retrieval
- Configuration: `elasticsearch.yml` in project root

### Data Processing

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.3.3 | Data manipulation and CSV handling |
| `numpy` | 2.4.0 | Numerical operations |
| `pyarrow` | 22.0.0 | Fast data serialization |

### Document Processing

| Package | Version | Purpose |
|---------|---------|---------|
| `PyMuPDF` (fitz) | 1.26.7 | PDF text extraction |
| `pyspellchecker` | (optional) | Spell checking for OCR text |

### Text Processing & Matching

| Package | Version | Purpose |
|---------|---------|---------|
| `RapidFuzz` | 3.14.3 | Fast fuzzy string matching for evaluation |
| `regex` | 2025.11.3 | Advanced regex patterns |
| `ftfy` | 6.3.1 | Text encoding fixes |

### Visualization & Analysis

| Package | Version | Purpose |
|---------|---------|---------|
| `matplotlib` | 3.10.8 | Plotting and visualization |
| `seaborn` | 0.13.2 | Statistical visualizations |

### Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| `tqdm` | 4.67.1 | Progress bars |
| `PyYAML` | 6.0.3 | Configuration file parsing |
| `requests` | 2.32.5 | HTTP requests |
| `huggingface-hub` | 0.36.0 | Model downloading |
| `safetensors` | 0.7.0 | Safe tensor serialization |

## Model Downloads

The following models will be downloaded automatically on first run:

### Embedding Model
- **Model**: `BAAI/bge-m3`
- **Size**: ~2.2 GB
- **Dimensions**: 1024
- **Use**: Semantic search and hybrid retrieval

### Reranker Model
- **Model**: `ms-marco-MiniLM-L-6-v2` (CrossEncoder)
- **Size**: ~80 MB
- **Use**: Multi-dimensional scoring (relevance dimension)

### spaCy Model (Optional)
```bash
python -m spacy download en_core_web_sm
```
- Only needed if using spaCy (Python < 3.14)
- NLTK is used as fallback for Python 3.14+

## Creating requirements.txt

To generate a requirements.txt file from the current environment:

```bash
pip freeze > requirements.txt
```

**Minimal requirements.txt:**
```txt
# Core ML/NLP
sentence-transformers>=5.0.0
torch>=2.9.0
transformers>=4.50.0
nltk>=3.9.0
scikit-learn>=1.8.0

# Search & Database
elasticsearch>=8.12.0

# Data Processing
pandas>=2.3.0
numpy>=2.4.0

# Document Processing
PyMuPDF>=1.26.0

# Text Processing
RapidFuzz>=3.14.0
regex>=2025.11.0

# Visualization
matplotlib>=3.10.0
seaborn>=0.13.0

# Utilities
tqdm>=4.67.0
PyYAML>=6.0.0
requests>=2.32.0
```

## Optional Dependencies

### For Python < 3.14 (spaCy support)
```bash
pip install spacy>=3.8.0
python -m spacy download en_core_web_sm
```

### For OCR Spell Checking
```bash
pip install pyspellchecker
```

### For Jupyter Notebooks
```bash
pip install jupyter notebook ipykernel
```

## Troubleshooting

### Issue: Pydantic v1 vs v2 Conflict
**Problem**: spaCy requires Pydantic v1, but newer packages require v2

**Solution**: Use NLTK-based semantic analyzer (already implemented)
```python
# Code automatically falls back to NLTK if spaCy unavailable
from search.semantic_analyzer_nltk import SemanticAnalyzerNLTK
```

### Issue: NLTK Data Not Found
**Problem**: `LookupError` for NLTK resources

**Solution**: Download required data
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

### Issue: Elasticsearch Connection Failed
**Problem**: `ConnectionError` when connecting to Elasticsearch

**Solution**: 
1. Ensure Elasticsearch is running: `docker ps` or check service status
2. Verify configuration in `elasticsearch.yml`
3. Check firewall settings for port 9200

### Issue: CUDA/GPU Not Available
**Problem**: PyTorch not detecting GPU

**Solution**: Install CUDA-compatible PyTorch
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU-only (slower)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Torch Security Warning (CVE-2025-32434)
**Problem**: `ValueError: Due to a serious vulnerability issue in torch.load`

**Solution**: Upgrade to PyTorch 2.6+ or use safetensors
```bash
pip install torch>=2.6.0
```

## Verification

Verify installation:

```python
# Test core imports
import pandas as pd
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder
from elasticsearch import Elasticsearch

# Test NLTK data
nltk.data.find('tokenizers/punkt')
nltk.data.find('taggers/averaged_perceptron_tagger')

# Test model loading
model = SentenceTransformer('BAAI/bge-m3')
print("✓ All dependencies loaded successfully")
```

## Performance Considerations

### Memory Requirements
- **Minimum**: 8 GB RAM
- **Recommended**: 16 GB RAM (for full embedding model in memory)
- **GPU**: Optional, improves speed 3-5x

### Disk Space
- Models: ~3 GB
- Elasticsearch indices: Varies by document size
- NLTK data: ~50 MB

### Speed Optimization
- Use GPU for embeddings (3-5x faster)
- Increase batch size for bulk operations
- Cache embeddings for repeated queries
- Use smaller models for development (all-MiniLM-L6-v2: 80MB)

## Development Tools (Optional)

```bash
# Code formatting
pip install black isort

# Linting
pip install flake8 pylint

# Testing
pip install pytest pytest-cov

# Type checking
pip install mypy
```

## Docker Alternative

For reproducible environments, consider using Docker:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet'])"

COPY . .
CMD ["python", "src/qa/run_qa_batch.py"]
```

## License Considerations

- **PyTorch**: BSD-style license
- **Transformers**: Apache 2.0
- **NLTK**: Apache 2.0
- **Elasticsearch**: Elastic License 2.0 (note restrictions for commercial use)
- **spaCy**: MIT License
- **sentence-transformers**: Apache 2.0

## Support

For package-specific issues:
- PyTorch: https://github.com/pytorch/pytorch
- Transformers: https://github.com/huggingface/transformers
- Elasticsearch: https://github.com/elastic/elasticsearch-py
- NLTK: https://github.com/nltk/nltk
