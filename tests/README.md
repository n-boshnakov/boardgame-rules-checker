# Testing Suite

This directory contains all testing utilities for the boardgame rules checker system.

## Directory Structure

```
tests/
├── qa/              # Question answering system tests
│   ├── README.md
│   ├── quick_check.py              # Fast QA results validation
│   ├── test_retriever_accuracy.py  # Compare retriever vs batch (10 questions)
│   ├── test_full_accuracy.py       # Full validation (40 questions)
│   ├── compare_results.py          # Compare two QA result files
│   └── generate_qa_report.py       # Generate detailed QA report
│
└── chunking/        # PDF chunking tests
    ├── README.md
    ├── test_chunking.py            # Analyze chunk sizes and distribution
    └── check_chunks.py             # Search chunks for specific content
```

## Quick Start

### Test QA System Changes

After modifying answer generation in [`src/search/retriever.py`](../src/search/retriever.py):

```bash
# 1. Quick validation (first 10 questions)
python tests/qa/test_retriever_accuracy.py

# 2. Full validation (all 40 questions)
python tests/qa/test_full_accuracy.py

# 3. Production run
python scripts/run_qa_batch.py --max_questions 40

# 4. Quick check results
python tests/qa/quick_check.py
```

### Test Chunk Parsing Changes

After modifying [`src/parsers/pdf_parser.py`](../src/parsers/pdf_parser.py):

```bash
# Analyze chunk distribution
python tests/chunking/test_chunking.py

# Search for specific content
python tests/chunking/check_chunks.py
```

## Current Performance Benchmarks

### QA System (Dec 24, 2025)
- **Overall (40 questions):** 49.87% mean similarity
- **Answerable (first 20):** 59.76% mean similarity
- **≥70% threshold:** 8/40 questions (20%)
- **≥80% threshold:** 6/40 questions (15%)

### Key Test Cases
- **Q7** "Can anomalies move?" - Should achieve 90.3% (litmus test)
- **Q3** "How many players?" - Should achieve ~82%
- **Q2** "How do I win?" - Should achieve ~66%

## Architecture

The testing suite validates two main components:

1. **Answer Generation** ([`src/search/retriever.py`](../src/search/retriever.py))
   - Multi-chunk synthesis (10 chunks)
   - Sentence scoring and deduplication
   - Relevance thresholding (0.35)
   - Answer length (5-10 sentences)

2. **Chunk Parsing** ([`src/parsers/pdf_parser.py`](../src/parsers/pdf_parser.py))
   - PDF text extraction
   - Section-based chunking
   - Size limits (<2000 chars)
   - Overlap handling

## See Also

- [QA Testing Documentation](qa/README.md)
- [Chunking Testing Documentation](chunking/README.md)
- [Refactoring Summary](../REFACTORING_SUMMARY.md)
- [Multi-Chunk Synthesis](../MULTI_CHUNK_SYNTHESIS.md)
