# Testing Scripts

This directory contains utility scripts for testing and validating the Q&A system.

## Scripts

### check_final_results.py
Analyzes the latest 40-question test results from `qa_results.csv`:
- Mean similarity score
- Success rate (questions ≥80% similarity)
- Score distribution breakdown

**Usage:** `python scripts/testing/check_final_results.py`

### check_similarity.py
Compares computed similarity vs CSV score column:
- Validates similarity score consistency
- Shows per-question scores
- Useful for debugging score calculation issues

**Usage:** `python scripts/testing/check_similarity.py`

### compare_runs.py
Compares multiple archived test runs:
- Lists recent test results with metrics
- Shows quality and answer length trends
- Helps track performance over time

**Usage:** `python scripts/testing/compare_runs.py`

### validate_config.py
Validates retriever configuration consistency:
- Checks default parameters
- Verifies settings across files

**Usage:** `python scripts/testing/validate_config.py`

### verify_consistency.py
Verifies consistency between retriever.py and run_qa_batch.py:
- Checks parameter alignment
- Validates configuration settings

**Usage:** `python scripts/testing/verify_consistency.py`
