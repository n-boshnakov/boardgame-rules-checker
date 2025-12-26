# QA Testing & Analysis Scripts

This directory contains utility scripts for analyzing, comparing, and debugging QA system performance.

## Quick Start

**Basic Analysis:**
```bash
# Check latest results
python check_final_results.py

# Analyze failures
python analyze_mismatches.py

# Compare to baseline
python compare_results.py
```

---

## Analysis Scripts

### `analyze_mismatches.py`
**Purpose:** Display sample questions where predictions don't match ground truth

**Output:**
- Sample mismatches with scores < 0.7
- Ground truth vs predicted comparison
- Helps identify failure patterns

**Usage:**
```bash
python analyze_mismatches.py
```

### `compare_results.py`
**Purpose:** Compare performance metrics between baseline and new approaches

**Output:**
- Mean score comparison (baseline vs new)
- Passing rate changes
- Score distribution analysis
- Statistical improvements

**Usage:**
```bash
python compare_results.py
```

### `analyze_ground_truth.py`
**Purpose:** Analyze characteristics of ground truth answers

**Analyzes:**
- Answer length distribution
- Complexity patterns
- Common formats

### `check_final_results.py`
**Purpose:** Quick summary of latest QA evaluation

**Output:**
- Mean similarity score
- Passing rate (≥0.8 threshold)
- Score distribution by bins


---

## Diagnostic Scripts

### `diagnose_qa_issues.py`
**Purpose:** Comprehensive diagnostic for identifying system bottlenecks

**Checks:**
- Retrieval accuracy (are correct chunks found?)
- Answer generation quality
- Score calibration
- Common failure patterns

### `debug_sentence_extraction.py`
**Purpose:** Debug sentence-level extraction logic

**Use when:** Testing sentence extraction from chunks

### `debug_scoring.py`
**Purpose:** Examine similarity scoring mechanisms

**Use when:** Verifying scoring calculations are correct

---

## Verification Scripts

### `verify_consistency.py`
**Purpose:** Check consistency between retriever.py and run_qa_batch.py

**Validates:**
- Parameter alignment
- Configuration consistency

### `verify_archive_scores.py`
**Purpose:** Validate archived QA results have correct scores

**Use for:** Data integrity checks on historical results

### `validate_config.py`
**Purpose:** Validate configuration files and parameters

**Checks:**
- Default parameter values
- Settings consistency across modules

---

## Comparison Scripts

### `check_similarity.py`
### `check_similarity.py`
**Purpose:** Compare computed similarity vs stored CSV scores

**Validates:** Similarity score calculation consistency

**Use when:** Debugging score discrepancies

### `compare_runs.py`
**Purpose:** Compare multiple QA runs side-by-side

**Output:**
- Performance trends over time
- Quality metrics across runs
- Answer length changes

---

## Workflows

### Quick Analysis
```bash
# 1. Run QA evaluation
cd ../..
python src/qa/run_qa_batch.py --max_questions 40

# 2. Check results
cd src/qa/testing
python check_final_results.py

# 3. Analyze failures
python analyze_mismatches.py

# 4. Compare to baseline
python compare_results.py
```

### Deep Debugging
```bash
# 1. Run comprehensive diagnostics
python diagnose_qa_issues.py

# 2. Check specific components
python debug_sentence_extraction.py
python debug_scoring.py

# 3. Verify consistency
python verify_consistency.py
```

### Performance Comparison
```bash
# 1. Run multiple configurations
python src/qa/run_qa_batch.py --hybrid_weight 0.85
python src/qa/run_qa_batch.py --hybrid_weight 0.90

# 2. Compare results
python compare_runs.py

# 3. Validate improvements
python verify_archive_scores.py
```

---

## File Paths

Scripts expect this project structure:
```
boardgame-rules-checker/
├── data/processed/
│   ├── qa_results.csv          # Current results
│   ├── qa_results_clean.csv    # Clean baseline
│   └── archive/                # Historical results
├── src/
│   ├── qa/
│   │   ├── run_qa_batch.py     # Main evaluation script
│   │   └── testing/            # This directory
│   └── search/
│       └── retriever.py        # QA system
```

---

## Configuration Notes

- Most scripts auto-detect project root
- Some reference specific baseline files (update paths in script if needed)
- Scripts are **read-only** - safe to run multiple times
- Output is printed to console (not saved)

---

## Tips

**Best Practices:**
- Run `check_final_results.py` after any changes to validate
- Use `compare_results.py` before committing improvements
- Keep archived results for historical comparison
- Document significant configuration changes in filenames

**Troubleshooting:**
- If scores seem wrong: Run `check_similarity.py`
- If results inconsistent: Run `verify_consistency.py`
- If configuration unclear: Run `validate_config.py`
- For general issues: Run `diagnose_qa_issues.py`
