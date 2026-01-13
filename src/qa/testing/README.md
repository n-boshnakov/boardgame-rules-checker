# QA Testing & Analysis Scripts

This directory contains utility scripts for analyzing, comparing, and debugging QA system performance with multi-dimensional scoring.

## Quick Start

**Comprehensive Comparison (Recommended):**
```bash
# Compare baseline vs semantic search (10 questions)
python compare_semantic_vs_baseline.py

# Full test with visualizations (40 questions)
python compare_semantic_vs_baseline.py --questions 40 --visualize

# Custom output directory
python compare_semantic_vs_baseline.py -n 20 -v -o my_results
```

**Basic Analysis:**
```bash
# Check latest results
python check_final_results.py

# Analyze failures
python analyze_mismatches.py

# Compare different runs
python compare_runs.py
```

---

## Main Comparison Script

### `compare_semantic_vs_baseline.py` ⭐ NEW
**Purpose:** Comprehensive comparison of baseline vs semantic search with multi-dimensional scoring

**Features:**
- Side-by-side comparison (baseline vs semantic)
- Multi-dimensional analysis (relevance, completeness, accuracy, conciseness)
- Question-level breakdown showing improvements/regressions
- Question type analysis (procedural, definitional, etc.)
- Top improvements & regressions highlighted
- Saves detailed CSV and JSON results
- Generates 4-chart visualization (optional)

**Output:**
- Console report with metrics, improvements, top/worst questions
- CSV files with full baseline and semantic results
- JSON comparison data
- PNG visualization with 4 comparison charts

**Usage:**
```bash
# Quick test (10 questions, default)
python compare_semantic_vs_baseline.py

# Full test with all questions and visualizations
python compare_semantic_vs_baseline.py --questions 40 --visualize

# Custom options
python compare_semantic_vs_baseline.py -n 20 -v -o custom_output
```

**Arguments:**
- `--questions N` / `-n N`: Number of questions to test (default: 10, max: 40)
- `--visualize` / `-v`: Generate comparison visualizations
- `--output-dir DIR` / `-o DIR`: Output directory for results (default: comparison_output)

---

## Analysis Scripts

### `analyze_mismatches.py`
**Purpose:** Display sample questions where predictions don't match ground truth (for multi-dimensional scoring)

**Output:**
- Sample mismatches with overall scores < 0.7
- Ground truth vs predicted comparison
- Dimension-specific scores (relevance, completeness, accuracy, conciseness)
- Helps identify failure patterns

**Usage:**
```bash
python analyze_mismatches.py
```

### `compare_runs.py`
**Purpose:** Compare performance metrics between different QA runs (supports both old and new formats)

**Output:**
- Mean score comparison
- Passing rate changes (≥80% threshold)
- Score distribution analysis
- Statistical improvements

**Usage:**
```bash
python compare_runs.py
```

**Note:** Automatically detects multi-dimensional scoring format (overall_score column) or falls back to single-score format.

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
---

## Archived Scripts

Older testing scripts that used single-score evaluation format have been moved to `archive/`. These include:
- compare_results.py (replaced by compare_semantic_vs_baseline.py)
- debug_scoring.py
- verify_archive_scores.py
- verify_consistency.py
- check_similarity.py
- debug_sentence_extraction.py

See `archive/README.md` for details on archived scripts.

---

## Multi-Dimensional Scoring

Current scripts support the new multi-dimensional scoring format with:
- **Overall Score**: Weighted combination of all dimensions
- **Relevance (35%)**: Does answer address the question?
- **Completeness (30%)**: Does it cover all necessary information?
- **Accuracy (25%)**: Are the facts correct?
- **Conciseness (10%)**: Is it appropriately detailed?

Files with multi-dimensional scoring have columns: `overall_score`, `relevance_score`, `completeness_score`, `accuracy_score`, `conciseness_score`, `question_type`

Legacy files have a single `score` column (token_set_ratio similarity).

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
