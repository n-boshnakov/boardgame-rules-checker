# Archived Testing Scripts

This directory contains older testing scripts that used the single-score evaluation format (pre-multi-dimensional scoring).

## Why Archived?

These scripts were created before the implementation of multi-dimensional scoring (relevance, completeness, accuracy, conciseness). They use the old `score` column format instead of the new `overall_score` and dimension-specific columns.

## Archived Scripts

### compare_results.py
- **Original Purpose:** Compare baseline vs new approach performance
- **Reason for Archival:** Replaced by `compare_semantic_vs_baseline.py` with multi-dimensional support
- **Last Updated:** Pre-December 2025

### debug_scoring.py
- **Original Purpose:** Debug score calculation discrepancies
- **Reason for Archival:** Used old single-score format
- **Last Updated:** Pre-December 2025

### verify_archive_scores.py
- **Original Purpose:** Verify score accuracy in archive files
- **Reason for Archival:** Used old single-score format
- **Last Updated:** Pre-December 2025

### verify_consistency.py
- **Original Purpose:** Verify consistency between runs
- **Reason for Archival:** Used old single-score format
- **Last Updated:** Pre-December 2025

### check_similarity.py
- **Original Purpose:** Check similarity calculations
- **Reason for Archival:** Used old single-score format
- **Last Updated:** Pre-December 2025

### debug_sentence_extraction.py
- **Original Purpose:** Debug sentence extraction issues
- **Reason for Archival:** Used old answer generation approach
- **Last Updated:** Pre-December 2025

## Current Alternative

For comprehensive comparison with multi-dimensional scoring, use:
```bash
python compare_semantic_vs_baseline.py --questions 40 --visualize
```

This script provides:
- Multi-dimensional score comparison (4 dimensions)
- Question-level analysis
- Question type breakdown
- Visualization charts
- Detailed CSV/JSON output

## Restoration

If you need to restore these scripts for historical analysis:
```bash
# Move back to parent directory
mv archive/compare_results.py ../
```

However, note that they will only work correctly with old-format qa_results files that have a single `score` column.
