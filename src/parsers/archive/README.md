# Archived Parser Scripts

**Date Archived:** February 7, 2026

These scripts have been superseded by the universal scripts and are kept here for reference only.

## Archived Scripts

### merge_chunks.py
**Status:** ⚠️ Deprecated  
**Replaced by:** `merge_chunks_universal.py`  
**Reason:** Limited to PDF+OCR merging only. The universal version handles any combination of sources (PDF, OCR, Forum) with the same quality.

### merge_and_index_all.py
**Status:** ⚠️ Deprecated  
**Replaced by:** `merge_chunks_universal.py` + `embedder.py` + `index_chunks.py`  
**Reason:** Hardcoded file paths and combined too many responsibilities (merging + embedding + indexing). The new approach separates concerns and provides flexibility.

### reindex_forum_qa.py
**Status:** ⚠️ Deprecated  
**Replaced by:** Standard pipeline with `merge_chunks_universal.py --forum`  
**Reason:** No longer needed. Forum Q&A pairs are now handled by the standard merging and indexing pipeline.

## Migration Guide

See [documentation/SCRIPT_CONSOLIDATION_2026-02-07.md](../../documentation/SCRIPT_CONSOLIDATION_2026-02-07.md) for complete migration instructions.

## Do Not Use

These scripts are no longer maintained and may not work correctly with the current codebase. Use the universal scripts instead:

```powershell
# Instead of the old scripts, use:
python src/parsers/merge_chunks_universal.py --pdf <path> --ocr <path> --forum <path> -o <output>
python src/search/embedder.py <chunks>
python src/search/index_chunks.py <embeddings>
```
