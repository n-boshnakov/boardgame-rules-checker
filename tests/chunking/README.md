# Chunking Test Utilities

This directory contains scripts for testing and analyzing PDF chunk parsing.

## Scripts

### Test Chunking

**`test_chunking.py`** - Analyze current chunking implementation
```bash
python tests/chunking/test_chunking.py
```
- Shows total chunk count
- Analyzes chunk size distribution
- Identifies oversized chunks (>2000 chars)
- Provides recommendations for improvements

### Check Chunks

**`check_chunks.py`** - Search for specific content in chunks
```bash
python tests/chunking/check_chunks.py
```
- Searches chunks for specific keywords
- Shows context around found terms
- Useful for debugging chunk content

## Chunking Guidelines

- **Target chunk size:** <2000 characters (to fit sentence-transformer limits)
- **Overlap:** 200 characters between consecutive chunks
- **Sentence boundaries:** Chunks should break at sentence boundaries when possible
- **Section preservation:** Keep section headers with their content

## Related Files

- Chunk parser: [`src/parsers/pdf_parser.py`](../../src/parsers/pdf_parser.py)
- Processed chunks: [`data/processed/chunks.pkl`](../../data/processed/)
- View chunks script: [`view_chunks.py`](../../view_chunks.py)
