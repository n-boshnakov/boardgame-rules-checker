"""
Test script to demonstrate the improved chunking implementation.
Shows before/after comparison of chunk sizes and counts.
"""

import pickle
import json
from pathlib import Path

# Load existing chunks
chunks_path = Path("data/processed/chunks.pkl")

if chunks_path.exists():
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    
    print(f"=== Current Chunking Analysis ===")
    print(f"Total chunks: {len(chunks)}")
    
    # Analyze chunk sizes
    chunk_sizes = [len(chunk['text']) for chunk in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes)
    max_size = max(chunk_sizes)
    min_size = min(chunk_sizes)
    
    print(f"Average chunk size: {avg_size:.0f} characters")
    print(f"Max chunk size: {max_size} characters")
    print(f"Min chunk size: {min_size} characters")
    
    # Count chunks over 2000 chars (likely to exceed sentence-transformer limits)
    oversized = sum(1 for size in chunk_sizes if size > 2000)
    print(f"Chunks over 2000 chars: {oversized} ({oversized/len(chunks)*100:.1f}%)")
    
    # Show examples of large chunks
    print(f"\n=== Examples of large chunks ===")
    large_chunks = sorted(enumerate(chunks), key=lambda x: len(x[1]['text']), reverse=True)
    for idx, chunk in large_chunks[:3]:
        size = len(chunk['text'])
        preview = chunk['text'][:100].replace('\n', ' ')
        print(f"Chunk {idx}: {size} chars - {preview}...")
    
    print(f"\n=== Recommendations ===")
    print(f"The improved chunking will:")
    print(f"1. Split large chunks into smaller overlapping chunks (max 2000 chars)")
    print(f"2. Maintain context with 200-char overlap between chunks")
    print(f"3. Keep all PDF text without losing information")
    print(f"4. Fit within sentence-transformer limits (512 tokens ≈ 2000 chars)")
    print(f"\nExpected result: {oversized} large chunks will be split into ~{oversized * 2} smaller chunks")
    print(f"Total chunks after re-parsing: ~{len(chunks) + oversized} chunks")
    
else:
    print("No chunks.pkl found. Run the parser first:")
    print("python src/parsers/pdf_parser.py <pdf_path>")
