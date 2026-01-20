"""
Compare OCR vs PDF text extraction to identify gaps and quality differences.

This script analyzes chunks from both extraction methods to determine:
1. Which chunks are similar (duplicates)
2. Which chunks are unique to OCR (gaps in PDF extraction)
3. Which chunks are unique to PDF (OCR missing content)
4. Quality comparison for overlapping content

Usage:
    python src/parsers/compare_extraction_methods.py <pdf_chunks.pkl> <ocr_chunks.pkl>
"""

import pickle
import sys
from typing import List, Dict, Tuple
from collections import defaultdict
from rapidfuzz import fuzz
import re


def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra whitespace and lowercasing."""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    return text


def calculate_text_quality_score(text: str) -> float:
    """
    Calculate quality score for text based on various heuristics.
    Higher score = better quality.
    
    Checks for:
    - OCR errors (repeated characters, garbled text)
    - Proper word structure (vowel ratio)
    - Sentence structure
    """
    if not text or len(text) < 20:
        return 0.0
    
    score = 100.0
    
    # Penalize repeated character sequences (OCR artifacts)
    repeated_chars = len(re.findall(r'([a-z])\1{4,}', text.lower()))
    score -= repeated_chars * 5
    
    # Penalize low vowel ratio (garbled text)
    text_clean = re.sub(r'[^a-zA-Z ]', '', text)
    if len(text_clean) > 20:
        vowel_ratio = len(re.findall(r'[aeiouAEIOU ]', text_clean)) / len(text_clean)
        if vowel_ratio < 0.25:
            score -= 20
        elif vowel_ratio < 0.30:
            score -= 10
    
    # Penalize excessive punctuation noise
    excessive_dots = len(re.findall(r'\.{6,}', text))
    score -= excessive_dots * 5
    
    # Reward proper sentence structure
    sentences = re.split(r'[.!?]+', text)
    proper_sentences = sum(1 for s in sentences if 3 <= len(s.split()) <= 50)
    score += proper_sentences * 2
    
    return max(0.0, score)


def find_similar_chunks(
    chunks_a: List[Dict], 
    chunks_b: List[Dict], 
    similarity_threshold: float = 70.0
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Find similar chunks between two lists using fuzzy matching.
    
    Returns:
        - List of (idx_a, idx_b, similarity_score) tuples for matches
        - List of indices from chunks_a that have no match in chunks_b
        - List of indices from chunks_b that have no match in chunks_a
    """
    matches = []
    matched_a = set()
    matched_b = set()
    
    print(f"Comparing {len(chunks_a)} chunks from method A with {len(chunks_b)} chunks from method B...")
    
    # For each chunk in A, find best match in B
    for idx_a, chunk_a in enumerate(chunks_a):
        text_a = normalize_text(chunk_a['text'])
        best_similarity = 0.0
        best_idx_b = -1
        
        for idx_b, chunk_b in enumerate(chunks_b):
            if idx_b in matched_b:
                continue
            
            text_b = normalize_text(chunk_b['text'])
            
            # Use token_set_ratio for better handling of reordered/partial matches
            similarity = fuzz.token_set_ratio(text_a, text_b)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx_b = idx_b
        
        if best_similarity >= similarity_threshold and best_idx_b >= 0:
            matches.append((idx_a, best_idx_b, best_similarity))
            matched_a.add(idx_a)
            matched_b.add(best_idx_b)
    
    # Find unmatched indices
    unique_a = [i for i in range(len(chunks_a)) if i not in matched_a]
    unique_b = [i for i in range(len(chunks_b)) if i not in matched_b]
    
    return matches, unique_a, unique_b


def analyze_page_coverage(chunks: List[Dict]) -> Dict[int, int]:
    """Count how many chunks cover each page."""
    page_counts = defaultdict(int)
    for chunk in chunks:
        page = chunk.get('page', 0)
        if page:
            page_counts[page] += 1
    return dict(page_counts)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_extraction_methods.py <pdf_chunks.pkl> <ocr_chunks.pkl>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    ocr_path = sys.argv[2]
    
    # Load chunks
    print(f"\nLoading chunks...")
    with open(pdf_path, 'rb') as f:
        pdf_chunks = pickle.load(f)
    with open(ocr_path, 'rb') as f:
        ocr_chunks = pickle.load(f)
    
    print(f"PDF chunks: {len(pdf_chunks)}")
    print(f"OCR chunks: {len(ocr_chunks)}")
    
    # Find similar chunks
    print(f"\n{'='*70}")
    print("FINDING SIMILAR CHUNKS")
    print(f"{'='*70}")
    matches, unique_pdf, unique_ocr = find_similar_chunks(pdf_chunks, ocr_chunks)
    
    print(f"\nMatches found: {len(matches)}")
    print(f"Unique to PDF: {len(unique_pdf)}")
    print(f"Unique to OCR: {len(unique_ocr)}")
    
    # Quality comparison for matched chunks
    print(f"\n{'='*70}")
    print("QUALITY COMPARISON FOR MATCHED CHUNKS")
    print(f"{'='*70}")
    
    pdf_better = 0
    ocr_better = 0
    quality_samples = []
    
    for idx_pdf, idx_ocr, similarity in matches[:20]:  # Sample first 20
        pdf_text = pdf_chunks[idx_pdf]['text']
        ocr_text = ocr_chunks[idx_ocr]['text']
        
        pdf_quality = calculate_text_quality_score(pdf_text)
        ocr_quality = calculate_text_quality_score(ocr_text)
        
        if pdf_quality > ocr_quality + 5:
            pdf_better += 1
        elif ocr_quality > pdf_quality + 5:
            ocr_better += 1
        
        quality_samples.append({
            'pdf_idx': idx_pdf,
            'ocr_idx': idx_ocr,
            'similarity': similarity,
            'pdf_quality': pdf_quality,
            'ocr_quality': ocr_quality,
            'winner': 'PDF' if pdf_quality > ocr_quality else 'OCR' if ocr_quality > pdf_quality else 'TIE'
        })
    
    print(f"\nQuality comparison (first 20 matches):")
    print(f"  PDF better: {pdf_better}")
    print(f"  OCR better: {ocr_better}")
    print(f"  Similar: {20 - pdf_better - ocr_better}")
    
    # Show a few examples
    print(f"\n{'='*70}")
    print("QUALITY EXAMPLES (First 5 matches)")
    print(f"{'='*70}")
    
    for i, sample in enumerate(quality_samples[:5]):
        print(f"\n--- Match {i+1} (Similarity: {sample['similarity']:.1f}%, Winner: {sample['winner']}) ---")
        print(f"PDF Quality: {sample['pdf_quality']:.1f} | OCR Quality: {sample['ocr_quality']:.1f}")
        
        pdf_text = pdf_chunks[sample['pdf_idx']]['text'][:200]
        ocr_text = ocr_chunks[sample['ocr_idx']]['text'][:200]
        
        print(f"\nPDF: {pdf_text}...")
        print(f"\nOCR: {ocr_text}...")
    
    # Analyze page coverage
    print(f"\n{'='*70}")
    print("PAGE COVERAGE ANALYSIS")
    print(f"{'='*70}")
    
    pdf_pages = analyze_page_coverage(pdf_chunks)
    ocr_pages = analyze_page_coverage(ocr_chunks)
    
    all_pages = set(pdf_pages.keys()) | set(ocr_pages.keys())
    
    pdf_only_pages = [p for p in pdf_pages if p not in ocr_pages]
    ocr_only_pages = [p for p in ocr_pages if p not in pdf_pages]
    
    print(f"\nTotal pages in PDF: {len(pdf_pages)}")
    print(f"Total pages in OCR: {len(ocr_pages)}")
    print(f"Pages only in PDF: {len(pdf_only_pages)} {pdf_only_pages[:10] if pdf_only_pages else ''}")
    print(f"Pages only in OCR: {len(ocr_only_pages)} {ocr_only_pages[:10] if ocr_only_pages else ''}")
    
    # Check gaps - unique content
    print(f"\n{'='*70}")
    print("UNIQUE CONTENT ANALYSIS")
    print(f"{'='*70}")
    
    if unique_ocr:
        print(f"\nOCR has {len(unique_ocr)} unique chunks (potential gaps filled)")
        print("\nSample OCR-only chunks (first 3):")
        for i in unique_ocr[:3]:
            chunk = ocr_chunks[i]
            print(f"\n  Page {chunk.get('page', '?')}, Section: {chunk.get('section', 'Unknown')}")
            print(f"  Text: {chunk['text'][:150]}...")
            print(f"  Quality: {calculate_text_quality_score(chunk['text']):.1f}")
    
    if unique_pdf:
        print(f"\nPDF has {len(unique_pdf)} unique chunks (OCR missing content)")
        print("\nSample PDF-only chunks (first 3):")
        for i in unique_pdf[:3]:
            chunk = pdf_chunks[i]
            print(f"\n  Page {chunk.get('page', '?')}, Section: {chunk.get('section', 'Unknown')}")
            print(f"  Text: {chunk['text'][:150]}...")
            print(f"  Quality: {calculate_text_quality_score(chunk['text']):.1f}")
    
    # Summary and recommendation
    print(f"\n{'='*70}")
    print("SUMMARY & RECOMMENDATION")
    print(f"{'='*70}")
    
    match_rate = len(matches) / max(len(pdf_chunks), len(ocr_chunks)) * 100
    
    print(f"\nMatch rate: {match_rate:.1f}%")
    print(f"PDF chunks: {len(pdf_chunks)} total, {len(unique_pdf)} unique")
    print(f"OCR chunks: {len(ocr_chunks)} total, {len(unique_ocr)} unique")
    
    # Calculate average quality for all chunks
    pdf_avg_quality = sum(calculate_text_quality_score(c['text']) for c in pdf_chunks) / len(pdf_chunks)
    ocr_avg_quality = sum(calculate_text_quality_score(c['text']) for c in ocr_chunks) / len(ocr_chunks)
    
    print(f"\nAverage quality scores:")
    print(f"  PDF: {pdf_avg_quality:.1f}")
    print(f"  OCR: {ocr_avg_quality:.1f}")
    
    print(f"\nRecommendation:")
    if pdf_avg_quality > ocr_avg_quality + 10:
        print("  ✓ Use PDF extraction only - significantly better quality")
    elif ocr_avg_quality > pdf_avg_quality + 10:
        print("  ✓ Use OCR extraction only - significantly better quality")
    elif len(unique_ocr) > len(pdf_chunks) * 0.1 and ocr_avg_quality > pdf_avg_quality - 5:
        print("  ✓ Merge both methods - OCR fills significant gaps with acceptable quality")
        print(f"    OCR adds {len(unique_ocr)} chunks ({len(unique_ocr)/len(pdf_chunks)*100:.1f}% more content)")
    elif len(unique_pdf) > len(ocr_chunks) * 0.1 and pdf_avg_quality > ocr_avg_quality - 5:
        print("  ✓ Use PDF extraction - more complete with better quality")
    else:
        print("  ✓ Use PDF extraction with selective OCR fallback for problematic pages")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
