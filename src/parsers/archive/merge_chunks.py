"""
Merge chunks from PDF and OCR extraction methods.

For overlapping content, keeps the higher quality version.
For unique content from either source, includes both.
Adds metadata to track extraction source.

Usage:
    python src/parsers/merge_chunks.py <pdf_chunks.pkl> <ocr_chunks.pkl> <output.pkl>
"""

import pickle
import sys
import os
from typing import List, Dict
from rapidfuzz import fuzz
import re


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def calculate_quality_score(text: str) -> float:
    """Calculate text quality score (higher = better)."""
    if not text or len(text) < 20:
        return 0.0
    
    score = 100.0
    
    # Penalize OCR artifacts
    repeated_chars = len(re.findall(r'([a-z])\1{4,}', text.lower()))
    score -= repeated_chars * 5
    
    # Check vowel ratio
    text_clean = re.sub(r'[^a-zA-Z ]', '', text)
    if len(text_clean) > 20:
        vowel_ratio = len(re.findall(r'[aeiouAEIOU ]', text_clean)) / len(text_clean)
        if vowel_ratio < 0.25:
            score -= 20
        elif vowel_ratio < 0.30:
            score -= 10
    
    # Penalize excessive dots
    excessive_dots = len(re.findall(r'\.{6,}', text))
    score -= excessive_dots * 5
    
    # Reward proper sentences
    sentences = re.split(r'[.!?]+', text)
    proper_sentences = sum(1 for s in sentences if 3 <= len(s.split()) <= 50)
    score += proper_sentences * 2
    
    # Penalize garbled/corrupted text patterns
    # Check for nonsense patterns like "ona iltinesS aris bully"
    words = text.split()
    if len(words) >= 5:
        # Count words with unusual capitalization (mixed case mid-word)
        weird_caps = sum(1 for w in words if len(w) > 3 and any(c.isupper() for c in w[1:]) and any(c.islower() for c in w))
        score -= weird_caps * 3
    
    return max(0.0, score)


def has_semantic_overlap(text1: str, text2: str) -> bool:
    """
    Check if two texts discuss the same specific topic/rule.
    Returns True if they're about the same thing (should deduplicate).
    Returns False if they discuss different aspects (keep both).
    """
    # Extract key game terms and concepts
    game_terms_pattern = r'\b(water|window|anomaly|enemy|loot|radiation|movement|attack|weapon|artifact|grenade|bolt|decoy|attention|stalker|move through|line of sight|los)\b'
    
    terms1 = set(re.findall(game_terms_pattern, text1.lower()))
    terms2 = set(re.findall(game_terms_pattern, text2.lower()))
    
    if not terms1 or not terms2:
        return True  # If no key terms, treat as generic overlap
    
    # Calculate term overlap
    overlap = len(terms1 & terms2) / max(len(terms1), len(terms2))
    
    # If less than 60% term overlap, they discuss different aspects
    return overlap >= 0.6


def merge_chunks(
    pdf_chunks: List[Dict],
    ocr_chunks: List[Dict],
    similarity_threshold: float = 80.0  # Increased from 70 to 80 for more conservative deduplication
) -> List[Dict]:
    """
    Merge chunks from both sources with conservative deduplication.
    
    Strategy:
    1. Find matching chunks between PDF and OCR (80%+ similarity)
    2. For matches, check if they discuss the same semantic topic
    3. If different topics despite similarity, keep both
    4. If same topic, keep the higher quality version
    5. Add all unique chunks from both sources
    6. Tag each chunk with extraction_method metadata
    """
    
    merged = []
    used_ocr_indices = set()
    kept_both_count = 0
    
    print(f"Merging {len(pdf_chunks)} PDF chunks with {len(ocr_chunks)} OCR chunks...")
    print(f"Using conservative similarity threshold: {similarity_threshold}%")
    
    # Process each PDF chunk
    for pdf_idx, pdf_chunk in enumerate(pdf_chunks):
        pdf_text = normalize_text(pdf_chunk['text'])
        pdf_quality = calculate_quality_score(pdf_chunk['text'])
        
        # Find best matching OCR chunk
        best_match_idx = -1
        best_similarity = 0.0
        
        for ocr_idx, ocr_chunk in enumerate(ocr_chunks):
            if ocr_idx in used_ocr_indices:
                continue
            
            ocr_text = normalize_text(ocr_chunk['text'])
            similarity = fuzz.token_set_ratio(pdf_text, ocr_text)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = ocr_idx
        
        # Decide what to keep
        if best_similarity >= similarity_threshold and best_match_idx >= 0:
            # Match found - but check if they discuss different semantic topics
            ocr_chunk = ocr_chunks[best_match_idx]
            ocr_quality = calculate_quality_score(ocr_chunk['text'])
            
            # Check semantic overlap
            same_topic = has_semantic_overlap(pdf_chunk['text'], ocr_chunk['text'])
            
            if not same_topic:
                # Different topics despite text similarity - keep both!
                # This prevents losing important nuanced information
                pdf_chunk_copy = pdf_chunk.copy()
                pdf_chunk_copy['extraction_method'] = 'pdf'
                pdf_chunk_copy['quality_score'] = pdf_quality
                pdf_chunk_copy['alt_source_available'] = False
                pdf_chunk_copy['semantic_variant'] = True
                merged.append(pdf_chunk_copy)
                
                # Don't mark as used - OCR chunk will be added separately
                kept_both_count += 1
                
            elif abs(pdf_quality - ocr_quality) <= 5:
                # Very similar quality and same topic - slightly prefer PDF for consistency
                chunk = pdf_chunk.copy()
                chunk['extraction_method'] = 'pdf'
                chunk['quality_score'] = pdf_quality
                chunk['alt_source_available'] = True
                chunk['similarity_to_alt'] = best_similarity
                merged.append(chunk)
                used_ocr_indices.add(best_match_idx)
                
            elif pdf_quality > ocr_quality:
                # Keep PDF version (higher quality)
                chunk = pdf_chunk.copy()
                chunk['extraction_method'] = 'pdf'
                chunk['quality_score'] = pdf_quality
                chunk['alt_source_available'] = True
                chunk['similarity_to_alt'] = best_similarity
                merged.append(chunk)
                used_ocr_indices.add(best_match_idx)
            else:
                # Keep OCR version (higher quality)
                chunk = ocr_chunk.copy()
                chunk['extraction_method'] = 'ocr'
                chunk['quality_score'] = ocr_quality
                chunk['alt_source_available'] = True
                chunk['similarity_to_alt'] = best_similarity
                merged.append(chunk)
                used_ocr_indices.add(best_match_idx)
            
        else:
            # No match - keep PDF chunk as unique
            chunk = pdf_chunk.copy()
            chunk['extraction_method'] = 'pdf'
            chunk['quality_score'] = pdf_quality
            chunk['alt_source_available'] = False
            merged.append(chunk)
    
    # Add remaining unique OCR chunks
    for ocr_idx, ocr_chunk in enumerate(ocr_chunks):
        if ocr_idx not in used_ocr_indices:
            chunk = ocr_chunk.copy()
            chunk['extraction_method'] = 'ocr'
            chunk['quality_score'] = calculate_quality_score(chunk['text'])
            chunk['alt_source_available'] = False
            merged.append(chunk)
    
    # Sort by page and chunk_index for consistency
    merged.sort(key=lambda x: (x.get('page', 0), x.get('chunk_index', 0)))
    
    if kept_both_count > 0:
        print(f"Kept both versions for {kept_both_count} chunks (different semantic topics)")
    
    return merged


def main():
    if len(sys.argv) != 4:
        print("Usage: python merge_chunks.py <pdf_chunks.pkl> <ocr_chunks.pkl> <output.pkl>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    ocr_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Load chunks
    print(f"\nLoading chunks...")
    with open(pdf_path, 'rb') as f:
        pdf_chunks = pickle.load(f)
    with open(ocr_path, 'rb') as f:
        ocr_chunks = pickle.load(f)
    
    print(f"PDF chunks: {len(pdf_chunks)}")
    print(f"OCR chunks: {len(ocr_chunks)}")
    
    # Merge
    merged_chunks = merge_chunks(pdf_chunks, ocr_chunks)
    
    # Statistics
    pdf_kept = sum(1 for c in merged_chunks if c['extraction_method'] == 'pdf')
    ocr_kept = sum(1 for c in merged_chunks if c['extraction_method'] == 'ocr')
    
    print(f"\n{'='*70}")
    print("MERGE RESULTS")
    print(f"{'='*70}")
    print(f"Total merged chunks: {len(merged_chunks)}")
    print(f"  From PDF: {pdf_kept} ({pdf_kept/len(merged_chunks)*100:.1f}%)")
    print(f"  From OCR: {ocr_kept} ({ocr_kept/len(merged_chunks)*100:.1f}%)")
    print(f"\nCoverage increase: {len(merged_chunks) - len(pdf_chunks)} chunks (+{(len(merged_chunks) - len(pdf_chunks))/len(pdf_chunks)*100:.1f}%)")
    
    # Save merged chunks
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(merged_chunks, f)
    
    # Also save JSON for inspection
    json_path = output_path.replace('.pkl', '.json')
    import json
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved merged chunks to:")
    print(f"  {output_path}")
    print(f"  {json_path}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
