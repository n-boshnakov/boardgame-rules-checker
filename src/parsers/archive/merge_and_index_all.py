"""
Unified Chunk Merger and Indexer

This script merges PDF, OCR, and Forum chunks into a single high-quality index
with intelligent deduplication and quality-based ranking.

Strategy:
1. Merge PDF and OCR chunks with quality-based selection
2. Add quality metadata to improve search relevance
3. Add extraction_confidence scores
4. Include forum Q&A pairs
5. Create a unified index with proper weighting

Usage:
    python src/parsers/merge_and_index_all.py
"""

import pickle
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
from rapidfuzz import fuzz
import re
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def calculate_quality_score(text: str) -> float:
    """Calculate text quality score (0-100, higher = better).
    
    Evaluates:
    - OCR artifacts (repeated chars, garbled text)
    - Vowel ratio (natural text has ~40% vowels)
    - Sentence structure
    - Capitalization patterns
    """
    if not text or len(text) < 20:
        return 0.0
    
    score = 100.0
    
    # Penalize OCR artifacts
    repeated_chars = len(re.findall(r'([a-z])\1{4,}', text.lower()))
    score -= repeated_chars * 5
    
    # Check vowel ratio (healthy ratio is 35-45%)
    text_clean = re.sub(r'[^a-zA-Z ]', '', text)
    if len(text_clean) > 20:
        vowel_ratio = len(re.findall(r'[aeiouAEIOU ]', text_clean)) / len(text_clean)
        if vowel_ratio < 0.25:
            score -= 20  # Too few vowels - likely corrupted
        elif vowel_ratio < 0.30:
            score -= 10
        elif vowel_ratio > 0.55:
            score -= 15  # Too many vowels - also unusual
    
    # Penalize excessive dots (common OCR artifact)
    excessive_dots = len(re.findall(r'\.{6,}', text))
    score -= excessive_dots * 5
    
    # Reward proper sentences (3-50 words each)
    sentences = re.split(r'[.!?]+', text)
    proper_sentences = sum(1 for s in sentences if 3 <= len(s.split()) <= 50)
    score += min(proper_sentences * 2, 20)  # Cap bonus at 20
    
    # Penalize garbled capitalization
    words = text.split()
    if len(words) >= 5:
        weird_caps = sum(1 for w in words if len(w) > 3 and 
                        any(c.isupper() for c in w[1:]) and 
                        any(c.islower() for c in w))
        score -= weird_caps * 3
    
    # Penalize very short chunks (likely incomplete)
    if len(text) < 100:
        score -= 10
    
    return max(0.0, min(100.0, score))


def calculate_extraction_confidence(chunk: Dict) -> float:
    """Calculate extraction confidence (0-1) based on multiple factors.
    
    Higher confidence means the chunk is more reliable for search.
    """
    quality = calculate_quality_score(chunk.get('text', ''))
    
    # Base confidence from quality
    confidence = quality / 100.0
    
    # Boost for PDF extraction (generally more reliable)
    if chunk.get('extraction_method') == 'pdf':
        confidence *= 1.1
    
    # Boost if chunk has good section metadata
    if chunk.get('section') and len(chunk.get('section', '')) > 3:
        confidence *= 1.05
    
    # Penalize if it's a semantic variant (potential duplicate)
    if chunk.get('semantic_variant', False):
        confidence *= 0.9
    
    return min(1.0, confidence)


def has_semantic_overlap(text1: str, text2: str, threshold: float = 0.6) -> bool:
    """Check if two texts discuss the same specific topic/rule.
    
    Returns True if they're about the same thing (>60% term overlap).
    Returns False if they discuss different aspects (keep both).
    """
    # Extract key game terms
    game_terms_pattern = r'\b(water|window|anomaly|enemy|loot|radiation|movement|attack|weapon|artifact|grenade|bolt|decoy|attention|stalker|move through|line of sight|los|action|turn|phase|round|card|tile|damage|heal|search|objective)\b'
    
    terms1 = set(re.findall(game_terms_pattern, text1.lower()))
    terms2 = set(re.findall(game_terms_pattern, text2.lower()))
    
    if not terms1 or not terms2:
        return True  # If no key terms, treat as generic overlap
    
    # Calculate term overlap
    overlap = len(terms1 & terms2) / max(len(terms1), len(terms2))
    
    return overlap >= threshold


def merge_pdf_ocr_chunks(
    pdf_chunks: List[Dict],
    ocr_chunks: List[Dict],
    similarity_threshold: float = 80.0
) -> List[Dict]:
    """Merge PDF and OCR chunks with intelligent deduplication.
    
    Strategy:
    1. Find matching chunks (80%+ text similarity)
    2. Check if they discuss the same semantic topic
    3. If different topics, keep both (semantic variants)
    4. If same topic, keep higher quality version
    5. Add quality and confidence metadata
    """
    merged = []
    used_ocr_indices = set()
    kept_both_count = 0
    
    print(f"\n[Merger] Merging {len(pdf_chunks)} PDF chunks with {len(ocr_chunks)} OCR chunks...")
    print(f"[Merger] Using similarity threshold: {similarity_threshold}%")
    
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
            ocr_chunk = ocr_chunks[best_match_idx]
            ocr_quality = calculate_quality_score(ocr_chunk['text'])
            
            # Check semantic overlap
            same_topic = has_semantic_overlap(pdf_chunk['text'], ocr_chunk['text'])
            
            if not same_topic:
                # Different topics - keep both as semantic variants
                pdf_chunk_copy = pdf_chunk.copy()
                pdf_chunk_copy['extraction_method'] = 'pdf'
                pdf_chunk_copy['quality_score'] = pdf_quality
                pdf_chunk_copy['extraction_confidence'] = calculate_extraction_confidence(pdf_chunk_copy)
                pdf_chunk_copy['semantic_variant'] = True
                pdf_chunk_copy['source_type'] = 'rulebook'
                merged.append(pdf_chunk_copy)
                
                kept_both_count += 1
            
            elif pdf_quality >= ocr_quality:
                # Keep PDF version (higher or equal quality)
                chunk = pdf_chunk.copy()
                chunk['extraction_method'] = 'pdf'
                chunk['quality_score'] = pdf_quality
                chunk['extraction_confidence'] = calculate_extraction_confidence(chunk)
                chunk['source_type'] = 'rulebook'
                chunk['alt_source_available'] = True
                chunk['similarity_to_alt'] = best_similarity
                merged.append(chunk)
                used_ocr_indices.add(best_match_idx)
            else:
                # Keep OCR version (higher quality)
                chunk = ocr_chunk.copy()
                chunk['extraction_method'] = 'ocr'
                chunk['quality_score'] = ocr_quality
                chunk['extraction_confidence'] = calculate_extraction_confidence(chunk)
                chunk['source_type'] = 'rulebook'
                chunk['alt_source_available'] = True
                chunk['similarity_to_alt'] = best_similarity
                merged.append(chunk)
                used_ocr_indices.add(best_match_idx)
        else:
            # No match - keep PDF chunk as unique
            chunk = pdf_chunk.copy()
            chunk['extraction_method'] = 'pdf'
            chunk['quality_score'] = pdf_quality
            chunk['extraction_confidence'] = calculate_extraction_confidence(chunk)
            chunk['source_type'] = 'rulebook'
            chunk['alt_source_available'] = False
            merged.append(chunk)
    
    # Add remaining unique OCR chunks
    for ocr_idx, ocr_chunk in enumerate(ocr_chunks):
        if ocr_idx not in used_ocr_indices:
            chunk = ocr_chunk.copy()
            chunk['extraction_method'] = 'ocr'
            chunk['quality_score'] = calculate_quality_score(chunk['text'])
            chunk['extraction_confidence'] = calculate_extraction_confidence(chunk)
            chunk['source_type'] = 'rulebook'
            chunk['alt_source_available'] = False
            merged.append(chunk)
    
    # Sort by page and chunk_index for consistency
    merged.sort(key=lambda x: (x.get('page', 0), x.get('chunk_index', 0)))
    
    # Statistics
    pdf_kept = sum(1 for c in merged if c['extraction_method'] == 'pdf')
    ocr_kept = sum(1 for c in merged if c['extraction_method'] == 'ocr')
    high_confidence = sum(1 for c in merged if c.get('extraction_confidence', 0) >= 0.8)
    
    print(f"[Merger] Merge complete:")
    print(f"  Total chunks: {len(merged)}")
    print(f"  From PDF: {pdf_kept} ({pdf_kept/len(merged)*100:.1f}%)")
    print(f"  From OCR: {ocr_kept} ({ocr_kept/len(merged)*100:.1f}%)")
    print(f"  High confidence (≥80%): {high_confidence} ({high_confidence/len(merged)*100:.1f}%)")
    if kept_both_count > 0:
        print(f"  Semantic variants kept: {kept_both_count}")
    
    return merged


def add_forum_qa_chunks(merged_chunks: List[Dict], forum_qa_path: str) -> List[Dict]:
    """Add forum Q&A pairs to the merged chunks.
    
    Forum chunks are kept separate (not merged with rulebook) but included
    in the same index for dual-source search.
    """
    print(f"\n[Merger] Loading forum Q&A pairs from {forum_qa_path}...")
    
    try:
        with open(forum_qa_path, 'r', encoding='utf-8') as f:
            forum_data = json.load(f)
        
        print(f"[Merger] Found {len(forum_data)} forum Q&A pairs")
        
        # Forum pairs already have proper structure from forum_qa_pairs.json
        # Just add necessary metadata for consistency
        forum_chunks = []
        for qa in forum_data:
            chunk = {
                'text': qa['question'],  # Question text for search
                'answer': qa['answer'],  # Answer text
                'source_type': 'forum',
                'extraction_method': 'forum_scraper',
                'quality_score': qa.get('metadata', {}).get('useful_answer_count', 0) * 10,  # Scale to 0-100
                'extraction_confidence': min(1.0, qa.get('metadata', {}).get('useful_answer_count', 0) / 10),
                'thread_id': qa['thread_id'],
                'url': qa['thread_url'],
                'answer_user': qa.get('raw_answers', [{}])[0].get('author', '') if qa.get('raw_answers') else '',
                'qa_id': qa['id'],
                # Forum chunks don't have page/section
                'page': None,
                'section': None,
                'doc_type': 'forum_qa'
            }
            forum_chunks.append(chunk)
        
        print(f"[Merger] Processed {len(forum_chunks)} forum chunks")
        
        # Combine with rulebook chunks
        all_chunks = merged_chunks + forum_chunks
        
        print(f"\n[Merger] Combined index statistics:")
        print(f"  Rulebook chunks: {len(merged_chunks)}")
        print(f"  Forum Q&A pairs: {len(forum_chunks)}")
        print(f"  Total chunks: {len(all_chunks)}")
        
        return all_chunks
        
    except FileNotFoundError:
        print(f"[Merger] Warning: Forum Q&A file not found at {forum_qa_path}")
        print(f"[Merger] Continuing with only rulebook chunks...")
        return merged_chunks
    except Exception as e:
        print(f"[Merger] Error loading forum data: {e}")
        print(f"[Merger] Continuing with only rulebook chunks...")
        return merged_chunks


def save_merged_chunks(chunks: List[Dict], output_dir: str = "data/processed"):
    """Save merged chunks in multiple formats for different uses."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle for fast loading
    pkl_path = os.path.join(output_dir, "chunks_unified.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"\n[Merger] Saved unified chunks to {pkl_path}")
    
    # Save as JSON for inspection
    json_path = os.path.join(output_dir, "chunks_unified.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[Merger] Saved unified chunks to {json_path}")
    
    # Save quality report
    report_path = os.path.join(output_dir, "chunks_unified_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("UNIFIED CHUNK QUALITY REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall statistics
        rulebook_chunks = [c for c in chunks if c.get('source_type') == 'rulebook']
        forum_chunks = [c for c in chunks if c.get('source_type') == 'forum']
        
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"  Rulebook: {len(rulebook_chunks)}\n")
        f.write(f"  Forum: {len(forum_chunks)}\n\n")
        
        # Quality breakdown for rulebook chunks
        if rulebook_chunks:
            qualities = [c.get('quality_score', 0) for c in rulebook_chunks]
            confidences = [c.get('extraction_confidence', 0) for c in rulebook_chunks]
            
            f.write("Rulebook Chunk Quality:\n")
            f.write(f"  Mean quality score: {np.mean(qualities):.2f}\n")
            f.write(f"  Median quality score: {np.median(qualities):.2f}\n")
            f.write(f"  High quality (≥80): {sum(1 for q in qualities if q >= 80)}\n")
            f.write(f"  Medium quality (60-80): {sum(1 for q in qualities if 60 <= q < 80)}\n")
            f.write(f"  Low quality (<60): {sum(1 for q in qualities if q < 60)}\n\n")
            
            f.write("Extraction Confidence:\n")
            f.write(f"  Mean confidence: {np.mean(confidences):.2%}\n")
            f.write(f"  High confidence (≥0.8): {sum(1 for c in confidences if c >= 0.8)}\n")
            f.write(f"  Medium confidence (0.6-0.8): {sum(1 for c in confidences if 0.6 <= c < 0.8)}\n")
            f.write(f"  Low confidence (<0.6): {sum(1 for c in confidences if c < 0.6)}\n\n")
            
            # Extraction method breakdown
            pdf_count = sum(1 for c in rulebook_chunks if c.get('extraction_method') == 'pdf')
            ocr_count = sum(1 for c in rulebook_chunks if c.get('extraction_method') == 'ocr')
            
            f.write("Extraction Methods:\n")
            f.write(f"  PDF: {pdf_count} ({pdf_count/len(rulebook_chunks)*100:.1f}%)\n")
            f.write(f"  OCR: {ocr_count} ({ocr_count/len(rulebook_chunks)*100:.1f}%)\n\n")
        
        # Forum chunk statistics
        if forum_chunks:
            forum_qualities = [c.get('quality_score', 0) for c in forum_chunks]
            f.write("Forum Q&A Quality:\n")
            f.write(f"  Mean quality score: {np.mean(forum_qualities):.2f}\n")
            f.write(f"  High quality (≥70): {sum(1 for q in forum_qualities if q >= 70)}\n")
    
    print(f"[Merger] Saved quality report to {report_path}")


def main():
    """Main workflow for merging and preparing all chunks for indexing."""
    print("="*70)
    print("UNIFIED CHUNK MERGER")
    print("Merges PDF, OCR, and Forum chunks into a single optimized index")
    print("="*70)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    pdf_chunks_path = project_root / "data" / "processed" / "chunks.pkl"
    ocr_chunks_path = project_root / "data" / "processed" / "archive" / "chunks_ocr_2026-01-13_15-46-36.pkl"
    forum_qa_path = project_root / "data" / "processed" / "forum_qa" / "forum_qa_pairs.json"
    output_dir = project_root / "data" / "processed"
    
    # Load PDF chunks
    print(f"\n[Merger] Loading PDF chunks from {pdf_chunks_path}...")
    try:
        with open(pdf_chunks_path, 'rb') as f:
            pdf_chunks = pickle.load(f)
        print(f"[Merger] Loaded {len(pdf_chunks)} PDF chunks")
    except FileNotFoundError:
        print(f"[Merger] Error: PDF chunks not found at {pdf_chunks_path}")
        sys.exit(1)
    
    # Load OCR chunks
    print(f"[Merger] Loading OCR chunks from {ocr_chunks_path}...")
    try:
        with open(ocr_chunks_path, 'rb') as f:
            ocr_chunks = pickle.load(f)
        print(f"[Merger] Loaded {len(ocr_chunks)} OCR chunks")
    except FileNotFoundError:
        print(f"[Merger] Warning: OCR chunks not found at {ocr_chunks_path}")
        print(f"[Merger] Continuing with only PDF chunks...")
        ocr_chunks = []
    
    # Merge PDF and OCR chunks
    merged_chunks = merge_pdf_ocr_chunks(pdf_chunks, ocr_chunks)
    
    # Add forum Q&A pairs
    all_chunks = add_forum_qa_chunks(merged_chunks, str(forum_qa_path))
    
    # Save unified chunks
    save_merged_chunks(all_chunks, str(output_dir))
    
    print("\n" + "="*70)
    print("MERGE COMPLETE")
    print("="*70)
    print(f"\nNext steps:")
    print("1. Generate embeddings: python src/search/generate_embeddings.py")
    print("2. Index to Elasticsearch: python src/search/indexer_unified.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
