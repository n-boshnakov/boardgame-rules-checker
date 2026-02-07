"""
Universal Chunk Merger

Flexible script that can merge chunks from any combination of sources:
- PDF chunks only
- OCR chunks only  
- PDF + OCR chunks
- Forum Q&A pairs only
- PDF + Forum
- OCR + Forum
- PDF + OCR + Forum (all three)

Usage:
    # Merge PDF + OCR
    python src/parsers/merge_chunks_universal.py --pdf data/processed/chunks.pkl --ocr data/processed/chunks_ocr_corrected.pkl --output data/processed/chunks_merged.pkl
    
    # Merge PDF + Forum
    python src/parsers/merge_chunks_universal.py --pdf data/processed/chunks.pkl --forum data/processed/forum_qa/forum_qa_pairs.json --output data/processed/chunks_with_forum.pkl
    
    # Merge all three sources
    python src/parsers/merge_chunks_universal.py --pdf data/processed/chunks.pkl --ocr data/processed/chunks_ocr_corrected.pkl --forum data/processed/forum_qa/forum_qa_pairs.json --output data/processed/chunks_unified.pkl
    
    # Just reformat a single source
    python src/parsers/merge_chunks_universal.py --pdf data/processed/chunks.pkl --output data/processed/chunks_formatted.pkl
"""

import pickle
import json
import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from rapidfuzz import fuzz
import re
import numpy as np


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
    # Extract key game terms (customize for your domain)
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


def load_pdf_chunks(path: str) -> List[Dict]:
    """Load PDF chunks from pickle file."""
    print(f"\n[Merger] Loading PDF chunks from {path}...")
    with open(path, 'rb') as f:
        chunks = pickle.load(f)
    print(f"[Merger] Loaded {len(chunks)} PDF chunks")
    
    # Ensure proper metadata
    for chunk in chunks:
        if 'extraction_method' not in chunk:
            chunk['extraction_method'] = 'pdf'
        if 'source_type' not in chunk:
            chunk['source_type'] = 'rulebook'
        if 'quality_score' not in chunk:
            chunk['quality_score'] = calculate_quality_score(chunk.get('text', ''))
        if 'extraction_confidence' not in chunk:
            chunk['extraction_confidence'] = calculate_extraction_confidence(chunk)
    
    return chunks


def load_ocr_chunks(path: str) -> List[Dict]:
    """Load OCR chunks from pickle file."""
    print(f"\n[Merger] Loading OCR chunks from {path}...")
    with open(path, 'rb') as f:
        chunks = pickle.load(f)
    print(f"[Merger] Loaded {len(chunks)} OCR chunks")
    
    # Ensure proper metadata
    for chunk in chunks:
        if 'extraction_method' not in chunk:
            chunk['extraction_method'] = 'ocr'
        if 'source_type' not in chunk:
            chunk['source_type'] = 'rulebook'
        if 'quality_score' not in chunk:
            chunk['quality_score'] = calculate_quality_score(chunk.get('text', ''))
        if 'extraction_confidence' not in chunk:
            chunk['extraction_confidence'] = calculate_extraction_confidence(chunk)
    
    return chunks


def load_forum_chunks(path: str) -> List[Dict]:
    """Load forum Q&A pairs from JSON file and convert to chunk format."""
    print(f"\n[Merger] Loading forum Q&A pairs from {path}...")
    
    with open(path, 'r', encoding='utf-8') as f:
        forum_data = json.load(f)
    
    print(f"[Merger] Found {len(forum_data)} forum Q&A pairs")
    
    # Convert to chunk format
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
    return forum_chunks


def save_merged_chunks(chunks: List[Dict], output_path: str, generate_report: bool = True):
    """Save merged chunks in multiple formats."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle for fast loading
    with open(output_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"\n[Merger] Saved merged chunks to {output_path}")
    
    # Save as JSON for inspection
    json_path = output_path.replace('.pkl', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[Merger] Saved merged chunks to {json_path}")
    
    if generate_report:
        # Save quality report
        report_path = output_path.replace('.pkl', '_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MERGED CHUNK QUALITY REPORT\n")
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
                f.write(f"  High quality (≥80): {sum(1 for q in qualities if q >= 80)} ({sum(1 for q in qualities if q >= 80)/len(qualities)*100:.1f}%)\n")
                f.write(f"  Medium quality (60-80): {sum(1 for q in qualities if 60 <= q < 80)} ({sum(1 for q in qualities if 60 <= q < 80)/len(qualities)*100:.1f}%)\n")
                f.write(f"  Low quality (<60): {sum(1 for q in qualities if q < 60)} ({sum(1 for q in qualities if q < 60)/len(qualities)*100:.1f}%)\n\n")
                
                f.write("Extraction Confidence:\n")
                f.write(f"  Mean confidence: {np.mean(confidences):.2%}\n")
                f.write(f"  High confidence (≥0.8): {sum(1 for c in confidences if c >= 0.8)} ({sum(1 for c in confidences if c >= 0.8)/len(confidences)*100:.1f}%)\n")
                f.write(f"  Medium confidence (0.6-0.8): {sum(1 for c in confidences if 0.6 <= c < 0.8)} ({sum(1 for c in confidences if 0.6 <= c < 0.8)/len(confidences)*100:.1f}%)\n")
                f.write(f"  Low confidence (<0.6): {sum(1 for c in confidences if c < 0.6)} ({sum(1 for c in confidences if c < 0.6)/len(confidences)*100:.1f}%)\n\n")
                
                # Extraction method breakdown
                pdf_count = sum(1 for c in rulebook_chunks if c.get('extraction_method') == 'pdf')
                ocr_count = sum(1 for c in rulebook_chunks if c.get('extraction_method') == 'ocr')
                
                if pdf_count > 0 or ocr_count > 0:
                    f.write("Extraction Methods:\n")
                    if pdf_count > 0:
                        f.write(f"  PDF: {pdf_count} ({pdf_count/len(rulebook_chunks)*100:.1f}%)\n")
                    if ocr_count > 0:
                        f.write(f"  OCR: {ocr_count} ({ocr_count/len(rulebook_chunks)*100:.1f}%)\n")
                    f.write("\n")
            
            # Forum chunk statistics
            if forum_chunks:
                forum_qualities = [c.get('quality_score', 0) for c in forum_chunks]
                f.write("Forum Q&A Quality:\n")
                f.write(f"  Mean quality score: {np.mean(forum_qualities):.2f}\n")
                f.write(f"  High quality (≥70): {sum(1 for q in forum_qualities if q >= 70)} ({sum(1 for q in forum_qualities if q >= 70)/len(forum_qualities)*100:.1f}%)\n")
        
        print(f"[Merger] Saved quality report to {report_path}")


def main():
    """Main workflow for merging chunks from various sources."""
    parser = argparse.ArgumentParser(
        description='Universal chunk merger - merge chunks from any combination of PDF, OCR, and Forum sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge PDF + OCR
  python merge_chunks_universal.py --pdf chunks.pkl --ocr chunks_ocr.pkl -o chunks_merged.pkl
  
  # Merge PDF + Forum
  python merge_chunks_universal.py --pdf chunks.pkl --forum forum_qa_pairs.json -o chunks_with_forum.pkl
  
  # Merge all three sources
  python merge_chunks_universal.py --pdf chunks.pkl --ocr chunks_ocr.pkl --forum forum_qa_pairs.json -o chunks_unified.pkl
  
  # Just reformat a single source
  python merge_chunks_universal.py --pdf chunks.pkl -o chunks_formatted.pkl
        """
    )
    
    parser.add_argument('--pdf', type=str, help='Path to PDF chunks pickle file')
    parser.add_argument('--ocr', type=str, help='Path to OCR chunks pickle file')
    parser.add_argument('--forum', type=str, help='Path to forum Q&A pairs JSON file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output path for merged chunks')
    parser.add_argument('--similarity-threshold', type=float, default=80.0, help='Similarity threshold for deduplication (default: 80.0)')
    parser.add_argument('--no-report', action='store_true', help='Skip generating quality report')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not any([args.pdf, args.ocr, args.forum]):
        print("[Merger] Error: At least one source (--pdf, --ocr, or --forum) must be specified")
        sys.exit(1)
    
    print("="*70)
    print("UNIVERSAL CHUNK MERGER")
    print("="*70)
    
    # Load chunks from specified sources
    all_chunks = []
    
    # Load PDF chunks if specified
    pdf_chunks = []
    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"[Merger] Error: PDF chunks file not found: {args.pdf}")
            sys.exit(1)
        pdf_chunks = load_pdf_chunks(args.pdf)
        all_chunks.extend(pdf_chunks)
    
    # Load OCR chunks if specified
    ocr_chunks = []
    if args.ocr:
        if not os.path.exists(args.ocr):
            print(f"[Merger] Error: OCR chunks file not found: {args.ocr}")
            sys.exit(1)
        ocr_chunks = load_ocr_chunks(args.ocr)
    
    # Merge PDF and OCR if both provided
    if pdf_chunks and ocr_chunks:
        # Replace pdf_chunks in all_chunks with merged version
        all_chunks = merge_pdf_ocr_chunks(pdf_chunks, ocr_chunks, args.similarity_threshold)
    elif ocr_chunks:
        # Only OCR chunks provided
        all_chunks.extend(ocr_chunks)
    
    # Load forum chunks if specified
    if args.forum:
        if not os.path.exists(args.forum):
            print(f"[Merger] Error: Forum Q&A file not found: {args.forum}")
            sys.exit(1)
        forum_chunks = load_forum_chunks(args.forum)
        all_chunks.extend(forum_chunks)
    
    # Summary
    print(f"\n{'='*70}")
    print("MERGE SUMMARY")
    print(f"{'='*70}")
    print(f"Total chunks: {len(all_chunks)}")
    
    rulebook_count = sum(1 for c in all_chunks if c.get('source_type') == 'rulebook')
    forum_count = sum(1 for c in all_chunks if c.get('source_type') == 'forum')
    
    if rulebook_count > 0:
        print(f"  Rulebook chunks: {rulebook_count}")
        pdf_count = sum(1 for c in all_chunks if c.get('extraction_method') == 'pdf')
        ocr_count = sum(1 for c in all_chunks if c.get('extraction_method') == 'ocr')
        if pdf_count > 0:
            print(f"    From PDF: {pdf_count} ({pdf_count/rulebook_count*100:.1f}%)")
        if ocr_count > 0:
            print(f"    From OCR: {ocr_count} ({ocr_count/rulebook_count*100:.1f}%)")
    
    if forum_count > 0:
        print(f"  Forum Q&A pairs: {forum_count}")
    
    print(f"{'='*70}\n")
    
    # Save merged chunks
    save_merged_chunks(all_chunks, args.output, generate_report=not args.no_report)
    
    print("\n✓ Merge complete!")
    print(f"\nNext step: Generate embeddings and index to Elasticsearch")
    print(f"  python src/search/embedder.py {args.output}")
    print(f"  python src/search/index_chunks.py {args.output.replace('.pkl', '_embeddings.parquet')}")


if __name__ == "__main__":
    main()
