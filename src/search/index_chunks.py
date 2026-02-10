"""
Universal Elasticsearch Indexer

Flexible indexer that can index chunks from any source:
- PDF chunks only
- OCR chunks only
- PDF + OCR merged chunks
- Forum Q&A pairs only
- Mixed rulebook + forum chunks
- Any combination of sources

The indexer automatically detects the chunk format and creates appropriate
Elasticsearch documents with proper field mappings.

Usage:
    # Basic usage - auto-detects chunk format
    python src/search/index_chunks.py data/processed/chunks_embeddings.parquet
    
    # Specify custom index name
    python src/search/index_chunks.py data/processed/chunks_embeddings.parquet --index my_custom_index
    
    # Specify Elasticsearch host
    python src/search/index_chunks.py data/processed/chunks_embeddings.parquet --es-host http://localhost:9200
    
    # Load chunks from separate files (auto-merge embeddings)
    python src/search/index_chunks.py data/processed/chunks_embeddings.parquet --chunks data/processed/chunks_unified.pkl
"""

import pandas as pd
from elasticsearch import Elasticsearch, helpers
import numpy as np
import pickle
import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Index configuration
EMBEDDING_DIMS = 1024  # BAAI/bge-m3

# Enhanced mapping with quality and confidence fields
MAPPING_TEMPLATE = {
    "mappings": {
        "properties": {
            # Common fields (all chunk types)
            "text": {"type": "text"},
            "source_type": {"type": "keyword"},  # "rulebook", "faq", or "forum"
            "doc_type": {"type": "keyword"},
            "priority": {"type": "integer"},  # 75 (faq) > 50 (rulebook) > 30 (forum)
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "index": True,
                "similarity": "cosine"
            },
            
            # Quality and confidence (for all sources)
            "quality_score": {"type": "float"},  # 0-100
            "extraction_confidence": {"type": "float"},  # 0-1
            "extraction_method": {"type": "keyword"},  # "pdf", "ocr", "forum_scraper", "manual"
            
            # Rulebook-specific fields
            "page": {"type": "integer"},
            "section": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "semantic_variant": {"type": "boolean"},
            "alt_source_available": {"type": "boolean"},
            
            # Forum-specific fields
            "answer": {"type": "text"},
            "thread_id": {"type": "keyword"},
            "url": {"type": "keyword"},
            "answer_user": {"type": "keyword"},
            "qa_id": {"type": "keyword"},
            
            # FAQ-specific fields
            "faq_id": {"type": "keyword"},
            "source_file": {"type": "keyword"},

            "old_text": {"type": "text"},
            "new_text": {"type": "text"}
        }
    }
}


def create_index(es: Elasticsearch, index_name: str):
    """Create or recreate index with comprehensive mappings."""
    # Delete existing index if present
    if es.indices.exists(index=index_name):
        print(f"[Indexer] Deleting existing index: {index_name}")
        es.indices.delete(index=index_name)
    
    # Create new index
    print(f"[Indexer] Creating index: {index_name}")
    es.indices.create(index=index_name, body=MAPPING_TEMPLATE)
    print(f"[Indexer] Index '{index_name}' created with comprehensive mappings")


def detect_chunk_format(chunks: list) -> Dict[str, Any]:
    """Analyze chunks to detect their format and source types.
    
    Returns:
        Dictionary with format information:
        - has_rulebook: bool
        - has_faq: bool
        - has_forum: bool
        - has_pdf: bool
        - has_ocr: bool
        - has_quality_scores: bool
        - has_priority: bool
    """
    format_info = {
        'has_rulebook': False,
        'has_faq': False,
        'has_forum': False,
        'has_pdf': False,
        'has_ocr': False,
        'has_quality_scores': False,
        'has_priority': False,
        'total_chunks': len(chunks)
    }
    
    if not chunks:
        return format_info
    
    # Analyze sample chunks
    for chunk in chunks[:min(20, len(chunks))]:
        source_type = chunk.get('source_type', 'rulebook')
        extraction_method = chunk.get('extraction_method', 'unknown')
        
        if source_type == 'rulebook':
            format_info['has_rulebook'] = True
        elif source_type == 'faq':
            format_info['has_faq'] = True
        elif source_type == 'forum':
            format_info['has_forum'] = True
        
        if extraction_method == 'pdf':
            format_info['has_pdf'] = True
        elif extraction_method == 'ocr':
            format_info['has_ocr'] = True
        
        if 'quality_score' in chunk:
            format_info['has_quality_scores'] = True
        
        if 'priority' in chunk:
            format_info['has_priority'] = True
    
    return format_info


def load_chunks_from_pickle(pkl_path: str) -> list:
    """Load chunks from pickle file."""
    print(f"[Indexer] Loading chunks from {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"[Indexer] Loaded {len(chunks)} chunks")
    return chunks


def load_embeddings(parquet_path: str) -> pd.DataFrame:
    """Load embeddings from parquet file."""
    print(f"[Indexer] Loading embeddings from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"[Indexer] Loaded {len(df)} embeddings")
    return df


def merge_chunks_with_embeddings(chunks: list, embeddings_df: pd.DataFrame) -> list:
    """Merge chunk metadata with embeddings.
    
    Assumes chunks and embeddings are aligned by index.
    """
    print(f"[Indexer] Merging {len(chunks)} chunks with {len(embeddings_df)} embeddings...")
    
    if len(chunks) != len(embeddings_df):
        print(f"[Indexer] Warning: Chunk count ({len(chunks)}) != embedding count ({len(embeddings_df)})")
        print(f"[Indexer] Will use minimum length: {min(len(chunks), len(embeddings_df))}")
    
    merged = []
    for idx in range(min(len(chunks), len(embeddings_df))):
        chunk = chunks[idx].copy()
        
        # Get embedding from same index position
        embedding = embeddings_df.iloc[idx].get('embedding')
        
        # Convert to list if needed
        if embedding is not None and hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        elif embedding is not None and isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        chunk['embedding'] = embedding
        merged.append(chunk)
    
    return merged


def doc_generator(chunks: list, index_name: str):
    """Generate Elasticsearch documents from chunks.
    
    Automatically detects chunk type and creates appropriate documents.
    """
    skipped_count = 0
    indexed_count = 0
    
    for idx, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        embedding = chunk.get('embedding')
        
        # Validate required fields
        if not text:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"[Indexer] Skipping chunk {idx}: empty text")
            continue
        
        if embedding is None or not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIMS:
            skipped_count += 1
            if skipped_count <= 5:
                emb_type = type(embedding).__name__ if embedding is not None else 'None'
                emb_len = len(embedding) if isinstance(embedding, list) else 'N/A'
                print(f"[Indexer] Skipping chunk {idx}: invalid embedding (type={emb_type}, len={emb_len}, expected {EMBEDDING_DIMS})")
            continue
        
        # Determine document ID and type
        source_type = chunk.get('source_type', 'rulebook')
        
        # Generate document ID based on source type
        if source_type == 'forum':
            doc_id = f"forum_{chunk.get('qa_id', idx)}"
        elif source_type == 'faq':
            doc_id = f"faq_{chunk.get('faq_id', idx)}"
        else:  # rulebook
            page = chunk.get('page', 0)
            doc_id = f"rulebook_{page}_{idx}"
        
        # Build base document with common fields
        doc_source = {
            "text": text,
            "source_type": source_type,
            "doc_type": chunk.get('doc_type', ''),
            "embedding": embedding,
            "quality_score": chunk.get('quality_score', 50.0),
            "extraction_confidence": chunk.get('extraction_confidence', 0.5),
            "extraction_method": chunk.get('extraction_method', 'unknown'),
            "priority": chunk.get('priority', 50)  # Default to rulebook priority
        }
        
        # Add rulebook-specific fields
        if source_type == 'rulebook':
            doc_source.update({
                "page": chunk.get('page'),
                "section": chunk.get('section', ''),
                "chunk_index": chunk.get('chunk_index', idx),
                "semantic_variant": chunk.get('semantic_variant', False),
                "alt_source_available": chunk.get('alt_source_available', False)
            })
        
        # Add forum-specific fields
        elif source_type == 'forum':
            doc_source.update({
                "answer": chunk.get('answer', ''),
                "thread_id": chunk.get('thread_id', ''),
                "url": chunk.get('url', ''),
                "answer_user": chunk.get('answer_user', ''),
                "qa_id": chunk.get('qa_id', '')
            })
        
        # Add FAQ-specific fields
        elif source_type == 'faq':
            doc_source.update({
                "faq_id": chunk.get('faq_id', f"faq_{idx}"),
                "source_file": chunk.get('source_file', ''),
                "answer": chunk.get('answer', ''),  # FAQ answer field
                "section": chunk.get('section', '')  # FAQ category/section
            })
        
        yield {
            "_index": index_name,
            "_id": doc_id,
            "_source": doc_source
        }
        
        indexed_count += 1
    
    if skipped_count > 0:
        print(f"[Indexer] Skipped {skipped_count} chunks (missing required fields)")
    print(f"[Indexer] Generated {indexed_count} documents for indexing")


def print_index_statistics(es: Elasticsearch, index_name: str):
    """Query and display index statistics."""
    print(f"\n[Indexer] Verifying index statistics...")
    
    # Refresh index first
    es.indices.refresh(index=index_name)
    
    # Total documents
    total_docs = es.count(index=index_name)['count']
    
    if total_docs == 0:
        print("\n[Indexer] Warning: No documents were indexed!")
        return
    
    # Count by source type
    rulebook_docs = es.count(
        index=index_name,
        body={"query": {"term": {"source_type": "rulebook"}}}
    )['count']
    
    forum_docs = es.count(
        index=index_name,
        body={"query": {"term": {"source_type": "forum"}}}
    )['count']
    
    faq_docs = es.count(
        index=index_name,
        body={"query": {"term": {"source_type": "faq"}}}
    )['count']
    
    # Quality statistics for rulebook chunks
    high_quality_docs = 0
    high_confidence_docs = 0
    
    if rulebook_docs > 0:
        high_quality_docs = es.count(
            index=index_name,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"source_type": "rulebook"}},
                            {"range": {"quality_score": {"gte": 80}}}
                        ]
                    }
                }
            }
        )['count']
        
        high_confidence_docs = es.count(
            index=index_name,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"source_type": "rulebook"}},
                            {"range": {"extraction_confidence": {"gte": 0.8}}}
                        ]
                    }
                }
            }
        )['count']
    
    # Display statistics
    print(f"\n{'='*70}")
    print("INDEX STATISTICS")
    print(f"{'='*70}")
    print(f"Index name: {index_name}")
    print(f"Total documents: {total_docs}")
    
    if faq_docs > 0:
        print(f"\nFAQ Q&A pairs: {faq_docs} ({faq_docs/total_docs*100:.1f}%) [Priority: 75]")
    
    if rulebook_docs > 0:
        print(f"\nRulebook chunks: {rulebook_docs} ({rulebook_docs/total_docs*100:.1f}%) [Priority: 50]")
        print(f"  High quality (≥80): {high_quality_docs} ({high_quality_docs/rulebook_docs*100:.1f}%)")
        print(f"  High confidence (≥0.8): {high_confidence_docs} ({high_confidence_docs/rulebook_docs*100:.1f}%)")
    
    if forum_docs > 0:
        print(f"\nForum Q&A pairs: {forum_docs} ({forum_docs/total_docs*100:.1f}%) [Priority: 30]")
    
    print(f"{'='*70}\n")


def main():
    """Main indexing workflow."""
    parser = argparse.ArgumentParser(
        description='Universal Elasticsearch indexer - index chunks from any source',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index embeddings (auto-detects format)
  python index_chunks.py data/processed/chunks_embeddings.parquet
  
  # Index with custom index name
  python index_chunks.py data/processed/chunks_embeddings.parquet --index my_index
  
  # Load chunks from separate pickle file
  python index_chunks.py data/processed/chunks_embeddings.parquet --chunks data/processed/chunks_unified.pkl
  
  # Specify Elasticsearch host
  python index_chunks.py data/processed/chunks_embeddings.parquet --es-host http://my-es-server:9200
        """
    )
    
    parser.add_argument('embeddings', type=str, help='Path to embeddings parquet file')
    parser.add_argument('--chunks', type=str, help='Optional: path to chunks pickle file (if separate from embeddings)')
    parser.add_argument('--index', type=str, default='rulebook_chunks', help='Elasticsearch index name (default: rulebook_chunks)')
    parser.add_argument('--es-host', type=str, default='http://localhost:9200', help='Elasticsearch host URL (default: http://localhost:9200)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("UNIVERSAL ELASTICSEARCH INDEXER")
    print("="*70 + "\n")
    
    # Validate files exist
    if not os.path.exists(args.embeddings):
        print(f"[Indexer] Error: Embeddings file not found: {args.embeddings}")
        sys.exit(1)
    
    # Connect to Elasticsearch
    print(f"[Indexer] Connecting to Elasticsearch at {args.es_host}...")
    es = Elasticsearch(args.es_host)
    
    if not es.ping():
        print(f"[Indexer] Error: Cannot connect to Elasticsearch at {args.es_host}")
        print(f"[Indexer] Please make sure Elasticsearch is running")
        sys.exit(1)
    
    print(f"[Indexer] Connected successfully\n")
    
    # Load embeddings
    embeddings_df = load_embeddings(args.embeddings)
    
    # Load chunks if separate file provided
    if args.chunks:
        if not os.path.exists(args.chunks):
            print(f"[Indexer] Error: Chunks file not found: {args.chunks}")
            sys.exit(1)
        
        chunks_list = load_chunks_from_pickle(args.chunks)
        merged_chunks = merge_chunks_with_embeddings(chunks_list, embeddings_df)
    else:
        # Embeddings file should contain all necessary data
        # Convert DataFrame to list of dicts
        merged_chunks = embeddings_df.to_dict('records')
        
        # Ensure embeddings are in list format
        for chunk in merged_chunks:
            if 'embedding' in chunk:
                emb = chunk['embedding']
                if hasattr(emb, 'tolist'):
                    chunk['embedding'] = emb.tolist()
                elif isinstance(emb, np.ndarray):
                    chunk['embedding'] = emb.tolist()
    
    # Detect chunk format
    format_info = detect_chunk_format(merged_chunks)
    
    print(f"\n[Indexer] Detected chunk format:")
    print(f"  Total chunks: {format_info['total_chunks']}")
    if format_info['has_rulebook']:
        print(f"  Contains rulebook chunks: Yes")
        if format_info['has_pdf']:
            print(f"    - PDF extraction: Yes")
        if format_info['has_ocr']:
            print(f"    - OCR extraction: Yes")
    if format_info['has_forum']:
        print(f"  Contains forum Q&A pairs: Yes")
    if format_info['has_quality_scores']:
        print(f"  Has quality scores: Yes")
    
    # Create index
    create_index(es, args.index)
    
    # Index documents
    print(f"\n[Indexer] Indexing documents...")
    start_time = time.time()
    
    success_count, errors = helpers.bulk(
        es,
        doc_generator(merged_chunks, args.index),
        raise_on_error=False
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n[Indexer] Indexing complete:")
    print(f"  Successfully indexed: {success_count} documents")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"  Indexing speed: {success_count/elapsed_time:.1f} docs/sec")
    
    if errors:
        print(f"  Errors: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"    {error}")
    
    # Display statistics
    print_index_statistics(es, args.index)
    
    print("✓ Indexing complete!")
    print(f"\nThe index '{args.index}' is ready for hybrid search!")


if __name__ == "__main__":
    main()
