"""
Unified Indexer for PDF, OCR, and Forum chunks

Indexes merged chunks from merge_and_index_all.py into Elasticsearch
with proper mappings and quality-based boosting.
"""

import pandas as pd
from elasticsearch import Elasticsearch, helpers
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Index configuration
ES_INDEX = "rulebook_chunks"
EMBEDDING_DIMS = 1024  # BAAI/bge-m3

# Enhanced mapping with quality and confidence fields
MAPPING = {
    "mappings": {
        "properties": {
            # Common fields
            "text": {"type": "text"},
            "source_type": {"type": "keyword"},  # "rulebook" or "forum"
            "doc_type": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "index": True,
                "similarity": "cosine"
            },
            
            # Quality and confidence (NEW - for boosting)
            "quality_score": {"type": "float"},  # 0-100
            "extraction_confidence": {"type": "float"},  # 0-1
            "extraction_method": {"type": "keyword"},  # "pdf", "ocr", "forum_scraper"
            
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
            "qa_id": {"type": "keyword"}
        }
    }
}


def create_index(es: Elasticsearch, index_name: str = ES_INDEX):
    """Create or recreate index with enhanced mappings."""
    # Delete existing index if present
    if es.indices.exists(index=index_name):
        print(f"[Indexer] Deleting existing index: {index_name}")
        es.indices.delete(index=index_name)
    
    # Create new index
    print(f"[Indexer] Creating index: {index_name}")
    es.indices.create(index=index_name, body=MAPPING)
    print(f"[Indexer] Index created with enhanced quality mappings")


def load_unified_chunks(pkl_path: str) -> list:
    """Load unified chunks from pickle file."""
    print(f"[Indexer] Loading unified chunks from {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"[Indexer] Loaded {len(chunks)} chunks")
    
    # Statistics
    rulebook_count = sum(1 for c in chunks if c.get('source_type') == 'rulebook')
    forum_count = sum(1 for c in chunks if c.get('source_type') == 'forum')
    
    print(f"[Indexer] Chunk breakdown:")
    print(f"  Rulebook: {rulebook_count}")
    print(f"  Forum: {forum_count}")
    
    return chunks


def load_embeddings(parquet_path: str) -> pd.DataFrame:
    """Load embeddings parquet file."""
    print(f"[Indexer] Loading embeddings from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"[Indexer] Loaded {len(df)} embeddings")
    return df


def doc_generator_unified(chunks: list, embeddings_df: pd.DataFrame):
    """Generate Elasticsearch documents from unified chunks with embeddings.
    
    This generator handles both rulebook and forum chunks, applying
    quality-based boosting during indexing.
    """
    skipped_count = 0
    indexed_count = 0
    
    # Create embedding lookup by index (they should be aligned)
    print(f"[Indexer] Matching {len(chunks)} chunks with {len(embeddings_df)} embeddings...")
    
    for idx, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        if not text:
            skipped_count += 1
            continue
        
        # Get embedding from same index position
        if idx < len(embeddings_df):
            embedding = embeddings_df.iloc[idx].get('embedding')
            # Convert to list if it's a numpy array
            if embedding is not None and hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            elif embedding is not None and isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
        else:
            embedding = None
        
        if embedding is None or not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIMS:
            if skipped_count < 5:
                emb_type = type(embedding).__name__ if embedding is not None else 'None'
                emb_len = len(embedding) if isinstance(embedding, list) else 'N/A'
                print(f"[Indexer] Skipping chunk {idx}: invalid embedding (type={emb_type}, len={emb_len}, expected {EMBEDDING_DIMS})")
            skipped_count += 1
            continue
        
        # Determine document ID based on source type
        if chunk.get('source_type') == 'forum':
            doc_id = f"forum_{chunk.get('qa_id', idx)}"
        else:
            page = chunk.get('page', 0)
            doc_id = f"rulebook_{page}_{idx}"
        
        # Build base document
        doc_source = {
            "text": text,
            "source_type": chunk.get('source_type', 'rulebook'),
            "doc_type": chunk.get('doc_type', ''),
            "embedding": embedding,
            "quality_score": chunk.get('quality_score', 50.0),
            "extraction_confidence": chunk.get('extraction_confidence', 0.5),
            "extraction_method": chunk.get('extraction_method', 'unknown')
        }
        
        # Add rulebook-specific fields
        if chunk.get('source_type') == 'rulebook':
            doc_source.update({
                "page": chunk.get('page'),
                "section": chunk.get('section', ''),
                "chunk_index": chunk.get('chunk_index', idx),
                "semantic_variant": chunk.get('semantic_variant', False),
                "alt_source_available": chunk.get('alt_source_available', False)
            })
        
        # Add forum-specific fields
        elif chunk.get('source_type') == 'forum':
            doc_source.update({
                "answer": chunk.get('answer', ''),
                "thread_id": chunk.get('thread_id', ''),
                "url": chunk.get('url', ''),
                "answer_user": chunk.get('answer_user', ''),
                "qa_id": chunk.get('qa_id', '')
            })
        
        yield {
            "_index": ES_INDEX,
            "_id": doc_id,
            "_source": doc_source
        }
        
        indexed_count += 1
    
    if skipped_count > 0:
        print(f"[Indexer] Skipped {skipped_count} chunks (no valid embeddings)")
    print(f"[Indexer] Generated {indexed_count} documents for indexing")


def main(
    unified_chunks_path: str = None,
    embeddings_path: str = None,
    es_host: str = "http://localhost:9200"
):
    """Main indexing workflow."""
    print("="*70)
    print("UNIFIED INDEXER")
    print("Indexes PDF, OCR, and Forum chunks with quality-based boosting")
    print("="*70 + "\n")
    
    # Default paths
    project_root = Path(__file__).parent.parent.parent
    if not unified_chunks_path:
        unified_chunks_path = project_root / "data" / "processed" / "chunks_unified.pkl"
    if not embeddings_path:
        embeddings_path = project_root / "data" / "processed" / "chunks_unified_embeddings.parquet"
    
    # Check if files exist
    if not os.path.exists(unified_chunks_path):
        print(f"[Indexer] Error: Unified chunks not found at {unified_chunks_path}")
        print(f"[Indexer] Please run merge_and_index_all.py first")
        sys.exit(1)
    
    if not os.path.exists(embeddings_path):
        print(f"[Indexer] Error: Embeddings not found at {embeddings_path}")
        print(f"[Indexer] Please run generate_embeddings.py first")
        sys.exit(1)
    
    # Connect to Elasticsearch
    print(f"[Indexer] Connecting to Elasticsearch at {es_host}...")
    es = Elasticsearch(es_host)
    
    if not es.ping():
        print(f"[Indexer] Error: Cannot connect to Elasticsearch at {es_host}")
        print(f"[Indexer] Please make sure Elasticsearch is running")
        sys.exit(1)
    
    print(f"[Indexer] Connected successfully\n")
    
    # Create index
    create_index(es)
    
    # Load data
    chunks = load_unified_chunks(str(unified_chunks_path))
    embeddings_df = load_embeddings(str(embeddings_path))
    
    # Index documents
    print(f"\n[Indexer] Indexing documents...")
    success_count, errors = helpers.bulk(
        es,
        doc_generator_unified(chunks, embeddings_df),
        raise_on_error=False
    )
    
    print(f"\n[Indexer] Indexing complete:")
    print(f"  Successfully indexed: {success_count} documents")
    if errors:
        print(f"  Errors: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"    {error}")
    
    # Refresh index
    es.indices.refresh(index=ES_INDEX)
    
    # Verify and display statistics
    print(f"\n[Indexer] Verifying index statistics...")
    
    total_docs = es.count(index=ES_INDEX)['count']
    rulebook_docs = es.count(
        index=ES_INDEX,
        body={"query": {"term": {"source_type": "rulebook"}}}
    )['count']
    forum_docs = es.count(
        index=ES_INDEX,
        body={"query": {"term": {"source_type": "forum"}}}
    )['count']
    
    # Quality statistics for rulebook chunks
    high_quality_docs = es.count(
        index=ES_INDEX,
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
        index=ES_INDEX,
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
    
    print(f"\n{'='*70}")
    print("INDEX STATISTICS")
    print(f"{'='*70}")
    print(f"Total documents: {total_docs}")
    if total_docs > 0:
        print(f"  Rulebook chunks: {rulebook_docs} ({rulebook_docs/total_docs*100:.1f}%)")
        print(f"  Forum Q&A pairs: {forum_docs} ({forum_docs/total_docs*100:.1f}%)")
        print(f"\nQuality metrics (rulebook only):")
        if rulebook_docs > 0:
            print(f"  High quality (≥80): {high_quality_docs} ({high_quality_docs/rulebook_docs*100:.1f}%)")
            print(f"  High confidence (≥0.8): {high_confidence_docs} ({high_confidence_docs/rulebook_docs*100:.1f}%)")
    else:
        print("  No documents indexed!")
    print(f"{'='*70}\n")
    
    print("✓ Indexing complete!")
    print("\nThe index now contains:")
    print("  - PDF and OCR chunks (intelligently merged)")
    print("  - Quality scores for ranking")
    print("  - Extraction confidence for boosting")
    print("  - Forum Q&A pairs for dual-source search")
    print("\nReady for hybrid search with quality-based ranking!")


if __name__ == "__main__":
    main()
