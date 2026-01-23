"""Index forum Q&A pairs into Elasticsearch.

This module loads forum Q&A data and creates embeddings for both questions
and answers, then indexes them into Elasticsearch for hybrid search.
"""
import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers
import numpy as np
import sys

# Index configuration
ES_INDEX = "rulebook_chunks"  # Unified index with rulebook data
MODEL_NAME = "BAAI/bge-m3"  # Same model as rulebook for consistency
EMBEDDING_DIMS = 1024


def load_forum_qa_pairs(json_path: str) -> list:
    """Load forum Q&A pairs from JSON file.
    
    Args:
        json_path: Path to forum_qa_pairs.json
        
    Returns:
        List of Q&A pair dictionaries
    """
    print(f"[ForumIndexer] Loading Q&A pairs from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    print(f"[ForumIndexer] Loaded {len(qa_pairs)} Q&A pairs")
    return qa_pairs


def create_embeddings(qa_pairs: list, model: SentenceTransformer) -> pd.DataFrame:
    """Create embeddings for forum Q&A pairs.
    
    For each Q&A pair, we embed the question text. The question embedding
    will be used for semantic search to find similar user questions.
    
    Args:
        qa_pairs: List of Q&A pair dictionaries
        model: SentenceTransformer model for embeddings
        
    Returns:
        DataFrame with Q&A pairs and embeddings
    """
    print(f"[ForumIndexer] Creating embeddings for {len(qa_pairs)} questions...")
    
    # Extract questions for embedding
    questions = [qa['question'] for qa in qa_pairs]
    
    # Generate embeddings (batch processing for efficiency)
    embeddings = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Unit vectors for cosine similarity
        show_progress_bar=True
    )
    
    # Create DataFrame
    df_data = []
    for qa, embedding in zip(qa_pairs, embeddings):
        df_data.append({
            'qa_id': qa['id'],  # Field is 'id' not 'qa_id'
            'thread_id': qa['thread_id'],
            'question': qa['question'],
            'answer': qa['answer'],
            'url': qa['thread_url'],  # Field is 'thread_url' not 'url'
            'score': qa.get('metadata', {}).get('useful_answer_count', 0),
            'answer_user': qa.get('raw_answers', [{}])[0].get('author', '') if qa.get('raw_answers') else '',
            'embedding': embedding.tolist()
        })
    
    df = pd.DataFrame(df_data)
    print(f"[ForumIndexer] Created embeddings with shape: {embeddings.shape}")
    return df


def doc_generator_forum(df: pd.DataFrame):
    """Generate Elasticsearch documents from forum Q&A DataFrame.
    
    Args:
        df: DataFrame with forum Q&A pairs and embeddings
        
    Yields:
        Document dictionaries for bulk indexing
    """
    skipped_count = 0
    
    for idx, row in df.iterrows():
        embedding = row.get("embedding", None)
        
        # Validate embedding
        if embedding is None or not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIMS:
            skipped_count += 1
            if skipped_count <= 5:
                emb_len = len(embedding) if isinstance(embedding, list) else 'N/A'
                print(f"[ForumIndexer] Skipping row {idx}: invalid embedding (len={emb_len}, expected {EMBEDDING_DIMS})")
            continue
        
        # Generate document with source_type field
        yield {
            "_index": ES_INDEX,
            "_id": f"forum_{row['qa_id']}",  # Prefix with 'forum_' to distinguish from rulebook
            "_source": {
                "text": row["question"],  # Question text for keyword search
                "answer": row["answer"],  # Answer text
                "source_type": "forum",  # NEW: Distinguish from rulebook
                "thread_id": row["thread_id"],
                "url": row["url"],
                "score": float(row["score"]),
                "answer_user": row["answer_user"],
                "embedding": embedding  # Question embedding for semantic search
            }
        }
    
    if skipped_count > 0:
        print(f"[ForumIndexer] Skipped {skipped_count} rows with invalid embeddings")


def update_index_mapping(es: Elasticsearch, index_name: str):
    """Update index mapping to include forum-specific fields.
    
    This adds fields needed for forum Q&A pairs while maintaining
    compatibility with existing rulebook documents.
    
    Args:
        es: Elasticsearch client
        index_name: Name of the index to update
    """
    # Check if index exists
    if not es.indices.exists(index=index_name):
        print(f"[ForumIndexer] Error: Index '{index_name}' does not exist")
        print("[ForumIndexer] Please run the rulebook indexer first to create the index")
        sys.exit(1)
    
    print(f"[ForumIndexer] Updating mapping for index: {index_name}")
    
    # Add new fields to existing mapping
    mapping_update = {
        "properties": {
            "source_type": {"type": "keyword"},  # "rulebook" or "forum"
            "answer": {"type": "text"},  # Forum answer text
            "thread_id": {"type": "keyword"},  # Forum thread ID
            "url": {"type": "keyword"},  # Forum thread URL
            "score": {"type": "float"},  # Answer quality score
            "answer_user": {"type": "keyword"}  # User who provided answer
        }
    }
    
    try:
        es.indices.put_mapping(index=index_name, body=mapping_update)
        print("[ForumIndexer] Mapping updated successfully")
    except Exception as e:
        print(f"[ForumIndexer] Mapping update error (may be safe to ignore): {e}")


def add_source_type_to_rulebook(es: Elasticsearch, index_name: str):
    """Add source_type='rulebook' to existing rulebook documents.
    
    This updates all documents that don't have a source_type field
    (i.e., existing rulebook chunks) to have source_type='rulebook'.
    
    Args:
        es: Elasticsearch client
        index_name: Name of the index
    """
    print(f"[ForumIndexer] Adding source_type to existing rulebook documents...")
    
    update_query = {
        "script": {
            "source": "ctx._source.source_type = 'rulebook'",
            "lang": "painless"
        },
        "query": {
            "bool": {
                "must_not": {
                    "exists": {
                        "field": "source_type"
                    }
                }
            }
        }
    }
    
    try:
        result = es.update_by_query(index=index_name, body=update_query, conflicts="proceed")
        updated_count = result.get('updated', 0)
        print(f"[ForumIndexer] Updated {updated_count} rulebook documents with source_type='rulebook'")
    except Exception as e:
        print(f"[ForumIndexer] Error updating rulebook documents: {e}")


def main(
    qa_json_path: str,
    es_host: str = "http://localhost:9200",
    update_rulebook: bool = True
):
    """Main forum indexing workflow.
    
    Args:
        qa_json_path: Path to forum_qa_pairs.json
        es_host: Elasticsearch host URL
        update_rulebook: Whether to add source_type to existing rulebook docs
    """
    # Initialize Elasticsearch client
    es = Elasticsearch(es_host)
    
    # Update index mapping to support forum fields
    update_index_mapping(es, ES_INDEX)
    
    # Add source_type to existing rulebook documents
    if update_rulebook:
        add_source_type_to_rulebook(es, ES_INDEX)
    
    # Load embedding model (same as rulebook for consistency)
    print(f"[ForumIndexer] Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    # Load and process forum Q&A pairs
    qa_pairs = load_forum_qa_pairs(qa_json_path)
    df = create_embeddings(qa_pairs, model)
    
    # Index forum data
    print(f"[ForumIndexer] Indexing {len(df)} forum Q&A pairs...")
    success_count, errors = helpers.bulk(es, doc_generator_forum(df), raise_on_error=False)
    
    print(f"\n[ForumIndexer] Indexing complete:")
    print(f"  - Successfully indexed: {success_count} forum Q&A pairs")
    if errors:
        print(f"  - Errors: {len(errors)}")
    
    # Verify index stats
    stats = es.count(index=ES_INDEX)
    total_docs = stats['count']
    
    forum_stats = es.count(index=ES_INDEX, body={"query": {"term": {"source_type": "forum"}}})
    forum_docs = forum_stats['count']
    
    rulebook_stats = es.count(index=ES_INDEX, body={"query": {"term": {"source_type": "rulebook"}}})
    rulebook_docs = rulebook_stats['count']
    
    print(f"\n[ForumIndexer] Index '{ES_INDEX}' statistics:")
    print(f"  - Total documents: {total_docs}")
    print(f"  - Rulebook chunks: {rulebook_docs}")
    print(f"  - Forum Q&A pairs: {forum_docs}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python forum_indexer.py <forum_qa_json> [es_host] [--skip-rulebook-update]")
        print("\nExample:")
        print("  python forum_indexer.py data/processed/forum_qa/forum_qa_pairs.json")
        print("  python forum_indexer.py data/processed/forum_qa/forum_qa_pairs.json http://localhost:9200")
        print("  python forum_indexer.py data/processed/forum_qa/forum_qa_pairs.json --skip-rulebook-update")
        print("\nOptions:")
        print("  --skip-rulebook-update: Don't add source_type to existing rulebook documents")
        sys.exit(1)
    
    qa_file = sys.argv[1]
    
    # Parse arguments
    es_host = "http://localhost:9200"
    update_rulebook = True
    
    for arg in sys.argv[2:]:
        if arg.startswith("http"):
            es_host = arg
        elif arg == "--skip-rulebook-update":
            update_rulebook = False
    
    main(qa_file, es_host, update_rulebook)
