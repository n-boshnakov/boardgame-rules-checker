"""Index text chunks with embeddings into Elasticsearch.

This module creates and populates an Elasticsearch index with text chunks
and their embeddings for hybrid search (semantic + keyword).
"""
import pandas as pd
from elasticsearch import Elasticsearch, helpers
import numpy as np
import os
import sys

# Index configuration
ES_INDEX = "rulebook_chunks"
EMBEDDING_DIMS = 1024  # BAAI/bge-m3 model dimension

# Elasticsearch mapping schema
MAPPING = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "page": {"type": "integer"},
            "section": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "index": True,
                "similarity": "cosine"  # Enable cosine similarity for semantic search
            }
        }
    }
}


def create_index(es, index_name="rulebook_chunks", mapping=None):
    """Create or recreate Elasticsearch index with proper mappings.
    
    Args:
        es: Elasticsearch client instance
        index_name: Name of the index to create
        mapping: Index mapping schema (uses MAPPING constant if None)
    """
    if mapping is None:
        mapping = MAPPING

    # Delete existing index if present
    if es.indices.exists(index=index_name):
        print(f"[Indexer] Deleting existing index: {index_name}")
        es.indices.delete(index=index_name)

    # Create new index with mappings
    print(f"[Indexer] Creating index: {index_name}")
    es.indices.create(index=index_name, body=mapping)


def load_embeddings(parquet_path: str) -> pd.DataFrame:
    """Load embeddings from parquet file and ensure proper format.
    
    Args:
        parquet_path: Path to parquet file containing embeddings
        
    Returns:
        DataFrame with properly formatted embeddings
    """
    df = pd.read_parquet(parquet_path)
    
    # Convert embeddings to list format if needed
    if 'embedding' in df.columns:
        def to_list(embedding):
            """Convert various embedding formats to list."""
            # NumPy array
            if isinstance(embedding, np.ndarray):
                return embedding.tolist()
            
            # Pandas series or similar
            if hasattr(embedding, "to_numpy"):
                return embedding.to_numpy().tolist()
            
            # Already list or tuple
            if isinstance(embedding, (list, tuple)):
                return list(embedding)
            
            # String representation (try to parse)
            if isinstance(embedding, str):
                import ast
                try:
                    parsed = ast.literal_eval(embedding)
                    if isinstance(parsed, (list, tuple, np.ndarray)):
                        return list(parsed)
                except Exception:
                    pass
            
            return None
        
        df['embedding'] = df['embedding'].apply(to_list)
    
    return df


def doc_generator(df: pd.DataFrame):
    """Generate Elasticsearch documents from DataFrame.
    
    Args:
        df: DataFrame with text chunks and embeddings
        
    Yields:
        Document dictionaries for bulk indexing
    """
    skipped_count = 0
    
    for idx, row in df.iterrows():
        embedding = row.get("embedding", None)
        
        # Validate embedding
        if embedding is None or not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIMS:
            skipped_count += 1
            
            # Show details for first few skipped rows
            if skipped_count <= 5:
                emb_len = len(embedding) if isinstance(embedding, list) else 'N/A'
                print(f"[Indexer] Skipping row {idx}: invalid embedding (len={emb_len}, expected {EMBEDDING_DIMS})")
            
            continue
        
        # Handle missing values
        section = row.get("section", "")
        section = "" if pd.isna(section) else section
        
        doc_type = row.get("doc_type", "")
        doc_type = "" if pd.isna(doc_type) else doc_type

        # Generate document
        yield {
            "_index": ES_INDEX,
            "_id": f"{int(row['page'])}_{idx}",
            "_source": {
                "text": row["text"],
                "page": int(row["page"]),
                "section": section,
                "doc_type": doc_type,
                "embedding": embedding
            }
        }
    
    # Summary of skipped rows
    if skipped_count > 0:
        print(f"[Indexer] Skipped {skipped_count} rows with invalid embeddings")


def main(parquet_path: str, es_host: str = "http://localhost:9200"):
    """Main indexing workflow.
    
    Args:
        parquet_path: Path to parquet file with embeddings
        es_host: Elasticsearch host URL
    """
    # Initialize Elasticsearch client
    es = Elasticsearch(es_host)
    
    # Create index
    create_index(es, ES_INDEX, MAPPING)
    
    # Load and index data
    df = load_embeddings(parquet_path)
    helpers.bulk(es, doc_generator(df))
    
    print(f"[Indexer] Successfully indexed {len(df)} chunks into '{ES_INDEX}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python indexer.py <embeddings_parquet> [es_host]")
        print("\nExample:")
        print("  python indexer.py data/processed/chunks_embeddings.parquet")
        print("  python indexer.py data/processed/chunks_embeddings.parquet http://localhost:9200")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    elasticsearch_host = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9200"
    
    main(parquet_file, elasticsearch_host)