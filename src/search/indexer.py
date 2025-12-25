import pandas as pd
from elasticsearch import Elasticsearch, helpers
import numpy as np
import os

ES_INDEX = "rulebook_chunks"
EMBEDDING_DIMS = 768  # all-mpnet-base-v2 outputs 768-dim vectors

MAPPING = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "page": {"type": "integer"},
            "section": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            # Enable vector search compatibility; required if using cosineSimilarity in script_score
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

def create_index(es, index_name="rulebook_chunks", mapping=None):
    """Create or recreate the Elasticsearch index with proper mappings."""
    if mapping is None:
        mapping = MAPPING

    if es.indices.exists(index=index_name):
        print(f"[Indexer] Deleting existing index: {index_name}")
        es.indices.delete(index=index_name)

    print(f"[Indexer] Creating index: {index_name}")
    es.indices.create(index=index_name, body=mapping)

def load_embeddings(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # Ensure embeddings are lists of floats
    if 'embedding' in df.columns:
        def to_list(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if hasattr(x, "to_numpy"):
                return x.to_numpy().tolist()
            if isinstance(x, (list, tuple)):
                return list(x)
            # If stored as string, try to eval safely
            if isinstance(x, str):
                import ast
                try:
                    val = ast.literal_eval(x)
                    return list(val) if isinstance(val, (list, tuple, np.ndarray)) else None
                except Exception:
                    return None
            return None
        df['embedding'] = df['embedding'].apply(to_list)
    return df

def doc_generator(df: pd.DataFrame):
    skipped = 0
    for i, row in df.iterrows():
        emb = row.get("embedding", None)
        if emb is None or not isinstance(emb, list) or len(emb) != EMBEDDING_DIMS:
            skipped += 1
            # Helpful message for first few skips
            if skipped <= 5:
                print(f"[Indexer] Skipping row {i}: invalid embedding (len={len(emb) if isinstance(emb, list) else 'N/A'})")
            continue
        section = row.get("section", "")
        if pd.isna(section):
            section = ""
        doc_type = row.get("doc_type", "")
        if pd.isna(doc_type):
            doc_type = ""

        yield {
            "_index": ES_INDEX,
            "_id": f"{int(row['page'])}_{i}",
            "_source": {
                "text": row["text"],
                "page": int(row["page"]),
                "section": section,
                "doc_type": doc_type,
                "embedding": emb
            }
        }
    if skipped:
        print(f"[Indexer] Skipped {skipped} rows due to invalid/mismatched embeddings")

def main(parquet_path: str, es_host: str = "http://localhost:9200"):
    es = Elasticsearch(es_host)
    create_index(es, ES_INDEX, MAPPING)
    df = load_embeddings(parquet_path)
    helpers.bulk(es, doc_generator(df))
    print(f"Indexed {len(df)} chunks into '{ES_INDEX}'")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python indexer.py <embeddings_parquet> [es_host]")
        exit(1)
    parquet_path = sys.argv[1]
    es_host = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9200"
    main(parquet_path, es_host)