import pandas as pd
from elasticsearch import Elasticsearch, helpers
import numpy as np
import os

ES_INDEX = "rulebook_chunks"

MAPPING = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "page": {"type": "integer"},
            "section": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims": 384}  # 384 for all-MiniLM-L6-v2
        }
    }
}

def create_index(es: Elasticsearch, index_name: str, mapping: dict):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=mapping)

def load_embeddings(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)

def doc_generator(df: pd.DataFrame):
    for i, row in df.iterrows():
        yield {
            "_index": ES_INDEX,
            "_id": f"{row['page']}_{i}",
            "_source": {
                "text": row["text"],
                "page": int(row["page"]),
                "section": row["section"] or "",
                "doc_type": row["doc_type"],
                "embedding": np.array(row["embedding"]).tolist()
            }
        }

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
