from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import numpy as np
from typing import List, Dict, Optional

ES_INDEX = "rulebook_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"

class RulebookRetriever:
    def __init__(self, es_host: str = "http://localhost:9200", model_name: str = MODEL_NAME):
        self.es = Elasticsearch(es_host)
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True)[0]

    def search(self, query: str, section: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        query_vec = self.embed_query(query)
        script_query = {
            "script_score": {
                "query": {"bool": {"must": []}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vec.tolist()}
                }
            }
        }
        if section:
            script_query["script_score"]["query"]["bool"]["filter"] = [{"term": {"section": section}}]
        res = self.es.search(index=ES_INDEX, body={"size": top_k, "query": script_query})
        hits = res["hits"]["hits"]
        return [{
            "score": hit["_score"],
            **hit["_source"]
        } for hit in hits]

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python retriever.py <query> [section] [es_host]")
        exit(1)
    query = sys.argv[1]
    section = sys.argv[2] if len(sys.argv) > 2 else None
    es_host = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:9200"
    retriever = RulebookRetriever(es_host)
    results = retriever.search(query, section)
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (score={r['score']:.3f}) ---\nSection: {r['section']}\nPage: {r['page']}\nText: {r['text'][:300]}\n")
