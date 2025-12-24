from sentence_transformers import SentenceTransformer, CrossEncoder
from elasticsearch import Elasticsearch
import numpy as np
from typing import List, Dict, Optional

ES_INDEX = "rulebook_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class RulebookRetriever:
    def __init__(self, es_host: str = "http://localhost:9200", model_name: str = MODEL_NAME, use_reranker: bool = True):
        # Use HTTP, no SSL, no auth, force API version 8 headers
        from elasticsearch import Elasticsearch
        self.es = Elasticsearch(es_host, headers={
            "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
            "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
        })
        self.model = SentenceTransformer(model_name)
        self.use_reranker = use_reranker
        if use_reranker:
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True)[0]

    def search(self, query: str, section: Optional[str] = None, top_k: int = 5, search_type: str = "hybrid", hybrid_weight: float = 0.7) -> List[Dict]:
        """
        search_type: 'vector', 'keyword', or 'hybrid'.
        hybrid_weight: weight for vector score (0.0-1.0), 1-hybrid_weight for keyword score. Controls the balance between vector and keyword scores (0.0 = only keyword, 1.0 = only vector).
        """
        if search_type == "vector":
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
            es_query = {"size": top_k, "query": script_query}
            res = self.es.search(index=ES_INDEX, body=es_query)
            hits = res["hits"]["hits"]
            return [{
                "score": hit["_score"],
                **hit["_source"]
            } for hit in hits]
        elif search_type == "keyword":
            keyword_query = {
                "bool": {
                    "must": [{"match": {"text": query}}]
                }
            }
            if section:
                keyword_query["bool"]["filter"] = [{"term": {"section": section}}]
            es_query = {"size": top_k, "query": keyword_query}
            res = self.es.search(index=ES_INDEX, body=es_query)
            hits = res["hits"]["hits"]
            return [{
                "score": hit["_score"],
                **hit["_source"]
            } for hit in hits]
        elif search_type == "hybrid":
            # Run both searches, combine scores
            query_vec = self.embed_query(query)
            # Vector search (get more results for merging)
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
            vector_res = self.es.search(index=ES_INDEX, body={"size": top_k*3, "query": script_query})
            vector_hits = vector_res["hits"]["hits"]
            # Keyword search (get more results for merging)
            keyword_query = {
                "bool": {
                    "must": [{"match": {"text": query}}]
                }
            }
            if section:
                keyword_query["bool"]["filter"] = [{"term": {"section": section}}]
            keyword_res = self.es.search(index=ES_INDEX, body={"size": top_k*3, "query": keyword_query})
            keyword_hits = keyword_res["hits"]["hits"]
            # Build dicts for merging
            def doc_id(hit):
                return hit.get("_id") or hit["_source"].get("id") or (hit["_source"].get("page"), hit["_source"].get("section"), hit["_source"].get("text")[:30])
            vector_dict = {doc_id(h): h for h in vector_hits}
            keyword_dict = {doc_id(h): h for h in keyword_hits}
            all_ids = set(vector_dict) | set(keyword_dict)
            # Normalize scores
            def norm_scores(hits):
                scores = [h["_score"] for h in hits]
                if not scores:
                    return {}
                min_s, max_s = min(scores), max(scores)
                if max_s == min_s:
                    return {doc_id(h): 1.0 for h in hits}
                return {doc_id(h): (h["_score"] - min_s) / (max_s - min_s) for h in hits}
            v_norm = norm_scores(vector_hits)
            k_norm = norm_scores(keyword_hits)
            # Combine
            combined = []
            for _id in all_ids:
                v_score = v_norm.get(_id, 0.0)
                k_score = k_norm.get(_id, 0.0)
                final_score = hybrid_weight * v_score + (1 - hybrid_weight) * k_score
                # Prefer vector, else keyword
                h = vector_dict.get(_id) or keyword_dict.get(_id)
                combined.append({
                    "score": final_score,
                    **h["_source"]
                })
            # Sort by combined score
            combined.sort(key=lambda x: x["score"], reverse=True)
            chunks = combined[:top_k]
        else:
            raise ValueError(f"Unknown search_type: {search_type}")
        
        # Apply cross-encoder re-ranking if enabled
        if self.use_reranker and search_type != "keyword":
            chunks = self._rerank_with_cross_encoder(query, chunks)
        
        return chunks
    
    def _rerank_with_cross_encoder(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Re-rank chunks using cross-encoder for better relevance."""
        if not chunks:
            return chunks
        
        # Create query-chunk pairs
        pairs = [[query, chunk['text']] for chunk in chunks]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Add cross-encoder scores and re-sort
        for chunk, score in zip(chunks, scores):
            chunk['cross_encoder_score'] = float(score)
            chunk['original_score'] = chunk['score']
            chunk['score'] = float(score)  # Replace with cross-encoder score
        
        # Sort by cross-encoder score
        chunks.sort(key=lambda x: x['score'], reverse=True)
        return chunks

if __name__ == "__main__":
    import sys
    import io
    # Ensure UTF-8 output encoding on Windows
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if len(sys.argv) < 2:
        print("Usage: python retriever.py <query> [section] [es_host] [search_type] [hybrid_weight]")
        print("search_type: 'vector' (default), 'keyword', or 'hybrid'")
        print("hybrid_weight: float between 0.0 and 1.0 (default 0.5, only for hybrid)")
        exit(1)
    query = sys.argv[1]
    section = sys.argv[2] if len(sys.argv) > 2 else None
    es_host = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:9200"
    search_type = sys.argv[4] if len(sys.argv) > 4 else "hybrid"
    hybrid_weight = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    retriever = RulebookRetriever(es_host)
    results = retriever.search(query, section, search_type=search_type, hybrid_weight=hybrid_weight)
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (score={r['score']:.3f}) ---\nSection: {r['section']}\nPage: {r['page']}\nText: {r['text'][:300]}\n")
