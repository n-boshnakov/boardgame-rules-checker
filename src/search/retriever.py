from sentence_transformers import SentenceTransformer, CrossEncoder, util
from elasticsearch import Elasticsearch
from elasticsearch import BadRequestError
from typing import List, Dict, Optional
import numpy as np
import re

ES_INDEX = "rulebook_chunks"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768 dims for quality

class RulebookRetriever:
    def __init__(self, es_host: str = "http://localhost:9200", model_name: str = MODEL_NAME, use_reranker: bool = True):
        self.es = Elasticsearch(es_host)
        print(f"[Retriever] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.use_reranker = use_reranker
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_reranker else None

    def embed_query(self, query: str) -> np.ndarray:
        # Normalize to unit vectors for cosine similarity stability
        return self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    def _vector_search_script(self, query_vec: np.ndarray, size: int, section: Optional[str] = None):
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
        return self.es.search(index=ES_INDEX, body={"size": size, "query": script_query})

    def search(self, query: str, section: Optional[str] = None, top_k: int = 25, search_type: str = "hybrid", hybrid_weight: float = 0.85) -> List[Dict]:
        query_vec = self.embed_query(query)

        def parse_hits(res):
            hits = res.get("hits", {}).get("hits", [])
            return [{"score": h.get("_score", 0.0), **h["_source"]} for h in hits]

        if search_type == "vector":
            res = self._vector_search_script(query_vec, size=top_k, section=section)
            return parse_hits(res)

        elif search_type == "keyword":
            # Prefer precise phrasing while retaining recall
            keyword_query = {
                "bool": {
                    "must": [
                        {"match": {"text": {"query": query, "operator": "and"}}}
                    ],
                    "should": [
                        {"match_phrase": {"text": {"query": query, "slop": 1, "boost": 2.0}}},
                        {"match": {"text": {"query": query, "boost": 0.8}}}
                    ],
                    "minimum_should_match": 0
                }
            }
            if section:
                keyword_query["bool"]["filter"] = [{"term": {"section": section}}]
            res = self.es.search(index=ES_INDEX, body={"size": top_k, "query": keyword_query})
            return parse_hits(res)

        elif search_type == "hybrid":
            # Vector side: fetch more for merging (balanced, low overhead)
            vec_size = max(top_k * 5, 25)
            vector_res = self._vector_search_script(query_vec, size=vec_size, section=section)
            vector_hits = vector_res.get("hits", {}).get("hits", [])

            # Keyword side: prefer phrasing + and-operator; same pool size
            keyword_query = {
                "bool": {
                    "must": [
                        {"match": {"text": {"query": query, "operator": "and"}}}
                    ],
                    "should": [
                        {"match_phrase": {"text": {"query": query, "slop": 1, "boost": 2.0}}},
                        {"match": {"text": {"query": query, "boost": 0.8}}}
                    ],
                    "minimum_should_match": 0
                }
            }
            if section:
                keyword_query["bool"]["filter"] = [{"term": {"section": section}}]
            keyword_res = self.es.search(index=ES_INDEX, body={"size": vec_size, "query": keyword_query})
            keyword_hits = keyword_res.get("hits", {}).get("hits", [])

            # Merge with normalized scores
            def doc_id(hit):
                src = hit.get("_source", {})
                return hit.get("_id") or (src.get("page"), src.get("section"), (src.get("text") or "")[:30])

            def norm_scores(hits):
                scores = [h.get("_score", 0.0) for h in hits]
                if not scores:
                    return {}
                mn, mx = min(scores), max(scores)
                if mx == mn:
                    return {doc_id(h): 1.0 for h in hits}
                return {doc_id(h): (h.get("_score", 0.0) - mn) / (mx - mn) for h in hits}

            v_norm = norm_scores(vector_hits)
            k_norm = norm_scores(keyword_hits)
            v_dict = {doc_id(h): h for h in vector_hits}
            k_dict = {doc_id(h): h for h in keyword_hits}
            all_ids = set(v_dict) | set(k_dict)

            combined = []
            for _id in all_ids:
                v_score = v_norm.get(_id, 0.0)
                k_score = k_norm.get(_id, 0.0)
                final_score = hybrid_weight * v_score + (1 - hybrid_weight) * k_score
                h = v_dict.get(_id) or k_dict.get(_id)
                combined.append({"score": final_score, **h["_source"]})

            # Sort by score descending and take top_k
            combined = sorted(combined, key=lambda x: x["score"], reverse=True)[:top_k]

        else:
            raise ValueError(f"Unknown search_type: {search_type}")

        # Optional cross-encoder rerank for vector/hybrid
        if self.use_reranker and search_type != "keyword":
            # Rerank top 10 for better precision
            top_n = min(10, len(combined))
            pairs = [[query, c["text"]] for c in combined[:top_n]]
            try:
                scores = self.cross_encoder.predict(pairs)
                for i in range(top_n):
                    combined[i]["cross_encoder_score"] = float(scores[i])
                # Default score for the rest to preserve order
                for j in range(top_n, len(combined)):
                    combined[j]["cross_encoder_score"] = combined[j].get("score", 0.0)
                combined.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
            except Exception:
                # Graceful fallback: keep combined order
                pass

        return combined

    def generate_answer(self, question: str, chunks: List[Dict], multi_chunk_synthesis: bool = False) -> str:
        """Return the text from the top-ranked chunk(s) - simpler and more accurate."""
        if not chunks:
            return "No relevant information found."

        # Combine text from top 15 chunks for maximum coverage
        texts = []
        for chunk in chunks[:15]:
            text = chunk.get("text", "")
            if text and text not in texts:  # Avoid duplicates
                texts.append(text)
        
        answer = " ".join(texts) if texts else "No relevant information found."
        
        # Clean whitespace
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer