from sentence_transformers import SentenceTransformer, CrossEncoder, util
from elasticsearch import Elasticsearch
from elasticsearch import BadRequestError
from typing import List, Dict, Optional
import numpy as np
import re
import json
from pathlib import Path

ES_INDEX = "rulebook_chunks"
MODEL_NAME = "BAAI/bge-m3"  # 1024 dims, better cross-domain

class RulebookRetriever:
    def __init__(self, es_host: str = "http://localhost:9200", model_name: str = MODEL_NAME, use_reranker: bool = True):
        self.es = Elasticsearch(es_host, headers={"accept": "application/json", "content-type": "application/json"})
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
                        {"match_phrase": {"text": {"query": query, "slop": 2, "boost": 2.0}}},
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
                        {"match_phrase": {"text": {"query": query, "slop": 2, "boost": 2.0}}},
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
                for i, score in enumerate(scores):
                    combined[i]["cross_encoder_score"] = float(score)
                # Default score for the rest to preserve order
                for j in range(top_n, len(combined)):
                    combined[j]["cross_encoder_score"] = combined[j].get("score", 0.0)
                combined.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
            except Exception:
                # Graceful fallback: keep combined order
                pass

        return combined

    def generate_answer(self, question: str, chunks: List[Dict]) -> str:
        """Generate answer by concatenating top relevant chunks (extractive method)."""
        if not chunks:
            return "No relevant information found."
        
        return self._generate_answer_extractive(question, chunks)
    
    def _generate_answer_sentence_level(self, question: str, chunks: List[Dict]) -> str:
        """Extract most relevant sentences from chunks using semantic similarity."""
        # Extract all sentences from top chunks with their scores
        sentence_candidates = []
        
        # Encode question once
        question_embedding = self.model.encode(question, convert_to_tensor=True, normalize_embeddings=True)
        
        for chunk_idx, chunk in enumerate(chunks[:10]):
            text = chunk.get("text", "")
            if not text:
                continue
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                # Filter out very short or all-caps sentences
                if len(sentence.split()) < 4 or (sentence.isupper() and len(sentence) < 50):
                    continue
                
                # Encode sentence and compute similarity to question
                sent_embedding = self.model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
                similarity = util.cos_sim(question_embedding, sent_embedding)[0][0].item()
                
                # Boost score for sentences in top-ranked chunks
                position_boost = 1.0 / (chunk_idx + 1) * 0.3
                final_score = similarity + position_boost
                
                sentence_candidates.append({
                    'sentence': sentence,
                    'score': final_score,
                    'chunk_idx': chunk_idx,
                    'sent_idx': sent_idx
                })
        
        if not sentence_candidates:
            return "No relevant information found."
        
        # Sort by score and select top sentences
        sentence_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Build answer from top sentences (up to 600 chars for focused answers)
        answer_parts = []
        current_length = 0
        max_length = 600
        seen_sentences = set()
        
        for candidate in sentence_candidates:
            sentence = candidate['sentence']
            
            # Skip duplicates or near-duplicates
            sentence_normalized = re.sub(r'\s+', ' ', sentence.lower()).strip()
            if sentence_normalized in seen_sentences:
                continue
            
            # Add sentence if it fits
            if current_length + len(sentence) <= max_length:
                answer_parts.append(sentence)
                current_length += len(sentence) + 1
                seen_sentences.add(sentence_normalized)
                
                # Stop after getting 3-5 high-quality sentences
                if len(answer_parts) >= 5 or (len(answer_parts) >= 3 and current_length > 300):
                    break
        
        if not answer_parts:
            # Fallback: use top chunk if no sentences passed filters
            return chunks[0].get("text", "No relevant information found.")[:600]
        
        answer = " ".join(answer_parts)
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer
    
    def _generate_answer_extractive(self, question: str, chunks: List[Dict]) -> str:
        """Legacy extractive method - kept for backwards compatibility."""
        # Build answer from chunks up to ~800 characters
        answer_parts = []
        current_length = 0
        max_length = 800
        
        for chunk in chunks[:15]:  # Use top 15 chunks for good coverage
            text = chunk.get("text", "")
            if not text:
                continue
            
            if current_length + len(text) > max_length:
                # Add partial chunk text up to the limit
                remaining = max_length - current_length
                if remaining > 50:  # Only add if meaningful amount remains
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    for sent in sentences:
                        if current_length + len(sent) <= max_length:
                            answer_parts.append(sent)
                            current_length += len(sent) + 1
                        else:
                            break
                break
            else:
                answer_parts.append(text)
                current_length += len(text) + 1
        
        answer = " ".join(answer_parts) if answer_parts else "No relevant information found."
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer
