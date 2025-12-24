from sentence_transformers import SentenceTransformer, CrossEncoder, util
from elasticsearch import Elasticsearch
import numpy as np
import re
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
    
    def generate_answer(self, question: str, chunks: List[Dict], multi_chunk_synthesis: bool = True) -> str:
        """
        Generate answer from retrieved chunks using multi-chunk synthesis.
        Same logic as run_qa_batch.py for consistency.
        """
        if not chunks:
            return "No relevant information found."
        
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        question_keywords = set(question.lower().split())
        
        # Load section headers to filter them out
        section_headers = set()
        try:
            with open("data/processed/section_headers.txt", "r", encoding="utf-8") as f:
                section_headers = {line.strip() for line in f}
        except FileNotFoundError:
            pass
        
        all_candidates = []
        
        # Process top 10 chunks for multi-chunk synthesis to handle ES ranking variability
        # (ensures best chunks are captured even if ranking fluctuates)
        chunks_to_process = chunks[:10] if multi_chunk_synthesis else chunks[:1]
        
        for chunk_rank, chunk in enumerate(chunks_to_process):
            text = chunk['text']
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            if not sentences:
                continue
            
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]
            
            # Calculate section relevance boost
            section_boost = 0
            if chunk.get('section'):
                section_keywords = set(chunk['section'].lower().split())
                common_keywords = question_keywords.intersection(section_keywords)
                if common_keywords:
                    section_boost = 0.1 * len(common_keywords)
            
            # Give slight advantage to sentences from higher-ranked chunks
            rank_boost = 0.05 * (10 - chunk_rank) / 10 if multi_chunk_synthesis else 0
            
            for i, sentence in enumerate(sentences):
                # Skip section headers or too-short sentences
                if sentence.strip() in section_headers or len(sentence.split()) < 4:
                    continue
                
                # Hybrid score
                semantic_score = cosine_scores[i].item()
                
                # Penalties
                penalty = 0
                if len(sentence.split()) < 5:
                    penalty += 0.1
                if sentence.isupper() and len(sentence.split()) < 10:
                    penalty += 0.2
                
                final_score = semantic_score + section_boost + rank_boost - penalty
                
                all_candidates.append({
                    "score": final_score,
                    "sentence": sentence,
                    "index": i,
                    "sentences": sentences,
                    "chunk": chunk,
                    "chunk_rank": chunk_rank
                })
        
        # Sort all candidates by score
        sorted_candidates = sorted(all_candidates, key=lambda x: x['score'], reverse=True)
        
        if not sorted_candidates:
            return "No relevant information found."
        
        best_candidate = sorted_candidates[0]
        highest_score = best_candidate['score']
        
        if multi_chunk_synthesis:
            # Multi-chunk synthesis: extract relevant sentences from multiple chunks
            relevance_threshold = highest_score * 0.35  # Lower threshold for better coverage
            
            answer_sentences = []
            seen_sentences = set()
            
            for candidate in sorted_candidates:
                if candidate['score'] >= relevance_threshold:
                    sentence_text = candidate['sentence'].strip()
                    
                    # Deduplicate
                    is_duplicate = False
                    sentence_lower = sentence_text.lower()
                    for seen in seen_sentences:
                        words1 = set(sentence_lower.split())
                        words2 = set(seen.lower().split())
                        if len(words1.intersection(words2)) / len(words1.union(words2)) > 0.8:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate and len(sentence_text.split()) > 4:
                        answer_sentences.append(sentence_text)
                        seen_sentences.add(sentence_text)
                        
                        # Limit to 10 sentences for comprehensive coverage (increased from 8)
                        if len(answer_sentences) >= 10:
                            break
            
            # Ensure minimum answer length
            if len(answer_sentences) < 5:
                # Fall back to extracting from best chunk with context
                best_index = best_candidate['index']
                sentences = best_candidate['sentences']
                start_idx = max(0, best_index - 1)
                end_idx = min(len(sentences), best_index + 3)
                answer_sentences = [s.strip() for s in sentences[start_idx:end_idx] 
                                  if len(s.split()) > 3]
            
            best_answer = " ".join(answer_sentences)
        else:
            # Single-chunk extraction
            best_index = best_candidate['index']
            sentences = best_candidate['sentences']
            
            relevance_threshold = highest_score * 0.5
            chunk_candidates = [c for c in sorted_candidates 
                               if c['sentences'] is sentences 
                               and c['score'] >= relevance_threshold]
            
            relevant_indices = {c['index'] for c in chunk_candidates}
            answer_indices = {best_index}
            
            for i in range(best_index - 1, max(0, best_index - 3), -1):
                if i in relevant_indices or len(answer_indices) < 3:
                    answer_indices.add(i)
                else:
                    break
            
            for i in range(best_index + 1, min(len(sentences), best_index + 5)):
                if i in relevant_indices or len(answer_indices) < 4:
                    answer_indices.add(i)
                else:
                    break
            
            sorted_indices = sorted(answer_indices)
            context_sentences = [sentences[i] for i in sorted_indices if len(sentences[i].split()) > 3]
            best_answer = " ".join(s.strip() for s in context_sentences)
        
        # Clean up artifacts
        best_answer = re.sub(r'\s+', ' ', best_answer).strip()
        return best_answer

if __name__ == "__main__":
    import sys
    import io
    # Ensure UTF-8 output encoding on Windows
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if len(sys.argv) < 2:
        print("Usage: python retriever.py <query> [section] [es_host] [search_type] [hybrid_weight]")
        print("search_type: 'vector', 'keyword', or 'hybrid' (default)")
        print("hybrid_weight: float between 0.0 and 1.0 (default 0.7, only for hybrid)")
        exit(1)
    query = sys.argv[1]
    section = sys.argv[2] if len(sys.argv) > 2 else None
    es_host = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:9200"
    search_type = sys.argv[4] if len(sys.argv) > 4 else "hybrid"
    hybrid_weight = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
    
    retriever = RulebookRetriever(es_host, use_reranker=False)  # Disable cross-encoder reranking
    results = retriever.search(query, section, top_k=10, search_type=search_type, hybrid_weight=hybrid_weight)
    
    print("="*80)
    print(f"QUESTION: {query}")
    print("="*80)
    
    # Generate answer using multi-chunk synthesis
    answer = retriever.generate_answer(query, results, multi_chunk_synthesis=True)
    print(f"\nGENERATED ANSWER:")
    print("-"*80)
    print(answer)
    print("-"*80)
    
    print(f"\nRETRIEVED CHUNKS (top 5 of 10):")
    print("="*80)
    for i, r in enumerate(results[:5], 1):
        print(f"\n--- Chunk {i} (score={r['score']:.3f}) ---")
        print(f"Section: {r.get('section', 'N/A')}")
        print(f"Page: {r.get('page', 'N/A')}")
        print(f"Text: {r['text'][:300]}...")
        print()
