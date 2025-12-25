from sentence_transformers import SentenceTransformer, CrossEncoder, util
from elasticsearch import Elasticsearch
from elasticsearch import BadRequestError
from typing import List, Dict, Optional
import numpy as np
import re

ES_INDEX = "rulebook_chunks"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768 dims for quality

class RulebookRetriever:
    def __init__(self, es_host: str = "http://localhost:9200", model_name: str = MODEL_NAME, use_reranker: bool = True, use_llm: bool = False):
        self.es = Elasticsearch(es_host)
        print(f"[Retriever] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.use_reranker = use_reranker
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_reranker else None
        
        # Optional LLM for concise answer generation
        self.use_llm = use_llm
        self.llm_model = None
        self.llm_tokenizer = None
        
        if use_llm:
            print("[Retriever] Loading Phi-3-mini model for answer generation...")
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                # Use Phi-3-mini (3.8B) - no authentication required, fast and good quality
                model_id = "microsoft/Phi-3-mini-4k-instruct"
                self.llm_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    trust_remote_code=True
                )
                # Move to GPU and use FP16 if available
                if torch.cuda.is_available():
                    self.llm_model = self.llm_model.half()  # Use FP16 for speed
                print("[Retriever] LLM loaded successfully")
            except Exception as e:
                print(f"[Retriever] Warning: Could not load LLM: {e}")
                print("[Retriever] Falling back to extractive answers")
                self.use_llm = False

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
        """Generate concise answer using LLM or extractive method."""
        if not chunks:
            return "No relevant information found."
        
        if self.use_llm and self.llm_model:
            return self._generate_answer_llm(question, chunks)
        else:
            return self._generate_answer_extractive(question, chunks)
    
    def _generate_answer_llm(self, question: str, chunks: List[Dict]) -> str:
        """Generate concise 2-3 sentence answer using Phi-3-mini."""
        import torch
        
        # Collect context from top 5 chunks (limit to ~2000 chars for speed)
        context_parts = []
        total_chars = 0
        for chunk in chunks[:5]:
            text = chunk.get("text", "")
            if text and total_chars + len(text) < 2000:
                context_parts.append(text)
                total_chars += len(text)
        
        context = "\n\n".join(context_parts)
        
        # Phi-3 format
        prompt = f"""<|system|>
You are a helpful assistant that answers questions about board game rules. Provide concise, accurate answers in 2-3 sentences based only on the given rulebook context.<|end|>
<|user|>
Rulebook Context:
{context}

Question: {question}

Provide a clear, concise answer in 2-3 sentences using only information from the context above.<|end|>
<|assistant|>"""
        
        try:
            inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
            
            # Generate without caching to avoid compatibility issues
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=150,  # ~2-3 sentences
                    do_sample=False,  # Deterministic for consistency
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    use_cache=False  # Disable caching to avoid 'seen_tokens' errors
                )
            
            # Decode and extract answer
            full_output = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response (after the last <|assistant|> tag)
            if "<|assistant|>" in full_output:
                answer = full_output.split("<|assistant|>")[-1].strip()
            else:
                # Fallback: take everything after the prompt
                answer = full_output[len(prompt):].strip()
            
            # Clean up any remaining tags
            answer = answer.replace("<|end|>", "").strip()
            answer = re.sub(r'\s+', ' ', answer).strip()
            
            return answer if answer else self._generate_answer_extractive(question, chunks)
            
        except Exception as e:
            print(f"[Retriever] LLM generation failed: {e}, falling back to extractive")
            return self._generate_answer_extractive(question, chunks)
    
    def _generate_answer_extractive(self, question: str, chunks: List[Dict]) -> str:
        """Extract text from top chunks with 800-char limit for fast, concise answers."""
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
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer