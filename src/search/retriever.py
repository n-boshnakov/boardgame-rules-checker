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
    def __init__(self, es_host: str = "http://localhost:9200", model_name: str = MODEL_NAME, use_reranker: bool = True, use_semantic_analysis: bool = False):
        self.es = Elasticsearch(es_host, headers={"accept": "application/json", "content-type": "application/json"})
        print(f"[Retriever] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.use_reranker = use_reranker
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_reranker else None
        
        # Initialize semantic analyzer for advanced question understanding (default: False - use hybrid search only)
        self.use_semantic_analysis = use_semantic_analysis
        self.semantic_analyzer = None
        if use_semantic_analysis:
            import sys
            import os
            from pathlib import Path
            
            # Ensure src is in path
            src_path = Path(__file__).parent.parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            # Load NLTK-based semantic analyzer (Python 3.14+ compatible)
            try:
                import importlib
                module = importlib.import_module('search.semantic_analyzer_nltk')
                self.semantic_analyzer = module.SemanticAnalyzerNLTK()
                print("[Retriever] Semantic analysis enabled (NLTK)")
            except Exception as e:
                print(f"[Retriever] Failed to load semantic analyzer: {e}")
                print("[Retriever] Continuing without semantic analysis")
            
            if not self.semantic_analyzer:
                print(f"[Retriever] WARNING: All semantic analyzers failed to load, disabling semantic analysis")
                self.use_semantic_analysis = False
    
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

    def search(self, query: str, section: Optional[str] = None, top_k: int = 25, search_type: str = "hybrid", hybrid_weight: float = 0.85, use_semantic: bool = None) -> List[Dict]:
        # Apply minimal semantic enhancement (max 1 term to avoid query drift)
        # Allow per-request override of semantic analysis setting
        should_use_semantic = use_semantic if use_semantic is not None else self.use_semantic_analysis
        
        enhanced_query = query
        if should_use_semantic and self.semantic_analyzer:
            try:
                enhanced_query = self.semantic_analyzer.enhance_query(query, max_additions=1)
            except Exception:
                pass
        
        query_vec = self.embed_query(enhanced_query)

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

    def generate_answer(self, question: str, chunks: List[Dict], use_semantic: bool = None) -> str:
        """Generate answer by concatenating top relevant chunks (extractive method).
        
        Args:
            question: The question to answer
            chunks: Retrieved chunks
            use_semantic: Override for semantic sentence selection (None = auto-detect based on init)
        """
        if not chunks:
            return "No relevant information found."
        
        # Determine if we should use semantic selection
        should_use_semantic = use_semantic if use_semantic is not None else (self.use_semantic_analysis and self.semantic_analyzer)
        
        if should_use_semantic:
            try:
                return self._generate_answer_semantic(question, chunks)
            except Exception:
                # Fall back to extractive on error
                return self._generate_answer_extractive(question, chunks)
        return self._generate_answer_extractive(question, chunks)
    
    def _generate_answer_semantic(self, question: str, chunks: List[Dict]) -> str:
        """Generate answer using semantic analysis to understand question intent.
        
        Combines semantic understanding with sentence-level relevance scoring.
        """
        if not self.semantic_analyzer:
            return self._generate_answer_sentence_level(question, chunks)
        
        # Analyze the question
        analysis = self.semantic_analyzer.analyze(question)
        intent = self.semantic_analyzer.get_question_intent(question)
        
        # Different strategies based on question type
        if intent.get('needs_procedural'):
            # For "how to" questions, look for step-by-step content
            return self._generate_procedural_answer(question, chunks, analysis)
        elif intent.get('needs_definition'):
            # For "what is" questions, look for definitional content
            return self._generate_definitional_answer(question, chunks, analysis)
        elif intent.get('needs_quantitative'):
            # For quantity questions, look for numbers
            return self._generate_quantitative_answer(question, chunks, analysis)
        else:
            # Default to semantic sentence selection
            return self._generate_answer_sentence_level(question, chunks)
    
    def _generate_procedural_answer(self, question: str, chunks: List[Dict], analysis: Dict) -> str:
        """Generate answer for procedural "how to" questions."""
        # Extract all sentences from top chunks
        sentence_candidates = []
        
        # Key terms to look for in procedural answers
        procedural_markers = ['first', 'then', 'next', 'after', 'must', 'should', 'to', 'step']
        action_verbs = set(analysis['action_verbs'])
        
        question_embedding = self.model.encode(question, convert_to_tensor=True, normalize_embeddings=True)
        
        for chunk_idx, chunk in enumerate(chunks[:8]):
            text = chunk.get("text", "")
            if not text:
                continue
            
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sent in sentences:
                sent = sent.strip()
                if len(sent.split()) < 4:
                    continue
                
                # Calculate base similarity
                sent_embedding = self.model.encode(sent, convert_to_tensor=True, normalize_embeddings=True)
                similarity = util.cos_sim(question_embedding, sent_embedding)[0][0].item()
                
                # Boost for procedural markers
                proc_boost = sum(0.05 for marker in procedural_markers if marker in sent.lower())
                
                # Boost for action verbs from question
                action_boost = sum(0.08 for verb in action_verbs if verb in sent.lower())
                
                # Position boost
                position_boost = 1.0 / (chunk_idx + 1) * 0.2
                
                final_score = similarity + proc_boost + action_boost + position_boost
                
                sentence_candidates.append({
                    'sentence': sent,
                    'score': final_score,
                    'chunk_idx': chunk_idx
                })
        
        # Sort and build answer
        sentence_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        answer_parts = []
        current_length = 0
        max_length = 700
        seen = set()
        
        for candidate in sentence_candidates[:10]:
            sent = candidate['sentence']
            sent_norm = re.sub(r'\s+', ' ', sent.lower()).strip()
            
            if sent_norm in seen:
                continue
            
            if current_length + len(sent) <= max_length:
                answer_parts.append(sent)
                current_length += len(sent) + 1
                seen.add(sent_norm)
                
                if len(answer_parts) >= 5:
                    break
        
        return " ".join(answer_parts) if answer_parts else chunks[0].get("text", "No relevant information found.")[:700]
    
    def _generate_definitional_answer(self, question: str, chunks: List[Dict], analysis: Dict) -> str:
        """Generate answer for definitional "what is" questions."""
        # Look for sentences that define or describe the concept
        sentence_candidates = []
        
        # The focus is typically what needs to be defined
        focus_terms = set(analysis['focus'].lower().split()) if analysis.get('focus') else set()
        game_concepts = set(analysis['game_concepts'])
        
        definitional_markers = ['is', 'are', 'means', 'refers to', 'represents', 'called', 'known as']
        
        question_embedding = self.model.encode(question, convert_to_tensor=True, normalize_embeddings=True)
        
        for chunk_idx, chunk in enumerate(chunks[:8]):
            text = chunk.get("text", "")
            if not text:
                continue
            
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sent in sentences:
                sent = sent.strip()
                if len(sent.split()) < 4:
                    continue
                
                sent_lower = sent.lower()
                
                # Calculate base similarity
                sent_embedding = self.model.encode(sent, convert_to_tensor=True, normalize_embeddings=True)
                similarity = util.cos_sim(question_embedding, sent_embedding)[0][0].item()
                
                # Boost for definitional markers
                def_boost = sum(0.06 for marker in definitional_markers if marker in sent_lower)
                
                # Boost for focus terms
                focus_boost = sum(0.08 for term in focus_terms if term in sent_lower)
                
                # Boost for game concepts
                concept_boost = sum(0.07 for concept in game_concepts if concept in sent_lower)
                
                # Position boost
                position_boost = 1.0 / (chunk_idx + 1) * 0.2
                
                final_score = similarity + def_boost + focus_boost + concept_boost + position_boost
                
                sentence_candidates.append({
                    'sentence': sent,
                    'score': final_score,
                    'chunk_idx': chunk_idx
                })
        
        # Sort and build answer
        sentence_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        answer_parts = []
        current_length = 0
        max_length = 600
        seen = set()
        
        for candidate in sentence_candidates[:8]:
            sent = candidate['sentence']
            sent_norm = re.sub(r'\s+', ' ', sent.lower()).strip()
            
            if sent_norm in seen:
                continue
            
            if current_length + len(sent) <= max_length:
                answer_parts.append(sent)
                current_length += len(sent) + 1
                seen.add(sent_norm)
                
                if len(answer_parts) >= 4:
                    break
        
        return " ".join(answer_parts) if answer_parts else chunks[0].get("text", "No relevant information found.")[:600]
    
    def _generate_quantitative_answer(self, question: str, chunks: List[Dict], analysis: Dict) -> str:
        """Generate answer for quantitative questions (how many, how much)."""
        # Look for sentences containing numbers
        sentence_candidates = []
        
        question_embedding = self.model.encode(question, convert_to_tensor=True, normalize_embeddings=True)
        
        for chunk_idx, chunk in enumerate(chunks[:8]):
            text = chunk.get("text", "")
            if not text:
                continue
            
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sent in sentences:
                sent = sent.strip()
                if len(sent.split()) < 3:
                    continue
                
                # Calculate base similarity
                sent_embedding = self.model.encode(sent, convert_to_tensor=True, normalize_embeddings=True)
                similarity = util.cos_sim(question_embedding, sent_embedding)[0][0].item()
                
                # Boost for containing numbers
                number_boost = 0.15 if re.search(r'\d+', sent) else 0.0
                
                # Position boost
                position_boost = 1.0 / (chunk_idx + 1) * 0.2
                
                final_score = similarity + number_boost + position_boost
                
                sentence_candidates.append({
                    'sentence': sent,
                    'score': final_score,
                    'chunk_idx': chunk_idx
                })
        
        # Sort and build answer
        sentence_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        answer_parts = []
        current_length = 0
        max_length = 500
        seen = set()
        
        for candidate in sentence_candidates[:6]:
            sent = candidate['sentence']
            sent_norm = re.sub(r'\s+', ' ', sent.lower()).strip()
            
            if sent_norm in seen:
                continue
            
            if current_length + len(sent) <= max_length:
                answer_parts.append(sent)
                current_length += len(sent) + 1
                seen.add(sent_norm)
                
                if len(answer_parts) >= 3:
                    break
        
        return " ".join(answer_parts) if answer_parts else chunks[0].get("text", "No relevant information found.")[:500]
    
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


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Search rulebook and answer questions")
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument("--use_semantic_analysis", action="store_true", 
                        help="Enable semantic query analysis (NLTK-based)")
    args = parser.parse_args()
    
    question = " ".join(args.question)
    use_semantic = args.use_semantic_analysis
    
    # Initialize retriever
    retriever = RulebookRetriever(use_reranker=True, use_semantic_analysis=use_semantic)
    
    # Search for relevant chunks
    print(f"\n{'='*70}")
    print(f"Question: {question}")
    print(f"Semantic Analysis: {'ENABLED' if use_semantic else 'DISABLED'}")
    print(f"{'='*70}\n")
    
    chunks = retriever.search(question, top_k=25, search_type="hybrid", hybrid_weight=0.85, use_semantic=use_semantic)
    
    if not chunks:
        print("No relevant chunks found.")
        sys.exit(0)
    
    # Generate answer
    answer = retriever.generate_answer(question, chunks, use_semantic=use_semantic)
    
    print(f"Answer:\n{answer}")
    print(f"\n{'='*70}\n")
