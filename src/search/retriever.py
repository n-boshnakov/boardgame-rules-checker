from sentence_transformers import SentenceTransformer, CrossEncoder, util
from elasticsearch import Elasticsearch
from elasticsearch import BadRequestError
from typing import List, Dict, Optional, Tuple
import numpy as np
import re
import json
from pathlib import Path
import sys
import os

ES_INDEX = "rulebook_chunks"
MODEL_NAME = "BAAI/bge-m3"  # 1024 dims, better cross-domain

class RulebookRetriever:
    def __init__(self, es_host: str = "http://localhost:9200", model_name: str = MODEL_NAME, use_reranker: bool = True, use_semantic_analysis: bool = False):
        # Force v8 API compatibility by setting request headers directly on each call
        # Using Elasticsearch 8.x client with API version compatibility
        self.es = Elasticsearch(es_host)
        
        # Patch the client to add v8 compatibility headers to every request
        original_perform_request = self.es.perform_request
        def patched_perform_request(method, path, **kwargs):
            # Force v8 API version on every request
            if 'headers' not in kwargs:
                kwargs['headers'] = {}
            kwargs['headers']['accept'] = 'application/vnd.elasticsearch+json; compatible-with=8'
            kwargs['headers']['content-type'] = 'application/vnd.elasticsearch+json; compatible-with=8'
            return original_perform_request(method, path, **kwargs)
        
        self.es.perform_request = patched_perform_request
        
        print(f"[Retriever] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.use_reranker = use_reranker
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_reranker else None
        
        # Initialize semantic analyzer for advanced question understanding (default: False - use hybrid search only)
        self.use_semantic_analysis = use_semantic_analysis
        self.semantic_analyzer = None
        
        # Ensure src is in path
        src_path = Path(__file__).parent.parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        if use_semantic_analysis:
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
        
        # Initialize spellchecker for question correction
        self.spellchecker = None
        self.unique_terms_path = None
        try:
            from parsers.spellcheck_utils import correct_spelling
            self.spellchecker = correct_spelling
            # Load game-specific terms to avoid "correcting" them
            project_root = Path(__file__).parent.parent.parent
            unique_terms = project_root / "data" / "processed" / "unique_terms.csv"
            if unique_terms.exists():
                self.unique_terms_path = str(unique_terms)
                print("[Retriever] Spellchecker enabled with game-specific dictionary")
            else:
                print("[Retriever] Spellchecker enabled (no custom dictionary)")
        except ImportError:
            print("[Retriever] Spellchecker not available (install pyspellchecker)")
        except Exception as e:
            print(f"[Retriever] Spellchecker initialization warning: {e}")
    
    def spellcheck_question(self, question: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Correct spelling mistakes in the question.
        
        Named entities and unique game terms are protected from correction.
        
        Args:
            question: The original question text
            
        Returns:
            Tuple of (corrected_question, list of (original, corrected) word pairs)
        """
        if not self.spellchecker:
            return question, []
        
        try:
            # Build set of protected words from unique terms + NER entities
            protected_words = set()
            
            # Load unique terms from CSV
            if self.unique_terms_path and os.path.exists(self.unique_terms_path):
                try:
                    import csv
                    with open(self.unique_terms_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if row:
                                protected_words.add(row[0].lower())
                except Exception:
                    pass
            
            # Extract entities using NER if available
            if self.semantic_analyzer:
                try:
                    entities = self.semantic_analyzer.extract_game_entities(question)
                    # Add all found entities to protected set
                    for entity_type, entity_list in entities.items():
                        for entity in entity_list:
                            protected_words.add(entity.lower())
                            # Also add individual words from multi-word entities
                            for word in entity.split():
                                protected_words.add(word.lower())
                except Exception:
                    pass
            
            # Run spellchecker
            result = self.spellchecker(
                question,
                language='en',
                generate_corrections_file=False,
                unique_terms_file=self.unique_terms_path
            )
            corrected = result.get('corrected_text', question)
            all_corrections = result.get('checked_words', [])
            
            # Filter out corrections for protected words
            filtered_corrections = []
            for orig, corr in all_corrections:
                if orig != corr and orig.lower() not in protected_words:
                    filtered_corrections.append((orig, corr))
            
            # Reconstruct corrected text, applying only non-protected corrections
            if filtered_corrections:
                import re
                corrected = question
                for orig, corr in filtered_corrections:
                    pattern = r'\b' + re.escape(orig) + r'\b'
                    corrected = re.sub(pattern, corr, corrected, flags=re.IGNORECASE)
                print(f"[Retriever] Spellcheck corrections: {filtered_corrections}")
            
            return corrected, filtered_corrections
        except Exception as e:
            print(f"[Retriever] Spellcheck error: {e}")
            return question, []
    
    def embed_query(self, query: str) -> np.ndarray:
        # Normalize to unit vectors for cosine similarity stability
        return self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    def _vector_search_script(self, query_vec: np.ndarray, size: int = 25, section: Optional[str] = None, source_type: Optional[str] = None, include_faq: bool = False, exclude_forum: bool = False):
        """Execute vector search using script_score with cosine similarity."""
        script_query = {
            "script_score": {
                "query": {"bool": {"must": []}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vec.tolist()}
                }
            }
        }
        filters = []
        if section:
            filters.append({"term": {"section": section}})
        if source_type:
            # Specific source type requested
            filters.append({"term": {"source_type": source_type}})
        else:
            # No specific source type - apply include/exclude logic
            must_not_filters = []
            if not include_faq:
                must_not_filters.append({"term": {"source_type": "faq"}})
            if exclude_forum:
                must_not_filters.append({"term": {"source_type": "forum"}})
            if must_not_filters:
                filters.append({"bool": {"must_not": must_not_filters}})
        if filters:
            script_query["script_score"]["query"]["bool"]["filter"] = filters
        return self.es.search(index=ES_INDEX, body={"size": size, "query": script_query})

    def search(self, query: str, section: Optional[str] = None, top_k: int = 25, search_type: str = "hybrid", hybrid_weight: float = 0.85, use_semantic: bool = None, source_type: Optional[str] = None, skip_entity_boosting: bool = False, include_faq: bool = False, exclude_forum: bool = False, skip_priority_boost: bool = False) -> List[Dict]:
        # Spellcheck the query first
        corrected_query, corrections = self.spellcheck_question(query)
        if corrections:
            print(f"[Retriever] Original query: {query}")
            print(f"[Retriever] Corrected query: {corrected_query}")
            query = corrected_query
        
        # Apply minimal semantic enhancement (max 1 term to avoid query drift)
        # Allow per-request override of semantic analysis setting
        should_use_semantic = use_semantic if use_semantic is not None else self.use_semantic_analysis
        
        # Extract domain keywords from question for answer filtering
        domain_keywords = []
        if should_use_semantic and self.semantic_analyzer:
            try:
                domain_keywords = self.semantic_analyzer.extract_domain_keywords(query)
            except Exception:
                pass
        
        enhanced_query = query
        if should_use_semantic and self.semantic_analyzer:
            try:
                enhanced_query = self.semantic_analyzer.enhance_query(query, max_additions=1)
            except Exception:
                pass
        
        query_vec = self.embed_query(enhanced_query)

        def parse_hits(res):
            hits = res.get("hits", {}).get("hits", [])
            parsed = []
            for h in hits:
                doc = dict(h["_source"])
                # Preserve ES relevance score, rename forum quality score if present
                if "score" in doc:
                    doc["forum_quality_score"] = doc.pop("score")
                doc["score"] = h.get("_score", 0.0)
                parsed.append(doc)
            return parsed

        if search_type == "vector":
            res = self._vector_search_script(query_vec, size=top_k, section=section, source_type=source_type, include_faq=include_faq, exclude_forum=exclude_forum)
            combined = parse_hits(res)

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
            filters = []
            if section:
                filters.append({"term": {"section": section}})
            if source_type:
                filters.append({"term": {"source_type": source_type}})
            else:
                # Apply FAQ/forum filtering if specified
                must_not_filters = []
                if not include_faq:
                    must_not_filters.append({"term": {"source_type": "faq"}})
                if exclude_forum:
                    must_not_filters.append({"term": {"source_type": "forum"}})
                if must_not_filters:
                    filters.append({"bool": {"must_not": must_not_filters}})
            if filters:
                keyword_query["bool"]["filter"] = filters
            res = self.es.search(index=ES_INDEX, body={"size": top_k, "query": keyword_query})
            return parse_hits(res)

        elif search_type == "hybrid":
            # Vector side: fetch more for merging (balanced, low overhead)
            vec_size = max(top_k * 5, 25)
            vector_res = self._vector_search_script(query_vec, size=vec_size, section=section, source_type=source_type, include_faq=include_faq, exclude_forum=exclude_forum)
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
            filters = []
            if section:
                filters.append({"term": {"section": section}})
            if source_type:
                filters.append({"term": {"source_type": source_type}})
            else:
                # Apply FAQ/forum filtering if specified
                must_not_filters = []
                if not include_faq:
                    must_not_filters.append({"term": {"source_type": "faq"}})
                if exclude_forum:
                    must_not_filters.append({"term": {"source_type": "forum"}})
                if must_not_filters:
                    filters.append({"bool": {"must_not": must_not_filters}})
            if filters:
                keyword_query["bool"]["filter"] = filters
            keyword_res = self.es.search(index=ES_INDEX, body={"size": vec_size, "query": keyword_query})
            keyword_hits = keyword_res.get("hits", {}).get("hits", [])

            # Merge with normalized scores
            def doc_id(hit):
                src = hit.get("_source", {})
                return hit.get("_id") or (src.get("page"), src.get("section"), (src.get("text") or "")[:30])

            def norm_scores(hits):
                """Normalize scores to [0, 1] range, preserving relative differences.
                For vector scores using script_score (cosineSimilarity + 1.0), we normalize from [1.0, 2.0].
                For keyword scores, we use min-max within the batch.
                """
                scores = [h.get("_score", 0.0) for h in hits]
                if not scores:
                    return {}
                mn, mx = min(scores), max(scores)
                
                # Check if scores are in [1.0, 2.0] range (vector search with script_score)
                # If so, normalize from absolute range to preserve cross-batch comparisons
                if mn >= 0.9 and mx <= 2.1:  # Allow small margin for floating point
                    # Vector scores: normalize from [1.0, 2.0] to [0, 1]
                    return {doc_id(h): max(0.0, min(1.0, (h.get("_score", 0.0) - 1.0))) for h in hits}
                else:
                    # Keyword scores: use min-max normalization
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
                
                # Create document dict and handle score collision
                doc = dict(h["_source"])
                if "score" in doc:
                    doc["forum_quality_score"] = doc.pop("score")
                doc["score"] = final_score
                combined.append(doc)

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
        
        # Filter and boost results containing domain keywords
        if domain_keywords:
            # Boost chunks that contain critical keywords
            for chunk in combined:
                text_lower = chunk.get("text", "").lower()
                keyword_matches = sum(1 for kw in domain_keywords if kw in text_lower)
                if keyword_matches > 0:
                    # Boost score based on keyword presence
                    boost_factor = 1.0 + (0.2 * keyword_matches)  # 20% boost per keyword
                    chunk["score"] = chunk.get("score", 0.0) * boost_factor
                    chunk["keyword_matches"] = keyword_matches
                else:
                    chunk["keyword_matches"] = 0
            
            # Re-sort after boosting
            combined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Apply priority-based score boosting (FAQ > Rulebook > Forum)
        # Priority values: FAQ=60, Rulebook=50, Forum=30
        # Boost multiplier: priority / 50 (base priority)
        # Skip in dual-source mode to allow fair comparison between sources
        if not skip_priority_boost:
            for chunk in combined:
                priority = chunk.get("priority", 50)
                # Apply priority boost: higher priority sources get higher scores
                priority_multiplier = priority / 50.0  # FAQ=1.5x, Rulebook=1.0x, Forum=0.6x
                chunk["score"] = chunk.get("score", 0.0) * priority_multiplier
                chunk["priority_multiplier"] = priority_multiplier
            
            # Re-sort after priority boosting
            combined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Apply entity-based score boosting for better semantic matching
        # Skip if explicitly disabled (e.g., in dual-source mode for rulebook to avoid unfair advantage)
        if should_use_semantic and self.semantic_analyzer and not skip_entity_boosting:
            try:
                self._apply_entity_boosting(query, combined)
            except Exception as e:
                print(f"[Retriever] Entity boosting failed: {e}")

        return combined
    
    def _apply_entity_boosting(self, query: str, chunks: List[Dict]) -> None:
        """
        Apply entity-based score boosting to chunks.
        Boosts chunks that share named entities with the query.
        
        Args:
            query: Original query text
            chunks: List of chunk dicts to boost (modified in-place)
        """
        # Extract entities from query
        query_entities = self.semantic_analyzer.extract_game_entities(query)
        
        # Skip if no entities found in query
        if not query_entities:
            return
        
        # Calculate total query entities for normalization
        total_query_entities = sum(len(entities) for entities in query_entities.values())
        
        for chunk in chunks:
            # Extract entities from chunk text
            chunk_entities = self.semantic_analyzer.extract_game_entities(chunk.get("text", ""))
            
            # Calculate entity overlap across all categories (excluding UNKNOWN)
            entity_overlap = 0
            matched_categories = []
            
            for entity_type, query_ents in query_entities.items():
                # Skip UNKNOWN entities as they are unreliable
                if entity_type == 'UNKNOWN':
                    continue
                    
                chunk_ents = chunk_entities.get(entity_type, [])
                if chunk_ents:
                    # Count matching entities (case-insensitive)
                    query_ents_lower = [e.lower() for e in query_ents]
                    chunk_ents_lower = [e.lower() for e in chunk_ents]
                    matches = len(set(query_ents_lower) & set(chunk_ents_lower))
                    
                    if matches > 0:
                        entity_overlap += matches
                        matched_categories.append(entity_type)
            
            # Apply boost based on entity overlap
            # Require minimum 2 entity matches to avoid over-boosting common single entities
            if entity_overlap >= 2:
                # 10% boost per matched entity, capped at 20% total boost
                boost_factor = 1.0 + min(0.20, 0.10 * entity_overlap)
                original_score = chunk.get("score", 0.0)
                chunk["score"] = original_score * boost_factor
                chunk["entity_matches"] = entity_overlap
                chunk["matched_entity_types"] = matched_categories
                
                # Debug info
                print(f"[EntityBoost] +{(boost_factor-1)*100:.0f}% for {entity_overlap} entities: {matched_categories}")
            elif entity_overlap > 0:
                # Store entity info but don't boost for single entity match
                chunk["entity_matches"] = entity_overlap
                chunk["matched_entity_types"] = matched_categories
            else:
                chunk["entity_matches"] = 0
        
        # Re-sort after entity boosting
        chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    def search_dual_source(self, query: str, top_k: int = 5, forum_weight: float = 0.45, use_semantic: bool = None, include_faq: bool = False) -> tuple:
        """Search both forum Q&A and rulebook, returning best source.
        
        Args:
            query: User question
            top_k: Number of results per source
            forum_weight: Weight for forum results (0-1), rulebook gets (1-weight)
            use_semantic: Enable semantic query expansion (None = use instance default)
            include_faq: Include FAQ results in search (default: True)
        
        Returns:
            Tuple: (chunks, source, forum_confidence, rulebook_confidence)
                chunks: List of result dicts from chosen source
                source: "forum" or "rulebook"
                forum_confidence: Float 0-1
                rulebook_confidence: Float 0-1
        """
        # Determine semantic setting
        should_use_semantic = use_semantic if use_semantic is not None else self.use_semantic_analysis
        # Spellcheck once
        corrected_query, corrections = self.spellcheck_question(query)
        if corrections:
            print(f"[DualSearch] Corrected: {query} -> {corrected_query}")
            query = corrected_query
        
        # Search forum for similar questions
        print(f"[DualSearch] Searching forum Q&A...")
        forum_results = self.search(
            query, 
            top_k=top_k, 
            search_type="vector",  # Pure semantic for question matching
            use_semantic=False,  # No query expansion for forum
            source_type="forum",  # Filter to forum only
            include_faq=False,  # Forum search doesn't include FAQ
            skip_priority_boost=True  # Skip priority boost for fair comparison
        )
        
        # Search rulebook + FAQ (if enabled) with hybrid approach
        # When include_faq=True, search both rulebook and FAQ chunks together
        # Exclude forum chunks to keep sources separate
        # Disable entity boosting for rulebook to prevent unfair advantage over forum
        print(f"[DualSearch] Searching rulebook{' + FAQ' if include_faq else ''}...")
        rulebook_results = self.search(
            query,
            top_k=top_k,
            search_type="hybrid",
            hybrid_weight=0.85,
            use_semantic=should_use_semantic,
            source_type="rulebook" if not include_faq else None,  # Allow FAQ when enabled
            skip_entity_boosting=True,  # Disable entity boosting in dual-source mode
            include_faq=include_faq,
            exclude_forum=True,  # Always exclude forum from rulebook search
            skip_priority_boost=True  # Skip priority boost for fair comparison
        )
        
        # Calculate confidence scores
        forum_confidence = forum_results[0].get('score', 0.0) if forum_results else 0.0
        rulebook_confidence = rulebook_results[0].get('score', 0.0) if rulebook_results else 0.0
        
        # Normalize scores to 0-1 range for consistent comparison
        # Vector search uses script_score which returns: cosineSimilarity() + 1.0
        # This gives scores in range [1.0, 2.0] where 1.0 = no similarity, 2.0 = perfect match
        # Hybrid search now uses improved normalization that preserves absolute scores
        
        # For debugging: print raw scores
        print(f"[DualSearch] Raw scores - Forum: {forum_confidence:.3f}, Rulebook: {rulebook_confidence:.3f}")
        
        # Normalize forum scores (pure vector search)
        if forum_results and forum_confidence > 1.0:
            # Vector search: normalize from [1.0, 2.0] to [0, 1]
            forum_confidence = max(0, min(1, (forum_confidence - 1.0)))
        
        # Normalize rulebook scores (hybrid search)
        if rulebook_results and rulebook_confidence > 1.0:
            # If using script_score format, normalize from [1.0, 2.0] to [0, 1]
            rulebook_confidence = max(0, min(1, (rulebook_confidence - 1.0)))
        # else: already in [0, 1] range from hybrid normalization
        
        # Apply source weights
        forum_weighted = forum_confidence * forum_weight
        rulebook_weighted = rulebook_confidence * (1 - forum_weight)
        
        print(f"[DualSearch] Forum confidence: {forum_confidence:.3f} (weighted: {forum_weighted:.3f})")
        print(f"[DualSearch] Rulebook confidence: {rulebook_confidence:.3f} (weighted: {rulebook_weighted:.3f})")
        
        # Decision logic: Choose source with higher weighted confidence
        # Only prefer rulebook if it's clearly better (>0.1 difference after weighting)
        # This ensures forum has fair chance when questions match well
        
        if not forum_results and not rulebook_results:
            return [], "none", 0.0, 0.0, None
        
        # Helper function to apply priority boost after source selection
        def apply_priority_boost(results):
            """Apply priority boost to results and re-sort."""
            for chunk in results:
                priority = chunk.get("priority", 50)
                priority_multiplier = priority / 50.0  # FAQ=1.5x, Rulebook=1.0x, Forum=0.6x
                chunk["score"] = chunk.get("score", 0.0) * priority_multiplier
                chunk["priority_multiplier"] = priority_multiplier
            return sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Use weighted scores for decision
        if forum_weighted > rulebook_weighted:
            # Forum has higher weighted score - use it
            print(f"[DualSearch] Selected: Forum (weighted {forum_weighted:.3f} > {rulebook_weighted:.3f})")
            forum_results = apply_priority_boost(forum_results)
            return forum_results, "forum", forum_confidence, rulebook_confidence, None
        elif rulebook_results:
            # Rulebook has higher or equal weighted score - use it
            print(f"[DualSearch] Selected: Rulebook (weighted {rulebook_weighted:.3f} >= {forum_weighted:.3f})")
            # Include top forum question for reference when rulebook is selected
            related_forum_question = forum_results[0].get('text') if forum_results else None
            rulebook_results = apply_priority_boost(rulebook_results)
            return rulebook_results, "rulebook", forum_confidence, rulebook_confidence, related_forum_question
        elif forum_results:
            # Only forum has results
            print(f"[DualSearch] Selected: Forum (only source available)")
            forum_results = apply_priority_boost(forum_results)
            return forum_results, "forum", forum_confidence, rulebook_confidence, None
        else:
            return [], "none", 0.0, 0.0, None

    def generate_answer(self, question: str, chunks: List[Dict], use_semantic: bool = None):
        """Generate answer by concatenating top relevant chunks (extractive method).
        
        Args:
            question: The question to answer
            chunks: Retrieved chunks
            use_semantic: Override for semantic sentence selection (None = auto-detect based on init)
        
        Returns:
            Tuple of (answer_text, metadata_dict) where metadata contains:
            - used_chunk_indices: List of chunk indices that contributed to the answer
            - highest_score_chunk_idx: Index of highest-scoring chunk that was used
        """
        if not chunks:
            return "No relevant information found.", {"used_chunk_indices": [], "highest_score_chunk_idx": None}
        
        # Determine if we should use semantic selection
        should_use_semantic = use_semantic if use_semantic is not None else (self.use_semantic_analysis and self.semantic_analyzer)
        
        if should_use_semantic:
            try:
                return self._generate_answer_semantic(question, chunks)
            except Exception:
                # Fall back to extractive on error
                return self._generate_answer_extractive(question, chunks)
        return self._generate_answer_extractive(question, chunks)
    
    def _generate_answer_semantic(self, question: str, chunks: List[Dict]):
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
        used_chunk_indices = []
        
        for candidate in sentence_candidates[:10]:
            sent = candidate['sentence']
            sent_norm = re.sub(r'\s+', ' ', sent.lower()).strip()
            
            if sent_norm in seen:
                continue
            
            if current_length + len(sent) <= max_length:
                answer_parts.append(sent)
                current_length += len(sent) + 1
                seen.add(sent_norm)
                used_chunk_indices.append(candidate['chunk_idx'])
                
                if len(answer_parts) >= 5:
                    break
        
        if answer_parts:
            # Find highest-scoring chunk among those used
            highest_score_idx = min(set(used_chunk_indices)) if used_chunk_indices else 0
            return " ".join(answer_parts), {"used_chunk_indices": used_chunk_indices, "highest_score_chunk_idx": highest_score_idx}
        else:
            return chunks[0].get("text", "No relevant information found.")[:700], {"used_chunk_indices": [0], "highest_score_chunk_idx": 0}
    
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
        used_chunk_indices = []
        
        for candidate in sentence_candidates[:8]:
            sent = candidate['sentence']
            sent_norm = re.sub(r'\s+', ' ', sent.lower()).strip()
            
            if sent_norm in seen:
                continue
            
            if current_length + len(sent) <= max_length:
                answer_parts.append(sent)
                current_length += len(sent) + 1
                seen.add(sent_norm)
                used_chunk_indices.append(candidate['chunk_idx'])
                
                if len(answer_parts) >= 4:
                    break
        
        if answer_parts:
            highest_score_idx = min(set(used_chunk_indices)) if used_chunk_indices else 0
            return " ".join(answer_parts), {"used_chunk_indices": used_chunk_indices, "highest_score_chunk_idx": highest_score_idx}
        else:
            return chunks[0].get("text", "No relevant information found.")[:600], {"used_chunk_indices": [0], "highest_score_chunk_idx": 0}
    
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
        used_chunk_indices = []
        
        for candidate in sentence_candidates[:6]:
            sent = candidate['sentence']
            sent_norm = re.sub(r'\s+', ' ', sent.lower()).strip()
            
            if sent_norm in seen:
                continue
            
            if current_length + len(sent) <= max_length:
                answer_parts.append(sent)
                current_length += len(sent) + 1
                seen.add(sent_norm)
                used_chunk_indices.append(candidate['chunk_idx'])
                
                if len(answer_parts) >= 3:
                    break
        
        if answer_parts:
            highest_score_idx = min(set(used_chunk_indices)) if used_chunk_indices else 0
            return " ".join(answer_parts), {"used_chunk_indices": used_chunk_indices, "highest_score_chunk_idx": highest_score_idx}
        else:
            return chunks[0].get("text", "No relevant information found.")[:500], {"used_chunk_indices": [0], "highest_score_chunk_idx": 0}
    
    def _generate_answer_sentence_level(self, question: str, chunks: List[Dict]) -> str:
        """Extract most relevant sentences from chunks using semantic similarity."""
        # Extract domain keywords from question for filtering
        domain_keywords = []
        if self.semantic_analyzer:
            try:
                domain_keywords = self.semantic_analyzer.extract_domain_keywords(question)
            except Exception:
                pass
        
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
                
                # Boost score if sentence contains domain keywords
                keyword_boost = 0.0
                sent_lower = sentence.lower()
                keyword_count = sum(1 for kw in domain_keywords if kw in sent_lower)
                if keyword_count > 0:
                    keyword_boost = 0.15 * keyword_count  # 15% boost per keyword
                
                final_score = similarity + position_boost + keyword_boost
                
                sentence_candidates.append({
                    'sentence': sentence,
                    'score': final_score,
                    'chunk_idx': chunk_idx,
                    'sent_idx': sent_idx,
                    'keyword_count': keyword_count
                })
        
        if not sentence_candidates:
            return "No relevant information found.", {"used_chunk_indices": [], "highest_score_chunk_idx": None}
        
        # Sort by score and select top sentences
        sentence_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Build answer from top sentences (up to 600 chars for focused answers)
        answer_parts = []
        current_length = 0
        max_length = 600
        seen_sentences = set()
        used_chunk_indices = []
        
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
                used_chunk_indices.append(candidate['chunk_idx'])
                
                # Stop after getting 3-5 high-quality sentences
                if len(answer_parts) >= 5 or (len(answer_parts) >= 3 and current_length > 300):
                    break
        
        if not answer_parts:
            # Fallback: use top chunk if no sentences passed filters
            return chunks[0].get("text", "No relevant information found.")[:600], {"used_chunk_indices": [0], "highest_score_chunk_idx": 0}
        
        answer = " ".join(answer_parts)
        answer = re.sub(r"\s+", " ", answer).strip()
        highest_score_idx = min(set(used_chunk_indices)) if used_chunk_indices else 0
        return answer, {"used_chunk_indices": used_chunk_indices, "highest_score_chunk_idx": highest_score_idx}
    
    def _generate_answer_extractive(self, question: str, chunks: List[Dict]):
        """Legacy extractive method - kept for backwards compatibility."""
        # Build answer from chunks up to ~800 characters
        answer_parts = []
        current_length = 0
        max_length = 800
        used_chunk_indices = []
        
        for chunk_idx, chunk in enumerate(chunks[:15]):  # Use top 15 chunks for good coverage
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
                            if chunk_idx not in used_chunk_indices:
                                used_chunk_indices.append(chunk_idx)
                        else:
                            break
                break
            else:
                answer_parts.append(text)
                current_length += len(text) + 1
                used_chunk_indices.append(chunk_idx)
        
        answer = " ".join(answer_parts) if answer_parts else "No relevant information found."
        answer = re.sub(r"\s+", " ", answer).strip()
        highest_score_idx = min(used_chunk_indices) if used_chunk_indices else 0
        return answer, {"used_chunk_indices": used_chunk_indices, "highest_score_chunk_idx": highest_score_idx}


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
    answer, metadata = retriever.generate_answer(question, chunks, use_semantic=use_semantic)
    
    print(f"Answer:\n{answer}")
    if metadata.get('highest_score_chunk_idx') is not None:
        highest_chunk = chunks[metadata['highest_score_chunk_idx']]
        print(f"\nSource: Page {highest_chunk.get('page', 'N/A')}, Section: {highest_chunk.get('section', 'N/A')}")
    print(f"\n{'='*70}\n")
