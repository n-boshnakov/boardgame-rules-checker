"""
Flask Web UI for Boardgame Rules Checker
Provides a simple interface to ask questions about board game rules.
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from search.retriever import RulebookRetriever
from qa.multi_dimensional_scorer import MultiDimensionalScorer

app = Flask(__name__)

# Disable caching for all responses
@app.after_request
def add_no_cache_headers(response):
    """Add headers to prevent browser caching of responses."""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Initialize retriever and scorer (reuse across requests)
print("[UI] Initializing retriever and scorer...")
print("[UI] This may take 1-2 minutes on first run (downloading models)...")
print("[UI] Loading embedding model (BAAI/bge-m3)...")
# Reranker always on for quality, load semantic analyzer for debug display (but don't use in search by default)
retriever = RulebookRetriever(use_reranker=True, use_semantic_analysis=True)
print("[UI] Loading scorer...")
scorer = MultiDimensionalScorer()
print("[UI] Initialization complete!")
print("[UI] Ready to accept requests!")

# Game information
GAME_INFO = {
    "name": "S.T.A.L.K.E.R.: The Board Game",
    "description": "A cooperative survival board game set in the Zone",
    "sources": ["Rulebook", "BoardGameGeek Forums"],
    "index": "rulebook_chunks"
}


@app.route('/')
def index():
    """Render the main UI page."""
    return render_template('index.html', game=GAME_INFO)


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Handle question submission and return answer with metadata.
    
    Expected JSON payload:
        {
            "question": "How many players can play?",
            "use_semantic": false (optional)
        }
    
    Returns JSON:
        {
            "answer": "The answer text...",
            "source": "rulebook",
            "page": 5,
            "confidence": 0.85,
            "scores": {...},
            "chunks": [...],
            "processing_time": 1.23
        }
    """
    import time
    start_time = time.time()
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        use_semantic = data.get('use_semantic', False)
        use_dual_source = data.get('use_dual_source', False)
        forum_weight = data.get('forum_weight', 0.45)  # Default slightly favors rulebook (0.55)
        include_faq = data.get('include_faq', False)  # FAQ disabled by default - enable via checkbox
        
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        # Debug logging
        print(f"[UI] Question: {question}")
        print(f"[UI] Semantic analysis requested: {use_semantic}")
        print(f"[UI] Dual-source search requested: {use_dual_source}")
        print(f"[UI] Include FAQ: {include_faq}")
        
        # Check for spelling corrections
        spellcheck_corrections = []
        original_question = question
        if hasattr(retriever, 'spellcheck_question'):
            corrected_question, corrections = retriever.spellcheck_question(question)
            if corrections:
                spellcheck_corrections = [{"original": orig, "corrected": corr} for orig, corr in corrections]
                question = corrected_question
                print(f"[UI] Spellcheck applied: {len(corrections)} corrections")
                print(f"[UI] Original: {original_question}")
                print(f"[UI] Corrected: {question}")
        
        # Prepare semantic analysis debug info (only when semantic analysis is enabled)
        semantic_debug = {}
        if use_semantic and retriever.semantic_analyzer:
            try:
                import string
                # Get semantic analysis
                analysis = retriever.semantic_analyzer.analyze(question)
                intent = retriever.semantic_analyzer.get_question_intent(question)
                enhanced_query = retriever.semantic_analyzer.enhance_query(question, max_additions=1)
                domain_keywords = retriever.semantic_analyzer.extract_domain_keywords(question)
                
                # Extract query words (cleaned)
                query_words = list(set(word.strip(string.punctuation) for word in question.lower().split()))
                
                # Extract entities from query
                query_entities = retriever.semantic_analyzer.extract_game_entities(question)
                
                semantic_debug = {
                    "original_query": question,
                    "enhanced_query": enhanced_query,
                    "query_words": sorted(query_words),
                    "question_type": intent.get('question_type', 'unknown'),
                    "intent_flags": {k: v for k, v in intent.items() if k != 'question_type'},
                    "key_nouns": analysis.get('key_nouns', []),
                    "action_verbs": analysis.get('action_verbs', []),
                    "game_concepts": analysis.get('game_concepts', []),
                    "domain_keywords": domain_keywords,
                    "extracted_entities": query_entities
                }
                
                print(f"[UI] Enhanced query: {enhanced_query}")
                print(f"[UI] Domain keywords: {domain_keywords}")
            except Exception as e:
                print(f"[UI] Semantic analysis debug failed: {e}")
                semantic_debug = {"error": str(e)}
        
        # Retrieve relevant chunks with detailed metadata
        # Always calculate confidence for all three sources when any comparison is enabled
        answer_source = "rulebook"  # Default
        forum_confidence = 0.0
        rulebook_confidence = 0.0
        faq_confidence = 0.0
        forum_metadata = None
        related_forum_question = None  # Initialize to avoid reference errors
        show_source_comparison = False  # Flag to show confidence bars
        
        print(f"[UI] Search options - FAQ enabled: {include_faq}, Dual-source enabled: {use_dual_source}")
        
        if use_dual_source or include_faq:
            # Calculate confidences for all three sources for comparison
            print(f"[UI] Calculating confidence for all sources...")
            
            # Search FAQ only
            faq_results = retriever.search(
                query=question,
                top_k=5,
                search_type="hybrid",
                hybrid_weight=0.85,
                use_semantic=use_semantic,
                source_type="faq",
                skip_priority_boost=True
            )
            
            # Search rulebook only
            rulebook_results = retriever.search(
                query=question,
                top_k=5,
                search_type="hybrid",
                hybrid_weight=0.85,
                use_semantic=use_semantic,
                source_type="rulebook",
                skip_priority_boost=True
            )
            
            # Search forum only
            forum_results = retriever.search(
                query=question,
                top_k=5,
                search_type="vector",  # Pure semantic for question matching
                use_semantic=False,
                source_type="forum",
                skip_priority_boost=True
            )
            
            # Calculate confidences
            faq_confidence = faq_results[0].get('score', 0.0) if faq_results else 0.0
            rulebook_confidence = rulebook_results[0].get('score', 0.0) if rulebook_results else 0.0
            forum_confidence = forum_results[0].get('score', 0.0) if forum_results else 0.0
            
            print(f"[UI] FAQ: {faq_confidence:.4f}, Rulebook: {rulebook_confidence:.4f}, Forum: {forum_confidence:.4f}")
            show_source_comparison = True
            print(f"[UI] show_source_comparison set to True")
        
        if use_dual_source:
            # Dual-source search: search both forum and rulebook (confidences already calculated above)
            print(f"[UI] Using dual-source search (forum_weight={forum_weight})")
            chunks, answer_source, forum_conf, rulebook_conf, related_forum_q = retriever.search_dual_source(
                query=question,
                top_k=25,
                forum_weight=forum_weight,
                use_semantic=use_semantic,
                include_faq=include_faq
            )
            # Update confidences from dual-source (may differ slightly from pre-calculated)
            forum_confidence = forum_conf if forum_conf > 0 else forum_confidence
            rulebook_confidence = rulebook_conf if rulebook_conf > 0 else rulebook_confidence
            print(f"[UI] Source selected: {answer_source}")
            
            # Extract forum metadata if source is forum
            if answer_source == "forum" and chunks:
                top_chunk = chunks[0]
                quality_score = top_chunk.get('forum_quality_score')
                # Ensure quality score is an integer 1-10, handle None or invalid values
                if quality_score is not None:
                    quality_score = int(round(float(quality_score)))
                    # Clamp to 1-10 range
                    quality_score = max(1, min(10, quality_score))
                forum_metadata = {
                    "thread_id": top_chunk.get('thread_id'),
                    "url": top_chunk.get('url'),
                    "answer_user": top_chunk.get('answer_user'),
                    "quality_score": quality_score,
                    "thread_question": top_chunk.get('text', '')  # Original forum question
                }
            # Store related forum question if rulebook was selected
            elif answer_source == "rulebook" and related_forum_q:
                related_forum_question = related_forum_q
                print(f"[UI] Related forum question: {related_forum_q[:100]}...")
        else:
            # Standard hybrid search (rulebook + FAQ if enabled)
            # Confidences already calculated above if include_faq is True
            # Now do the full search with priority boost to get final results
            if include_faq:
                chunks = retriever.search(
                    query=question,
                    top_k=25,
                    search_type="hybrid",
                    hybrid_weight=0.85,
                    include_faq=include_faq,
                    use_semantic=use_semantic,
                    source_type=None  # Allow both FAQ and rulebook
                )
            else:
                # Filter to only search rulebook chunks (exclude forum and FAQ)
                chunks = retriever.search(
                    query=question,
                    top_k=25,
                    search_type="hybrid",
                    hybrid_weight=0.85,
                    include_faq=False,
                    use_semantic=use_semantic,
                    source_type="rulebook"
                )
        
        print(f"[UI] Retrieved {len(chunks)} chunks")
        if chunks:
            print(f"[UI] Top chunk score: {chunks[0].get('score', 0.0):.4f}")
            print(f"[UI] Top chunk text preview: {chunks[0].get('text', '')[:100]}...")
        
        if not chunks:
            return jsonify({
                "answer": "I couldn't find any relevant information in the rulebook.",
                "source": "none",
                "page": None,
                "confidence": 0.0,
                "scores": {},
                "chunks": [],
                "processing_time": time.time() - start_time,
                "semantic_analysis_used": use_semantic
            })
        
        # Generate answer from chunks
        # For forum and FAQ results, extract the answer field directly
        if answer_source == "forum" and chunks:
            answer = chunks[0].get('answer', 'No answer found.')
            answer_metadata = {"used_chunk_indices": [0], "highest_score_chunk_idx": 0}
            print(f"[UI] Extracted forum answer: {answer[:100]}...")
        elif chunks and chunks[0].get('source_type') == 'faq':
            # FAQ has dedicated answer field
            answer = chunks[0].get('answer', 'No answer found.')
            answer_metadata = {"used_chunk_indices": [0], "highest_score_chunk_idx": 0}
            print(f"[UI] Extracted FAQ answer: {answer[:100]}...")
        else:
            # For rulebook, generate answer by concatenating relevant chunks
            answer, answer_metadata = retriever.generate_answer(question, chunks, use_semantic=use_semantic)
            print(f"[UI] Generated rulebook answer length: {len(answer)}")
            print(f"[UI] Answer preview: {answer[:100]}...")
            print(f"[UI] Answer metadata: {answer_metadata}")
        
        # Get ground truth (placeholder - in production, this would be optional)
        ground_truth = ""  # Not available in UI mode
        
        # Score the answer using multi-dimensional scorer
        scores = scorer.score_answer(question, answer, ground_truth)
        
        # Extract source information from the chunk that contributed most to the answer
        # Use the highest-scoring chunk that was actually used in the answer
        highest_score_chunk_idx = answer_metadata.get("highest_score_chunk_idx")
        if highest_score_chunk_idx is not None and highest_score_chunk_idx < len(chunks):
            top_chunk = chunks[highest_score_chunk_idx]
        else:
            # Fallback to first chunk if metadata is missing
            top_chunk = chunks[0]
        
        source_info = top_chunk.get('source', {})
        
        # Determine actual source type from chunk
        actual_source_type = top_chunk.get('source_type', 'rulebook')
        
        # Initialize FAQ metadata
        faq_metadata = None
        
        # For forums and FAQ, page/section don't exist - handle appropriately
        if answer_source == "forum" or actual_source_type == "faq":
            page = None
            section = top_chunk.get('section') if actual_source_type == "faq" else None  # FAQ has section categories
            
            # Extract FAQ metadata if source is FAQ
            if actual_source_type == "faq":
                faq_metadata = {
                    "faq_id": top_chunk.get('faq_id'),
                    "question": top_chunk.get('text', ''),  # Original FAQ question
                    "section": top_chunk.get('section', 'General')
                }
        else:
            page = source_info.get('page', top_chunk.get('page'))
            section = top_chunk.get('section', 'Unknown')
        
        # Use actual source type from chunk, fallback to answer_source
        if actual_source_type in ['faq', 'forum']:
            source_type = actual_source_type
        else:
            source_type = answer_source if use_dual_source else source_info.get('type', 'rulebook')
        
        # Prepare chunk details for debug info (top 5)
        chunk_details = []
        for i, chunk in enumerate(chunks[:5]):
            chunk_source_type = chunk.get('source_type', 'rulebook')
            
            # For forum chunks, show answer snippet in debug
            if chunk_source_type == 'forum':
                answer_preview = chunk.get('answer', '')[:200]
                # Normalize forum score if needed (cosine similarity + 1.0 → [0,1])
                raw_score = chunk.get('score', 0.0)
                if raw_score > 1.0:
                    norm_score = max(0.0, min(1.0, raw_score - 1.0))
                else:
                    norm_score = raw_score
                # Validate quality score for forum chunks
                quality_score = chunk.get('forum_quality_score', 'N/A')
                if quality_score != 'N/A' and quality_score is not None:
                    quality_score = int(round(float(quality_score)))
                    quality_score = max(1, min(10, quality_score))
                # For forum chunks, show answer_user and url instead of page/section
                chunk_details.append({
                    "rank": i + 1,
                    "text": answer_preview,
                    "score": norm_score,
                    "answered_by": chunk.get('answer_user', 'Unknown'),
                    "thread_url": chunk.get('url', ''),
                    "quality_score": quality_score,
                    "original_question": chunk.get('text', ''),  # Original forum question
                    "hybrid_breakdown": chunk.get('hybrid_breakdown', {}),
                    "entity_matches": chunk.get('entity_matches', 0),
                    "matched_entity_types": chunk.get('matched_entity_types', []),
                    "source_type": 'forum'
                })
            elif chunk_source_type == 'faq':
                # FAQ chunks: show question and answer
                chunk_details.append({
                    "rank": i + 1,
                    "text": chunk.get('answer', '')[:200],  # Show answer preview
                    "score": chunk.get('score', 0.0),
                    "faq_id": chunk.get('faq_id', 'N/A'),
                    "question": chunk.get('text', ''),  # Original FAQ question
                    "section": chunk.get('section', 'N/A'),  # FAQ category
                    "priority": chunk.get('priority', 60),
                    "hybrid_breakdown": chunk.get('hybrid_breakdown', {}),
                    "entity_matches": chunk.get('entity_matches', 0),
                    "matched_entity_types": chunk.get('matched_entity_types', []),
                    "source_type": 'faq'
                })
            else:
                # Rulebook chunks
                text_preview = chunk.get('text', '')
                chunk_details.append({
                    "rank": i + 1,
                    "text": text_preview,
                    "score": chunk.get('score', 0.0),
                    "page": chunk.get('page'),
                    "section": chunk.get('section', 'Unknown'),
                    "hybrid_breakdown": chunk.get('hybrid_breakdown', {}),
                    "entity_matches": chunk.get('entity_matches', 0),
                    "matched_entity_types": chunk.get('matched_entity_types', []),
                    "source_type": 'rulebook'
                })
        
        # Calculate overall confidence using scorer's dimension weights
        # Current weights: relevance 30%, completeness 25%, accuracy 40%, conciseness 5%
        confidence = (
            scores.get('relevance', 0.5) * scorer.weights['relevance'] +
            scores.get('completeness', 0.5) * scorer.weights['completeness'] +
            scores.get('accuracy', 0.5) * scorer.weights['accuracy'] +
            scores.get('conciseness', 0.5) * scorer.weights['conciseness']
        )
        
        processing_time = time.time() - start_time
        
        response_data = {
            "answer": answer,
            "source": source_type,
            "page": page,
            "confidence": round(confidence, 2),
            "scores": {
                "overall": round(scores.get('overall', 0.0), 2),
                "relevance": round(scores.get('relevance', 0.0), 2),
                "completeness": round(scores.get('completeness', 0.0), 2),
                "accuracy": round(scores.get('accuracy', 0.0), 2),
                "conciseness": round(scores.get('conciseness', 0.0), 2)
            },
            "score_weights": {
                "relevance": f"{scorer.weights['relevance']*100:.0f}%",
                "completeness": f"{scorer.weights['completeness']*100:.0f}%",
                "accuracy": f"{scorer.weights['accuracy']*100:.0f}%",
                "conciseness": f"{scorer.weights['conciseness']*100:.0f}%"
            },
            "chunks_retrieved": len(chunks),
            "chunk_details": chunk_details,
            "processing_time": round(processing_time, 2),
            "semantic_analysis_used": use_semantic,
            "dual_source_used": use_dual_source,
            "spellcheck_corrections": spellcheck_corrections,
            "original_question": original_question if spellcheck_corrections else question,
            "corrected_question": question if spellcheck_corrections else None
        }
        
        # Add FAQ metadata if source is FAQ
        if faq_metadata:
            response_data["faq_metadata"] = faq_metadata
        
        # Add source confidence comparison if any comparison mode is enabled
        if show_source_comparison:
            response_data["faq_confidence"] = round(faq_confidence, 2)
            response_data["rulebook_confidence"] = round(rulebook_confidence, 2)
            response_data["forum_confidence"] = round(forum_confidence, 2)
            response_data["source_comparison_enabled"] = True
            print(f"[UI] Adding source comparison to response: FAQ={faq_confidence:.2f}, Rulebook={rulebook_confidence:.2f}, Forum={forum_confidence:.2f}")
            print(f"[UI] Response will include source_comparison_enabled=True")
        else:
            print(f"[UI] NOT adding source comparison (show_source_comparison={show_source_comparison})")
        
        # Add dual-source specific data if enabled (forum metadata, related questions)
        if use_dual_source:
            if forum_metadata:
                response_data["forum_metadata"] = forum_metadata
            # Add related forum question when rulebook is selected
            if answer_source == "rulebook" and related_forum_question:
                response_data["related_forum_question"] = related_forum_question
        
        # Add semantic debug info if available
        if semantic_debug:
            response_data["semantic_debug"] = semantic_debug
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print(f"[ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "game": GAME_INFO["name"],
        "retriever_initialized": retriever is not None,
        "scorer_initialized": scorer is not None
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("BOARDGAME RULES CHECKER - WEB UI")
    print("="*70)
    print(f"Game: {GAME_INFO['name']}")
    print(f"Index: {GAME_INFO['index']}")
    print("\nStarting server at http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
