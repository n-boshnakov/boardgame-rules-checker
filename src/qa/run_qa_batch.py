import csv
import sys
import time
import os
from datetime import datetime
import shutil
import argparse
import pandas as pd
from tqdm import tqdm

# This script is in src/qa/, so go up 2 levels to reach project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from search.retriever import RulebookRetriever
from qa.multi_dimensional_scorer import MultiDimensionalScorer

CSV_PATH = os.path.join(PROJECT_ROOT, "data/processed/qa_results.csv")
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "data/processed/archive")
CSV_CLEAN_PATH = os.path.join(PROJECT_ROOT, "data/processed/qa_results_clean.csv")

# Ensure archive directory exists
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Always start by copying the clean file to qa_results.csv
if os.path.exists(CSV_CLEAN_PATH):
    shutil.copy(CSV_CLEAN_PATH, CSV_PATH)

# NOTE: Answer generation logic is centralized in retriever.py
# RulebookRetriever.generate_answer() is the single source of truth

def main(args):
    """Run QA evaluation batch on questions from CSV file.
    
    Args:
        args: Command-line arguments from argparse
        
    Results:
        Saves results to qa_results.csv and archives with timestamp
    """
    start_time = time.time()

    # Use hybrid search by default; enable semantic analysis only when flag is provided
    use_semantic = args.use_semantic_analysis

    # Initialize retriever with reranking enabled (CrossEncoder)
    retriever = RulebookRetriever(
        use_reranker=True,
        use_semantic_analysis=use_semantic
    )
    
    # Initialize multi-dimensional scorer (it will load its own CrossEncoder)
    scorer = MultiDimensionalScorer()

    # Load questions from clean CSV
    questions_df = pd.read_csv(CSV_PATH)
    if args.max_questions:
        questions_df = questions_df.head(args.max_questions)
        print(f"Processing first {args.max_questions} questions...")

    results = []
    for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Processing questions"):
        question = row['question']
        ground_truth = row['ground_truth']
        
        # Choose search mode: dual-source or single-source
        if args.use_dual_source:
            # Use dual-source search (forum + rulebook)
            # Returns: (chunks, source, forum_confidence, rulebook_confidence, related_forum_question)
            chunks, answer_source, forum_conf, rulebook_conf, related_forum_q = retriever.search_dual_source(
                question,
                top_k=25,
                forum_weight=args.forum_weight,
                use_semantic=args.use_semantic_analysis
            )
            
            if answer_source == 'none' or not chunks:
                # No relevant chunks found - record failure
                results.append((question, ground_truth, "No chunks found", 0, None, None, None, None, 'none', None, None, None, None, 'none', 0, 0))
                continue
            
            # Generate answer from chunks based on source
            if answer_source == 'forum':
                # For forum, use the answer field from top chunk
                predicted_answer = chunks[0].get('answer', 'No answer found.')
            else:
                # For rulebook, generate answer from chunks
                predicted_answer = retriever.generate_answer(question, chunks, use_semantic=args.use_semantic_analysis)
            
            # Set source confidence
            source_confidence = forum_conf if answer_source == 'forum' else rulebook_conf
            
            # For compatibility, use chunks as retrieved_chunks
            retrieved_chunks = chunks
        else:
            # Use traditional single-source search (rulebook only)
            retrieved_chunks = retriever.search(
                question, 
                top_k=25, 
                search_type="hybrid", 
                hybrid_weight=0.85,
                use_semantic=use_semantic,
                source_type="rulebook"  # Explicit filter to rulebook
            )
            
            if not retrieved_chunks:
                # No relevant chunks found - record failure
                results.append((question, ground_truth, "No chunks found", 0, None, None, None, None, 'none', None, None, None, None, 'rulebook', 0, 0))
                continue
            
            # Generate answer from retrieved chunks
            predicted_answer = retriever.generate_answer(question, retrieved_chunks, use_semantic=use_semantic)
            answer_source = 'rulebook'
            source_confidence = retrieved_chunks[0].get('score', 0) if retrieved_chunks else 0
            forum_conf = 0
            rulebook_conf = source_confidence

        # Diagnostic: Find highest-ranked chunk that matches ground truth well
        # This helps identify if retrieval is the bottleneck vs answer generation
        best_chunk_rank = None
        try:
            from rapidfuzz import fuzz
            for rank, chunk in enumerate(retrieved_chunks, 1):
                chunk_text = str(chunk.get('text', ''))
                similarity = fuzz.token_set_ratio(chunk_text, str(ground_truth))
                if similarity >= 70:  # Found a chunk with >70% match to answer
                    best_chunk_rank = rank
                    break
        except Exception:
            pass  # Silently continue if diagnostic fails
        
        # Convert ground_truth to string, handle NaN/None
        gt_string = str(ground_truth) if ground_truth and str(ground_truth) != 'nan' else None
        
        # Use multi-dimensional scoring instead of simple text similarity
        score_result = scorer.score_answer(question, predicted_answer, gt_string)
        
        # Extract individual scores
        overall_score = score_result['overall']
        relevance_score = score_result['relevance']
        completeness_score = score_result['completeness']
        accuracy_score = score_result['accuracy']
        conciseness_score = score_result['conciseness']
        
        # Detect question type for analysis
        question_type = scorer._detect_question_type(question.lower())
        
        # Extract metadata from top-ranked chunk for reference
        top_chunk = retrieved_chunks[0] if retrieved_chunks else {}
        chunk_page = top_chunk.get('page')
        chunk_section = top_chunk.get('section')
        chunk_index = top_chunk.get('chunk_index')

        results.append((
            question, 
            ground_truth, 
            predicted_answer, 
            overall_score,
            relevance_score,
            completeness_score,
            accuracy_score,
            conciseness_score,
            question_type,
            chunk_page, 
            chunk_section, 
            chunk_index, 
            best_chunk_rank,
            answer_source,  # 'forum' or 'rulebook'
            forum_conf,
            rulebook_conf
        ))

    # Create DataFrame with results
    results_df = pd.DataFrame(
        results, 
        columns=[
            'question', 
            'ground_truth', 
            'predicted', 
            'overall_score',
            'relevance_score',
            'completeness_score', 
            'accuracy_score',
            'conciseness_score',
            'question_type',
            'page', 
            'section', 
            'chunk_index', 
            'best_chunk_rank',
            'answer_source',  # 'forum' or 'rulebook'
            'forum_confidence',
            'rulebook_confidence'
        ]
    )

    # Calculate performance metrics
    elapsed_time = int(time.time() - start_time)
    mean_overall_score = results_df['overall_score'].mean()
    mean_relevance = results_df['relevance_score'].mean()
    mean_completeness = results_df['completeness_score'].mean()
    mean_accuracy = results_df['accuracy_score'].mean()
    mean_conciseness = results_df['conciseness_score'].mean()
    passing_count = (results_df['overall_score'] >= 0.8).sum()
    
    # Generate timestamped filename for archive
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_path = os.path.join(
        PROJECT_ROOT,
        f"data/processed/archive/qa_results_{timestamp}_{elapsed_time}s.csv"
    )
    current_path = os.path.join(PROJECT_ROOT, "data/processed/qa_results.csv")
    
    # Save results to both current and archive locations
    results_df.to_csv(current_path, index=False)
    results_df.to_csv(archive_path, index=False)
    
    # Calculate source distribution if dual-source enabled
    if args.use_dual_source:
        forum_count = (results_df['answer_source'] == 'forum').sum()
        rulebook_count = (results_df['answer_source'] == 'rulebook').sum()
        source_dist = f"Forum: {forum_count} ({forum_count/len(results_df):.1%}), Rulebook: {rulebook_count} ({rulebook_count/len(results_df):.1%})"
    else:
        source_dist = "Rulebook only (dual-source disabled)"
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"QA Batch Evaluation Complete - Multi-Dimensional Scoring")
    if args.use_dual_source:
        print(f"Search Mode: Dual-Source (Forum + Rulebook)")
        print(f"Forum Weight: {args.forum_weight:.2f}")
    else:
        print(f"Search Mode: Hybrid (85% semantic + 15% BM25, Rulebook only)")
    if use_semantic:
        print(f"Semantic Analysis: ENABLED")
    else:
        print(f"Semantic Analysis: DISABLED (default)")
    print(f"{'='*70}")
    print(f"Questions processed: {len(results_df)}")
    print(f"Answer sources: {source_dist}")
    print(f"Mean overall score: {mean_overall_score:.2%}")
    print(f"  - Relevance ({scorer.weights['relevance']:.0%}):     {mean_relevance:.2%}")
    print(f"  - Completeness ({scorer.weights['completeness']:.0%}):  {mean_completeness:.2%}")
    print(f"  - Accuracy ({scorer.weights['accuracy']:.0%}):      {mean_accuracy:.2%}")
    print(f"  - Conciseness ({scorer.weights['conciseness']:.0%}):   {mean_conciseness:.2%}")
    print(f"Passing (≥0.8): {passing_count}/{len(results_df)} ({passing_count/len(results_df):.1%})")
    print(f"Processing time: {elapsed_time}s")
    print(f"\nResults saved to:")
    print(f"  Current: {current_path}")
    print(f"  Archive: {archive_path}")
    print(f"{'='*70}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run QA batch evaluation on rulebook questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_qa_batch.py --max_questions 10
  python run_qa_batch.py --hybrid_weight 0.90 --rerank_top_n 15
        """
    )
    parser.add_argument(
        "--max_questions", 
        type=int, 
        default=None, 
        help="Maximum number of questions to process (default: all)"
    )
    parser.add_argument(
        "--rerank_top_n", 
        type=int, 
        default=10, 
        help="Number of top chunks to rerank with CrossEncoder (default: 10)"
    )
    parser.add_argument(
        "--hybrid_weight", 
        type=float, 
        default=0.85, 
        help="Semantic weight in hybrid search: 0.85 = 85%% semantic + 15%% BM25 (default: 0.85)"
    )
    parser.add_argument(
        "--search_type", 
        type=str, 
        default="hybrid", 
        choices=["vector", "hybrid", "keyword"],
        help="Search strategy: vector (semantic only), hybrid (semantic+BM25), keyword (BM25 only)"
    )
    parser.add_argument(
        "--semantic_selection", 
        action="store_true", 
        help="Use semantic sentence selection for answer generation (experimental)"
    )
    parser.add_argument(
        "--use_semantic_analysis",
        action="store_true",
        help="Enable semantic query analysis (NLTK-based question understanding) - Optional NLP enhancement"
    )
    parser.add_argument(
        "--use_dual_source",
        action="store_true",
        help="Enable dual-source search (forum + rulebook) - Uses forum Q&A when more relevant"
    )
    parser.add_argument(
        "--forum_weight",
        type=float,
        default=0.45,
        help="Weight for forum results in dual-source mode (0-1, default: 0.45 - slightly favors rulebook)"
    )
    
    args = parser.parse_args()
    main(args)
