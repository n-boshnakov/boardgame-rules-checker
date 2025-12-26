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

CSV_PATH = os.path.join(PROJECT_ROOT, "data/processed/qa_results.csv")
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "data/processed/archive")
CSV_CLEAN_PATH = os.path.join(PROJECT_ROOT, "data/processed/qa_results_clean.csv")

# Ensure archive directory exists
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Always start by copying the clean file to qa_results.csv
if os.path.exists(CSV_CLEAN_PATH):
    shutil.copy(CSV_CLEAN_PATH, CSV_PATH)

# NOTE: Answer generation logic has been moved to retriever.py
# All answer generation now happens in RulebookRetriever.generate_answer()
# This ensures a single source of truth for answer generation logic
# NOTE: Answer generation logic has been moved to retriever.py
# All answer generation now happens in RulebookRetriever.generate_answer()
# This ensures a single source of truth for answer generation logic

def main(args):
    # Track start time
    start_time = time.time()

    # Initialize Retriever with baseline configuration
    retriever = RulebookRetriever(use_reranker=True)

    df = pd.read_csv(CSV_PATH)
    if args.max_questions:
        df = df.head(args.max_questions)

    results = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        question = row['question']
        ground_truth = row['ground_truth']
        
        # Search for relevant chunks with baseline configuration
        answer_chunks = retriever.search(question, top_k=25, search_type="hybrid", hybrid_weight=0.85)
        
        if not answer_chunks:
            results.append((question, ground_truth, "No chunks found", 0, None, None, None, None))
            continue

        # Find which rank contains the best matching chunk (for diagnostics)
        best_chunk_rank = None
        try:
            from rapidfuzz import fuzz
            for rank, chunk in enumerate(answer_chunks, 1):
                sim = fuzz.token_set_ratio(str(chunk['text']), str(ground_truth))
                if sim >= 70:  # If chunk has >70% similarity to ground truth
                    best_chunk_rank = rank
                    break
        except:
            pass

        # Generate answer using retriever's generate_answer method
        # This is the single source of truth for answer generation
        predicted_answer = retriever.generate_answer(question, answer_chunks)
        
        # Calculate answer similarity to ground truth
        try:
            from rapidfuzz import fuzz
            similarity_score = fuzz.token_set_ratio(str(predicted_answer), str(ground_truth)) / 100.0
        except:
            similarity_score = 0.0
        
        # Get metadata from first chunk for reference
        first_chunk = answer_chunks[0] if answer_chunks else {}

        results.append((question, ground_truth, predicted_answer, similarity_score, 
                       first_chunk.get('page'), first_chunk.get('section'), 
                       first_chunk.get('chunk_index'), best_chunk_rank))

    # Create a new DataFrame for the results
    results_df = pd.DataFrame(results, columns=['question', 'ground_truth', 'predicted', 'score', 'page', 'section', 'chunk_index', 'best_chunk_rank'])

    # Calculate processing time
    elapsed_time = int(time.time() - start_time)
    
    # Save results with format: qa_results_YYYY-MM-DD_HH-MM-SS_XXs.csv
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"data/processed/archive/qa_results_{timestamp}_{elapsed_time}s.csv"
    
    # Overwrite qa_results.csv
    results_df.to_csv("data/processed/qa_results.csv", index=False)
    # Save archive copy
    results_df.to_csv(output_filename, index=False)
    print(f"\nQA results saved to {output_filename} and overwritten qa_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QA batch script.")
    parser.add_argument("--max_questions", type=int, default=None, help="Maximum number of questions to process.")
    parser.add_argument("--rerank_top_n", type=int, default=10, help="Number of top chunks to rerank with CrossEncoder (default: 10)")
    parser.add_argument("--hybrid_weight", type=float, default=0.85, help="Weight for vector vs keyword in hybrid search (default: 0.85 = 85%% semantic, 15%% BM25)")
    parser.add_argument("--search_type", type=str, default="hybrid", choices=["vector", "hybrid", "keyword"], help="Search type: vector (semantic only), hybrid (semantic+BM25), keyword (BM25 only)")
    parser.add_argument("--semantic_selection", action="store_true", help="Use semantic sentence selection for answer generation (scores sentences by relevance)")
    args = parser.parse_args()
    main(args)
    args = parser.parse_args()
    main(args)
