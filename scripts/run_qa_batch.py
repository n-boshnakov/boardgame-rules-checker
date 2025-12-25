import csv
import sys
import time
import os
from datetime import datetime
import shutil
import argparse
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from search.retriever import RulebookRetriever

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search.retriever import RulebookRetriever

CSV_PATH = "data/processed/qa_results.csv"
ARCHIVE_DIR = "data/processed/archive"
CSV_CLEAN_PATH = "data/processed/qa_results_clean.csv"

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

    # Initialize Retriever (disable cross-encoder for consistency)
    retriever = RulebookRetriever(use_reranker=False)
    
    # Get hybrid weight from args or use default
    hybrid_weight = float(args.hybrid_weight) if hasattr(args, 'hybrid_weight') else 0.8  # bias toward vector
    retriever = RulebookRetriever(use_reranker=True)  # enable cross-encoder reranking

    df = pd.read_csv(CSV_PATH)
    if args.max_questions:
        df = df.head(args.max_questions)

    results = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        question = row['question']
        ground_truth = row['ground_truth']
        
        # Search for relevant chunks with specified hybrid weight
        # Retrieve 20 chunks for maximum coverage, favor semantic search
        answer_chunks = retriever.search(question, top_k=20, search_type="hybrid", hybrid_weight=0.8)
        
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
        predicted_answer = retriever.generate_answer(question, answer_chunks, multi_chunk_synthesis=True)
        
        # Get metadata from first chunk for reference
        first_chunk = answer_chunks[0] if answer_chunks else {}
        score = first_chunk.get('score', 0)

        results.append((question, ground_truth, predicted_answer, score, 
                       first_chunk.get('page'), first_chunk.get('section'), 
                       first_chunk.get('chunk_index'), best_chunk_rank))

    # Create a new DataFrame for the results
    results_df = pd.DataFrame(results, columns=['question', 'ground_truth', 'predicted', 'score', 'page', 'section', 'chunk_index', 'best_chunk_rank'])

    # Calculate processing time
    elapsed_time = int(time.time() - start_time)
    
    # Save results with new format: qa_results_YYYY-MM-DD_HH-MM-SS_XXs_wWW.csv
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    weight_suffix = f"_w{int(hybrid_weight*100)}" if hybrid_weight != 0.7 else ""
    output_filename = f"data/processed/archive/qa_results_{timestamp}_{elapsed_time}s{weight_suffix}.csv"
    # Overwrite qa_results.csv
    results_df.to_csv("data/processed/qa_results.csv", index=False)
    # Save archive copy
    results_df.to_csv(output_filename, index=False)
    print(f"\nQA results saved to {output_filename} and overwritten qa_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QA batch script.")
    parser.add_argument("--max_questions", type=int, default=None, help="Maximum number of questions to process.")
    parser.add_argument("--hybrid_weight", type=float, default=0.7, help="Weight for vector vs keyword search (0.0=keyword only, 1.0=vector only). Default: 0.7 (optimal: 70%% vector, 30%% keyword)")
    args = parser.parse_args()
    main(args)
