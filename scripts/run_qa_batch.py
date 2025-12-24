import csv
import sys
import time
import os
from datetime import datetime
import shutil
import argparse
import re
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from elasticsearch import Elasticsearch

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search.retriever import RulebookRetriever, ES_INDEX

CSV_PATH = "data/processed/qa_results.csv"
ARCHIVE_DIR = "data/processed/archive"
CSV_CLEAN_PATH = "data/processed/qa_results_clean.csv"

# Ensure archive directory exists
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Always start by copying the clean file to qa_results.csv
if os.path.exists(CSV_CLEAN_PATH):
    shutil.copy(CSV_CLEAN_PATH, CSV_PATH)

def get_best_answer_from_chunks(question: str, chunks: list, model: SentenceTransformer):
    """
    Finds the best sentence from chunks based on a hybrid scoring model
    that considers semantic similarity and section relevance.
    Extracts a multi-sentence answer for better context.
    """
    if not chunks:
        return "No relevant information found.", 0, {}

    question_embedding = model.encode(question, convert_to_tensor=True)
    question_keywords = set(question.lower().split())

    # Load section headers to filter them out from answers
    section_headers = set()
    try:
        with open("data/processed/section_headers.txt", "r", encoding="utf-8") as f:
            section_headers = {line.strip() for line in f}
    except FileNotFoundError:
        print("[Warning] section_headers.txt not found. Section headers may appear in answers.")

    all_candidates = []

    for chunk in chunks:
        text = chunk['text']
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            continue

        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]

        # Calculate section relevance boost
        section_boost = 0
        if chunk.get('section'):
            section_keywords = set(chunk['section'].lower().split())
            common_keywords = question_keywords.intersection(section_keywords)
            if common_keywords:
                section_boost = 0.1 * len(common_keywords) # Boost for each matching keyword

        for i, sentence in enumerate(sentences):
            # Skip sentences that are just section headers or too short
            if sentence.strip() in section_headers or len(sentence.split()) < 4:
                continue

            # Hybrid score: combine semantic score with other heuristics
            semantic_score = cosine_scores[i].item()
            
            # Penalize short sentences and all-caps sentences (likely headers)
            penalty = 0
            if len(sentence.split()) < 5:
                penalty += 0.1
            if sentence.isupper() and len(sentence.split()) < 10:
                penalty += 0.2

            final_score = semantic_score + section_boost - penalty
            
            all_candidates.append({
                "score": final_score,
                "sentence": sentence,
                "index": i,
                "sentences": sentences,
                "chunk": chunk
            })

    # Sort all candidates by score
    sorted_candidates = sorted(all_candidates, key=lambda x: x['score'], reverse=True)

    if not sorted_candidates:
        return "No relevant information found.", 0, {}

    # Get the best candidate
    best_candidate = sorted_candidates[0]
    highest_score = best_candidate['score']
    best_chunk_info = best_candidate['chunk']
    
    # Smart extraction: balance completeness with relevance
    best_index = best_candidate['index']
    sentences = best_candidate['sentences']
    
    # Create a relevance score map for sentences around the best one
    relevance_threshold = highest_score * 0.4  # Lower threshold to include more context
    
    # Find all candidates from the same chunk and sentence list
    chunk_candidates = [c for c in sorted_candidates 
                       if c['sentences'] is sentences 
                       and c['score'] >= relevance_threshold]
    
    # Get indices of relevant sentences
    relevant_indices = {c['index'] for c in chunk_candidates}
    
    # Build answer by including consecutive relevant sentences around the best one
    # Start from best sentence and expand outward
    answer_indices = {best_index}
    
    # Expand backward - include at least 1 sentence before if available
    for i in range(best_index - 1, max(0, best_index - 3), -1):
        if i in relevant_indices or len(answer_indices) < 2:
            answer_indices.add(i)
        else:
            break  # Stop at first irrelevant sentence (unless we need more context)
    
    # Expand forward - include at least 2 sentences after if available
    for i in range(best_index + 1, min(len(sentences), best_index + 5)):
        if i in relevant_indices or len(answer_indices) < 3:
            answer_indices.add(i)
        else:
            break  # Stop at first irrelevant sentence (unless we need more context)
    
    # Sort indices and extract sentences
    sorted_indices = sorted(answer_indices)
    context_sentences = [sentences[i] for i in sorted_indices if len(sentences[i].split()) > 3]
    
    # Post-process to form a coherent paragraph
    best_answer = " ".join(s.strip() for s in context_sentences)
    
    # Clean up potential artifacts
    best_answer = re.sub(r'\s+', ' ', best_answer).strip()

    return best_answer, highest_score, best_chunk_info

def main(args):
    # Track start time
    start_time = time.time()
    
    # Initialize Retriever
    retriever = RulebookRetriever()
    answer_selection_model = retriever.model
    
    # Get hybrid weight from args or use default
    hybrid_weight = getattr(args, 'hybrid_weight', 0.7)

    df = pd.read_csv(CSV_PATH)
    if args.max_questions:
        df = df.head(args.max_questions)

    results = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        question = row['question']
        ground_truth = row['ground_truth']
        
        # Search for relevant chunks with specified hybrid weight
        answer_chunks = retriever.search(question, top_k=5, search_type="hybrid", hybrid_weight=hybrid_weight)
        
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

        # Get the best answer from the chunks
        predicted_answer, score, chunk_info = get_best_answer_from_chunks(question, answer_chunks, answer_selection_model)

        results.append((question, ground_truth, predicted_answer, score, chunk_info.get('page'), chunk_info.get('section'), chunk_info.get('chunk_index'), best_chunk_rank))

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
