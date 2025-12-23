import csv
import sys
import subprocess
import time
import os
from datetime import datetime
import shutil

CSV_PATH = "data/processed/qa_results.csv"
ARCHIVE_DIR = "data/processed/archive"
RETRIEVER_PATH = "src/search/retriever.py"
CSV_CLEAN_PATH = "data/processed/qa_results_clean.csv"

# Ensure archive directory exists
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Always start by copying the clean file to qa_results.csv
if os.path.exists(CSV_CLEAN_PATH):
    shutil.copy(CSV_CLEAN_PATH, CSV_PATH)

import argparse

parser = argparse.ArgumentParser(description="Run QA batch script with optional question limit.")
parser.add_argument("--max_questions", type=int, default=None, help="Maximum number of questions to answer from the file.")
args = parser.parse_args()

# Read all questions from the CSV
rows = []
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Optionally limit the number of questions
if args.max_questions is not None:
    rows = rows[:args.max_questions]

total = len(rows)
start_time = time.time()


# Load section headers for filtering
section_headers_path = "data/processed/section_headers.txt"
if os.path.exists(section_headers_path):
    with open(section_headers_path, "r", encoding="utf-8") as shf:
        section_headers = set(line.strip() for line in shf if line.strip())
else:
    section_headers = set()

# For each question, run the retriever and capture the top answer
for idx, row in enumerate(rows, 1):
    import re
    question = row['question']
    # Use the same Python executable that's running this script
    python_exe = sys.executable
    # Run subprocess with UTF-8 encoding
    result = subprocess.run([
        python_exe, RETRIEVER_PATH, question
    ], capture_output=True, encoding='utf-8', errors='replace', cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Collect all answer chunks from output
    answer_chunks = []
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Text: "):
            # Extract full text (may be truncated at 300 chars in output)
            text = line[6:].strip()
            if text:
                answer_chunks.append(text)
    # For each chunk, split into sentences and score each sentence
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer('all-MiniLM-L6-v2')
    def hybrid_score(q, s):
        # Hybrid: 0.7 vector + 0.3 keyword (BM25-like, here: simple token overlap)
        q_emb = model.encode([q], convert_to_numpy=True)[0]
        s_emb = model.encode([s], convert_to_numpy=True)[0]
        v_score = np.dot(q_emb, s_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(s_emb) + 1e-8)
        # Simple keyword overlap as BM25 proxy
        q_tokens = set(re.findall(r'\w+', q.lower()))
        s_tokens = set(re.findall(r'\w+', s.lower()))
        k_score = len(q_tokens & s_tokens) / (len(q_tokens | s_tokens) + 1e-8)
        score = 0.7 * v_score + 0.3 * k_score
        # Penalty for short sentences (<= 15 chars)
        if len(s.strip()) <= 15:
            score -= 0.25
        # Penalty for all uppercase (ignoring punctuation)
        s_alpha = re.sub(r'[^A-Za-z]', '', s)
        if s_alpha.isupper() and len(s_alpha) > 0:
            score -= 0.25
        # Penalty for matching section headers
        if s.strip() in section_headers:
            score -= 0.5
        return score
    
    # Collect all candidate sentences with their scores
    candidates = []
    for chunk in answer_chunks:
        # Split chunk into sentences (lenient regex)
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        for sent in sentences:
            sent = sent.strip()
            # Skip very short sentences
            if len(sent) < 10:
                continue
            score = hybrid_score(question, sent)
            candidates.append((score, sent, chunk))
        # Also consider the whole chunk as a candidate
        if len(chunk.strip()) >= 10:
            score = hybrid_score(question, chunk)
            candidates.append((score, chunk, chunk))
    
    # Select best answer by extracting multiple relevant sentences
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Get the best scored sentence and its source chunk
        best_score, best_sent, source_chunk = candidates[0]
        
        # Extract context: find the best sentence in its chunk and include surrounding sentences
        answer_sentences = []
        chunk_sentences = re.split(r'(?<=[.!?])\s+', source_chunk)
        
        # Find where the best sentence appears in the chunk
        best_sent_idx = -1
        for i, sent in enumerate(chunk_sentences):
            if best_sent.strip() in sent.strip() or sent.strip() in best_sent.strip():
                best_sent_idx = i
                break
        
        if best_sent_idx >= 0:
            # Include the best sentence plus surrounding context (1 before, 2 after)
            start_idx = max(0, best_sent_idx - 1)
            end_idx = min(len(chunk_sentences), best_sent_idx + 3)
            
            for i in range(start_idx, end_idx):
                sent = chunk_sentences[i].strip()
                if len(sent) >= 10:
                    # Skip if it's a section header
                    if sent not in section_headers:
                        answer_sentences.append(sent)
        else:
            # Fallback: use the best sentence
            answer_sentences = [best_sent]
        
        # Combine sentences into coherent answer (max 3-4 sentences)
        best_sent = ' '.join(answer_sentences[:4])
        
        # If answer is too long, try to trim to most relevant sentences
        if len(best_sent) > 800:
            # Re-score each sentence and keep only top 3
            scored_sents = [(hybrid_score(question, s), s) for s in answer_sentences[:4]]
            scored_sents.sort(key=lambda x: x[0], reverse=True)
            best_sent = ' '.join([s for _, s in scored_sents[:3]])
            
    elif answer_chunks:
        # If no valid sentences, use the first non-empty chunk
        best_sent = next((chunk for chunk in answer_chunks if chunk.strip()), answer_chunks[0] if answer_chunks else "[No answer found]")
    else:
        best_sent = "[No answer found]"
    
    row['predicted'] = best_sent
    # Progress tracker
    print(f"[{idx}/{total}] Processed: {question[:50]}{'...' if len(question) > 50 else ''}")

end_time = time.time()
elapsed = end_time - start_time

# Write back to CSV
with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Always save archive with timestamp and elapsed time in filename, even if no new predictions are made
now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
elapsed_str = f"{int(elapsed)}s"
archive_path = os.path.join(
    ARCHIVE_DIR,
    f"qa_results_{now_str}_{elapsed_str}.csv"
)
with open(archive_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nArchive saved to {archive_path}")

print(f"\nUpdated predictions written to {CSV_PATH}")
print(f"Archive saved to {archive_path}")
print(f"Total time: {elapsed:.2f} seconds for {total} questions.")
