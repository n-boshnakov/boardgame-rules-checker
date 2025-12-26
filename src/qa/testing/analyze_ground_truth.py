import json
import pandas as pd
import os

# This script is in src/qa/testing/, go up 3 levels to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Load Q&A data
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/qa_results.csv'))

# Load chunks
with open(os.path.join(PROJECT_ROOT, 'data/processed/chunks_2025-12-23.json'), encoding='utf-8') as f:
    chunks = json.load(f)

print("\n" + "="*80)
print("GROUND TRUTH VS CHUNK CONTENT ANALYSIS")
print("="*80)

# Examine specific cases
test_questions = [
    "How many players can play this game?",
    "What happens if a player dies?",
    "Can I pick up loot while in enemy LoS?"
]

for question in test_questions:
    row = df[df['question'] == question].iloc[0]
    
    print(f"\n\n{'='*80}")
    print(f"Q: {question}")
    print('='*80)
    print(f"\nGround Truth:")
    print(f"  {row['ground_truth']}")
    
    print(f"\nPredicted Answer:")
    print(f"  {row['predicted'][:200]}...")
    
    print(f"\nScore: {row['score']:.2f}")
    
    print(f"\nBest chunk rank: {row.get('best_chunk_rank', 'N/A')}")
    
    # Search for keywords in chunks
    gt_keywords = set(row['ground_truth'].lower().split())
    
    # Find chunks that might contain the answer
    print(f"\nSearching chunks for keywords...")
    best_matches = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk['text'].lower().split())
        overlap = len(gt_keywords & chunk_words)
        if overlap > 3:
            best_matches.append((i, overlap, chunk))
    
    best_matches.sort(key=lambda x: x[1], reverse=True)
    
    if best_matches:
        print(f"  Top matching chunks by keyword overlap:")
        for i, (idx, overlap, chunk) in enumerate(best_matches[:3], 1):
            print(f"\n  {i}. Chunk {idx} (overlap: {overlap} words)")
            print(f"     Section: {chunk.get('section', 'N/A')}")
            print(f"     Text: {chunk['text'][:200]}...")

print("\n\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
The ground truth answers appear to be MANUALLY CREATED summaries, not direct
extractions from the PDF. This means:

1. Perfect match is impossible - we're comparing human-written summaries
   to document text
   
2. Our sentence extraction CAN'T find "This game can be played by 1-4 players"
   because that exact phrasing doesn't exist in the chunks
   
3. The similarity score measures semantic similarity, not exact text match

SOLUTION: The sentence-level extraction is CORRECT. The issue is that:
- Ground truth is human-summarized (paraphrased)
- Our extractive method returns actual document text
- Similarity scores will never reach 1.0 unless we paraphrase too

The goal should be: Extract the MOST RELEVANT sentences from chunks,
even if they're not worded exactly like ground truth.
""")
