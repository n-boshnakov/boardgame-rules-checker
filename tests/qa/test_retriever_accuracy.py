import pandas as pd
from rapidfuzz import fuzz
from src.search.retriever import RulebookRetriever
from tqdm import tqdm

# Load the batch QA results for comparison
batch_results = pd.read_csv('data/processed/qa_results.csv')

# Initialize retriever
print("Initializing retriever...")
retriever = RulebookRetriever(use_reranker=False)

# Test on first 10 questions for speed
print("\nTesting retriever on first 10 questions...")
results = []

for idx, row in tqdm(batch_results.head(10).iterrows(), total=10):
    question = row['question']
    ground_truth = row['ground_truth']
    batch_answer = row['predicted']
    
    # Get answer from retriever
    chunks = retriever.search(question, top_k=10, search_type="hybrid", hybrid_weight=0.7)
    retriever_answer = retriever.generate_answer(question, chunks, multi_chunk_synthesis=True)
    
    # Calculate similarities
    batch_sim = fuzz.token_set_ratio(str(ground_truth), str(batch_answer))
    retriever_sim = fuzz.token_set_ratio(str(ground_truth), str(retriever_answer))
    
    results.append({
        'question': question,
        'batch_similarity': batch_sim,
        'retriever_similarity': retriever_sim,
        'difference': retriever_sim - batch_sim,
        'batch_answer_len': len(str(batch_answer)),
        'retriever_answer_len': len(str(retriever_answer))
    })

# Create dataframe
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("RETRIEVER vs BATCH COMPARISON")
print("="*80)
print(f"\nBatch Mean Similarity:     {results_df['batch_similarity'].mean():.2f}")
print(f"Retriever Mean Similarity: {results_df['retriever_similarity'].mean():.2f}")
print(f"Average Difference:        {results_df['difference'].mean():+.2f}")

print(f"\nBatch ≥70% similarity:     {(results_df['batch_similarity'] >= 70).sum()}/10")
print(f"Retriever ≥70% similarity: {(results_df['retriever_similarity'] >= 70).sum()}/10")

print(f"\nBatch ≥80% similarity:     {(results_df['batch_similarity'] >= 80).sum()}/10")
print(f"Retriever ≥80% similarity: {(results_df['retriever_similarity'] >= 80).sum()}/10")

print("\n" + "="*80)
print("DETAILED RESULTS:")
print("="*80)
for idx, row in results_df.iterrows():
    print(f"\nQ{idx+1}: {row['question'][:60]}...")
    print(f"  Batch:     {row['batch_similarity']:.1f}% ({row['batch_answer_len']} chars)")
    print(f"  Retriever: {row['retriever_similarity']:.1f}% ({row['retriever_answer_len']} chars)")
    print(f"  Diff:      {row['difference']:+.1f}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
if abs(results_df['difference'].mean()) < 2:
    print("✓ Retriever matches batch accuracy closely (within 2 points)")
elif results_df['difference'].mean() > 0:
    print(f"✓ Retriever performs BETTER than batch by {results_df['difference'].mean():.2f} points")
else:
    print(f"✗ Retriever performs WORSE than batch by {abs(results_df['difference'].mean()):.2f} points")
    print("  Investigate differences in answer generation logic")
