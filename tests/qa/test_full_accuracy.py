"""
Test retriever accuracy on all 40 questions to ensure 
the changes improve overall performance
"""

import pandas as pd
from rapidfuzz import fuzz
from src.search.retriever import RulebookRetriever
from tqdm import tqdm

# Load the batch QA results
batch_results = pd.read_csv('data/processed/qa_results.csv')

# Initialize retriever
print("Initializing retriever...")
retriever = RulebookRetriever(use_reranker=False)

# Test on all questions
print(f"\nTesting retriever on all {len(batch_results)} questions...")
results = []

for idx, row in tqdm(batch_results.iterrows(), total=len(batch_results)):
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
        'difference': retriever_sim - batch_sim
    })

# Create dataframe
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("FULL RETRIEVER vs BATCH COMPARISON (ALL QUESTIONS)")
print("="*80)

print(f"\nOVERALL METRICS:")
print(f"  Batch Mean Similarity:     {results_df['batch_similarity'].mean():.2f}")
print(f"  Retriever Mean Similarity: {results_df['retriever_similarity'].mean():.2f}")
print(f"  Average Difference:        {results_df['difference'].mean():+.2f}")

print(f"\nFIRST 20 QUESTIONS (answerable):")
first_20 = results_df.head(20)
print(f"  Batch Mean:     {first_20['batch_similarity'].mean():.2f}")
print(f"  Retriever Mean: {first_20['retriever_similarity'].mean():.2f}")
print(f"  Difference:     {first_20['difference'].mean():+.2f}")

print(f"\nTHRESHOLD ANALYSIS:")
print(f"  Batch ≥70%:     {(results_df['batch_similarity'] >= 70).sum()}/{len(results_df)}")
print(f"  Retriever ≥70%: {(results_df['retriever_similarity'] >= 70).sum()}/{len(results_df)}")
print(f"  Batch ≥80%:     {(results_df['batch_similarity'] >= 80).sum()}/{len(results_df)}")
print(f"  Retriever ≥80%: {(results_df['retriever_similarity'] >= 80).sum()}/{len(results_df)}")

print(f"\nQUESTIONS WHERE RETRIEVER IMPROVED (+10 or more):")
improved = results_df[results_df['difference'] >= 10].sort_values('difference', ascending=False)
for idx, row in improved.iterrows():
    print(f"  Q{idx+1}: {row['question'][:60]}...")
    print(f"    Batch: {row['batch_similarity']:.1f}% → Retriever: {row['retriever_similarity']:.1f}% (+{row['difference']:.1f})")

print(f"\nQUESTIONS WHERE RETRIEVER DECLINED (-10 or worse):")
declined = results_df[results_df['difference'] <= -10].sort_values('difference')
for idx, row in declined.iterrows():
    print(f"  Q{idx+1}: {row['question'][:60]}...")
    print(f"    Batch: {row['batch_similarity']:.1f}% → Retriever: {row['retriever_similarity']:.1f}% ({row['difference']:.1f})")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
if abs(results_df['difference'].mean()) < 1:
    print("✓ Retriever MATCHES batch accuracy (within 1 point)")
elif results_df['difference'].mean() > 0:
    print(f"✓ Retriever performs BETTER than batch by {results_df['difference'].mean():.2f} points")
else:
    print(f"✗ Retriever performs worse than batch by {abs(results_df['difference'].mean()):.2f} points")

# Save detailed results
results_df.to_csv('data/processed/retriever_vs_batch_comparison.csv', index=False)
print("\nDetailed results saved to: data/processed/retriever_vs_batch_comparison.csv")
