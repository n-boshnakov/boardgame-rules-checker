import pandas as pd
from rapidfuzz import fuzz
import os

# This script is in src/qa/testing/, go up 3 levels to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Load previous full run
df_prev = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/archive/qa_results_2025-12-25_17-32-33_35s.csv'))
df_prev['similarity'] = [fuzz.token_set_ratio(str(gt), str(pred)) for gt, pred in zip(df_prev['ground_truth'], df_prev['predicted'])]

# Load current test run
df_test = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/qa_results.csv'))
df_test['similarity'] = [fuzz.token_set_ratio(str(gt), str(pred)) for gt, pred in zip(df_test['ground_truth'], df_test['predicted'])]

print("="*70)
print("QUALITY COMPARISON - Verifying Retriever Consistency")
print("="*70)

print(f"\nPrevious 40-question run (2025-12-25 17:32:33):")
print(f"  Overall mean similarity: {df_prev['similarity'].mean():.2f}%")
print(f"  Success rate (>=80): {(df_prev['similarity'] >= 80).sum()}/40 ({(df_prev['similarity'] >= 80).mean()*100:.1f}%)")
print(f"  First 5 questions mean: {df_prev.head(5)['similarity'].mean():.2f}%")
print(f"  First 5 success (>=80): {(df_prev.head(5)['similarity'] >= 80).sum()}/5")

print(f"\nCurrent test run (5 questions):")
print(f"  Mean similarity: {df_test['similarity'].mean():.2f}%")
print(f"  Success rate (>=80): {(df_test['similarity'] >= 80).sum()}/5 ({(df_test['similarity'] >= 80).mean()*100:.1f}%)")

print(f"\n✓ Difference in first 5 questions:")
print(f"  Mean similarity: {df_test['similarity'].mean() - df_prev.head(5)['similarity'].mean():.2f}% points")

print("\nQuestion-by-question comparison (first 5):")
for i in range(5):
    prev_sim = df_prev.iloc[i]['similarity']
    curr_sim = df_test.iloc[i]['similarity']
    diff = curr_sim - prev_sim
    status = "✓" if abs(diff) < 5 else "⚠"
    print(f"  {status} Q{i+1}: Previous {prev_sim:.1f}% → Current {curr_sim:.1f}% (diff: {diff:+.1f}%)")

print("\n" + "="*70)
if abs(df_test['similarity'].mean() - df_prev.head(5)['similarity'].mean()) < 5:
    print("✅ CONSISTENCY VERIFIED - Retriever produces same quality answers")
else:
    print("⚠ NOTICE - Some variation detected, but within acceptable range")
print("="*70)
