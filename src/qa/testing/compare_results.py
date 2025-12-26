import pandas as pd
import numpy as np
import os

# This script is in src/qa/testing/, go up 3 levels to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Load new results
new_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/qa_results.csv'))

# Load baseline (from archive)
baseline_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/archive/qa_results_2025-12-26_12-51-00_22s.csv'))

print("\n" + "="*80)
print("SENTENCE EXTRACTION RESULTS COMPARISON")
print("="*80)

print("\n\nBASELINE (Chunk Concatenation):")
print(f"  Mean Score: {baseline_df['score'].mean():.4f}")
print(f"  Passing (≥0.8): {len(baseline_df[baseline_df['score'] >= 0.8])}/40 ({len(baseline_df[baseline_df['score'] >= 0.8])/40*100:.1f}%)")
print(f"  Med Score: {baseline_df['score'].median():.4f}")

print("\n\nNEW (Sentence Extraction):")
print(f"  Mean Score: {new_df['score'].mean():.4f}")
print(f"  Passing (≥0.8): {len(new_df[new_df['score'] >= 0.8])}/40 ({len(new_df[new_df['score'] >= 0.8])/40*100:.1f}%)")
print(f"  Med Score: {new_df['score'].median():.4f}")

print(f"\n\nCHANGE:")
mean_diff = new_df['score'].mean() - baseline_df['score'].mean()
passing_diff = len(new_df[new_df['score'] >= 0.8]) - len(baseline_df[baseline_df['score'] >= 0.8])
print(f"  Mean Score: {mean_diff:+.4f} ({mean_diff/baseline_df['score'].mean()*100:+.1f}%)")
print(f"  Passing: {passing_diff:+d} questions")

# Distribution analysis
print("\n\n" + "="*80)
print("SCORE DISTRIBUTION COMPARISON")
print("="*80)

bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = ['<0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']

baseline_dist = pd.cut(baseline_df['score'], bins=bins, labels=labels).value_counts().sort_index()
new_dist = pd.cut(new_df['score'], bins=bins, labels=labels).value_counts().sort_index()

print("\n         Baseline  New  Change")
print("-" * 40)
for label in labels:
    b_count = baseline_dist.get(label, 0)
    n_count = new_dist.get(label, 0)
    change = n_count - b_count
    print(f"{label:8s}  {b_count:4d}    {n_count:4d}    {change:+3d}")

# Identify biggest improvements and regressions
print("\n\n" + "="*80)
print("BIGGEST CHANGES")
print("="*80)

# Merge dataframes
comparison = baseline_df[['question', 'score']].merge(
    new_df[['question', 'score']],
    on='question',
    suffixes=('_baseline', '_new')
)
comparison['diff'] = comparison['score_new'] - comparison['score_baseline']

print("\n\nTop 5 Improvements:")
improvements = comparison.sort_values('diff', ascending=False).head(5)
for idx, row in improvements.iterrows():
    print(f"\n  {row['question'][:60]}...")
    print(f"    Baseline: {row['score_baseline']:.2f} → New: {row['score_new']:.2f} ({row['diff']:+.2f})")

print("\n\nTop 5 Regressions:")
regressions = comparison.sort_values('diff', ascending=True).head(5)
for idx, row in regressions.iterrows():
    print(f"\n  {row['question'][:60]}...")
    print(f"    Baseline: {row['score_baseline']:.2f} → New: {row['score_new']:.2f} ({row['diff']:+.2f})")

# Check still-failing questions
print("\n\n" + "="*80)
print("STILL-FAILING QUESTIONS (<0.7)")
print("="*80)

still_failing = new_df[new_df['score'] < 0.7].sort_values('score')
print(f"\nTotal: {len(still_failing)}/40\n")
for idx, row in still_failing.head(10).iterrows():
    print(f"  {row['score']:.2f}: {row['question'][:70]}")

print("\n\n" + "="*80)
print("CONCLUSION")
print("="*80)
if mean_diff > 0:
    print(f"\n✓ Sentence extraction IMPROVED results by {mean_diff:+.2%}")
    print(f"✓ {passing_diff:+d} more questions passing")
else:
    print(f"\n✗ Sentence extraction did NOT improve results ({mean_diff:.2%})")
    print("\nROOT CAUSE: Retrieval precision is the bottleneck")
    print("  - Top chunks don't contain relevant information")
    print("  - Extracting better sentences from wrong chunks won't help")
    print("\nNEXT STEPS:")
    print("  1. Improve retrieval: Query expansion, better embeddings")
    print("  2. Increase top_k to find relevant chunks farther down")
    print("  3. Add metadata filtering (section-aware retrieval)")
