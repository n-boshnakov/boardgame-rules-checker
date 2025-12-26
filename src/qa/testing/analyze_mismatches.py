import pandas as pd
import os

# This script is in src/qa/testing/, go up 3 levels to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/qa_results.csv'))

print("\n=== SAMPLE MISMATCHES ===\n")
for idx, row in df[df['score'] < 0.7].head(5).iterrows():
    print(f"\nQ: {row['question']}")
    print(f"\nGROUND TRUTH:\n{row['ground_truth'][:300]}")
    print(f"\nPREDICTED:\n{row['predicted'][:300]}")
    print(f"\nScore: {row['score']:.2f}")
    print('-'*80)
