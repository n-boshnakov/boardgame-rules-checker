"""Analyze questions where predicted answers don't match ground truth.

Displays sample mismatches to help identify patterns in failures
and understand where the QA system struggles.
"""
import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RESULTS_FILE = os.path.join(PROJECT_ROOT, 'data/processed/qa_results.csv')

def analyze_mismatches(score_threshold=0.7, num_samples=5, max_text_length=300):
    """Analyze and display sample mismatches from QA results.
    
    Args:
        score_threshold: Show questions with scores below this (default: 0.7)
        num_samples: Number of sample mismatches to display (default: 5)
        max_text_length: Maximum characters to display per text field (default: 300)
    """
    df = pd.read_csv(RESULTS_FILE)
    mismatches = df[df['score'] < score_threshold]
    
    print(f"\n{'='*80}")
    print(f"SAMPLE MISMATCHES (Score < {score_threshold})")
    print(f"{'='*80}")
    print(f"\nTotal mismatches: {len(mismatches)}/{len(df)} ({len(mismatches)/len(df)*100:.1f}%)\n")
    
    for idx, row in mismatches.head(num_samples).iterrows():
        print(f"\nQuestion {idx + 1}: {row['question']}")
        print(f"\nGROUND TRUTH:")
        print(f"{str(row['ground_truth'])[:max_text_length]}")
        print(f"\nPREDICTED:")
        print(f"{str(row['predicted'])[:max_text_length]}")
        print(f"\nScore: {row['score']:.2%}")
        print('-' * 80)

if __name__ == "__main__":
    analyze_mismatches()
