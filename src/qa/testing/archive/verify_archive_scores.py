import pandas as pd
from rapidfuzz import fuzz
import glob

print("=== CHECKING SCORE ACCURACY IN ARCHIVE FILES ===\n")

# Check a few archive files
archive_files = [
    'data/processed/archive/qa_results_2025-12-26_12-51-00_22s.csv',  # The "baseline"
    'data/processed/archive/qa_results_2025-12-26_18-13-15_21s.csv',  # Recent run
    'data/processed/archive/qa_results_2025-12-26_18-17-03_20s.csv',  # Latest run
]

for filepath in archive_files:
    try:
        df = pd.read_csv(filepath)
        filename = filepath.split('/')[-1]
        
        # Check first question
        gt = str(df.iloc[1]['ground_truth'])
        pred = str(df.iloc[1]['predicted'])
        stored_score = df.iloc[1]['score']
        calc_score = fuzz.token_set_ratio(gt, pred) / 100.0
        
        mismatch = abs(stored_score - calc_score) > 0.01
        status = "❌ INCORRECT" if mismatch else "✅ CORRECT"
        
        print(f"{status} - {filename}")
        print(f"  Q2 stored: {stored_score:.4f}, calculated: {calc_score:.4f}")
        print(f"  Mean stored: {df['score'].mean():.4f}")
        print()
    except Exception as e:
        print(f"Error reading {filepath}: {e}\n")
