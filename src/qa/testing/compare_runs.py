import pandas as pd
import glob
import os

# This script is in src/qa/testing/, go up 3 levels to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

files = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'data/processed/archive/qa_results_2025-12-25*.csv')), reverse=True)[:15]

print("Recent test results:")
for f in files:
    df = pd.read_csv(f)
    name = f.split('\\')[-1]
    mean = df['score'].mean()
    passing = (df['score'] >= 0.8).sum()
    avg_len = df['predicted'].str.len().mean()
    print(f'{name}: {mean:.4f} ({mean*100:.1f}%), {passing}/20 pass, {avg_len:.0f} chars')
