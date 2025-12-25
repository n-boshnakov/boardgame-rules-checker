import pandas as pd

df = pd.read_csv('data/processed/qa_results.csv')

print('40-question results (800-char extractive):')
print(f'Mean score: {df["score"].mean():.4f} ({df["score"].mean()*100:.2f}%)')
print(f'Success (>=0.8): {(df["score"] >= 0.8).sum()}/40 ({(df["score"] >= 0.8).sum()/40*100:.1f}%)')
print(f'Avg answer length: {df["predicted"].str.len().mean():.0f} chars')
print(f'\nScore distribution:')
print(f'  >=0.9: {(df["score"] >= 0.9).sum()}')
print(f'  0.8-0.9: {((df["score"] >= 0.8) & (df["score"] < 0.9)).sum()}')
print(f'  0.7-0.8: {((df["score"] >= 0.7) & (df["score"] < 0.8)).sum()}')
print(f'  0.6-0.7: {((df["score"] >= 0.6) & (df["score"] < 0.7)).sum()}')
print(f'  <0.6: {(df["score"] < 0.6).sum()}')
