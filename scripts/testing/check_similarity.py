import pandas as pd
from rapidfuzz import fuzz

df = pd.read_csv('data/processed/qa_results.csv')

# Compute actual similarity scores
sims = [fuzz.token_set_ratio(str(gt), str(pred)) for gt, pred in zip(df['ground_truth'], df['predicted'])]

print(f'Computed similarity (token_set_ratio): {sum(sims)/len(sims):.2f}')
print(f'Passing (>=80): {sum(1 for s in sims if s >= 80)}/40')
print(f'\nScore column mean: {df["score"].mean():.4f} (normalized 0-1)')
print(f'Score column as percentage: {df["score"].mean()*100:.2f}%')
print(f'Passing score (>=0.8): {(df["score"] >= 0.8).sum()}/40')

print(f'\nFirst 5 rows:')
for i in range(5):
    print(f'Q{i+1}: computed_sim={sims[i]:.1f}, csv_score={df.iloc[i]["score"]:.4f}')
