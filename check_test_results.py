import pandas as pd
from rapidfuzz import fuzz

df = pd.read_csv('data/processed/qa_results.csv')
df['similarity'] = [fuzz.token_set_ratio(str(gt), str(pred)) for gt, pred in zip(df['ground_truth'], df['predicted'])]

print(f'First 5 questions - Mean similarity: {df["similarity"].mean():.2f}%')
print(f'Success rate (>=80): {(df["similarity"] >= 80).sum()}/{len(df)}')
print('\nQuestion-by-question:')
for i, row in df.iterrows():
    print(f'{i+1}. Similarity: {row["similarity"]:.1f}%')
