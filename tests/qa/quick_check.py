import pandas as pd
from rapidfuzz import fuzz

df = pd.read_csv('data/processed/qa_results.csv')
df['similarity'] = [fuzz.token_set_ratio(str(gt), str(pred)) 
                    for gt, pred in zip(df['ground_truth'], df['predicted'])]
df['correct'] = df['similarity'] >= 80

print(f"Total Questions: {len(df)}")
print(f"Correct Answers: {df['correct'].sum()}")
print(f"Success Rate: {df['correct'].mean()*100:.1f}%")
print(f"Mean Similarity: {df['similarity'].mean():.2f}")
print(f"Median Similarity: {df['similarity'].median():.2f}")
print(f"\nFirst 20 Questions:")
print(f"Correct: {df.head(20)['correct'].sum()}/20")
print(f"Mean Similarity: {df.head(20)['similarity'].mean():.2f}")
