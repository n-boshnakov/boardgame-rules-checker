import pandas as pd
from rapidfuzz import fuzz

df_old = pd.read_csv('data/processed/archive/qa_results_2025-12-24_16-14-38_74s.csv')
df_new = pd.read_csv('data/processed/qa_results.csv')

df_old['sim'] = [fuzz.token_set_ratio(str(gt), str(pred)) for gt, pred in zip(df_old['ground_truth'], df_old['predicted'])]
df_new['sim'] = [fuzz.token_set_ratio(str(gt), str(pred)) for gt, pred in zip(df_new['ground_truth'], df_new['predicted'])]

print('='*80)
print('COMPARISON: Before vs After Cross-Encoder + 8 Chunks')
print('='*80)
print(f"\nFirst 20 Questions:")
print(f"  Before: Mean={df_old.head(20)['sim'].mean():.2f}, ≥70%: {(df_old.head(20)['sim']>=70).sum()}/20")
print(f"  After:  Mean={df_new.head(20)['sim'].mean():.2f}, ≥70%: {(df_new.head(20)['sim']>=70).sum()}/20")
print(f"  Change: {df_new.head(20)['sim'].mean() - df_old.head(20)['sim'].mean():+.2f}")

print('\n' + '='*80)
print('Questions that got WORSE (>5 point drop):')
print('='*80)
for i in range(20):
    diff = df_new.iloc[i]['sim'] - df_old.iloc[i]['sim']
    if diff < -5:
        print(f"\nQ{i+1}: {df_old.iloc[i]['sim']:.1f}% -> {df_new.iloc[i]['sim']:.1f}% ({diff:+.1f})")
        print(f"  Question: {df_old.iloc[i]['question']}")
        print(f"  Old answer length: {len(str(df_old.iloc[i]['predicted']))} chars")
        print(f"  New answer length: {len(str(df_new.iloc[i]['predicted']))} chars")

print('\n' + '='*80)
print('Questions that got BETTER (>5 point gain):')
print('='*80)
for i in range(20):
    diff = df_new.iloc[i]['sim'] - df_old.iloc[i]['sim']
    if diff > 5:
        print(f"\nQ{i+1}: {df_old.iloc[i]['sim']:.1f}% -> {df_new.iloc[i]['sim']:.1f}% ({diff:+.1f})")
        print(f"  Question: {df_old.iloc[i]['question']}")
