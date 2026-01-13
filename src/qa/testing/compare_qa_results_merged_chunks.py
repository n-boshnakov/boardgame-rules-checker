import pandas as pd

# Load both results
pdf = pd.read_csv('data/processed/archive/qa_results_2026-01-13_18-32-57_543s.csv')
merged = pd.read_csv('data/processed/archive/qa_results_2026-01-13_23-07-01_569s.csv')

print('='*70)
print('PDF-ONLY CHUNKS (with semantic analysis)')
print('='*70)
print(f'Passing (>=0.8): {(pdf["overall_score"] >= 0.8).sum()}/40 ({(pdf["overall_score"] >= 0.8).sum()/40*100:.1f}%)')
print(f'Mean overall: {pdf["overall_score"].mean():.2%}')
print(f'  - Relevance:    {pdf["relevance_score"].mean():.2%}')
print(f'  - Completeness: {pdf["completeness_score"].mean():.2%}')
print(f'  - Accuracy:     {pdf["accuracy_score"].mean():.2%}')
print(f'  - Conciseness:  {pdf["conciseness_score"].mean():.2%}')

print('\n' + '='*70)
print('MERGED CHUNKS (PDF + OCR, with semantic analysis)')
print('='*70)
print(f'Passing (>=0.8): {(merged["overall_score"] >= 0.8).sum()}/40 ({(merged["overall_score"] >= 0.8).sum()/40*100:.1f}%)')
print(f'Mean overall: {merged["overall_score"].mean():.2%}')
print(f'  - Relevance:    {merged["relevance_score"].mean():.2%}')
print(f'  - Completeness: {merged["completeness_score"].mean():.2%}')
print(f'  - Accuracy:     {merged["accuracy_score"].mean():.2%}')
print(f'  - Conciseness:  {merged["conciseness_score"].mean():.2%}')

print('\n' + '='*70)
print('IMPROVEMENT (Merged - PDF-only)')
print('='*70)
pass_diff = (merged["overall_score"] >= 0.8).sum() - (pdf["overall_score"] >= 0.8).sum()
print(f'Passing questions: {pass_diff:+d} ({pass_diff/40*100:+.1f}%)')
print(f'Overall score:     {(merged["overall_score"].mean() - pdf["overall_score"].mean()):.2%}')
print(f'  - Relevance:     {(merged["relevance_score"].mean() - pdf["relevance_score"].mean()):.2%}')
print(f'  - Completeness:  {(merged["completeness_score"].mean() - pdf["completeness_score"].mean()):.2%}')
print(f'  - Accuracy:      {(merged["accuracy_score"].mean() - pdf["accuracy_score"].mean()):.2%}')
print(f'  - Conciseness:   {(merged["conciseness_score"].mean() - pdf["conciseness_score"].mean()):.2%}')

# Question-by-question comparison
print('\n' + '='*70)
print('QUESTION-BY-QUESTION COMPARISON')
print('='*70)

improved = 0
worsened = 0
unchanged = 0

for i in range(len(pdf)):
    pdf_score = pdf.iloc[i]['overall_score']
    merged_score = merged.iloc[i]['overall_score']
    diff = merged_score - pdf_score
    
    if diff > 0.05:
        improved += 1
    elif diff < -0.05:
        worsened += 1
    else:
        unchanged += 1

print(f'Questions improved with merged chunks: {improved}')
print(f'Questions worsened with merged chunks: {worsened}')
print(f'Questions unchanged: {unchanged}')

# Show biggest improvements
print('\n' + '='*70)
print('TOP 5 IMPROVEMENTS (Merged better than PDF-only)')
print('='*70)

comparison = pd.DataFrame({
    'question': pdf['question'],
    'pdf_score': pdf['overall_score'],
    'merged_score': merged['overall_score'],
    'improvement': merged['overall_score'] - pdf['overall_score']
})

top_improvements = comparison.nlargest(5, 'improvement')
for idx, row in top_improvements.iterrows():
    print(f"\n{row['question']}")
    print(f"  PDF-only: {row['pdf_score']:.2%} | Merged: {row['merged_score']:.2%} | Δ: {row['improvement']:+.2%}")

# Show biggest regressions
print('\n' + '='*70)
print('TOP 5 REGRESSIONS (PDF-only better than Merged)')
print('='*70)

top_regressions = comparison.nsmallest(5, 'improvement')
for idx, row in top_regressions.iterrows():
    print(f"\n{row['question']}")
    print(f"  PDF-only: {row['pdf_score']:.2%} | Merged: {row['merged_score']:.2%} | Δ: {row['improvement']:+.2%}")

print('\n' + '='*70)
