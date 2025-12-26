import pandas as pd
from rapidfuzz import fuzz

# Load both files
baseline = pd.read_csv('data/processed/archive/qa_results_2025-12-26_12-51-00_22s.csv')
current = pd.read_csv('data/processed/qa_results.csv')

print("=== SCORING DEBUG ===\n")

for i in range(min(5, len(baseline))):
    q = baseline.iloc[i]['question']
    
    # Get data from both files
    gt_base = str(baseline.iloc[i]['ground_truth'])
    pred_base = str(baseline.iloc[i]['predicted'])
    score_base_stored = baseline.iloc[i]['score']
    
    gt_curr = str(current.iloc[i]['ground_truth'])
    pred_curr = str(current.iloc[i]['predicted'])
    score_curr_stored = current.iloc[i]['score']
    
    # Recalculate scores
    score_base_calc = fuzz.token_set_ratio(gt_base, pred_base) / 100.0
    score_curr_calc = fuzz.token_set_ratio(gt_curr, pred_curr) / 100.0
    
    print(f"Q{i+1}: {q}")
    print(f"  GT same? {gt_base == gt_curr}")
    print(f"  Pred same? {pred_base == pred_curr}")
    print(f"  Baseline: stored={score_base_stored:.4f}, calc={score_base_calc:.4f}")
    print(f"  Current:  stored={score_curr_stored:.4f}, calc={score_curr_calc:.4f}")
    
    if score_curr_stored != score_curr_calc:
        print(f"  ⚠️ MISMATCH: Stored score doesn't match recalculated score!")
    print()
