import pandas as pd
import json
import os

# This script is in src/qa/testing/, go up 3 levels to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Load Q&A data
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/qa_results.csv'))

# Load chunk data to examine content
with open(os.path.join(PROJECT_ROOT, 'data/processed/chunks_2025-12-23.json'), 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Create searchable index
chunk_texts = {i: chunk['text'] for i, chunk in enumerate(chunks)}

print("\n" + "="*80)
print("IDENTIFYING ROOT CAUSES OF ANSWER MISMATCHES")
print("="*80)

# Analyze low-scoring questions
low_scores = df[df['score'] < 0.7].sort_values('score')

print(f"\n\nTotal low-scoring questions (<0.7): {len(low_scores)}")
print(f"Total questions: {len(df)}")
print(f"Percentage: {len(low_scores)/len(df)*100:.1f}%")

# Pattern analysis
print("\n\n" + "="*80)
print("PATTERN ANALYSIS: Why do answers not match ground truth?")
print("="*80)

issues = {
    'retrieval_wrong_chunk': 0,
    'answer_too_verbose': 0,
    'answer_too_short': 0,
    'answer_wrong_focus': 0,
    'ground_truth_short_predicted_long': 0,
}

for idx, row in low_scores.head(15).iterrows():
    gt_len = len(row['ground_truth'])
    pred_len = len(row['predicted'])
    
    # Check if predicted answer is way longer than ground truth
    if gt_len < 200 and pred_len > 400:
        issues['ground_truth_short_predicted_long'] += 1
        
print("\n\nDETECTED ISSUES:")
for issue, count in issues.items():
    if count > 0:
        print(f"  {issue}: {count} cases")

# Examine specific cases
print("\n\n" + "="*80)
print("DETAILED ANALYSIS OF KEY FAILURES")
print("="*80)

# Case 1: Simple factual question
q1 = df[df['question'] == 'How many players can play this game?'].iloc[0]
print(f"\n1. SIMPLE FACTUAL QUESTION")
print(f"   Q: {q1['question']}")
print(f"   Ground Truth ({len(q1['ground_truth'])} chars): {q1['ground_truth']}")
print(f"   Predicted ({len(q1['predicted'])} chars): {q1['predicted'][:200]}...")
print(f"   Score: {q1['score']:.2f}")
print(f"\n   ISSUE: Ground truth is very short (42 chars), predicted is long (677 chars)")
print(f"   ROOT CAUSE: Extractive method just concatenates top chunks, doesn't extract precise answer")

# Case 2: Yes/No question
q2 = df[df['question'] == 'Can I pick up loot while in enemy LoS?'].iloc[0]
print(f"\n2. YES/NO QUESTION")
print(f"   Q: {q2['question']}")
print(f"   Ground Truth ({len(q2['ground_truth'])} chars): {q2['ground_truth']}")
print(f"   Predicted ({len(q2['predicted'])} chars): {q2['predicted'][:200]}...")
print(f"   Score: {q2['score']:.2f}")
print(f"\n   ISSUE: Ground truth is 54 chars ('Yes, you can...'), predicted is wrong chunk")
print(f"   ROOT CAUSE: Retrieved wrong chunk + no answer extraction logic")

# Case 3: Specific rule question
q3 = df[df['question'] == 'When do I increase my radiation?'].iloc[0]
print(f"\n3. SPECIFIC RULE QUESTION")
print(f"   Q: {q3['question']}")
print(f"   Ground Truth ({len(q3['ground_truth'])} chars): {q3['ground_truth'][:150]}...")
print(f"   Predicted ({len(q3['predicted'])} chars): {q3['predicted'][:150]}...")
print(f"   Score: {q3['score']:.2f}")
print(f"\n   ISSUE: Retrieved chunk talks about radiation but wrong context")
print(f"   ROOT CAUSE: Semantic similarity not precise enough + no answer extraction")

print("\n\n" + "="*80)
print("SUMMARY: TWO CORE PROBLEMS")
print("="*80)
print("""
1. RETRIEVAL PRECISION
   - Queries like "How many players" retrieve chunks about game phases, not player count
   - Semantic similarity matches keywords but not intent
   - Need: Query expansion or better query understanding

2. ANSWER EXTRACTION
   - Current: Concatenate first 800 chars from top 15 chunks
   - Problem: Doesn't extract the specific answer to the question
   - Examples:
     * "How many players?" needs "1-4 players" not full game phase description
     * "Can I pick up loot in LoS?" needs "Yes, you can" not enemy movement rules
     * "When do I increase radiation?" needs specific trigger, not general context
   - Need: Sentence-level extraction with semantic similarity to question
""")

print("\n\n" + "="*80)
print("PROPOSED SOLUTION")
print("="*80)
print("""
PHASE 1: Improve Answer Extraction (Will fix most issues)
----------
Current: _generate_answer_extractive() just concatenates chunks
Proposed: Extract most relevant SENTENCES from chunks

Algorithm:
1. Take top 10-15 chunks from retrieval
2. Split each chunk into sentences
3. Calculate semantic similarity of each sentence to the question
4. Select top 5-10 sentences by similarity
5. Order sentences by their original chunk rank (preserve context flow)
6. Concatenate up to 800 chars

Benefits:
- Extracts precise answers instead of full chunks
- "How many players?" would extract "This game can be played by 1-4 players"
- "Can I pick up loot in LoS?" would extract "Yes, you can pick up loot while in enemy Line of Sight"
- Maintains context coherence by respecting chunk order

PHASE 2: Query Expansion (If Phase 1 insufficient)
----------
- Build terminology synonym map (e.g., "player" = "Stalker")
- Expand short queries with domain terms
- Only needed if retrieval precision remains low
""")
