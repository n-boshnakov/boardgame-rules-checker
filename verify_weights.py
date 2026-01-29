"""Verify the current scorer weights."""
import sys
sys.path.append('c:/Users/I544554/Documents/GitHub/boardgame-rules-checker/src')

from qa.multi_dimensional_scorer import MultiDimensionalScorer

scorer = MultiDimensionalScorer()

print("="*80)
print("Current Scorer Weights Verification")
print("="*80)
print(f"\nWeights in MultiDimensionalScorer:")
for dim, weight in scorer.weights.items():
    print(f"  {dim:15s}: {weight:.0%}")

total = sum(scorer.weights.values())
print(f"\nTotal: {total:.2f} (should be 1.00)")

# Check if SentenceTransformer is loaded
print(f"\nSentenceTransformer loaded: {hasattr(scorer, 'sentence_transformer')}")
print(f"Model: {scorer.sentence_transformer if hasattr(scorer, 'sentence_transformer') else 'N/A'}")

print("\n" + "="*80)
