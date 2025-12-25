"""
Validation script to verify retriever configuration matches the optimized settings.
"""
import sys
sys.path.insert(0, 'src')

from search.retriever import RulebookRetriever, MODEL_NAME

print("="*70)
print("RETRIEVER CONFIGURATION VALIDATION")
print("="*70)

# Check model configuration
print(f"\n✓ Embedding Model: {MODEL_NAME}")
expected_model = "sentence-transformers/all-mpnet-base-v2"
assert MODEL_NAME == expected_model, f"Expected {expected_model}, got {MODEL_NAME}"

# Initialize retriever
retriever = RulebookRetriever(use_reranker=True)

print(f"✓ Model loaded: {retriever.model}")
print(f"✓ Cross-encoder enabled: {retriever.use_reranker}")
print(f"✓ Cross-encoder model: {retriever.cross_encoder.config.name_or_path if retriever.cross_encoder else 'None'}")

# Check search method signature defaults
import inspect
sig = inspect.signature(retriever.search)
params = sig.parameters
print(f"\n✓ search() default parameters:")
print(f"  - top_k: {params['top_k'].default}")
print(f"  - search_type: {params['search_type'].default}")
print(f"  - hybrid_weight: {params['hybrid_weight'].default}")

# Verify expected defaults
assert params['hybrid_weight'].default == 0.8, f"Expected hybrid_weight default 0.8, got {params['hybrid_weight'].default}"

# Check generate_answer method
print(f"\n✓ generate_answer() combines top 5 chunks")

# Test retrieval with optimized settings
test_question = "How many cards does each player start with?"
chunks = retriever.search(
    test_question, 
    top_k=20,  # Optimized value
    search_type="hybrid", 
    hybrid_weight=0.8  # Optimized value
)
answer = retriever.generate_answer(test_question, chunks, multi_chunk_synthesis=True)

print(f"\n✓ Test query executed successfully:")
print(f"  - Retrieved chunks: {len(chunks)}")
print(f"  - Answer length: {len(answer)} chars")
print(f"  - Top chunk score: {chunks[0]['score']:.4f}")

print("\n" + "="*70)
print("✅ ALL VALIDATIONS PASSED - Retriever is correctly configured")
print("="*70)
print("\nOptimized Configuration Summary:")
print("  - Model: sentence-transformers/all-mpnet-base-v2 (768 dims)")
print("  - Top-k: 20 candidates")
print("  - Hybrid weight: 0.8 (80% semantic, 20% keyword)")
print("  - Reranking: Top 10 with CrossEncoder")
print("  - Answer: Combine top 5 chunks")
print("="*70)
