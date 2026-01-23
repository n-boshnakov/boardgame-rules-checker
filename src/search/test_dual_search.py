"""Test dual-source search (forum + rulebook)."""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from search.retriever import RulebookRetriever
import json

def test_dual_search():
    """Test searching both forum and rulebook."""
    
    # Initialize retriever
    print("Initializing retriever...")
    retriever = RulebookRetriever(use_reranker=True, use_semantic_analysis=False)
    
    # Test questions
    test_questions = [
        "Does the bandit shotgun -2 include the space the target is on?",  # Forum has exact answer
        "How many actions do you get per turn?",  # Rulebook likely has this
        "Can you play solo with one stalker?",  # Forum has detailed answer
        "What happens when you enter an anomaly?",  # Rulebook rule
        "How does enemy AI work with noise tokens?",  # Could be in both
    ]
    
    results = []
    
    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")
        
        result = retriever.search_dual_source(question, top_k=5, forum_weight=0.5)
        
        print(f"\n✓ Source: {result['source'].upper()}")
        print(f"  Reason: {result['reason']}")
        print(f"\n  Answer:\n  {result['answer'][:300]}...")
        
        if result['source'] == 'forum':
            print(f"\n  Forum Question: {result.get('question', 'N/A')[:100]}...")
            print(f"  Thread URL: {result.get('thread_url', 'N/A')}")
        
        results.append({
            'question': question,
            'source': result['source'],
            'confidence': result['confidence'],
            'forum_conf': result['forum_confidence'],
            'rulebook_conf': result['rulebook_confidence'],
            'answer_preview': result['answer'][:150]
        })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    for r in results:
        print(f"Q: {r['question'][:60]}...")
        print(f"   Source: {r['source']} | Forum: {r['forum_conf']:.3f} | Rulebook: {r['rulebook_conf']:.3f}")
        print()
    
    # Save results
    with open('data/processed/dual_search_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("Results saved to: data/processed/dual_search_test_results.json")


if __name__ == "__main__":
    test_dual_search()
