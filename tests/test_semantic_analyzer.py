"""
Test script for the Semantic Analyzer (NLTK-based).
Demonstrates morphological analysis, POS tagging, lemmatization, and game vocabulary mapping.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from search.semantic_analyzer_nltk import SemanticAnalyzerNLTK


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def print_analysis(question: str, analyzer: SemanticAnalyzerNLTK):
    """Print detailed analysis of a question."""
    print(f"Question: {question}")
    print(f"{'-'*70}")
    
    analysis = analyzer.analyze(question)
    intent = analyzer.get_question_intent(question)
    
    # Question Type
    print(f"\nQuestion Type: {intent['question_type']}")
    
    # Question Intent Flags
    print(f"\nQuestion Intent:")
    needs = []
    if intent['needs_procedural']:
        needs.append("Procedural (how-to)")
    if intent['needs_definition']:
        needs.append("Definition (what-is)")
    if intent['needs_quantitative']:
        needs.append("Quantitative (how-many)")
    if intent['needs_temporal']:
        needs.append("Temporal (when)")
    if intent['needs_locational']:
        needs.append("Locational (where)")
    if intent['needs_permission']:
        needs.append("Permission (can/may)")
    print(f"   {', '.join(needs) if needs else 'General'}")
    
    # Linguistic Analysis
    print(f"\nLinguistic Analysis:")
    print(f"   Action Verbs: {', '.join(analysis['action_verbs'][:5]) if analysis['action_verbs'] else 'None'}")
    print(f"   Key Nouns: {', '.join(analysis['key_nouns'][:5]) if analysis['key_nouns'] else 'None'}")
    print(f"   Lemmas: {', '.join(analysis['lemmas'][:8]) if analysis['lemmas'] else 'None'}")
    
    # Game Concepts
    print(f"\nGame Concepts Identified:")
    if analysis['game_concepts']:
        for concept in analysis['game_concepts']:
            print(f"   - {concept}")
    else:
        print(f"   None")
    
    # Query Enhancement
    enhanced = analyzer.enhance_query(question, max_additions=2)
    if enhanced != question:
        addition = enhanced.replace(question, "").strip()
        print(f"\nQuery Enhancement:")
        print(f"   Original:  {question}")
        print(f"   Enhanced:  {enhanced}")
        print(f"   Added:     [{addition}]")
    else:
        print(f"\nQuery Enhancement: No additions needed")


def main():
    """Run semantic analyzer tests."""
    print_section("NLTK-Based Semantic Analyzer Test Suite")
    
    # Initialize analyzer
    print("Initializing SemanticAnalyzerNLTK...")
    analyzer = SemanticAnalyzerNLTK()
    print("[OK] Analyzer loaded successfully\n")
    
    # Test questions covering different types and game concepts
    test_questions = [
        # Procedural questions
        "How do I move my stalker?",
        "What are the steps to resolve combat?",
        
        # Definitional questions
        "What happens if a player dies?",
        "What does line of sight mean?",
        
        # Quantitative questions
        "How many cards can I draw per turn?",
        "How much damage does a shotgun deal?",
        
        # Permission questions
        "Can I attack multiple enemies?",
        "Is it possible to move through occupied spaces?",
        
        # Temporal questions
        "When does the mission end?",
        "When do I draw mission cards?",
        
        # Locational questions
        "Where do I place anomaly tokens?",
        "Where is the base located?",
        
        # Equipment/Combat questions
        "What equipment do I need for combat?",
        "How does the skill test work?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'-'*70}")
        print(f"TEST {i}/{len(test_questions)}")
        print_analysis(question, analyzer)
    
    # Summary
    print_section("TEST SUMMARY")
    print(f"[OK] Tested {len(test_questions)} questions")
    print(f"[OK] All analyses completed successfully")
    print("\nDemonstrated Features:")
    print("   * Morphological Analysis (lemmatization)")
    print("   * POS Tagging (part-of-speech)")
    print("   * Question Type Classification (6 types)")
    print("   * Intent Recognition (procedural, definitional, etc.)")
    print("   * Game Vocabulary Mapping (15 concept categories)")
    print("   * Query Enhancement (dynamic synonym expansion)")
    print("\nThese features improve:")
    print("   * Retrieval accuracy through better query understanding")
    print("   * Answer relevance through intent-aware strategies")
    print("   * Search coverage through game-specific synonym expansion")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nMake sure NLTK data is installed:")
        print("   python -c \"import nltk; nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet'])\"")
        sys.exit(1)
