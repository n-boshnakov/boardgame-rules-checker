#!/usr/bin/env python3
"""
Comparison Script: Baseline vs Semantic Search Implementation

This script demonstrates the improvements achieved by implementing semantic search
with multi-dimensional scoring and game-specific vocabulary mapping.

It runs both approaches on the same questions and compares:
- Overall quality scores
- Individual dimension scores (relevance, completeness, accuracy, conciseness)
- Pass rate (questions scoring ≥80%)
- Per-question improvements/regressions

Usage:
    python compare_semantic_vs_baseline.py [--questions N] [--visualize]
    
    --questions N: Number of questions to test (default: 10, max: 40)
    --visualize: Generate comparison visualizations
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from search.retriever import RulebookRetriever
from qa.multi_dimensional_scorer import MultiDimensionalScorer


def load_ground_truth(limit=None):
    """Load ground truth questions and answers."""
    gt_path = 'data/processed/qa_ground_truth_extractive.csv'
    df = pd.read_csv(gt_path)
    
    if limit:
        df = df.head(limit)
    
    print(f"Loaded {len(df)} questions from ground truth")
    return df


def run_qa_evaluation(use_semantic_analysis, questions_df, label):
    """
    Run QA evaluation with or without semantic analysis.
    
    Args:
        use_semantic_analysis: Boolean flag for semantic analysis
        questions_df: DataFrame with questions and ground truth
        label: Label for progress bar
    
    Returns:
        List of result dictionaries
    """
    print(f"\n{'='*70}")
    print(f"Running: {label}")
    print(f"{'='*70}")
    
    # Initialize retriever
    retriever = RulebookRetriever(
        use_semantic_analysis=use_semantic_analysis,
        use_reranker=True
    )
    
    # Initialize scorer
    scorer = MultiDimensionalScorer()
    
    results = []
    
    for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc=label):
        question = row['question']
        ground_truth = row['answer']
        
        try:
            # Retrieve and generate answer
            chunks = retriever.search(question, top_k=5)
            predicted = retriever.generate_answer(question, chunks)
            
            # Score the answer
            score_result = scorer.score_answer(question, predicted, ground_truth)
            
            # Store result
            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'overall_score': score_result['overall'],
                'relevance_score': score_result['relevance'],
                'completeness_score': score_result['completeness'],
                'accuracy_score': score_result['accuracy'],
                'conciseness_score': score_result['conciseness'],
                'question_type': score_result['question_type'],
                'top_chunk_score': chunks[0]['score'] if chunks else 0.0
            })
            
        except Exception as e:
            print(f"\nError on question {idx}: {e}")
            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'predicted': f"ERROR: {str(e)}",
                'overall_score': 0.0,
                'relevance_score': 0.0,
                'completeness_score': 0.0,
                'accuracy_score': 0.0,
                'conciseness_score': 0.0,
                'question_type': 'unknown',
                'top_chunk_score': 0.0
            })
    
    return results


def calculate_metrics(results_df):
    """Calculate summary metrics from results."""
    metrics = {
        'overall_mean': results_df['overall_score'].mean(),
        'relevance_mean': results_df['relevance_score'].mean(),
        'completeness_mean': results_df['completeness_score'].mean(),
        'accuracy_mean': results_df['accuracy_score'].mean(),
        'conciseness_mean': results_df['conciseness_score'].mean(),
        'passing_count': (results_df['overall_score'] >= 0.8).sum(),
        'passing_rate': (results_df['overall_score'] >= 0.8).mean(),
        'total_questions': len(results_df)
    }
    return metrics


def compare_results(baseline_df, semantic_df):
    """
    Compare baseline vs semantic results.
    
    Returns detailed comparison dictionary.
    """
    comparison = {
        'baseline': calculate_metrics(baseline_df),
        'semantic': calculate_metrics(semantic_df),
        'improvements': {},
        'question_level': []
    }
    
    # Calculate improvements
    base_metrics = comparison['baseline']
    sem_metrics = comparison['semantic']
    
    comparison['improvements'] = {
        'overall': (sem_metrics['overall_mean'] - base_metrics['overall_mean']) * 100,
        'relevance': (sem_metrics['relevance_mean'] - base_metrics['relevance_mean']) * 100,
        'completeness': (sem_metrics['completeness_mean'] - base_metrics['completeness_mean']) * 100,
        'accuracy': (sem_metrics['accuracy_mean'] - base_metrics['accuracy_mean']) * 100,
        'conciseness': (sem_metrics['conciseness_mean'] - base_metrics['conciseness_mean']) * 100,
        'passing_count': sem_metrics['passing_count'] - base_metrics['passing_count'],
        'passing_rate': (sem_metrics['passing_rate'] - base_metrics['passing_rate']) * 100
    }
    
    # Question-level comparison
    for idx in range(len(baseline_df)):
        base_row = baseline_df.iloc[idx]
        sem_row = semantic_df.iloc[idx]
        
        improvement = (sem_row['overall_score'] - base_row['overall_score']) * 100
        
        comparison['question_level'].append({
            'question': base_row['question'],
            'baseline_score': base_row['overall_score'] * 100,
            'semantic_score': sem_row['overall_score'] * 100,
            'improvement': improvement,
            'question_type': sem_row['question_type']
        })
    
    return comparison


def print_comparison_report(comparison):
    """Print detailed comparison report to console."""
    print("\n" + "="*70)
    print("COMPARISON REPORT: Baseline vs Semantic Search")
    print("="*70)
    
    base = comparison['baseline']
    sem = comparison['semantic']
    imp = comparison['improvements']
    
    print("\n📊 OVERALL METRICS")
    print("-"*70)
    print(f"{'Metric':<25} {'Baseline':<15} {'Semantic':<15} {'Change':<15}")
    print("-"*70)
    print(f"{'Overall Score':<25} {base['overall_mean']:>6.2%}         {sem['overall_mean']:>6.2%}         {imp['overall']:>+6.2f}%")
    print(f"{'  Relevance (35%)':<25} {base['relevance_mean']:>6.2%}         {sem['relevance_mean']:>6.2%}         {imp['relevance']:>+6.2f}%")
    print(f"{'  Completeness (30%)':<25} {base['completeness_mean']:>6.2%}         {sem['completeness_mean']:>6.2%}         {imp['completeness']:>+6.2f}%")
    print(f"{'  Accuracy (25%)':<25} {base['accuracy_mean']:>6.2%}         {sem['accuracy_mean']:>6.2%}         {imp['accuracy']:>+6.2f}%")
    print(f"{'  Conciseness (10%)':<25} {base['conciseness_mean']:>6.2%}         {sem['conciseness_mean']:>6.2%}         {imp['conciseness']:>+6.2f}%")
    print("-"*70)
    print(f"{'Passing (≥80%)':<25} {base['passing_count']:>3}/{base['total_questions']:<8} {sem['passing_count']:>3}/{sem['total_questions']:<8} {imp['passing_count']:>+3} ({imp['passing_rate']:>+5.1f}%)")
    print("-"*70)
    
    # Question type breakdown
    ql = comparison['question_level']
    improved = sum(1 for q in ql if q['improvement'] > 0)
    regressed = sum(1 for q in ql if q['improvement'] < 0)
    unchanged = sum(1 for q in ql if q['improvement'] == 0)
    
    print("\n📈 QUESTION-LEVEL IMPACT")
    print("-"*70)
    print(f"  Improved:   {improved:>3} questions ({improved/len(ql)*100:>5.1f}%)")
    print(f"  Regressed:  {regressed:>3} questions ({regressed/len(ql)*100:>5.1f}%)")
    print(f"  Unchanged:  {unchanged:>3} questions ({unchanged/len(ql)*100:>5.1f}%)")
    
    # Top improvements
    print("\n🎯 TOP 5 IMPROVEMENTS")
    print("-"*70)
    sorted_ql = sorted(ql, key=lambda x: x['improvement'], reverse=True)
    for i, q in enumerate(sorted_ql[:5], 1):
        print(f"{i}. [{q['question_type']}] {q['improvement']:>+6.2f}%")
        print(f"   Q: {q['question'][:60]}...")
        print(f"   Baseline: {q['baseline_score']:.1f}% → Semantic: {q['semantic_score']:.1f}%")
        print()
    
    # Top regressions
    if regressed > 0:
        print("⚠️  TOP 3 REGRESSIONS")
        print("-"*70)
        for i, q in enumerate(sorted_ql[-3:], 1):
            if q['improvement'] < 0:
                print(f"{i}. [{q['question_type']}] {q['improvement']:>+6.2f}%")
                print(f"   Q: {q['question'][:60]}...")
                print(f"   Baseline: {q['baseline_score']:.1f}% → Semantic: {q['semantic_score']:.1f}%")
                print()
    
    print("="*70)
    
    # Summary verdict
    print("\n🔍 SUMMARY")
    print("-"*70)
    if imp['overall'] > 0:
        print(f"✓ Semantic search improves overall score by {imp['overall']:+.2f}%")
    else:
        print(f"✗ Semantic search decreases overall score by {imp['overall']:+.2f}%")
    
    if imp['passing_count'] > 0:
        pct_increase = (imp['passing_count'] / base['passing_count']) * 100 if base['passing_count'] > 0 else float('inf')
        print(f"✓ {imp['passing_count']:+d} more questions pass (≥80%) - {pct_increase:+.0f}% increase")
    elif imp['passing_count'] < 0:
        print(f"✗ {abs(imp['passing_count'])} fewer questions pass (≥80%)")
    
    print(f"\nMost improved dimension: ", end="")
    dim_improvements = {
        'Relevance': imp['relevance'],
        'Completeness': imp['completeness'],
        'Accuracy': imp['accuracy'],
        'Conciseness': imp['conciseness']
    }
    best_dim = max(dim_improvements.items(), key=lambda x: x[1])
    print(f"{best_dim[0]} ({best_dim[1]:+.2f}%)")
    
    print("\n" + "="*70)


def visualize_comparison(comparison, output_path='comparison_results.png'):
    """Generate visualization comparing baseline vs semantic."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        base = comparison['baseline']
        sem = comparison['semantic']
        
        # 1. Dimension comparison bar chart
        ax = axes[0, 0]
        dimensions = ['Relevance', 'Completeness', 'Accuracy', 'Conciseness', 'Overall']
        baseline_values = [
            base['relevance_mean'] * 100,
            base['completeness_mean'] * 100,
            base['accuracy_mean'] * 100,
            base['conciseness_mean'] * 100,
            base['overall_mean'] * 100
        ]
        semantic_values = [
            sem['relevance_mean'] * 100,
            sem['completeness_mean'] * 100,
            sem['accuracy_mean'] * 100,
            sem['conciseness_mean'] * 100,
            sem['overall_mean'] * 100
        ]
        
        x = np.arange(len(dimensions))
        width = 0.35
        
        ax.bar(x - width/2, baseline_values, width, label='Baseline', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, semantic_values, width, label='Semantic', color='coral', alpha=0.8)
        ax.set_ylabel('Score (%)')
        ax.set_title('Dimension Scores: Baseline vs Semantic')
        ax.set_xticks(x)
        ax.set_xticklabels(dimensions, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
        
        # 2. Pass rate comparison
        ax = axes[0, 1]
        pass_data = [
            [base['passing_count'], base['total_questions'] - base['passing_count']],
            [sem['passing_count'], sem['total_questions'] - sem['passing_count']]
        ]
        colors = ['#2ecc71', '#e74c3c']
        
        x_pos = [0, 1]
        bottom = [0, 0]
        for i, color in enumerate(colors):
            values = [pass_data[j][i] for j in range(2)]
            ax.bar(x_pos, values, bottom=bottom, color=color, alpha=0.8)
            bottom = [bottom[j] + values[j] for j in range(2)]
        
        ax.set_ylabel('Number of Questions')
        ax.set_title('Pass Rate Comparison (≥80%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Baseline', 'Semantic'])
        ax.legend(['Passing', 'Failing'], loc='upper right')
        
        # Add counts on bars
        for i, (passing, total) in enumerate(pass_data):
            ax.text(i, passing/2, f'{passing}', ha='center', va='center', fontweight='bold', color='white')
            ax.text(i, passing + (total-passing)/2, f'{total-passing}', ha='center', va='center', fontweight='bold', color='white')
        
        # 3. Question-level improvement distribution
        ax = axes[1, 0]
        improvements = [q['improvement'] for q in comparison['question_level']]
        ax.hist(improvements, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
        ax.set_xlabel('Improvement (%)')
        ax.set_ylabel('Number of Questions')
        ax.set_title('Distribution of Question-Level Improvements')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Improvement by question type
        ax = axes[1, 1]
        ql = comparison['question_level']
        type_improvements = {}
        for q in ql:
            qtype = q['question_type']
            if qtype not in type_improvements:
                type_improvements[qtype] = []
            type_improvements[qtype].append(q['improvement'])
        
        types = list(type_improvements.keys())
        avg_improvements = [np.mean(type_improvements[t]) for t in types]
        colors_imp = ['coral' if x > 0 else 'steelblue' for x in avg_improvements]
        
        ax.barh(types, avg_improvements, color=colors_imp, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Average Improvement (%)')
        ax.set_title('Improvement by Question Type')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {output_path}")
        
    except ImportError:
        print("\n⚠️  matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"\n⚠️  Visualization error: {e}")


def save_results(baseline_df, semantic_df, comparison, output_dir='comparison_output'):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save dataframes
    baseline_df.to_csv(f'{output_dir}/baseline_results_{timestamp}.csv', index=False)
    semantic_df.to_csv(f'{output_dir}/semantic_results_{timestamp}.csv', index=False)
    
    # Save comparison JSON
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    comparison_serializable = json.loads(
        json.dumps(comparison, default=convert_types)
    )
    
    with open(f'{output_dir}/comparison_{timestamp}.json', 'w') as f:
        json.dump(comparison_serializable, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir}/")
    print(f"   - baseline_results_{timestamp}.csv")
    print(f"   - semantic_results_{timestamp}.csv")
    print(f"   - comparison_{timestamp}.json")


def main():
    parser = argparse.ArgumentParser(
        description='Compare baseline vs semantic search implementation'
    )
    parser.add_argument(
        '--questions', '-n',
        type=int,
        default=10,
        help='Number of questions to test (default: 10, max: 40)'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate comparison visualizations'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='comparison_output',
        help='Output directory for results (default: comparison_output)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("SEMANTIC SEARCH COMPARISON SCRIPT")
    print("="*70)
    print(f"Testing with {args.questions} questions")
    print(f"Visualization: {'Enabled' if args.visualize else 'Disabled'}")
    print("="*70)
    
    # Load questions
    questions_df = load_ground_truth(limit=args.questions)
    
    # Run baseline
    baseline_results = run_qa_evaluation(
        use_semantic_analysis=False,
        questions_df=questions_df,
        label="Baseline (No Semantic Analysis)"
    )
    baseline_df = pd.DataFrame(baseline_results)
    
    # Run semantic
    semantic_results = run_qa_evaluation(
        use_semantic_analysis=True,
        questions_df=questions_df,
        label="Semantic Search (Enhanced)"
    )
    semantic_df = pd.DataFrame(semantic_results)
    
    # Compare results
    comparison = compare_results(baseline_df, semantic_df)
    
    # Print report
    print_comparison_report(comparison)
    
    # Save results
    save_results(baseline_df, semantic_df, comparison, args.output_dir)
    
    # Visualize if requested
    if args.visualize:
        output_path = f'{args.output_dir}/comparison_visualization.png'
        visualize_comparison(comparison, output_path)
    
    print("\n✅ Comparison complete!")
    
    return comparison


if __name__ == '__main__':
    main()
