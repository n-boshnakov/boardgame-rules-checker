"""
Re-process all forum Q&A pairs.
Creates backup of existing data before processing.
"""

import json
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from forum_qa_extractor import ForumQAExtractor


def analyze_qa_quality(qa_pairs: List[Dict]) -> Dict:
    """Analyze quality metrics of Q&A pairs."""
    if not qa_pairs:
        return {}
    
    total_length = 0
    total_mentions = 0
    over_500_count = 0
    with_mentions_count = 0
    
    for qa in qa_pairs:
        answer = qa.get('answer', '')
        length = len(answer)
        mentions = len(re.findall(r'@\w+', answer))
        
        total_length += length
        total_mentions += mentions
        
        if length > 500:
            over_500_count += 1
        if mentions > 0:
            with_mentions_count += 1
    
    n = len(qa_pairs)
    
    return {
        'total_pairs': n,
        'avg_answer_length': total_length / n,
        'total_mentions': total_mentions,
        'avg_mentions_per_answer': total_mentions / n,
        'answers_over_500_chars': over_500_count,
        'answers_over_500_pct': (over_500_count / n * 100),
        'answers_with_mentions': with_mentions_count,
        'answers_with_mentions_pct': (with_mentions_count / n * 100),
    }


def backup_existing_file(file_path: Path) -> Path:
    """Create timestamped backup of existing file."""
    if not file_path.exists():
        print(f"No existing file to backup: {file_path}")
        return None
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    backup_path = file_path.parent / f"{file_path.stem}_BACKUP_BEFORE_PHASE1_{timestamp}{file_path.suffix}"
    
    shutil.copy2(file_path, backup_path)
    print(f"✓ Created backup: {backup_path.name}")
    
    return backup_path


def load_existing_qa_pairs(file_path: Path) -> List[Dict]:
    """Load existing Q&A pairs from file."""
    if not file_path.exists():
        print(f"File does not exist: {file_path}")
        return []
    
    try:
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return []


def print_comparison_report(old_metrics: Dict, new_metrics: Dict):
    """Print comparison between old and new Q&A metrics."""
    print("\n" + "="*80)
    print("RE-PROCESSING COMPARISON REPORT")
    print("="*80)
    
    print("\nTOTAL Q&A PAIRS:")
    print(f"  Old: {old_metrics.get('total_pairs', 0)}")
    print(f"  New: {new_metrics.get('total_pairs', 0)}")
    
    if old_metrics and new_metrics:
        print("\nAVERAGE ANSWER LENGTH:")
        old_len = old_metrics['avg_answer_length']
        new_len = new_metrics['avg_answer_length']
        reduction = ((old_len - new_len) / old_len * 100) if old_len > 0 else 0
        print(f"  Old: {old_len:.1f} chars")
        print(f"  New: {new_len:.1f} chars")
        print(f"  Change: {reduction:+.1f}%")
        
        print("\n@MENTIONS IN ANSWERS:")
        old_mentions = old_metrics['total_mentions']
        new_mentions = new_metrics['total_mentions']
        old_avg = old_metrics['avg_mentions_per_answer']
        new_avg = new_metrics['avg_mentions_per_answer']
        mention_reduction = ((old_avg - new_avg) / old_avg * 100) if old_avg > 0 else 0
        print(f"  Old: {old_mentions} total ({old_avg:.2f} per answer)")
        print(f"  New: {new_mentions} total ({new_avg:.2f} per answer)")
        print(f"  Change: {mention_reduction:+.1f}%")
        
        print("\nANSWERS WITH @MENTIONS:")
        print(f"  Old: {old_metrics['answers_with_mentions']} ({old_metrics['answers_with_mentions_pct']:.1f}%)")
        print(f"  New: {new_metrics['answers_with_mentions']} ({new_metrics['answers_with_mentions_pct']:.1f}%)")
        
        print("\nANSWERS OVER 500 CHARACTERS:")
        print(f"  Old: {old_metrics['answers_over_500_chars']} ({old_metrics['answers_over_500_pct']:.1f}%)")
        print(f"  New: {new_metrics['answers_over_500_chars']} ({new_metrics['answers_over_500_pct']:.1f}%)")
    
    print("\n" + "="*80)


def main():
    """Main re-processing function."""
    print("="*80)
    print("FORUM Q&A RE-PROCESSING")
    print("="*80)
    print("\nThis script will:")
    print("  1. Create backup of existing forum_qa_pairs.json")
    print("  2. Re-process all forum threads")
    print("  3. Generate comparison report")
    print("\n" + "="*80)
    
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    output_dir = project_root / "data" / "processed" / "forum_qa"
    current_file = output_dir / "forum_qa_pairs.json"
    
    # Step 1: Load existing Q&A pairs and analyze
    print("\n[Step 1] Loading existing Q&A pairs...")
    old_qa_pairs = load_existing_qa_pairs(current_file)
    
    if old_qa_pairs:
        print(f"Loaded {len(old_qa_pairs)} existing Q&A pairs")
        old_metrics = analyze_qa_quality(old_qa_pairs)
        print(f"  - Average answer length: {old_metrics['avg_answer_length']:.1f} chars")
        print(f"  - Answers with @mentions: {old_metrics['answers_with_mentions']} ({old_metrics['answers_with_mentions_pct']:.1f}%)")
        print(f"  - Answers over 500 chars: {old_metrics['answers_over_500_chars']} ({old_metrics['answers_over_500_pct']:.1f}%)")
    else:
        print("No existing Q&A pairs found (this is a fresh run)")
        old_metrics = {}
    
    # Step 2: Create backup
    print("\n[Step 2] Creating backup...")
    backup_path = backup_existing_file(current_file)
    
    # Step 3: Re-process all forum threads
    print("\n[Step 3] Re-processing all forum threads...")
    print("This may take a few minutes...\n")
    
    extractor = ForumQAExtractor(
        input_dir=str(project_root / "data" / "raw" / "forum_threads"),
        output_dir=str(output_dir)
    )
    
    # Process all threads (this will automatically create archive and save)
    new_qa_pairs = extractor.process_all_threads()
    extractor.save_qa_pairs(new_qa_pairs)
    
    # Step 4: Analyze new results
    print("\n[Step 4] Analyzing new results...")
    new_metrics = analyze_qa_quality(new_qa_pairs)
    
    # Step 5: Generate comparison report
    print_comparison_report(old_metrics, new_metrics)
    
    # Save detailed comparison
    comparison_file = output_dir / f"phase1_reprocessing_comparison_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    comparison_data = {
        'reprocessing_date': datetime.now().isoformat(),
        'backup_file': str(backup_path.name) if backup_path else None,
        'old_metrics': old_metrics,
        'new_metrics': new_metrics,
        'improvements': {
            'answer_length_reduction_pct': ((old_metrics.get('avg_answer_length', 0) - new_metrics.get('avg_answer_length', 0)) / old_metrics.get('avg_answer_length', 1) * 100) if old_metrics else 0,
            'mention_reduction_pct': ((old_metrics.get('avg_mentions_per_answer', 0) - new_metrics.get('avg_mentions_per_answer', 0)) / old_metrics.get('avg_mentions_per_answer', 1) * 100) if old_metrics else 0,
        }
    }
    
    with comparison_file.open('w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed comparison saved to: {comparison_file.name}")
    print("\n✓ Re-processing complete!")
    print(f"\nFiles created:")
    if backup_path:
        print(f"  - Backup: {backup_path.name}")
    print(f"  - New Q&A pairs: forum_qa_pairs.json")
    print(f"  - Archive: forum_qa_pairs_{datetime.now().strftime('%Y-%m-%d')}_*.json")
    print(f"  - Comparison: {comparison_file.name}")


if __name__ == "__main__":
    main()
