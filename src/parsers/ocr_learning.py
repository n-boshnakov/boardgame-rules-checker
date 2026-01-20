"""
OCR Learning Module - Automatically learns OCR correction patterns from spell corrections.
"""

import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set


def analyze_corrections_for_patterns(corrections_csv_path: str) -> Dict:
    """
    Analyze a corrections CSV file to identify OCR patterns.
    
    Args:
        corrections_csv_path: Path to the corrections CSV file with columns: original, corrected
    
    Returns:
        Dictionary with 'character_map' and 'word_patterns' entries
    """
    import csv
    
    if not os.path.exists(corrections_csv_path):
        print(f"[OCR Learning] Corrections file not found: {corrections_csv_path}")
        return {"character_map": {}, "word_patterns": []}
    
    character_substitutions = Counter()
    word_patterns = defaultdict(Counter)
    
    try:
        with open(corrections_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                original = row.get('original', '').strip()
                corrected = row.get('corrected', '').strip()
                
                if not original or not corrected or original == corrected:
                    continue
                
                # Analyze character-level substitutions
                if len(original) == len(corrected):
                    for orig_char, corr_char in zip(original, corrected):
                        if orig_char != corr_char:
                            character_substitutions[(orig_char, corr_char)] += 1
                
                # Analyze word-level patterns
                # Check for prefix/suffix patterns
                if len(original) > 3 and len(corrected) > 3:
                    # Check prefix substitution (e.g., ]talker -> Stalker)
                    if original[1:] == corrected[1:]:
                        pattern = f"^{re.escape(original[0])}"
                        word_patterns['prefix'][(original[0], corrected[0], original, corrected)] += 1
                    
                    # Check for full word substitution patterns
                    elif original.lower() != corrected.lower():
                        # Common word substitutions
                        word_patterns['word_replace'][(original, corrected)] += 1
    
    except Exception as e:
        print(f"[OCR Learning] Error analyzing corrections: {e}")
        return {"character_map": {}, "word_patterns": []}
    
    # Convert counters to configuration format
    config = {
        "character_map": {},
        "word_patterns": []
    }
    
    # Add character substitutions that appear at least 3 times
    for (orig_char, corr_char), count in character_substitutions.items():
        if count >= 3 and orig_char not in [' ', '\t', '\n']:
            config["character_map"][orig_char] = corr_char
            print(f"[OCR Learning] Learned character: '{orig_char}' -> '{corr_char}' (seen {count} times)")
    
    # Add prefix patterns that appear at least 2 times
    prefix_groups = defaultdict(list)
    for (orig_prefix, corr_prefix, orig_word, corr_word), count in word_patterns['prefix'].items():
        if count >= 2:
            prefix_groups[(orig_prefix, corr_prefix)].append((orig_word, corr_word, count))
    
    for (orig_prefix, corr_prefix), examples in prefix_groups.items():
        if len(examples) >= 2:
            # Create a general pattern for this prefix
            # Find common suffix pattern
            suffixes = [orig[1:] for orig, _, _ in examples]
            if suffixes:
                # Create regex pattern that matches words with this prefix
                pattern = f"\\{orig_prefix}\\w+"
                config["word_patterns"].append({
                    "pattern": pattern,
                    "type": "prefix_replace",
                    "prefix": orig_prefix,
                    "prefix_replacement": corr_prefix,
                    "examples": [f"{o} -> {c}" for o, c, _ in examples[:3]]
                })
                print(f"[OCR Learning] Learned prefix pattern: '{orig_prefix}' -> '{corr_prefix}' ({len(examples)} examples)")
    
    # Add word replacements that appear at least 3 times
    for (orig_word, corr_word), count in word_patterns['word_replace'].most_common(20):
        if count >= 3:
            pattern = f"\\b{re.escape(orig_word)}\\b"
            config["word_patterns"].append({
                "pattern": pattern,
                "replacement": corr_word,
                "type": "simple",
                "frequency": count
            })
            print(f"[OCR Learning] Learned word pattern: '{orig_word}' -> '{corr_word}' (seen {count} times)")
    
    return config


def merge_ocr_corrections(existing_config: Dict, learned_config: Dict) -> Dict:
    """
    Merge learned OCR corrections with existing configuration.
    Avoids duplicates and preserves manual entries.
    """
    merged = {
        "character_map": dict(existing_config.get("character_map", {})),
        "word_patterns": list(existing_config.get("word_patterns", []))
    }
    
    # Merge character maps (learned values don't override existing)
    for char, replacement in learned_config.get("character_map", {}).items():
        if char not in merged["character_map"]:
            merged["character_map"][char] = replacement
    
    # Merge word patterns (check for duplicates by pattern)
    existing_patterns = {p.get("pattern") for p in merged["word_patterns"]}
    for pattern_config in learned_config.get("word_patterns", []):
        pattern = pattern_config.get("pattern")
        if pattern and pattern not in existing_patterns:
            merged["word_patterns"].append(pattern_config)
            existing_patterns.add(pattern)
    
    return merged


def update_ocr_corrections_from_learning(
    corrections_csv_path: str,
    ocr_config_path: str = "data/processed/pdf_ocr_corrections.json",
    backup: bool = True
) -> bool:
    """
    Analyze corrections and update the OCR corrections configuration file.
    
    Args:
        corrections_csv_path: Path to corrections CSV from spell checking
        ocr_config_path: Path to OCR corrections JSON configuration
        backup: Whether to backup the existing config before updating
    
    Returns:
        True if updates were made, False otherwise
    """
    # Load existing configuration
    existing_config = {"character_map": {}, "word_patterns": []}
    if os.path.exists(ocr_config_path):
        try:
            with open(ocr_config_path, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
        except Exception as e:
            print(f"[OCR Learning] Error loading existing config: {e}")
            return False
    
    # Learn new patterns from corrections
    print(f"[OCR Learning] Analyzing corrections from {corrections_csv_path}...")
    learned_config = analyze_corrections_for_patterns(corrections_csv_path)
    
    if not learned_config["character_map"] and not learned_config["word_patterns"]:
        print("[OCR Learning] No new patterns learned.")
        return False
    
    # Merge configurations
    merged_config = merge_ocr_corrections(existing_config, learned_config)
    
    # Check if anything changed
    if merged_config == existing_config:
        print("[OCR Learning] No new patterns to add.")
        return False
    
    # Backup existing file
    if backup and os.path.exists(ocr_config_path):
        backup_path = ocr_config_path.replace('.json', '_backup.json')
        try:
            import shutil
            shutil.copy(ocr_config_path, backup_path)
            print(f"[OCR Learning] Backed up existing config to {backup_path}")
        except Exception as e:
            print(f"[OCR Learning] Warning: Could not create backup: {e}")
    
    # Save updated configuration
    try:
        os.makedirs(os.path.dirname(ocr_config_path), exist_ok=True)
        with open(ocr_config_path, 'w', encoding='utf-8') as f:
            json.dump(merged_config, f, indent=2, ensure_ascii=False)
        
        new_chars = len(merged_config["character_map"]) - len(existing_config.get("character_map", {}))
        new_patterns = len(merged_config["word_patterns"]) - len(existing_config.get("word_patterns", []))
        
        print(f"[OCR Learning] Updated {ocr_config_path}")
        print(f"[OCR Learning] Added {new_chars} character mappings, {new_patterns} word patterns")
        return True
        
    except Exception as e:
        print(f"[OCR Learning] Error saving updated config: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_learning.py <corrections_csv_path> [ocr_config_path]")
        print("Example: python ocr_learning.py data/processed/corrections_2025-12-23_21-06-17.csv")
        sys.exit(1)
    
    corrections_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/pdf_ocr_corrections.json"
    
    success = update_ocr_corrections_from_learning(corrections_path, config_path)
    
    if success:
        print("\n✓ OCR corrections updated successfully!")
        print(f"  Run the PDF parser again to apply the new corrections.")
    else:
        print("\n✗ No updates made to OCR corrections.")
