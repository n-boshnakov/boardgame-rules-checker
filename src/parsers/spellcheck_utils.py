import re
from typing import List
import csv
import os

try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None
def correct_spelling(
    text: str,
    language: str = 'en',
    generate_corrections_file: bool = True,
    corrections_output_path: str = None,
    unique_terms_file: str = None,
    word_fragments_file: str = None,
    deduplicate_corrections: bool = False
) -> dict:
    """
    Corrects spelling mistakes in the given text using pyspellchecker.
    Returns a dict with:
        - 'corrected_text': the corrected text
        - 'checked_words': list of (original, corrected) tuples
    Optionally, provide a corrections_file (txt or csv) with human-made corrections (word,correction per line).
    Optionally, provide a word_fragments_file (csv) with fragment,full_word mappings for partial word corrections.
    """
    if SpellChecker is None:
        return {'corrected_text': text, 'checked_words': []}  # fallback: do nothing if not installed
    spell = SpellChecker(language=language)
    unique_terms = set()
    unique_terms_lower = set()  # Case-insensitive lookup
    if unique_terms_file:
        try:
            with open(unique_terms_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or row[0].startswith('#'):
                        continue
                    word = row[0].strip()
                    if word:
                        unique_terms.add(word)
                        unique_terms_lower.add(word.lower())
        except Exception as e:
            print(f"Warning: Could not read unique_terms file: {e}")
    
    # Load word fragments mapping
    word_fragments = {}
    if word_fragments_file:
        try:
            with open(word_fragments_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    fragment = row[0].strip()
                    full_word = row[1].strip()
                    if fragment and full_word:
                        word_fragments[fragment] = full_word
        except Exception as e:
            print(f"Warning: Could not read word_fragments file: {e}")
    
    # Common typo mappings for short ambiguous words
    typo_map = {
        "od": "do",
        "eck": "deck"
    }
    
    checked_words = []
    def correct_word(word):
        if not word.isalpha():
            return word, word
        
        # Check unique terms (case-insensitive) - NEVER correct these
        if word.lower() in unique_terms_lower:
            return word, word
        
        # Check common typo mappings (for short ambiguous words)
        if word.lower() in typo_map:
            corrected = typo_map[word.lower()]
            # Preserve capitalization
            if word.isupper():
                return word, corrected.upper()
            elif word[0].isupper():
                return word, corrected.capitalize()
            else:
                return word, corrected
        
        # Check if word is very close to a unique term (likely typo of game term)
        # This handles cases like "Stalers" which should be "Stalkers"
        # But avoid correcting valid words to similar game terms (e.g., "dies" → "die")
        from difflib import get_close_matches
        
        # First check if the word itself is already known to be correct
        word_is_valid = word.lower() not in spell.unknown([word.lower()])
        
        # Only apply fuzzy matching to unknown words or check for game-specific terms
        if not word_is_valid:
            close_matches = get_close_matches(word.lower(), unique_terms_lower, n=1, cutoff=0.85)
            if close_matches:
                matched_term_lower = close_matches[0]
                # Make sure the match is actually better than the word itself
                # (e.g., "staler" is valid but "stalker" is more relevant for game queries)
                if matched_term_lower != word.lower():
                    for term in unique_terms:
                        if term.lower() == matched_term_lower:
                            # Preserve capitalization pattern from original word
                            if word.isupper():
                                return word, term.upper()
                            elif word[0].isupper():
                                return word, term.capitalize() if term[0].islower() else term
                            else:
                                return word, term.lower()
                    return word, close_matches[0]
        
        # Check word fragments first (exact match)
        if word in word_fragments:
            return word, word_fragments[word]
        
        # Check if word is in dictionary (no correction needed)
        # BUT: Even if valid, check if there's a much closer game term match
        word_is_valid = word.lower() not in spell.unknown([word.lower()])
        
        if word_is_valid:
            # Word is technically valid, but check if a game term is a better match
            # E.g., "staler" (valid) vs "Stalker" (game term), "od" (valid) vs "do" (common)
            from difflib import get_close_matches
            close_game_terms = get_close_matches(word.lower(), unique_terms_lower, n=1, cutoff=0.90)
            if close_game_terms and close_game_terms[0] != word.lower():
                # Found a very close game term - prefer it
                matched_term_lower = close_game_terms[0]
                for term in unique_terms:
                    if term.lower() == matched_term_lower:
                        # Preserve capitalization pattern
                        if word.isupper():
                            return word, term.upper()
                        elif word[0].isupper():
                            return word, term.capitalize() if term[0].islower() else term
                        else:
                            return word, term.lower()
            # No close game term match - word is fine as-is
            return word, word
        
        # Word is misspelled - get all candidates
        candidates = spell.candidates(word.lower())
        if not candidates or len(candidates) == 0:
            return word, word
        
        # Quality check: Find best candidate based on similarity
        from difflib import SequenceMatcher
        
        # For very short words (2-3 chars), use more lenient matching
        # but also check for common short word typos
        min_similarity = 0.5 if len(word) <= 3 else 0.65
        
        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            similarity = SequenceMatcher(None, word.lower(), candidate).ratio()
            # Prefer candidates that preserve more letters (e.g., "skip" over "sky" for "skp")
            length_penalty = abs(len(word) - len(candidate)) * 0.1
            score = similarity - length_penalty
            scored_candidates.append((score, similarity, candidate))
        
        # Get best candidate
        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        best_score, best_similarity, best_candidate = scored_candidates[0]
        
        # Require minimum similarity (lower for short words)
        if best_similarity < min_similarity:
            return word, word
        
        # Preserve original capitalization pattern
        if word.isupper():
            corrected = best_candidate.upper()
        elif word[0].isupper():
            corrected = best_candidate.capitalize()
        else:
            corrected = best_candidate
        
        return word, corrected
    tokens = re.findall(r"\w+|\W+", text)
    corrected_tokens = []
    for tok in tokens:
        if tok.isalpha():
            orig, corr = correct_word(tok)
            checked_words.append((orig, corr))
            corrected_tokens.append(corr)
        else:
            corrected_tokens.append(tok)
    corrected_text = ''.join(corrected_tokens)
    # Always append corrections, then deduplicate if requested
    if generate_corrections_file:
        out_path = corrections_output_path or 'corrections_autogen.csv'
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        # Append new corrections (but skip terms in unique_terms to avoid conflicts)
        try:
            with open(out_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                for orig, corr in checked_words:
                    # Never write corrections for terms in unique_terms (case-insensitive check)
                    if orig != corr and orig.lower() not in unique_terms_lower:
                        writer.writerow([orig, corr])
        except Exception as e:
            print(f"[SpellChecker] Warning: Could not append to corrections file: {e}")
        # Deduplicate and sort if requested
        if deduplicate_corrections:
            try:
                with open(out_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    pairs = set(tuple(row) for row in reader if row and row[0] != 'original')
                # Filter out any terms that are in unique_terms (safety check, case-insensitive)
                pairs = {(orig, corr) for orig, corr in pairs if orig.lower() not in unique_terms_lower}
                pairs = sorted(pairs, key=lambda x: (x[0].lower(), x[1].lower()))
                with open(out_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['original', 'correction'])
                    for orig, corr in pairs:
                        writer.writerow([orig, corr])
                print(f"[SpellChecker] Corrections file deduplicated and sorted: {os.path.abspath(out_path)}")
            except Exception as e:
                print(f"[SpellChecker] Warning: Could not deduplicate corrections file: {e}")
    return {'corrected_text': corrected_text, 'checked_words': checked_words}
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spellcheck utility with optional unique terms file.")
    parser.add_argument('--text', type=str, default=None, help='Text to check (if not given, uses example)')
    parser.add_argument('--corrections-output', type=str, default=None, help='Path to corrections CSV file (output, always appended)')
    parser.add_argument('--unique-terms', type=str, default=None, help='Path to unique terms CSV file (optional)')
    parser.add_argument('--word-fragments', type=str, default=None, help='Path to word fragments CSV file (optional)')
    parser.add_argument('--deduplicate', action='store_true', help='Deduplicate and sort corrections file after run')
    parser.add_argument('--no-generate', action='store_true', help='Do not generate corrections file')
    args = parser.parse_args()
    s = args.text if args.text else "Ths is a smple txt with erors."
    result = correct_spelling(
        s,
        generate_corrections_file=not args.no_generate,
        corrections_output_path=args.corrections_output,
        unique_terms_file=args.unique_terms,
        word_fragments_file=args.word_fragments,
        deduplicate_corrections=args.deduplicate
    )
    print("Corrected text:", result['corrected_text'])
    print("Checked words:", result['checked_words'])
    if not args.no_generate:
        print("Corrections file was updated. Use --deduplicate to clean and sort it.")
