import fitz  # PyMuPDF
from typing import List, Dict, Optional
import re
from spellcheck_utils import correct_spelling


def extract_section_from_content(text: str, prev_section: Optional[str] = None) -> Optional[str]:
    """
    Extracts section name from paragraph content using semantic patterns.
    Looks for common section indicators like numbered/titled headers, keywords, etc.
    """
    # Common section keywords in boardgame rulebooks
    section_keywords = [
        r"setup", r"game setup", r"objective", r"goal", r"components", r"overview",
        r"gameplay", r"game play", r"turn structure", r"player turn", r"phases",
        r"actions", r"movement", r"combat", r"trading", r"resources",
        r"winning", r"end game", r"end of game", r"victory", r"scoring",
        r"special rules", r"variants", r"solo mode", r"team play",
        r"reference", r"quick reference", r"glossary", r"FAQ", r"clarifications"
    ]
    
    # Check if the paragraph starts with a numbered section (e.g., "1. Setup", "2.1 Game Phases")
    numbered_section = re.match(r"^(\d+\.?\d*)\s+([A-Z][A-Za-z\s]{2,30})", text)
    if numbered_section:
        return numbered_section.group(2).strip()
    
    # Check if the first line is a short title (likely section header)
    first_line = text.split("\n")[0].strip()
    if len(first_line) < 50 and first_line[0].isupper():
        # Check if it contains section keywords
        for keyword in section_keywords:
            if re.search(keyword, first_line, re.IGNORECASE):
                return first_line
        # Check if it's title case and short (likely a section header)
        if len(first_line.split()) <= 5 and sum(1 for c in first_line if c.isupper()) >= 2:
            return first_line
    
    # Check for section keywords anywhere in the first 100 chars
    text_start = text[:100].lower()
    for keyword in section_keywords:
        match = re.search(keyword, text_start, re.IGNORECASE)
        if match:
            # Extract the matched keyword as section name (capitalized)
            return match.group(0).title()
    
    # If no section found, inherit from previous
    return prev_section


def parse_pdf_rulebook(pdf_path: str, doc_type: str = "rulebook") -> List[Dict]:
    """
    Parses a PDF rulebook and returns a list of chunks with metadata.
    Skips irrelevant sections (credits, table of contents, ads, thanks, etc.).
    Cleans headers/footers and ensures all content is chunked by paragraphs.
    Extracts section names from content semantically.
    """
    doc = fitz.open(pdf_path)
    chunks = []
    skip_patterns = [
        r"table of contents", r"contents", r"thank you", r"thanks", r"credits", r"designed by", r"illustrated by",
        r"advertisement", r"visit our website", r"customer service", r"all rights reserved", r"contact", r"support",
        r"^components$", r"^game components$", r"^box contents$", r"^in the box$", r"^component list$", r"^you should have$", r"^this game includes$"
    ]
    skip_regex = re.compile("|".join(skip_patterns), re.IGNORECASE)
    
    current_section = None

    from datetime import datetime
    from datetime import datetime
    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    corrections_csv_path = f"data/processed/corrections_{dt_str}.csv"
    unique_terms_path = "data/processed/unique_terms.csv"
    word_fragments_path = "data/processed/word_fragments.csv"
    section_headers = set()
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        lines = text.splitlines()
        if len(lines) > 4:
            lines = lines[1:-1]
        text = "\n".join(lines)

        for para in text.split("\n\n"):
            clean_para = para.strip()
            if len(clean_para) < 50:
                continue
            if skip_regex.search(clean_para):
                continue
            if len(clean_para) > 20 and clean_para == clean_para.upper():
                continue
            if re.fullmatch(r"\d{1,3}", clean_para):
                continue

            first_line = clean_para.split("\n")[0].strip()
            # Collect likely section headers for later filtering
            if (len(first_line) <= 40 and (first_line.isupper() or sum(1 for c in first_line if c.isupper()) > 3)):
                section_headers.add(first_line)
                continue

            spell_result = correct_spelling(
                clean_para,
                generate_corrections_file=True,
                corrections_output_path=corrections_csv_path,
                unique_terms_file=unique_terms_path,
                word_fragments_file=word_fragments_path
            )
            print(f"[Parser] Spellchecked paragraph (page {page_num+1}): {clean_para[:40]}...")
            corrected_text = spell_result['corrected_text'] if isinstance(spell_result, dict) else spell_result

            section = extract_section_from_content(corrected_text, current_section)
            if section:
                current_section = section

            chunk = {
                "text": corrected_text,
                "page": page_num + 1,
                "section": current_section,
                "doc_type": doc_type,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            chunks.append(chunk)
    # Save section headers for answer filtering
    section_headers_path = "data/processed/section_headers.txt"
    os.makedirs(os.path.dirname(section_headers_path), exist_ok=True)
    with open(section_headers_path, "w", encoding="utf-8") as shf:
        for header in sorted(section_headers):
            shf.write(header + "\n")
    # Deduplicate and sort corrections file at the end
    from spellcheck_utils import correct_spelling as _cs
    _cs('', generate_corrections_file=True, corrections_output_path=corrections_csv_path, unique_terms_file=unique_terms_path, word_fragments_file=word_fragments_path, deduplicate_corrections=True)
    return chunks

if __name__ == "__main__":
    import sys
    import os
    import pickle
    # Step 1: Parse PDF and extract unique terms from content
    unique_terms_path = "data/processed/unique_terms.csv"

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <pdf_path> [output_pickle]")
        sys.exit(1)
    from datetime import datetime
    pdf_path = sys.argv[1]
    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    else:
        out_path = os.path.join("data", "processed", "chunks.pkl")
    # Always create archive copy with date-time
    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_dir = os.path.join("data", "processed", "archive")
    os.makedirs(archive_dir, exist_ok=True)
    archive_pkl = os.path.join(archive_dir, f"chunks_{dt_str}.pkl")
    archive_json = os.path.join(archive_dir, f"chunks_{dt_str}.json")
    chunks = parse_pdf_rulebook(pdf_path)
    # Extract unique terms from all chunked text
    import re
    import requests
    all_text = " ".join(chunk["text"] for chunk in chunks)
    # Find capitalized words (not just sentence-initial) and all-uppercase words (acronyms/terms)
    sentence_end_re = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_end_re.split(all_text)
    capitalized_words = set()
    fully_upper_words = set()
    for sent in sentences:
        words_in_sent = re.findall(r"\b\w+\b", sent)
        for i, word in enumerate(words_in_sent):
            if i == 0:
                continue  # skip first word of sentence
            if re.match(r"^[A-Z][a-zA-Z0-9\-]{2,}$", word):
                capitalized_words.add(word)
            if re.match(r"^[A-Z]{2,}$", word):
                fully_upper_words.add(word)
    # Add all fully capitalized words from chunked text (excluding section titles)
    words = capitalized_words.union(fully_upper_words)
    # Download and use stopwords list to filter out common English words
    stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
    stopwords = set(stopwords_list.decode().splitlines())
    # Normalize all terms for comparison (lowercase, strip)
    def normalize(term):
        return term.lower().strip()
    normalized_stopwords = set(normalize(sw) for sw in stopwords)
    normalized_words = {normalize(w): w for w in words}
    filtered_words = [original for norm, original in normalized_words.items() if norm not in normalized_stopwords]
    # Write unique terms to CSV (original form, sorted by normalized)
    os.makedirs(os.path.dirname(unique_terms_path), exist_ok=True)
    for_write = [normalized_words[n] for n in sorted(normalized_words) if n in {normalize(w) for w in filtered_words}]
    with open(unique_terms_path, "w", encoding="utf-8") as f:
        for w in for_write:
            f.write(w + "\n")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Save main chunk file (no date-time)
    with open(out_path, "wb") as f:
        pickle.dump(chunks, f)
    # Save archive copy with date-time
    with open(archive_pkl, "wb") as f:
        pickle.dump(chunks, f)
    # Save archive JSON with date-time
    import json
    with open(archive_json, "w", encoding="utf-8") as jf:
        json.dump(chunks, jf, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks to {out_path}")
    print(f"Archived {len(chunks)} chunks to {archive_pkl} and {archive_json}")
    print(f"Extracted {len(chunks)} chunks.")
    print(f"Extracted {len(words)} unique terms to {unique_terms_path}")
    for c in chunks[:3]:
        print(c)
