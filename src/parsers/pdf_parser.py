import fitz  # PyMuPDF
from typing import List, Dict, Optional
import re


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
        r"advertisement", r"visit our website", r"customer service", r"all rights reserved", r"contact", r"support"
    ]
    skip_regex = re.compile("|".join(skip_patterns), re.IGNORECASE)
    
    current_section = None

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        # Remove common headers/footers (simple heuristic: lines repeated on every page)
        lines = text.splitlines()
        # Optionally, collect header/footer candidates from first/last lines
        if len(lines) > 4:
            lines = lines[1:-1]  # Remove first and last line (often header/footer)
        text = "\n".join(lines)

        # Split by double newlines (paragraphs)
        for para in text.split("\n\n"):
            clean_para = para.strip()
            # Skip empty or very short
            if len(clean_para) < 50:
                continue
            # Skip if matches skip patterns
            if skip_regex.search(clean_para):
                continue
            # Skip if mostly uppercase (often section titles, TOC)
            if len(clean_para) > 20 and clean_para == clean_para.upper():
                continue
            # Skip if looks like a page number
            if re.fullmatch(r"\d{1,3}", clean_para):
                continue
            
            # Extract section from content
            section = extract_section_from_content(clean_para, current_section)
            if section:
                current_section = section
            
            chunk = {
                "text": clean_para,
                "page": page_num + 1,
                "section": current_section,
                "doc_type": doc_type
            }
            chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    import sys
    import os
    import pickle
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <pdf_path> [output_pickle]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("data", "processed", "chunks.pkl")
    chunks = parse_pdf_rulebook(pdf_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {out_path}")
    print(f"Extracted {len(chunks)} chunks.")
    for c in chunks[:3]:
        print(c)
