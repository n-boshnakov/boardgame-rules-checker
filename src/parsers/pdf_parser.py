import fitz  # PyMuPDF
from typing import List, Dict, Optional
import re
import json
import os
from spellcheck_utils import correct_spelling

# Try to import ftfy for automatic unicode/encoding fixes
try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False
    print("[Warning] ftfy not installed. Install with: pip install ftfy")
    print("[Warning] Automatic unicode/encoding fixes will be limited.")


# Global cache for OCR corrections config
_OCR_CORRECTIONS_CACHE = None


def load_ocr_corrections(config_path: str = "data/processed/pdf_ocr_corrections.json") -> Dict:
    """
    Load OCR correction mappings from a JSON configuration file.
    If the file doesn't exist, returns empty mappings.
    """
    global _OCR_CORRECTIONS_CACHE
    
    # Return cached config if already loaded
    if _OCR_CORRECTIONS_CACHE is not None:
        return _OCR_CORRECTIONS_CACHE
    
    # Try to load from file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"[Parser] Loaded OCR corrections from {config_path}")
                _OCR_CORRECTIONS_CACHE = config
                return config
        except Exception as e:
            print(f"[Parser] Error loading OCR corrections from {config_path}: {e}")
            print(f"[Parser] No OCR corrections will be applied")
    else:
        print(f"[Parser] OCR corrections file not found at {config_path}")
        print(f"[Parser] Create this file to enable automatic OCR error correction")
    
    # Return empty config if file doesn't exist or failed to load
    empty_config = {"character_map": {}, "word_patterns": []}
    _OCR_CORRECTIONS_CACHE = empty_config
    return empty_config


def fix_encoding_issues(text: str, config_path: str = "data/processed/pdf_ocr_corrections.json") -> str:
    """
    Automatically fix encoding and OCR issues using ftfy and pattern matching.
    Loads correction mappings from a JSON configuration file.
    """
    # First, use ftfy if available - it automatically fixes unicode/encoding issues
    if HAS_FTFY:
        text = ftfy.fix_text(text)
    
    # Load OCR corrections from config file
    config = load_ocr_corrections(config_path)
    
    # Apply character map
    char_map = str.maketrans(config.get("character_map", {}))
    text = text.translate(char_map)
    
    # Apply word-level pattern corrections
    for pattern_config in config.get("word_patterns", []):
        pattern = pattern_config.get("pattern")
        pattern_type = pattern_config.get("type", "simple")
        
        if pattern_type == "prefix_replace":
            # Handle prefix replacement (e.g., ]talker -> Stalker)
            prefix = pattern_config.get("prefix")
            prefix_replacement = pattern_config.get("prefix_replacement")
            text = re.sub(pattern, lambda m: m.group(0).replace(prefix, prefix_replacement), text)
        elif pattern_type == "simple":
            # Simple string replacement
            replacement = pattern_config.get("replacement", "")
            text = re.sub(pattern, replacement, text)
    
    return text


def detect_and_format_table(text: str) -> str:
    """
    Detect table-like structures and format them more clearly.
    Tables often have multiple columns with bullet points or similar structure.
    """
    lines = text.split('\n')
    
    # Check if this looks like a table (multiple bullet points on consecutive lines)
    bullet_lines = [i for i, line in enumerate(lines) if line.strip().startswith('•')]
    
    if len(bullet_lines) >= 3:
        # This might be a table - try to add clearer separators
        formatted_lines = []
        for i, line in enumerate(lines):
            formatted_lines.append(line)
            # Add separator after bullet point groups
            if i in bullet_lines and (i + 1 >= len(lines) or not lines[i + 1].strip().startswith('•')):
                formatted_lines.append('')  # Add blank line for separation
        
        return '\n'.join(formatted_lines)
    
    return text


def extract_section_from_content(text: str, prev_section: Optional[str] = None) -> Optional[str]:
    """
    Extracts section name from paragraph content using semantic patterns.
    Looks for common section indicators like numbered/titled headers, keywords, etc.
    More robust detection to maintain section coherence across chunks.
    """
    # Common section keywords in boardgame rulebooks
    section_keywords = [
        r"setup", r"game setup", r"objective", r"goal", r"components", r"overview",
        r"gameplay", r"game play", r"turn structure", r"player turn", r"phases",
        r"actions", r"movement", r"combat", r"trading", r"resources",
        r"winning", r"end game", r"end of game", r"victory", r"scoring",
        r"special rules", r"variants", r"solo mode", r"team play",
        r"reference", r"quick reference", r"glossary", r"FAQ", r"clarifications",
        r"mission", r"enemies", r"anomalies", r"artifacts", r"equipment",
        r"weapons", r"armor", r"radiation", r"attention", r"status", r"tokens"
    ]
    
    # Check if the paragraph starts with a numbered section (e.g., "1. Setup", "2.1 Game Phases")
    numbered_section = re.match(r"^(\d+\.?\d*)\s+([A-Z][A-Za-z\s]{2,50})", text)
    if numbered_section:
        return numbered_section.group(2).strip()
    
    # Check for section markers like "| Section Name |"
    section_marker = re.match(r"^\|?\s*([A-Z][A-Za-z\s&]{3,50})\s*\|", text)
    if section_marker:
        return section_marker.group(1).strip()
    
    # Check if the first line is a short title (likely section header)
    lines = text.split("\n")
    first_line = lines[0].strip() if lines else ""
    
    if first_line and len(first_line) < 60:
        # Check if it's all caps or title case (section header pattern)
        if first_line.isupper() or (first_line[0].isupper() and sum(1 for c in first_line if c.isupper()) >= 2):
            # Verify it's not just a sentence fragment
            if not first_line.endswith('.') or len(first_line.split()) <= 6:
                # Check if it contains section keywords
                for keyword in section_keywords:
                    if re.search(keyword, first_line, re.IGNORECASE):
                        return first_line
                
                # If it looks like a header (short, capitalized, no period), use it
                if len(first_line.split()) <= 8 and not first_line.endswith('.'):
                    return first_line
    
    # Check for section keywords in the first 150 chars
    text_start = text[:150].lower()
    for keyword in section_keywords:
        match = re.search(r'\b' + keyword + r'\b', text_start, re.IGNORECASE)
        if match:
            # Try to extract a meaningful section name around the keyword
            # Look for the line containing the keyword
            for line in lines[:3]:  # Check first 3 lines
                if re.search(keyword, line, re.IGNORECASE):
                    cleaned = line.strip('|').strip()
                    if len(cleaned) < 60:
                        return cleaned
            # Fallback to capitalized keyword
            return match.group(0).title()
    
    # If no section found, inherit from previous
    return prev_section


def chunk_text_with_overlap(text: str, max_chars: int = 1000, overlap_chars: int = 150) -> List[str]:
    """
    Splits text into overlapping chunks that respect sentence boundaries.
    Aggressive chunking: 500-1000 chars per chunk with 150 char overlap for good balance.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk (default 1000 for optimal context)
        overlap_chars: Number of characters to overlap between chunks (default 150)
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]
    
    # Split into sentences (handle multiple sentence endings)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # If single sentence exceeds max_chars, split it by character limit
        if sentence_len > max_chars:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long sentence into smaller parts with overlap
            for i in range(0, len(sentence), max_chars - overlap_chars):
                chunk_part = sentence[i:i + max_chars]
                if chunk_part.strip():
                    chunks.append(chunk_part)
            continue
        
        # If adding this sentence would exceed limit, save current chunk
        if current_length + sentence_len > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap from previous chunk
            overlap_sentences = []
            overlap_length = 0
            for prev_sent in reversed(current_chunk):
                if overlap_length + len(prev_sent) <= overlap_chars:
                    overlap_sentences.insert(0, prev_sent)
                    overlap_length += len(prev_sent) + 1  # +1 for space
                else:
                    break
            
            current_chunk = overlap_sentences
            current_length = sum(len(s) + 1 for s in overlap_sentences) if overlap_sentences else 0
        
        current_chunk.append(sentence)
        current_length += sentence_len + 1  # +1 for space
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def is_all_caps_section(text: str) -> bool:
    """Check if text is an ALL CAPS section heading."""
    if not text or len(text) > 100:
        return False
    
    # Filter out card names and game component descriptions
    card_keywords = [
        'GRENADE', 'BLAST', 'CHEMICAL', 'MEDICINE', 'ARTIFACT', 'CONTAINER',
        'WEAPON', 'SHOTGUN', 'RIFLE', 'PISTOL', 'HUMAN', 'MUTANT', 'PSIONIC',
        'GRAVITATIONAL', 'IMPROVED', 'ADVANCED', 'BANDIT', 'CONTROLLER',
        'ATTACK SEQUENCE', 'ENEMY ACTIVATION', 'ATTENTION ON THE MAP'
    ]
    
    text_upper = text.upper()
    for keyword in card_keywords:
        if keyword in text_upper:
            return False
    
    # Remove punctuation and numbers for checking
    text_clean = re.sub(r'[^\w\s]', '', text)
    words = text_clean.split()
    
    if not words or len(words) > 15:
        return False
    
    # Must be all uppercase and at least 2 words
    if len(words) < 2:
        return False
    
    return all(w.isupper() and len(w) >= 2 for w in words)


def is_title_case_heading(text: str) -> bool:
    """Check if text is a Title Case heading (subsection)."""
    if not text or len(text) > 150:
        return False
    
    # Must be a single line or very short
    if '\n' in text.strip():
        return False
    
    words = text.split()
    if len(words) > 20 or len(words) < 2:
        return False
    
    # Skip if it contains colons (likely credits like "Design: Name")
    if ':' in text:
        return False
    
    # Skip if ends with comma or "and" (likely list items)
    if text.strip().endswith(',') or text.strip().endswith('and'):
        return False
    
    # Title case: most significant words start with uppercase
    significant_words = [w for w in words if len(w) > 2]
    if not significant_words:
        return False
    
    capitalized = sum(1 for w in significant_words if w[0].isupper())
    
    # At least 70% of significant words must be capitalized
    # AND not all caps (that would be a section heading)
    is_title = capitalized >= len(significant_words) * 0.7
    not_all_caps = not is_all_caps_section(text)
    
    return is_title and not_all_caps


def should_skip_example(text: str) -> bool:
    """
    Check if text is an Example paragraph that should be skipped.
    Returns True if the paragraph starts with 'Example:' and should be skipped.
    
    Note: Complex case handling is commented out - when important text follows
    an Example paragraph on the same line or block, we need more sophisticated
    parsing to distinguish example content from following important content.
    """
    # Simple case: paragraph starts with "Example:"
    if text.strip().startswith("Example:"):
        return True
    
    # Check if any sentence in the paragraph starts with Example:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        if sent.strip().startswith("Example:"):
            return True
    
    return False


def has_substantial_content_after(paragraphs: List[Dict], current_idx: int, min_sentences: int = 1) -> bool:
    """
    Check if there's substantial content (at least min_sentences) after the current index.
    This helps filter out image captions and card names that aren't real section headings.
    
    Changed to min_sentences=1 to be less aggressive and allow shorter subsections.
    """
    if current_idx >= len(paragraphs) - 1:
        return False
    
    # Look at the next paragraph
    next_para = paragraphs[current_idx + 1]['text'].strip()
    
    # Skip if next is empty
    if not next_para:
        return False
    
    # Allow if next is a Title Case heading (subsections often follow sections)
    if is_title_case_heading(next_para):
        return True
    
    # Skip if next is also an ALL CAPS section
    if is_all_caps_section(next_para):
        return False
    
    # Count sentences in the next paragraph
    sentences = re.split(r'(?<=[.!?])\s+', next_para)
    substantial_sentences = [s for s in sentences if len(s.split()) >= 5]
    
    return len(substantial_sentences) >= min_sentences


def chunk_by_sections(paragraphs: List[Dict]) -> List[Dict]:
    """
    Aggressive paragraph-level chunking:
    1. Find ALL CAPS section headings and use as current section
    2. Find Title Case subsection headings 
    3. Split content into 500-1000 char chunks with 150 char overlap
    4. Skip card scans (ALL CAPS with minimal content)
    5. Preserve table formatting
    Target: 80-120 chunks from 26 current chunks
    """
    chunks = []
    current_section = None
    current_subsection = None
    current_chunk_text = []
    current_page = None
    
    # Credits/component list detection patterns
    credits_patterns = [
        r"narrative\s+design:", r"writing:", r"proofreading:", r"graphic\s+design:",
        r"illustrations?:", r"3d\s+modelling:", r"dtp:", r"production:",
        r"tests?\s+and\s+development:", r"internal\s+testing:",
        r"based\s+on:", r"dedicated\s+to"
    ]
    credits_regex = re.compile("|".join(credits_patterns), re.IGNORECASE)
    
    in_credits_section = False
    in_components_section = False
    
    def save_current_chunk():
        """Helper to save and split current chunk if needed."""
        nonlocal current_chunk_text, current_page, current_section, current_subsection
        
        if not current_chunk_text:
            return
        
        chunk_text = '\n\n'.join(current_chunk_text)
        
        # Aggressive splitting: if chunk exceeds 1000 chars, split it
        if len(chunk_text) > 1000:
            sub_chunks = chunk_text_with_overlap(chunk_text, max_chars=1000, overlap_chars=150)
            for sub_chunk in sub_chunks:
                if sub_chunk.strip():
                    chunks.append({
                        'text': sub_chunk.strip(),
                        'page': current_page,
                        'section': current_section,
                        'subsection': current_subsection
                    })
        else:
            # Keep smaller chunks as-is
            chunks.append({
                'text': chunk_text,
                'page': current_page,
                'section': current_section,
                'subsection': current_subsection
            })
        
        current_chunk_text = []
    
    for idx, para_info in enumerate(paragraphs):
        text = para_info['text'].strip()
        page = para_info['page']
        
        if not text:
            continue
        
        # Check if we're entering credits or components section
        if credits_regex.search(text.lower()):
            in_credits_section = True
            print(f"[Parser] Entering credits section on page {page}")
            continue
        
        if re.search(r"component\s*list|game\s*components", text.lower()):
            in_components_section = True
            print(f"[Parser] Entering components section on page {page}")
            continue
        
        # Skip if in credits or components section
        if in_credits_section or in_components_section:
            # Check if we've left these sections (new ALL CAPS section found)
            if is_all_caps_section(text) and len(text.split()) >= 2:
                in_credits_section = False
                in_components_section = False
                print(f"[Parser] Exiting credits/components section")
            else:
                continue
        
        # Skip Example paragraphs
        if should_skip_example(text):
            print(f"[Parser] Skipping Example paragraph on page {page}")
            continue
        
        # Check if this is an ALL CAPS section heading (with content validation)
        if is_all_caps_section(text):
            # Validate that this is a real section heading, not a card scan
            if not has_substantial_content_after(paragraphs, idx):
                print(f"[Parser] Skipping card scan/image caption: {text[:50]}")
                continue
            
            # Save previous chunk
            save_current_chunk()
            
            # Update section
            current_section = text
            current_subsection = None
            current_page = page
            print(f"[Parser] Found section heading: {text.encode('ascii', errors='replace').decode('ascii')}")
            continue
        
        # Check if this is a Title Case subsection heading
        if is_title_case_heading(text):
            # Validate that this is a real subsection heading
            if not has_substantial_content_after(paragraphs, idx):
                print(f"[Parser] Skipping potential image caption: {text[:50]}")
                # Treat as regular content
                if current_page is None:
                    current_page = page
                current_chunk_text.append(text)
                continue
            
            # Save previous chunk
            save_current_chunk()
            
            # Start new chunk with subsection heading
            current_subsection = text
            current_chunk_text = [text]
            current_page = page
            print(f"[Parser] Found subsection heading: {text.encode('ascii', errors='replace').decode('ascii')}")
            continue
        
        # Regular content - add to current chunk
        if current_page is None:
            current_page = page
        
        # For very long paragraphs (tables, etc), consider splitting
        if len(text) > 800:
            # Save current chunk first
            save_current_chunk()
            
            # Split long paragraph and create chunks
            sub_chunks = chunk_text_with_overlap(text, max_chars=1000, overlap_chars=150)
            for sub_chunk in sub_chunks:
                if sub_chunk.strip():
                    chunks.append({
                        'text': sub_chunk.strip(),
                        'page': page,
                        'section': current_section,
                        'subsection': current_subsection
                    })
        else:
            # Check if adding this paragraph would make chunk too large
            current_text_length = sum(len(t) for t in current_chunk_text)
            if current_text_length + len(text) > 1000 and current_chunk_text:
                # Save current chunk and start new one
                save_current_chunk()
                current_page = page
            
            current_chunk_text.append(text)
    
    # Save final chunk
    save_current_chunk()
    
    print(f"[Parser] Created {len(chunks)} aggressive paragraph-level chunks from {len(paragraphs)} paragraphs")
    print(f"[Parser] Target: 80-120 chunks, Achieved: {len(chunks)} chunks")
    
    # Calculate and report statistics
    chunk_sizes = [len(c['text']) for c in chunks]
    if chunk_sizes:
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        max_size = max(chunk_sizes)
        min_size = min(chunk_sizes)
        print(f"[Parser] Chunk size stats - Avg: {avg_size:.0f} chars, Min: {min_size}, Max: {max_size}")
        over_1000 = sum(1 for s in chunk_sizes if s > 1000)
        print(f"[Parser] Chunks over 1000 chars: {over_1000}/{len(chunks)} ({over_1000/len(chunks)*100:.1f}%)")
    
    return chunks


def parse_pdf_rulebook(pdf_path: str, doc_type: str = "rulebook", max_chunk_chars: int = 1000, overlap_chars: int = 150) -> List[Dict]:
    """
    Parses a PDF rulebook and returns a list of chunks with metadata.
    Skips irrelevant sections (credits, table of contents, ads, thanks, etc.).
    Cleans headers/footers and ensures all content is chunked properly.
    Extracts section names from content semantically.
    
    Args:
        pdf_path: Path to the PDF file
        doc_type: Type of document (default "rulebook")
        max_chunk_chars: Maximum characters per chunk (default 1000 for optimal context)
        overlap_chars: Characters to overlap between chunks (default 150 for better context)
    
    Returns:
        List of chunk dictionaries with text, metadata, and context
    """
    doc = fitz.open(pdf_path)
    chunks = []
    skip_patterns = [
        r"table of contents", r"contents", r"^\s*thank you", r"special thanks", r"credits", r"designed by", r"illustrated by",
        r"advertisement", r"visit our website", r"customer service", r"all rights reserved", r"contact\s*us", r"customer\s*support|technical\s*support|spiritual\s*support",
        r"component list", r"components", r"^game components$", r"^box contents$", r"^in the box$", r"^you should have$", r"^this game includes$",
        # Enhanced TOC detection
        r"^\s*table\s+of\s+contents\s*$", r"^\s*contents\s*$",
        r"page\s+\d+", r"\.\s*\.\s*\.\s*\d+",  # Page number patterns in TOC
        r"^\d+\s*$",  # Standalone page numbers
        # Credits section patterns
        r"narrative\s+design:", r"writing:", r"proofreading:", r"graphic\s+design:",
        r"illustrations?:", r"3d\s+modelling:", r"dtp:", r"production:",
        r"tests?\s+and\s+development:", r"internal\s+testing:",
        r"rulebook\s+&\s+gameplay", r"game\s+world\s+team:",
        r"based\s+on:", r"dedicated\s+to",
        # Component list patterns
        r"^\d+x\s+", r"^\d+\s+x\s+", r"quantity", r"component\s+type"
    ]
    skip_regex = re.compile("|".join(skip_patterns), re.IGNORECASE)
    
    # Track if we're in a TOC section (usually first few pages)
    in_toc_section = False
    toc_page_limit = 3  # Usually TOC is within first 3 pages
    
    current_section = None

    from datetime import datetime
    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    corrections_csv_path = "data/processed/corrections.csv"
    corrections_archive_path = f"data/processed/archive/corrections_{dt_str}.csv"
    unique_terms_path = "data/processed/unique_terms.csv"
    word_fragments_path = "data/processed/word_fragments.csv"
    section_headers = set()
    
    # First pass: collect all paragraphs with their page numbers
    all_paragraphs = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Check if we're potentially in TOC section
        if page_num < toc_page_limit:
            in_toc_section = True
        else:
            in_toc_section = False
        
        # Use layout-aware text extraction with "dict" mode
        # This preserves text blocks and their positioning
        try:
            # Extract with layout preservation
            page_dict = page.get_text("dict")
            text_blocks = []
            
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    block_lines = []
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        if line_text.strip():
                            block_lines.append(line_text.strip())
                    
                    if block_lines:
                        # Join lines in a block with single newline
                        block_text = "\n".join(block_lines)
                        text_blocks.append(block_text)
            
            # Join blocks with double newline to separate paragraphs
            text = "\n\n".join(text_blocks)
            
        except Exception as e:
            # Fallback to simple text extraction if layout parsing fails
            print(f"[Parser] Layout extraction failed on page {page_num + 1}, using simple method: {e}")
            text = page.get_text("text")
        
        # Fix encoding issues BEFORE any other processing
        text = fix_encoding_issues(text)
        
        lines = text.splitlines()
        # Remove likely headers/footers (first and last line)
        if len(lines) > 4:
            lines = lines[1:-1]
        text = "\n".join(lines)

        # Split by double newlines to get paragraphs
        for para in text.split("\n\n"):
            clean_para = para.strip()
            
            # Skip very short paragraphs (but not too aggressively)
            if len(clean_para) < 20:
                continue
            
            # Skip sections matching skip patterns
            if skip_regex.search(clean_para):
                print(f"[Parser] Skipping paragraph (matched skip pattern): {clean_para[:60].encode('ascii', errors='replace').decode('ascii')}...")
                continue
            
            # Enhanced TOC detection: skip paragraphs with dotted lines and page numbers
            if in_toc_section and (re.search(r'\.\s*\.\s*\.', clean_para) or re.search(r'\s*\d+\s*$', clean_para)):
                print(f"[Parser] Skipping TOC entry: {clean_para[:60]}...")
                continue
            
            # Skip page numbers
            if re.fullmatch(r"\d{1,3}", clean_para):
                continue

            # Don't skip section headers anymore - we need them for section-based chunking
            # Just collect them for later reference
            first_line = clean_para.split("\n")[0].strip()
            if (len(first_line) <= 40 and (first_line.isupper() or sum(1 for c in first_line if c.isupper()) > 3)):
                section_headers.add(first_line)
            
            # Detect and format tables
            clean_para = detect_and_format_table(clean_para)
            
            all_paragraphs.append({
                'text': clean_para,
                'page': page_num + 1
            })
    
    print(f"[Parser] Extracted {len(all_paragraphs)} paragraphs from {len(doc)} pages")

    # Apply new section-based chunking logic
    section_chunks = chunk_by_sections(all_paragraphs)
    
    # Second pass: spellcheck each section chunk
    for idx, chunk_info in enumerate(section_chunks):
        chunk_text = chunk_info['text']
        page_num = chunk_info['page']
        section = chunk_info.get('section', 'Unknown')
        subsection = chunk_info.get('subsection')
        
        # Spellcheck the chunk
        spell_result = correct_spelling(
            chunk_text,
            generate_corrections_file=True,
            corrections_output_path=corrections_csv_path,
            unique_terms_file=unique_terms_path,
            word_fragments_file=word_fragments_path
        )
        print(f"[Parser] Spellchecked chunk {idx+1}/{len(section_chunks)} (page {page_num}, section: {section}): {chunk_text[:40]}...")
        corrected_text = spell_result['corrected_text'] if isinstance(spell_result, dict) else spell_result

        # Create final chunk entry
        chunk = {
            "text": corrected_text,
            "page": page_num,
            "section": section,
            "subsection": subsection,
            "doc_type": doc_type,
            "chunk_index": idx,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        chunks.append(chunk)
    
    print(f"[Parser] Created {len(chunks)} total chunks from {len(section_chunks)} section-based chunks")
    
    # Save section headers for answer filtering
    import os
    section_headers_path = "data/processed/section_headers.txt"
    os.makedirs(os.path.dirname(section_headers_path), exist_ok=True)
    with open(section_headers_path, "w", encoding="utf-8") as shf:
        for header in sorted(section_headers):
            shf.write(header + "\n")
    
    # Deduplicate and sort corrections file at the end
    from spellcheck_utils import correct_spelling as _cs
    _cs('', generate_corrections_file=True, corrections_output_path=corrections_csv_path, unique_terms_file=unique_terms_path, word_fragments_file=word_fragments_path, deduplicate_corrections=True)

    # Archive a timestamped copy of the deduplicated corrections file
    import shutil
    try:
        shutil.copyfile(corrections_csv_path, corrections_archive_path)
        print(f"[Parser] Archived corrections file to {corrections_archive_path}")
    except Exception as e:
        print(f"[Parser] Could not archive corrections file: {e}")

    # Learn OCR patterns from corrections and update configuration
    try:
        from ocr_learning import update_ocr_corrections_from_learning
        print("\n[Parser] Analyzing corrections to learn OCR patterns...")
        learned = update_ocr_corrections_from_learning(
            corrections_csv_path,
            ocr_config_path="data/processed/pdf_ocr_corrections.json"
        )
        if learned:
            print("[Parser] ✓ OCR corrections updated! Re-run parser to apply new patterns.\n")
    except Exception as e:
        print(f"[Parser] Could not update OCR corrections: {e}")
    
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
    # Split by sentence endings with better handling of sentence boundaries
    sentence_end_re = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_end_re.split(all_text)
    capitalized_words = set()
    fully_upper_words = set()
    for sent in sentences:
        # Strip leading/trailing whitespace and skip empty sentences
        sent = sent.strip()
        if not sent:
            continue
        
        words_in_sent = re.findall(r"\b\w+\b", sent)
        for i, word in enumerate(words_in_sent):
            # IMPORTANT: Skip first word of each sentence (sentence-initial capitalization)
            if i == 0:
                continue
            
            # Add all-uppercase words/acronyms FIRST (e.g., "HP", "XP", "ARTIFACT")
            # This catches 2+ character acronyms before the length check
            if re.match(r"^[A-Z]{2,}$", word):
                fully_upper_words.add(word)
                continue  # Skip remaining checks for this word
            
            # Only add words with 3+ characters to avoid single initials/abbreviations
            # (All-uppercase acronyms already handled above)
            if len(word) < 3:
                continue
            
            # Add capitalized words (e.g., "Stalker", "Anomaly", "Psionic")
            if re.match(r"^[A-Z][a-zA-Z0-9\-]{2,}$", word):
                capitalized_words.add(word)
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
