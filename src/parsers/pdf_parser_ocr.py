import fitz  # PyMuPDF - for PDF page rendering
from typing import List, Dict, Optional
import re
import json
import os
import logging
from PIL import Image
import io
from spellcheck_utils import correct_spelling

# OCR Libraries
try:
    import pytesseract
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError
    from pytesseract import TesseractNotFoundError
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("[Warning] OCR libraries not installed.")
    print("[Warning] Install with: pip install pytesseract pdf2image pillow")
    print("[Warning] Also install Tesseract-OCR: https://github.com/tesseract-ocr/tesseract")
    print("[Warning] Also install Poppler: https://github.com/oschwartz10612/poppler-windows/releases/")
    PDFInfoNotInstalledError = Exception  # Fallback for type hints
    TesseractNotFoundError = Exception  # Fallback for type hints

# Setup logging for skipped paragraphs
SKIPPED_LOG_FILE = "skipped_paragraphs_ocr.log"
logging.basicConfig(
    filename=SKIPPED_LOG_FILE,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s: %(message)s'
)
skipped_logger = logging.getLogger('skipped_paragraphs_ocr')

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
                print(f"[OCR Parser] Loaded OCR corrections from {config_path}")
                _OCR_CORRECTIONS_CACHE = config
                return config
        except Exception as e:
            print(f"[OCR Parser] Error loading OCR corrections from {config_path}: {e}")
            print(f"[OCR Parser] No OCR corrections will be applied")
    else:
        print(f"[OCR Parser] OCR corrections file not found at {config_path}")
        print(f"[OCR Parser] Create this file to enable automatic OCR error correction")
    
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
    
    # Apply character map - separate single-char and multi-char mappings
    character_map = config.get("character_map", {})
    single_char_map = {}
    multi_char_replacements = []
    
    for key, value in character_map.items():
        if len(key) == 1:
            single_char_map[key] = value
        else:
            # Multi-character mappings need to be applied separately
            multi_char_replacements.append((key, value))
    
    # Apply single-character translations
    if single_char_map:
        char_map = str.maketrans(single_char_map)
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


def extract_text_with_ocr(pdf_path: str, output_folder: str = "data/ocr_extracted", poppler_path: Optional[str] = None, tesseract_cmd: Optional[str] = None) -> List[Dict]:
    """
    Extract text from PDF using OCR (Tesseract).
    Saves each page's OCR text to a separate file in the output folder.
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Folder to save OCR-extracted text files
        poppler_path: Optional path to poppler binaries (Windows only)
        tesseract_cmd: Optional path to tesseract executable (e.g., r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    
    Returns:
        List of dictionaries with page number and extracted text
    """
    if not HAS_OCR:
        raise ImportError(
            "OCR libraries not available.\n"
            "Install with: pip install pytesseract pdf2image pillow\n"
            "Also install Tesseract-OCR: https://github.com/tesseract-ocr/tesseract\n"
            "Also install Poppler: https://github.com/oschwartz10612/poppler-windows/releases/"
        )
    
    # Set tesseract command path if provided
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    # Set tesseract command path if provided
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    # Validate poppler path if provided
    if poppler_path:
        if not os.path.exists(poppler_path):
            print(f"\n[OCR Parser] WARNING: Provided poppler_path does not exist: {poppler_path}")
            print(f"[OCR Parser] Please check the path and try again.")
            print(f"[OCR Parser] Common locations:")
            print(f"  - C:\\poppler\\Library\\bin")
            print(f"  - C:\\Program Files\\poppler\\Library\\bin")
            raise FileNotFoundError(f"Poppler path not found: {poppler_path}")
        else:
            print(f"[OCR Parser] Using poppler from: {poppler_path}")
    
    # Create output folder structure
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the PDF filename without extension for the subfolder
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_folder = os.path.join(output_folder, pdf_name)
    os.makedirs(pdf_output_folder, exist_ok=True)
    
    print(f"[OCR Parser] Converting PDF to images and extracting text with OCR...")
    print(f"[OCR Parser] Output folder: {pdf_output_folder}")
    
    # Open PDF to get page count
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()
    
    # Convert PDF pages to images
    print(f"[OCR Parser] Processing {num_pages} pages...")
    
    ocr_results = []
    
    # Process each page
    for page_num in range(num_pages):
        print(f"[OCR Parser] Processing page {page_num + 1}/{num_pages}...")
        
        try:
            # Convert single page to image using pdf2image
            # Add poppler_path parameter if provided (Windows)
            convert_kwargs = {
                'pdf_path': pdf_path,
                'first_page': page_num + 1,
                'last_page': page_num + 1,
                'dpi': 300  # Higher DPI for better OCR quality
            }
            
            if poppler_path:
                convert_kwargs['poppler_path'] = poppler_path
            
            images = convert_from_path(**convert_kwargs)
        except PDFInfoNotInstalledError:
            print("\n" + "="*70)
            print("ERROR: Poppler is not installed or not in PATH!")
            print("="*70)
            print("\nFor Windows users:")
            print("1. Download poppler from: https://github.com/oschwartz10612/poppler-windows/releases/")
            print("2. Extract the zip file (e.g., to C:\\poppler)")
            print("3. Either:")
            print("   a) Add C:\\poppler\\Library\\bin to your system PATH, OR")
            print("   b) Pass poppler_path parameter: extract_text_with_ocr(pdf_path, poppler_path=r'C:\\poppler\\Library\\bin')")
            print("\nFor Linux users:")
            print("   sudo apt-get install poppler-utils")
            print("\nFor Mac users:")
            print("   brew install poppler")
            print("="*70 + "\n")
            raise
        
        if not images:
            print(f"[OCR Parser] Warning: Could not convert page {page_num + 1} to image")
            continue
        
        # Extract text using Tesseract OCR
        page_image = images[0]
        
        try:
            ocr_text = pytesseract.image_to_string(page_image, lang='eng')
        except TesseractNotFoundError:
            print("\n" + "="*70)
            print("ERROR: Tesseract OCR is not installed or not in PATH!")
            print("="*70)
            print("\nFor Windows users:")
            print("1. Download Tesseract installer from:")
            print("   https://github.com/UB-Mannheim/tesseract/wiki")
            print("2. Install it (default location: C:\\Program Files\\Tesseract-OCR)")
            print("3. Either:")
            print("   a) Add C:\\Program Files\\Tesseract-OCR to your system PATH, OR")
            print("   b) Pass tesseract_cmd parameter when calling the function")
            print("\nFor Linux users:")
            print("   sudo apt-get install tesseract-ocr")
            print("\nFor Mac users:")
            print("   brew install tesseract")
            print("="*70 + "\n")
            raise
        
        # Save OCR text to file
        page_text_file = os.path.join(pdf_output_folder, f"page_{page_num + 1:03d}.txt")
        with open(page_text_file, 'w', encoding='utf-8') as f:
            f.write(ocr_text)
        
        print(f"[OCR Parser] Saved OCR text to {page_text_file}")
        
        ocr_results.append({
            'page': page_num + 1,
            'text': ocr_text,
            'file': page_text_file
        })
    
    print(f"[OCR Parser] OCR extraction complete. Extracted text from {len(ocr_results)} pages.")
    
    return ocr_results


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
        'GRAVITATIONAL', 'IMPROVED', 'ADVANCED', 'BANDIT', 'CONTROLLER'
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
        skipped_logger.info(f"[SKIPPED - TitleCase Entry due to :] {text}")
        # return False
    
    # Skip if ends with comma or "and" (likely list items)
    if text.strip().endswith(',') or text.strip().endswith('and'):
        skipped_logger.info(f"[SKIPPED - TitleCase Entry due to , or and] {text}")
        # return False
    
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
            skipped_logger.info(f"[SKIPPED - Example:] {text}")
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
            print(f"[OCR Parser] Entering credits section on page {page}")
            skipped_logger.info(f"[SKIPPED - Second credits filtering] {text}")
            continue
        
        if re.search(r"component\s*list|game\s*components", text.lower()):
            in_components_section = True
            print(f"[OCR Parser] Entering components section on page {page}")
            continue
        
        # Skip if in credits or components section
        if in_credits_section or in_components_section:
            # Check if we've left these sections (new ALL CAPS section found)
            if is_all_caps_section(text) and len(text.split()) >= 2:
                in_credits_section = False
                in_components_section = False
                print(f"[OCR Parser] Exiting credits/components section")
            else:
                continue
        
        # Skip Example paragraphs
        if should_skip_example(text):
            print(f"[OCR Parser] Skipping Example paragraph on page {page}")
            continue
        
        # Check if this is an ALL CAPS section heading (with content validation)
        if is_all_caps_section(text):
            # Validate that this is a real section heading, not a card scan
            if not has_substantial_content_after(paragraphs, idx):
                print(f"[OCR Parser] Skipping card scan/image caption: {text[:50]}")
                continue
            
            # Save previous chunk
            save_current_chunk()
            
            # Update section
            current_section = text
            current_subsection = None
            current_page = page
            print(f"[OCR Parser] Found section heading: {text.encode('ascii', errors='replace').decode('ascii')}")
            continue
        
        # Check if this is a Title Case subsection heading
        if is_title_case_heading(text):
            # Validate that this is a real subsection heading
            if not has_substantial_content_after(paragraphs, idx):
                print(f"[OCR Parser] Skipping potential image caption: {text[:50]}")
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
            print(f"[OCR Parser] Found subsection heading: {text.encode('ascii', errors='replace').decode('ascii')}")
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
    
    print(f"[OCR Parser] Created {len(chunks)} aggressive paragraph-level chunks from {len(paragraphs)} paragraphs")
    print(f"[OCR Parser] Target: 80-120 chunks, Achieved: {len(chunks)} chunks")
    
    # Calculate and report statistics
    chunk_sizes = [len(c['text']) for c in chunks]
    if chunk_sizes:
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        max_size = max(chunk_sizes)
        min_size = min(chunk_sizes)
        print(f"[OCR Parser] Chunk size stats - Avg: {avg_size:.0f} chars, Min: {min_size}, Max: {max_size}")
        over_1000 = sum(1 for s in chunk_sizes if s > 1000)
        print(f"[OCR Parser] Chunks over 1000 chars: {over_1000}/{len(chunks)} ({over_1000/len(chunks)*100:.1f}%)")
    
    return chunks


def parse_pdf_rulebook(pdf_path: str, doc_type: str = "rulebook", max_chunk_chars: int = 1000, overlap_chars: int = 150, poppler_path: Optional[str] = None, tesseract_cmd: Optional[str] = None) -> List[Dict]:
    """
    Parses a PDF rulebook using OCR and returns a list of chunks with metadata.
    Extracts text via OCR and saves to separate folder before processing.
    Skips irrelevant sections (credits, table of contents, ads, thanks, etc.).
    Cleans headers/footers and ensures all content is chunked properly.
    Extracts section names from content semantically.
    
    Args:
        pdf_path: Path to the PDF file
        doc_type: Type of document (default "rulebook")
        max_chunk_chars: Maximum characters per chunk (default 1000 for optimal context)
        overlap_chars: Characters to overlap between chunks (default 150 for better context)
        poppler_path: Optional path to poppler binaries (Windows only)
        tesseract_cmd: Optional path to tesseract executable (Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    
    Returns:
        List of chunk dictionaries with text, metadata, and context
    """
    if not HAS_OCR:
        raise ImportError(
            "OCR libraries not available.\n"
            "Install with: pip install pytesseract pdf2image pillow\n"
            "Also install Tesseract-OCR: https://github.com/tesseract-ocr/tesseract\n"
            "Also install Poppler: https://github.com/oschwartz10612/poppler-windows/releases/"
        )
    
    # Step 1: Extract text using OCR and save to folder
    ocr_results = extract_text_with_ocr(pdf_path, poppler_path=poppler_path, tesseract_cmd=tesseract_cmd)
    
    chunks = []
    skip_patterns = [
        r"table of contents", r"^\s*thank you", r"special thanks", r"credits", r"designed by", r"illustrated by",
        r"advertisement", r"visit our website", r"customer service", r"all rights reserved", r"contact\s*us", r"customer\s*support|technical\s*support",
        r"^\s*component list", r"^game components$", r"^box contents$", r"^in the box$", r"^you should have$", r"^this game includes$",
        # Enhanced TOC detection
        r"^\s*table\s+of\s+contents\s*$", r"^\s*contents\s*$",
        r"\.\s*\.\s*\.\s*\.+\s*\d+\s*$",  # TOC entries with dots leading to page numbers
        r"^\d+\s*$",  # Standalone page numbers
        # Credits section patterns
        r"narrative\s+design:", r"writing:", r"proofreading:", r"graphic\s+design:", 
        r"3d\s+modelling:", r"dtp:", r"production:",
        r"tests?\s+and\s+development:", r"internal\s+testing:",
        r"rulebook\s+&\s+gameplay", r"game\s+world\s+team:", r"dedicated\s+to",
        # Component list patterns
        r"^\d+x\s+", r"^\d+\s+x\s+", r"^\d+\s*x.*quantity", r"^\s*component\s+type"
    ]
    skip_regex = re.compile("|".join(skip_patterns), re.IGNORECASE)
    
    # Track if we're in a TOC section (usually first few pages)
    in_toc_section = False
    toc_page_limit = 3  # Usually TOC is within first 3 pages
    
    current_section = None

    from datetime import datetime
    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    corrections_csv_path = "data/processed/corrections_ocr.csv"
    corrections_archive_path = f"data/processed/archive/corrections_ocr_{dt_str}.csv"
    unique_terms_path = "data/processed/unique_terms_ocr.csv"
    word_fragments_path = "data/processed/word_fragments_ocr.csv"
    section_headers = set()
    
    # First pass: collect all paragraphs with their page numbers from OCR results
    all_paragraphs = []
    
    for ocr_page in ocr_results:
        page_num = ocr_page['page'] - 1  # Convert to 0-indexed
        text = ocr_page['text']
        
        # Check if we're potentially in TOC section
        if page_num < toc_page_limit:
            in_toc_section = True
        else:
            in_toc_section = False
        
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
                skipped_logger.info(f"[SKIPPED - Pattern Match] Page {page_num + 1}: {clean_para}")
                continue
            
            # Enhanced TOC detection: skip paragraphs with dotted lines and page numbers
            if in_toc_section and (re.search(r'\.\s*\.\s*\.', clean_para) or re.search(r'\s*\d+\s*$', clean_para)):
                skipped_logger.info(f"[SKIPPED - TOC Entry] Page {page_num + 1}: {clean_para}")
                continue
            
            # Skip page numbers
            if re.fullmatch(r"\d{1,3}", clean_para):
                continue

            # Don't skip section headers - we need them for section-based chunking
            first_line = clean_para.split("\n")[0].strip()
            if (len(first_line) <= 40 and (first_line.isupper() or sum(1 for c in first_line if c.isupper()) > 3)):
                section_headers.add(first_line)
            
            # Detect and format tables
            clean_para = detect_and_format_table(clean_para)
            
            all_paragraphs.append({
                'text': clean_para,
                'page': page_num + 1
            })
    
    print(f"[OCR Parser] Extracted {len(all_paragraphs)} paragraphs from {len(ocr_results)} OCR pages")

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
        print(f"[OCR Parser] Spellchecked chunk {idx+1}/{len(section_chunks)} (page {page_num}, section: {section}): {chunk_text[:40]}...")
        corrected_text = spell_result['corrected_text'] if isinstance(spell_result, dict) else spell_result

        # Create final chunk entry
        chunk = {
            "text": corrected_text,
            "page": page_num,
            "section": section,
            "subsection": subsection,
            "doc_type": doc_type,
            "chunk_index": idx,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "extraction_method": "OCR"
        }
        chunks.append(chunk)
    
    print(f"[OCR Parser] Created {len(chunks)} total chunks from {len(section_chunks)} section-based chunks")
    
    # Save section headers for answer filtering
    section_headers_path = "data/processed/section_headers_ocr.txt"
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
        print(f"[OCR Parser] Archived corrections file to {corrections_archive_path}")
    except Exception as e:
        print(f"[OCR Parser] Could not archive corrections file: {e}")

    # Learn OCR patterns from corrections and update configuration
    try:
        from ocr_learning import update_ocr_corrections_from_learning
        print("\n[OCR Parser] Analyzing corrections to learn OCR patterns...")
        learned = update_ocr_corrections_from_learning(
            corrections_csv_path,
            ocr_config_path="data/processed/pdf_ocr_corrections.json"
        )
        if learned:
            print("[OCR Parser] ✓ OCR corrections updated! Re-run parser to apply new patterns.\n")
    except Exception as e:
        print(f"[OCR Parser] Could not update OCR corrections: {e}")
    
    return chunks


if __name__ == "__main__":
    import sys
    import os
    import pickle
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser_ocr.py <pdf_path> [output_pickle] [poppler_path] [tesseract_path]")
        print("\nExample (with paths for Windows):")
        print('  python pdf_parser_ocr.py rulebook.pdf chunks.pkl "C:\\poppler-XX.XX.X\\Library\\bin" "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"')
        print("\nNote: Requires Tesseract OCR and Poppler to be installed.")
        print("  Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Poppler: https://github.com/oschwartz10612/poppler-windows/releases/")
        print("\nTo find your poppler path:")
        print("  1. Look in the folder where you extracted poppler")
        print("  2. Find the 'Library\\bin' subfolder (e.g., C:\\poppler-24.08.0\\Library\\bin)")
        print("  3. That folder should contain pdftoppm.exe and pdfinfo.exe")
        sys.exit(1)
    
    from datetime import datetime
    pdf_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    else:
        out_path = os.path.join("data", "processed", "chunks_ocr.pkl")
    
    # Optional poppler path for Windows
    poppler_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Optional tesseract path for Windows
    tesseract_cmd = sys.argv[4] if len(sys.argv) > 4 else None
    
    # Always create archive copy with date-time
    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_dir = os.path.join("data", "processed", "archive")
    os.makedirs(archive_dir, exist_ok=True)
    archive_pkl = os.path.join(archive_dir, f"chunks_ocr_{dt_str}.pkl")
    archive_json = os.path.join(archive_dir, f"chunks_ocr_{dt_str}.json")
    
    chunks = parse_pdf_rulebook(pdf_path, poppler_path=poppler_path, tesseract_cmd=tesseract_cmd)
    
    # Extract unique terms from all chunked text
    import re
    import requests
    all_text = " ".join(chunk["text"] for chunk in chunks)
    
    # Find capitalized words and all-uppercase words
    sentence_end_re = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_end_re.split(all_text)
    capitalized_words = set()
    fully_upper_words = set()
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        
        words_in_sent = re.findall(r"\b\w+\b", sent)
        for i, word in enumerate(words_in_sent):
            # Skip first word of each sentence
            if i == 0:
                continue
            
            # Add all-uppercase words/acronyms
            if re.match(r"^[A-Z]{2,}$", word):
                fully_upper_words.add(word)
                continue
            
            # Only add words with 3+ characters
            if len(word) < 3:
                continue
            
            # Add capitalized words
            if re.match(r"^[A-Z][a-zA-Z0-9\-]{2,}$", word):
                capitalized_words.add(word)
    
    words = capitalized_words.union(fully_upper_words)
    
    # Download stopwords list
    stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
    stopwords = set(stopwords_list.decode().splitlines())
    
    def normalize(term):
        return term.lower().strip()
    
    normalized_stopwords = set(normalize(sw) for sw in stopwords)
    normalized_words = {normalize(w): w for w in words}
    filtered_words = [original for norm, original in normalized_words.items() if norm not in normalized_stopwords]
    
    # Write unique terms to CSV
    unique_terms_path = "data/processed/unique_terms_ocr.csv"
    os.makedirs(os.path.dirname(unique_terms_path), exist_ok=True)
    for_write = [normalized_words[n] for n in sorted(normalized_words) if n in {normalize(w) for w in filtered_words}]
    with open(unique_terms_path, "w", encoding="utf-8") as f:
        for w in for_write:
            f.write(w + "\n")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save main chunk file
    with open(out_path, "wb") as f:
        pickle.dump(chunks, f)
    
    # Save archive copies
    with open(archive_pkl, "wb") as f:
        pickle.dump(chunks, f)
    
    with open(archive_json, "w", encoding="utf-8") as jf:
        json.dump(chunks, jf, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to {out_path}")
    print(f"Archived {len(chunks)} chunks to {archive_pkl} and {archive_json}")
    print(f"Extracted {len(chunks)} chunks.")
    print(f"Extracted {len(words)} unique terms to {unique_terms_path}")
    
    for c in chunks[:3]:
        print(c)
