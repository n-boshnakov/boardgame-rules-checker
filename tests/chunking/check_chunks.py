import pickle

# Load chunks
with open('data/processed/chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

# Find chunks with "marking" or "sticker"
marking_chunks = [(i, c) for i, c in enumerate(chunks) if 'marking' in c['text'].lower() or 'sticker' in c['text'].lower()]

print(f'Found {len(marking_chunks)} chunks with marking/sticker:\n')
for idx, (i, c) in enumerate(marking_chunks[:5]):
    print(f"{idx+1}. Chunk {i}: Section: {c['section']}, Subsection: {c['subsection']}")
    print(f"   Has 'marking': {'marking' in c['text'].lower()}")
    print(f"   Has 'sticker': {'sticker' in c['text'].lower()}")
    if 'marking' in c['text'].lower():
        # Find context around "marking"
        text_lower = c['text'].lower()
        pos = text_lower.find('marking')
        context = c['text'][max(0, pos-50):min(len(c['text']), pos+150)]
        print(f"   Context: ...{context}...")
    print()
