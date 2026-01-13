import sys
import os
# This script is in src/qa/testing/, go up 3 levels to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from search.retriever import RulebookRetriever
import re

# Initialize retriever
retriever = RulebookRetriever(use_reranker=True)

question = 'How many players can play this game?'
print(f"Question: {question}")

# Get results
results = retriever.search(question, top_k=25, search_type="hybrid")

print(f"\n\nTop 3 chunks:")
for i, chunk in enumerate(results[:3], 1):
    text = chunk.get('text', '')
    print(f"\n{i}. ({len(text)} chars): {text[:200]}...")
    
    # Test sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    print(f"   Sentences: {len(sentences)}")
    for j, sent in enumerate(sentences[:3], 1):
        print(f"     {j}. {sent[:100]}...")

# Test embedding model
print("\n\nTesting sentence similarity:")
from sentence_transformers import util
import numpy as np

question_emb = retriever.model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]

test_sentences = [
    "This game can be played by 1-4 players.",
    "The game has multiple phases in each round.",
    "S.T.A.L.K.E.R. The Board Game is a 1-4 player fully co-op, story-based zone crawler."
]

sent_embs = retriever.model.encode(test_sentences, convert_to_numpy=True, normalize_embeddings=True)
similarities = util.cos_sim(question_emb, sent_embs)[0].cpu().numpy()

for sent, sim in zip(test_sentences, similarities):
    print(f"  {sim:.4f}: {sent}")
