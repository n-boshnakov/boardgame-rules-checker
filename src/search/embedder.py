from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os
from typing import List, Dict

class RulebookEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: List[Dict]) -> pd.DataFrame:
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        df = pd.DataFrame(chunks)
        df["embedding"] = list(embeddings)
        return df

    def save_embeddings(self, df: pd.DataFrame, out_path: str):
        # Save as Parquet for efficient storage
        df.to_parquet(out_path, index=False)

if __name__ == "__main__":
    import sys
    import pickle
    if len(sys.argv) < 3:
        print("Usage: python embedder.py <chunks_pickle> <output_parquet>")
        exit(1)
    chunks_path = sys.argv[1]
    out_path = sys.argv[2]
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    embedder = RulebookEmbedder()
    df = embedder.embed_chunks(chunks)
    embedder.save_embeddings(df, out_path)
    print(f"Saved embeddings to {out_path}")
