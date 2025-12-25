import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import pickle

class Embedder:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):  # 768 dims for quality
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_chunks(self, chunks_file, output_file):
        print(f"[Embedder] Loading chunks from {chunks_file}")
        if not Path(chunks_file).exists():
            raise FileNotFoundError("Chunks file not found: " + chunks_file)
        with open(chunks_file, 'rb') as f:
            data = pickle.load(f)
        chunks_df = pd.DataFrame(data) if isinstance(data, list) else data

        print(f"[Embedder] Generating embeddings for {len(chunks_df)} chunks...")
        embeddings = self.model.encode(
            chunks_df['text'].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # ensure stored vectors are unit-normalized
        )
        chunks_df['embedding'] = embeddings.tolist()
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        chunks_df.to_parquet(output_file, index=False)
        print(f"[Embedder] Saved embeddings to {output_file}")
        return chunks_df

if __name__ == '__main__':
    embedder = Embedder()
    embedder.embed_chunks(
        'data/processed/chunks.pkl',
        'data/processed/chunks_embeddings.parquet'
    )