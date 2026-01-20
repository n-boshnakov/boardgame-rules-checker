"""Generate embeddings for text chunks using sentence transformers.

This module creates dense vector embeddings from text chunks for semantic search.
Uses the BAAI/bge-m3 model (1024 dimensions) - hybrid sparse/dense, excellent cross-domain.
"""
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import pickle


class Embedder:
    """Generate and manage text embeddings for semantic search.
    
    Attributes:
        model: SentenceTransformer model for encoding text
        model_name: Name/path of the embedding model
    """
    
    def __init__(self, model_name='BAAI/bge-m3'):
        """Initialize embedder with specified model.
        
        Args:
            model_name: HuggingFace model identifier (default: bge-m3 with 1024 dims)
        """
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_chunks(self, chunks_file, output_file):
        """Generate embeddings for chunks and save to parquet format.
        
        Args:
            chunks_file: Path to pickle file containing chunk data
            output_file: Path to save embeddings parquet file
            
        Returns:
            DataFrame with chunks and their embeddings
            
        Raises:
            FileNotFoundError: If chunks_file doesn't exist
        """
        print(f"[Embedder] Loading chunks from {chunks_file}")
        
        # Load chunks from pickle file
        chunks_path = Path(chunks_file)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
            
        with open(chunks_path, 'rb') as f:
            data = pickle.load(f)
        
        # Convert to DataFrame if needed
        chunks_df = pd.DataFrame(data) if isinstance(data, list) else data
        
        # Generate embeddings with progress bar
        print(f"[Embedder] Generating embeddings for {len(chunks_df)} chunks...")
        embeddings = self.model.encode(
            chunks_df['text'].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Unit-normalize for cosine similarity
        )
        
        # Add embeddings to DataFrame
        chunks_df['embedding'] = embeddings.tolist()
        
        # Save to parquet format
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_df.to_parquet(output_path, index=False)
        
        print(f"[Embedder] Saved embeddings to {output_file}")
        print(f"[Embedder] Embedding dimensions: {embeddings.shape[1]}")
        print(f"[Embedder] Total chunks embedded: {len(chunks_df)}")
        
        return chunks_df


if __name__ == '__main__':
    # Default usage: embed chunks from standard location
    embedder = Embedder()
    embedder.embed_chunks(
        chunks_file='data/processed/chunks.pkl',
        output_file='data/processed/chunks_embeddings.parquet'
    )