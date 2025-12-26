from sentence_transformers import SentenceTransformer, CrossEncoder, util
from elasticsearch import Elasticsearch
from elasticsearch import BadRequestError
from typing import List, Dict, Optional
import numpy as np
import re

ES_INDEX = "rulebook_chunks"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class RulebookRetriever:
    def __init__(self, es_host: str = "http://localhost:9200", model_name: str = MODEL_NAME, use_reranker: bool = True):
        self.es = Elasticsearch(es_host)
        print(f"[Retriever] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.use_reranker = use_reranker
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_reranker else None
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
