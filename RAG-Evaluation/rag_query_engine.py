import os
from typing import List, Dict, Tuple
from colorama import Fore, init
import chromadb
import requests
import numpy as np
import torch

init(autoreset=False)

class RAGQueryEngine:
    """Simple RAG Query Engine with retrieval and reranking using Ollama."""

    def __init__(
        self,
        db_path: str = "./chromadb_storage",
        ollama_model: str = "all-minilm:33m",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the RAG Query Engine.

        Args:
            db_path: Path to ChromaDB persistent storage
            ollama_model: Ollama embedding/rerank model (e.g., all-minilm:33m)
            ollama_base_url: Ollama server URL
        """
        self.db_path = db_path
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url

        self._initialize_chromadb()

    def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB client and get collection."""
        try:
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"ChromaDB storage not found at {self.db_path}")

            self.client = chromadb.PersistentClient(path=self.db_path)
            collections = self.client.list_collections()

            if not collections:
                raise ValueError("No collections found in ChromaDB")

            self.collection = collections[-1]

        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to initialize ChromaDB: {str(e)}" + Fore.RESET)
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama using the specified model."""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.ollama_model,
                    "prompt": text
                },
                timeout=120
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return embedding
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to get embedding from Ollama: {str(e)}" + Fore.RESET)
            raise

    def query(self, query_text: str, query_embedding: list = None, k: int = 10) -> List[Dict]:
        """Query ChromaDB and retrieve top-k results."""
        try:
            if query_embedding is None:
                query_embedding = self._get_embedding(query_text)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            chunks = []
            for doc, metadata, distance in zip(documents, metadatas, distances):
                chunks.append({
                    "text": doc,
                    "metadata": metadata,
                    "distance": distance
                })

            return chunks

        except Exception as e:
            print(Fore.RED + f"[ERROR] Query failed: {str(e)}" + Fore.RESET)
            raise

    def rerank(self, chunks: List[Dict], query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Rerank chunks using Ollama all-minilm:33m (cosine similarity).
        """
        try:
            # Get embedding for the query
            query_embedding = np.array(self._get_embedding(query))
            # Get embedding for each chunk
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_embeddings = [np.array(self._get_embedding(text)) for text in chunk_texts]
            # Cosine similarity
            scores = []
            for text, emb in zip(chunk_texts, chunk_embeddings):
                score = float(np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb)))
                scores.append((text, score))
            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:k]
        except Exception as e:
            print(Fore.RED + f"[ERROR] Ollama MiniLM reranking failed: {str(e)}" + Fore.RESET)
            raise
        
        
    def rerank_biencoder(self, chunks: List[Dict], query: str, k: int = 1) -> List[Tuple[str, float]]:
        """
        Bi-encoder reranking using cosine similarity, all embeddings from Ollama.
        """
        try:
            query_embedding = np.array(self._get_embedding(query))
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_embeddings = [np.array(self._get_embedding(text)) for text in chunk_texts]
            scores = []
            for text, emb in zip(chunk_texts, chunk_embeddings):
                score = float(np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb)))
                scores.append((text, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:k]
        except Exception as e:
            print(Fore.RED + f"[ERROR] Ollama biencoder reranking failed: {str(e)}" + Fore.RESET)
            raise


















# import os
# from typing import List, Dict, Tuple
# from colorama import Fore, init
# import chromadb
# from sentence_transformers import SentenceTransformer
# import requests
# import torch
# import numpy as np

# init(autoreset=False)

# class RAGQueryEngine:
#     """Simple RAG Query Engine with retrieval and reranking using Ollama Cross Encoder."""
    
#     def __init__(
#         self,
#         db_path: str = "./chromadb_storage",
#         ollama_cross_encoder: str = "all-minilm:33m",
#         ollama_model: str = "embeddinggemma:300m",
#         ollama_base_url: str = "http://localhost:11434"
#     ):
#         """
#         Initialize the RAG Query Engine.
        
#         Args:
#             db_path: Path to ChromaDB persistent storage
#             ollama_cross_encoder: Ollama model for reranking (e.g., all-minilm:33m)
#             ollama_model: Ollama embedding model (used for retrieval)
#             ollama_base_url: Ollama server URL
#         """
#         self.db_path = db_path
#         self.ollama_cross_encoder = ollama_cross_encoder
#         self.ollama_model = ollama_model
#         self.ollama_base_url = ollama_base_url
        
#         self._initialize_chromadb()
#         self._initialize_bi_encoder()
        
#     def _initialize_chromadb(self) -> None:
#         """Initialize ChromaDB client and get collection."""
#         try:
#             if not os.path.exists(self.db_path):
#                 raise FileNotFoundError(f"ChromaDB storage not found at {self.db_path}")
            
#             self.client = chromadb.PersistentClient(path=self.db_path)
#             collections = self.client.list_collections()
            
#             if not collections:
#                 raise ValueError("No collections found in ChromaDB")
            
#             self.collection = collections[-1]
            
#         except Exception as e:
#             print(Fore.RED + f"[ERROR] Failed to initialize ChromaDB: {str(e)}" + Fore.RESET)
#             raise

#     def _initialize_bi_encoder(self) -> None:
#         """Initialize bi-encoder model for semantic similarity fallback."""
#         try:
#             self.bi_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
#         except Exception as e:
#             print(Fore.RED + f"[ERROR] Failed to initialize bi-encoder: {str(e)}" + Fore.RESET)
#             raise
    
#     def _get_embedding(self, text: str) -> List[float]:
#         """Get embedding from Ollama using the same model as indexing."""
#         try:
#             response = requests.post(
#                 f"{self.ollama_base_url}/api/embeddings",
#                 json={
#                     "model": self.ollama_model,
#                     "prompt": text
#                 },
#                 timeout=120
#             )
#             response.raise_for_status()
#             embedding = response.json()["embedding"]
#             return embedding
#         except Exception as e:
#             print(Fore.RED + f"[ERROR] Failed to get embedding from Ollama: {str(e)}" + Fore.RESET)
#             raise
    
#     def query(self, query_text: str, query_embedding: list[float] = None, k: int = 10) -> List[Dict]:
#         """Query ChromaDB and retrieve top-k results."""
#         try:
#             if query_embedding is None:
#                 query_embedding = self._get_embedding(query_text)
            
#             results = self.collection.query(
#                 query_embeddings=[query_embedding],
#                 n_results=k,
#                 include=["documents", "metadatas", "distances"]
#             )
            
#             documents = results["documents"][0]
#             metadatas = results["metadatas"][0]
#             distances = results["distances"][0]
            
#             chunks = []
#             for doc, metadata, distance in zip(documents, metadatas, distances):
#                 chunks.append({
#                     "text": doc,
#                     "metadata": metadata,
#                     "distance": distance
#                 })
            
#             return chunks
            
#         except Exception as e:
#             print(Fore.RED + f"[ERROR] Query failed: {str(e)}" + Fore.RESET)
#             raise
    
#     def rerank(self, chunks: List[Dict], query: str, k: int) -> List[Tuple[str, float]]:
#         """
#         Rerank chunks locally using SentenceTransformer cosine similarity (no Ollama API).
#         """
#         try:
#             chunk_texts = [chunk["text"] for chunk in chunks]
#             query_embedding = self.bi_encoder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
#             chunk_embeddings = self.bi_encoder.encode(chunk_texts, convert_to_tensor=True, normalize_embeddings=True)
#             cosine_scores = torch.matmul(chunk_embeddings, query_embedding)
#             chunk_scores = [(text, score.item()) for text, score in zip(chunk_texts, cosine_scores)]
#             chunk_scores.sort(key=lambda x: x[1], reverse=True)
#             return chunk_scores[:k]
#         except Exception as e:
#             print(Fore.RED + f"[ERROR] Local MiniLM reranking failed: {str(e)}" + Fore.RESET)
#             raise