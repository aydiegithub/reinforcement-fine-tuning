import os
from typing import List, Dict, Tuple
from colorama import Fore, init
import chromadb
from sentence_transformers import CrossEncoder
import requests

init(autoreset=False)

class RAGQueryEngine:
    """Simple RAG Query Engine with retrieval and reranking."""
    
    def __init__(
        self,
        db_path: str = "./chromadb_storage",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        ollama_model: str = "qwen3-embedding:4b",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the RAG Query Engine.
        
        Args:
            db_path: Path to ChromaDB persistent storage
            cross_encoder_model: HuggingFace cross-encoder model for reranking
            ollama_model: Ollama embedding model (must match the one used during indexing)
            ollama_base_url: Ollama server URL
        """
        self.db_path = db_path
        self.cross_encoder_model = cross_encoder_model
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        
        self._initialize_chromadb()
        self._initialize_cross_encoder()
        
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
    
    def _initialize_cross_encoder(self) -> None:
        """Initialize cross-encoder model."""
        try:
            self.cross_encoder = CrossEncoder(self.cross_encoder_model)
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to initialize cross-encoder: {str(e)}" + Fore.RESET)
            raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama using the same model as indexing."""
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
    
    def query(self, query_text: str, k: int) -> List[Dict]:
        """
        Query ChromaDB and retrieve top-k results.
        
        Args:
            query_text: Query string
            k: Number of top results to retrieve
            
        Returns:
            List of dictionaries with document and metadata
        """
        try:
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
    
    def rerank(self, chunks: List[Dict], query: str, k: int) -> List[Tuple[str, float]]:
        """
        Rerank chunks using cross-encoder and return top-k with scores.
        
        Args:
            chunks: List of chunk dictionaries from query()
            query: Query string for reranking
            k: Number of top results to return after reranking
            
        Returns:
            List of tuples (chunk_text, rerank_score) sorted by score
        """
        try:
            chunk_texts = [chunk["text"] for chunk in chunks]
            
            # Create pairs for cross-encoder
            pairs = [[query, chunk_text] for chunk_text in chunk_texts]
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Create list of (text, score) tuples
            chunk_scores = list(zip(chunk_texts, scores))
            
            # Sort by score in descending order
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k
            top_k_results = chunk_scores[:k]
            
            return top_k_results
            
        except Exception as e:
            print(Fore.RED + f"[ERROR] Reranking failed: {str(e)}" + Fore.RESET)
            raise