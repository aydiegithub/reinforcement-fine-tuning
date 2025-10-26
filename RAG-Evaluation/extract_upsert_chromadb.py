import time
import json
import os
import shutil
from typing import List, Dict
from pathlib import Path
from datetime import datetime

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore, init
import chromadb
import requests

init(autoreset=False)

class PDFEmbedder:
    def __init__(
        self,
        pdf_path: str,
        db_path: str = "./chromadb_storage",
        chunk_size: int = 1500,
        chunk_overlap: int = 400,
        embedding_model: str = "qwen3-embedding:4b",
        ollama_base_url: str = "http://localhost:11434",
        clear_db: bool = False
    ):
        self.pdf_path = pdf_path
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        
        self.converter = DocumentConverter()
        self.chunker = HybridChunker(merge_peers=True)
        
        self.native_dim = None
        
        if clear_db:
            self._clear_database()
        
        self._initialize_chromadb()
        self._verify_embedding_model()
        
    def _clear_database(self) -> None:
        """Clear existing ChromaDB storage."""
        try:
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
                print(Fore.YELLOW + f"[INFO] Cleared existing ChromaDB at: {self.db_path}" + Fore.RESET)
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to clear database: {str(e)}" + Fore.RESET)
            raise
        
    def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB with persistent storage."""
        os.makedirs(self.db_path, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            collection_name = f"pdf_embeddings_{int(time.time())}"
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            print(Fore.GREEN + f"[SUCCESS] ChromaDB initialized at: {self.db_path}" + Fore.RESET)
            print(Fore.GREEN + f"[SUCCESS] Collection created: {collection_name}" + Fore.RESET)
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to initialize ChromaDB: {str(e)}" + Fore.RESET)
            raise
    
    def _verify_embedding_model(self) -> None:
        """Verify that the embedding model is available in Ollama."""
        try:
            print(Fore.CYAN + f"[INFO] Verifying Ollama model: {self.embedding_model}" + Fore.RESET)
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            available_models = [model["name"] for model in response.json().get("models", [])]
            
            model_base = self.embedding_model.split(":")[0]
            model_found = any(m.startswith(model_base) for m in available_models)
            
            if not model_found:
                print(Fore.YELLOW + f"[WARNING] Model {self.embedding_model} not found locally" + Fore.RESET)
                print(Fore.CYAN + f"[INFO] Pulling model {self.embedding_model}..." + Fore.RESET)
                try:
                    requests.post(
                        f"{self.ollama_base_url}/api/pull",
                        json={"name": self.embedding_model},
                        timeout=900
                    )
                    print(Fore.GREEN + f"[SUCCESS] Model pulled successfully" + Fore.RESET)
                except Exception as pull_error:
                    print(Fore.RED + f"[ERROR] Could not pull model: {str(pull_error)}" + Fore.RESET)
                    raise
            else:
                print(Fore.GREEN + f"[SUCCESS] Embedding model available: {self.embedding_model}" + Fore.RESET)
        except Exception as e:
            print(Fore.RED + f"[ERROR] Could not verify Ollama model: {str(e)}" + Fore.RESET)
            raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama for given text."""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                },
                timeout=120
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            
            if self.native_dim is None:
                self.native_dim = len(embedding)
                print(Fore.GREEN + f"[SUCCESS] Detected embedding dimension: {self.native_dim}" + Fore.RESET)
            
            return embedding
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to get embedding: {str(e)}" + Fore.RESET)
            raise
    
    def _extract_pdf_content(self):
        """Extract content from PDF using Docling."""
        print(Fore.CYAN + f"[INFO] Converting PDF: {self.pdf_path}" + Fore.RESET)
        
        try:
            result = self.converter.convert(self.pdf_path)
            doc = result.document
            print(Fore.GREEN + f"[SUCCESS] PDF converted successfully" + Fore.RESET)
            return doc
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to convert PDF: {str(e)}" + Fore.RESET)
            raise
    
    def _create_chunks(self, doc) -> List[Dict[str, str]]:
        """Create chunks from document with specified size and overlap."""
        print(Fore.CYAN + f"[INFO] Creating chunks (size: {self.chunk_size} words, overlap: {self.chunk_overlap} words)" + Fore.RESET)
        
        try:
            text_content = doc.export_to_markdown()
            words = text_content.split()
            
            processed_chunks = []
            step = self.chunk_size - self.chunk_overlap
            
            for i in range(0, len(words), step):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = " ".join(chunk_words)
                
                if len(chunk_text.strip()) > 0:
                    processed_chunks.append({
                        "text": chunk_text,
                        "start_idx": i,
                        "end_idx": min(i + self.chunk_size, len(words))
                    })
            
            print(Fore.GREEN + f"[SUCCESS] Created {len(processed_chunks)} chunks" + Fore.RESET)
            return processed_chunks
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to create chunks: {str(e)}" + Fore.RESET)
            raise
    
    def process_and_embed(self) -> None:
        """Main processing pipeline: extract, chunk, embed, and store."""
        print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
        print(Fore.MAGENTA + "PDF to ChromaDB Embedding Pipeline" + Fore.RESET)
        print(Fore.MAGENTA + f"Model: {self.embedding_model}" + Fore.RESET)
        print(Fore.MAGENTA + f"Chunk Size: {self.chunk_size} words | Overlap: {self.chunk_overlap} words" + Fore.RESET)
        print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
        
        try:
            doc = self._extract_pdf_content()
            chunks = self._create_chunks(doc)
            
            print(Fore.CYAN + "[INFO] Starting embedding and storage process" + Fore.RESET)
            print(Fore.MAGENTA + "-" * 80 + Fore.RESET)
            
            embeddings_data = []
            document_metadata = {
                "source_file": self.pdf_path,
                "processed_at": datetime.now().isoformat(),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": self.embedding_model,
                "total_chunks": len(chunks)
            }
            
            for idx, chunk in enumerate(chunks, 1):
                chunk_id = f"chunk_{idx:05d}"
                chunk_text = chunk["text"]
                
                print(Fore.CYAN + f"[{idx:03d}/{len(chunks):03d}]" + Fore.RESET, end=" ")
                print(Fore.BLUE + f"ID: {chunk_id}" + Fore.RESET, end=" | ")
                
                embedding = self._get_embedding(chunk_text)
                print(Fore.BLUE + f"Dims: {len(embedding)}" + Fore.RESET, end=" | ")
                
                metadata = {
                    "document": self.pdf_path,
                    "chunk_index": idx,
                    "chunk_id": chunk_id,
                    "start_word": chunk["start_idx"],
                    "end_word": chunk["end_idx"],
                    "word_count": len(chunk_text.split()),
                    "embedding_model": self.embedding_model,
                    "embedding_dimension": len(embedding),
                    "processed_at": datetime.now().isoformat()
                }
                
                self.collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[chunk_text]
                )
                
                embeddings_data.append({
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "text_preview": chunk_text[:200],
                    "embedding_dimension": len(embedding),
                    "metadata": metadata
                })
                
                print(Fore.GREEN + "STORED" + Fore.RESET)
            
            print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
            print(Fore.GREEN + f"[SUCCESS] Processed and stored {len(chunks)} chunks" + Fore.RESET)
            print(Fore.GREEN + f"[SUCCESS] Embedding dimension: {self.native_dim}" + Fore.RESET)
            print(Fore.GREEN + f"[SUCCESS] ChromaDB persisted to: {self.db_path}" + Fore.RESET)
            print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
            
            self._save_metadata(document_metadata, embeddings_data)
            
        except Exception as e:
            print(Fore.RED + f"[ERROR] Pipeline failed: {str(e)}" + Fore.RESET)
            raise
    
    def _save_metadata(self, document_metadata: Dict, embeddings_data: List[Dict]) -> None:
        """Save metadata and embedding information to JSON."""
        metadata_file = os.path.join(self.db_path, "embeddings_metadata.json")
        
        metadata_output = {
            "document_metadata": document_metadata,
            "chunks": embeddings_data,
            "total_embeddings": len(embeddings_data)
        }
        
        with open(metadata_file, "w") as f:
            json.dump(metadata_output, f, indent=2)
        
        print(Fore.CYAN + f"[INFO] Metadata saved to: {metadata_file}" + Fore.RESET)
    
    def query_similar(self, query_text: str, n_results: int = 5) -> Dict:
        """Query the vector database for similar chunks."""
        print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
        print(Fore.CYAN + f"[INFO] Query: {query_text}" + Fore.RESET)
        print(Fore.MAGENTA + "-" * 80 + Fore.RESET)
        
        try:
            query_embedding = self._get_embedding(query_text)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            print(Fore.GREEN + f"[SUCCESS] Found {len(results['documents'][0])} similar chunks" + Fore.RESET)
            print(Fore.MAGENTA + "-" * 80 + Fore.RESET)
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                print(Fore.CYAN + f"[RESULT {i}]" + Fore.RESET, end=" ")
                print(Fore.BLUE + f"Distance: {distance:.4f}" + Fore.RESET)
                print(Fore.BLUE + f"  Chunk ID: {metadata['chunk_id']}" + Fore.RESET)
                print(Fore.BLUE + f"  Word Count: {metadata['word_count']}" + Fore.RESET)
                print(Fore.BLUE + f"  Preview: {doc[:150]}..." + Fore.RESET)
                print()
            
            print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
            return results
        except Exception as e:
            print(Fore.RED + f"[ERROR] Query failed: {str(e)}" + Fore.RESET)
            raise


def main():
    """Main execution function."""
    start_time = time.time()
    
    pdf_path = "documents/tm1_dg_dvlpr-10pages.pdf"
    
    if not os.path.exists(pdf_path):
        print(Fore.RED + f"[ERROR] PDF file not found: {pdf_path}" + Fore.RESET)
        return
    
    print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
    print(Fore.MAGENTA + "Initializing PDF Embedder" + Fore.RESET)
    print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
    
    embedder = PDFEmbedder(
        pdf_path=pdf_path,
        db_path="./chromadb_storage",
        chunk_size=1500,
        chunk_overlap=400,
        embedding_model="qwen3-embedding:4b",
        clear_db=True
    )
    
    embedder.process_and_embed()
    
    execution_time = round(time.time() - start_time, 3)
    print(Fore.YELLOW + f"Total Execution Time: {execution_time}s" + Fore.RESET)
    
    print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
    print(Fore.CYAN + "Testing Query Functionality" + Fore.RESET)
    print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
    
    test_queries = [
        "What is the main topic of this document?",
        "Tell me about the key concepts covered",
        "What are the important details"
    ]
    
    for test_query in test_queries:
        results = embedder.query_similar(test_query, n_results=3)
        print()


if __name__ == "__main__":
    main()