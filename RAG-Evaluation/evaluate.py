import time
from colorama import Fore, init
from rag_query_engine import RAGQueryEngine

init(autoreset=False)

def main():
    """Main function to demonstrate RAG Query Engine."""
    
    start_time = time.time()
    
    print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
    print(Fore.MAGENTA + "Starting Evaluation" + Fore.RESET)
    print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
    print()
    
    try:
        # Initialize RAG Query Engine
        engine = RAGQueryEngine(
            db_path="./chromadb_storage",
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2"
        )
        
        # Test query
        query_text = input(Fore.GREEN + "Ask a question: " + Fore.RESET)
        
        print(Fore.MAGENTA + "-" * 80 + Fore.RESET)
        print()
        
        # Step 1: Query ChromaDB for top 15 documents
        retrieved_chunks = engine.query(query_text, k=15)
        
        # Step 2: Rerank and get top 5
        reranked_results = engine.rerank(retrieved_chunks, query_text, k=5)
        
        # Print results
        print(Fore.CYAN + "Reranked Top 5 Chunks:" + Fore.RESET)
        print(Fore.MAGENTA + "-" * 80 + Fore.RESET)
        for idx, (chunk_text, score) in enumerate(reranked_results, 1):
            print(Fore.BLUE + f"[{idx}] Score: {score:.4f}" + Fore.RESET)
            print(Fore.BLUE + f"    {chunk_text[:]}..." + Fore.RESET)
            print()
        
        print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
        
        execution_time = round(time.time() - start_time, 3)
        print(Fore.YELLOW + f"Execution Time: {execution_time}s" + Fore.RESET)
        
    except Exception as e:
        print(Fore.RED + f"[ERROR] Failed: {str(e)}" + Fore.RESET)
        raise


if __name__ == "__main__":
    main()