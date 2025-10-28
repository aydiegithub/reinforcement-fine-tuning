import time
from colorama import Fore
import json
from rag_query_engine import RAGQueryEngine
from ollama_qwen import OllamaFineTunedQwenChat
from pretrained_model import OllamaQwenChat
from evaluator_model import OllamaGemmaChat
import pandas as pd


evaluator_client = OllamaGemmaChat()
fine_tuned_client = OllamaFineTunedQwenChat()
pre_trained_client = OllamaQwenChat()
rag_query_engine = RAGQueryEngine()

def main():
    with open("evaluation_data/evaluation_data.json", "r") as f:
        qna_pairs = json.load(f)
        
    model_evaluation_scores = dict()
    
    cnt = 0
    for qna in qna_pairs:
        question = qna['question']
        answer = qna['answer']
        
        print(Fore.RED + f"\nQuestion {cnt+1}: {question}" + Fore.RESET)
        
        # Retrieve and rerank contexts
        contexts = rag_query_engine.query(query_text=question, k=15)
        contexts = rag_query_engine.rerank(chunks=contexts, query=question, k=5)
        contexts_text = " ".join([con[0] for con in contexts])
        
        # Generate evaluator response (ground truth reference)
        evaluator_response = evaluator_client.generate_response(
            prompt=question,
            context=contexts_text,
            expected_answer=answer
        )
        print(Fore.BLUE + f"Evaluator: {evaluator_response}" + Fore.RESET)
        
        # Generate pre-trained response
        pre_trained_response = pre_trained_client.generate_response(
            prompt=question
        )
        print(Fore.GREEN + f"Pre-trained: {pre_trained_response}" + Fore.RESET)
        
        # Generate fine-tuned response (no context)
        fine_tuned_response = fine_tuned_client.generate_response(
            prompt=question
        )
        print(Fore.CYAN + f"Fine-tuned: {fine_tuned_response}" + Fore.RESET)
        
        # Generate fine-tuned RAG response
        fine_tuned_rag_response = fine_tuned_client.generate_response(
            prompt=question,
            context=contexts_text[:3000]
        )
        print(Fore.YELLOW + f"Fine-tuned+RAG: {fine_tuned_rag_response}" + Fore.RESET)
        
        # Compute similarity scores using bi-encoder
        pre_trained_scores = rag_query_engine.rerank_biencoder(
            chunks=[{"text": evaluator_response}],
            query=pre_trained_response,
            k=1
        )
        
        fine_tuned_score = rag_query_engine.rerank_biencoder(
            chunks=[{"text": evaluator_response}], 
            query=fine_tuned_response,
            k=1
        )
        
        fine_tuned_rag_score = rag_query_engine.rerank_biencoder(
            chunks=[{"text": evaluator_response}], 
            query=fine_tuned_rag_response,
            k=1
        )
        
        pt_score = pre_trained_scores[0][-1]
        ft_score = fine_tuned_score[0][-1]
        ft_rag_score = fine_tuned_rag_score[0][-1]
        
        print(f"\nScores - PT: {pt_score:.4f}, FT: {ft_score:.4f}, FT_RAG: {ft_rag_score:.4f}")
        print(Fore.RED + "*" * 80 + Fore.RESET + "\n")
        
        model_evaluation_scores.setdefault('pre-trained', []).append(pt_score)
        model_evaluation_scores.setdefault('fine-tuned', []).append(ft_score)
        model_evaluation_scores.setdefault('fine-tuned-rag', []).append(ft_rag_score)
        
        cnt += 1
        if cnt == 10:
            break
    
    # Print summary
    print("\n" + Fore.MAGENTA + "="*80 + Fore.RESET)
    print(Fore.MAGENTA + "EVALUATION SUMMARY" + Fore.RESET)
    print(Fore.MAGENTA + "="*80 + Fore.RESET)
    
    for model_name in ['pre-trained', 'fine-tuned', 'fine-tuned-rag']:
        scores = model_evaluation_scores[model_name]
        avg_score = sum(scores) / len(scores)
        scores_str = ", ".join([f"{x:.4f}" for x in scores])
        print(Fore.BLUE + f"{model_name.upper()}: [{scores_str}] | Avg: {avg_score:.4f}" + Fore.RESET)

if __name__ == "__main__":
    main()