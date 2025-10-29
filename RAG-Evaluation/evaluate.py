import time
from colorama import Fore
import pandas as pd
import csv

from rag_query_engine import RAGQueryEngine
from ollama_qwen import OllamaFineTunedQwenChat
from pretrained_model import OllamaQwenChat
from evaluator_model import OllamaGemmaChat

evaluator_client = OllamaGemmaChat()
fine_tuned_client = OllamaFineTunedQwenChat()
pre_trained_client = OllamaQwenChat()
rag_query_engine = RAGQueryEngine()

REPORT_CSV = "evaluation_data/report.csv"

def main():
    results = []
    path = r"evaluation_data/qna_embedding.parquet"
    qna_pairs = pd.read_parquet(path=path)

    for idx, qna in qna_pairs.iterrows():
        question = qna['question']
        answer = qna['answers']
        embedding = qna['embedding']

        print(Fore.RED + f"\nQuestion {idx+1}: {question}" + Fore.RESET)

        # Retrieve and rerank contexts
        contexts = rag_query_engine.query(query_text=question, query_embedding=embedding, k=15)
        contexts = rag_query_engine.rerank(chunks=contexts, query=question, k=5)
        contexts_text = " ".join([con[0] for con in contexts])

        # Generate evaluator response (reference)
        evaluator_response = evaluator_client.generate_response(
            prompt=question,
            context=contexts_text,
            expected_answer=answer
        )
        print(Fore.BLUE + f"Evaluator: {evaluator_response[:100]}" + Fore.RESET)

        # Pre-trained model response
        pre_trained_response = pre_trained_client.generate_response(
            prompt=question
        )
        print(Fore.GREEN + f"\nPre-trained: {pre_trained_response[:100]}" + Fore.RESET)

        # Fine-tuned model response (no context)
        fine_tuned_response = fine_tuned_client.generate_response(
            prompt=question
        )
        print(Fore.CYAN + f"Fine-tuned: {fine_tuned_response[:100]}" + Fore.RESET)

        # Fine-tuned + RAG response
        fine_tuned_rag_response = fine_tuned_client.generate_response(
            prompt=question,
            # context=contexts_text[:1000]
        )
        print(Fore.YELLOW + f"Fine-tuned+RAG: {fine_tuned_rag_response[:100]}" + Fore.RESET)

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

        # Track all results for report
        results.append({
            "_id": idx,
            "question": question,
            "pre_trained_response": pre_trained_response,
            "pre_trained_score": pt_score,
            "fine_tuned_response": fine_tuned_response,
            "fine_tuned_score": ft_score,
            "fine_tuned__rag_response": fine_tuned_rag_response,
            "fine_tuned_score_rag": ft_rag_score
        })

        if idx + 1 == 100:
            break

    # Save to CSV
    df_report = pd.DataFrame(results)
    df_report.to_csv(REPORT_CSV, index=False)
    print(Fore.GREEN + f"\nSaved evaluation report to {REPORT_CSV}" + Fore.RESET)

    # Print summary
    print("\n" + Fore.MAGENTA + "="*80 + Fore.RESET)
    print(Fore.MAGENTA + "EVALUATION SUMMARY" + Fore.RESET)
    print(Fore.MAGENTA + "="*80 + Fore.RESET)
    for model_name, score_col in [
        ("pre-trained", "pre_trained_score"),
        ("fine-tuned", "fine_tuned_score"),
        ("fine-tuned-rag", "fine_tuned_score_rag"),
    ]:
        scores = df_report[score_col].tolist()
        avg_score = sum(scores) / len(scores)
        scores_str = ", ".join([f"{x:.4f}" for x in scores])
        print(Fore.BLUE + f"{model_name.upper()}: [{scores_str}] | Avg: {avg_score:.4f}" + Fore.RESET)


if __name__ == "__main__":
    st = time.time()
    main()
    en = time.time()
    print("\n", Fore.RED + f"{round(en-st, 4)}sec" + Fore.RESET)