from rag_query_engine import RAGQueryEngine
import pandas as pd
from pandas import DataFrame
import json
from colorama import Fore

class EmbeddingGenerator():
    def __init__(self):
        self.rag_engine = RAGQueryEngine()
        
    def generate_embedding(self, sentence: list[str] = None) -> DataFrame:
        embeddings = {}
        
        for sent in sentence:
            embeddings[sent] = self.rag_engine._get_embedding(sent)
            print(Fore.CYAN + "Generated: " + sent[:35] + f"{embeddings[sent][:4]}" + Fore.RESET)
        
        data_frame = pd.DataFrame({'question': list(embeddings.keys()), 'embedding': list(embeddings.values())})
        return data_frame


if __name__ == "__main__":
    with open("evaluation_data/evaluation_data.json", "r") as f:
        qna_pairs = json.load(f)
    
    questions = []
    answers = []

    for qna in qna_pairs:
        questions.append(qna['question'])
        answers.append(qna['answer'])
    
    emb_gen = EmbeddingGenerator()
    emb = emb_gen.generate_embedding(questions)
    emb['answers'] = answers
    emb.to_parquet("evaluation_data/qna_embedding.parquet")
    # emb.to_csv("evaluation_data/qna_embedding.csv")
