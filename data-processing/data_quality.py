import time
import json
from pydantic import BaseModel
from litellm import completion
from colorama import Fore

MODEL = "ollama/gemma3:4b"

class Score(BaseModel):
    score: int
    explanation: str
    
    
class Rank(BaseModel):
    accuracy: Score
    style: Score
    
def llm_call(record: str) -> dict:
    
    prompt = f"""
                You are an expert evaluator. Classify this instruction tuning record on **accuracy** and **style**, each on a scale from 1-10.  
                Provide concise explanations for both. The response must be self-contained and in JSON format matching the Rank schema.

                Scoring Guidelines:
                - Accuracy:
                - 0 if the record is not a question.
                - 1 if the answer does not adequately address the question.
                - Higher scores indicate correctness and relevance.
                - Style:
                - 1 if the question or answer is harmful, misleading, dishonest, or blank ("..." or empty).
                - Higher scores indicate clarity, helpfulness, and readability.

                Record: {record}
                Return only JSON output.
            """
            
    stream = completion(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ], 
        stream=True,
        options={"num_predict": 2000, "temperature": 0.2},
        format=Rank.model_json_schema(),
    )
    
    data = ""
    for x in stream:
        delta = x['choices'][0]['delta']['content']
        if delta is not None:
            print(Fore.LIGHTBLUE_EX + str(delta) + Fore.RESET, end=" ") 
            data += delta
            
    return json.loads(data)


if __name__ == "__main__":
    st = time.time()
    
    
    quality = []
    instructions = []
    with open('data/instruction.json', 'r') as f:
        data = json.load(f)
        for pair in data:
            print(Fore.YELLOW + str(pair) + Fore.RESET)
            result = llm_call(pair)
            
            if result['accuracy']['score'] >= 6 and result['style']['score'] >= 6:
                instructions.append(pair)
                quality.append({**pair, 'quality': result})
                
            
    with open('data/instructionquality.json', 'w') as f:
        json.dump(instructions, f)
        
    with open('qualityresults.json', 'w') as f:
        json.dump(quality, f)
    
    # test_record = {
    #     "question": "What are cubes in TM1 used for?",
    #     "answer": "Cubes in TM1 store business analysis data, with each ce;; containing a measure being tracked.",
    # }
        
    # llm_call(test_record)
    print("/n/n")
    print(Fore.RED + f"Total Execution Time: {round(time.time() - st, 3)} s")