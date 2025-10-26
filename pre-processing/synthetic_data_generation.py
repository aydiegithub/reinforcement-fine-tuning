import time

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore

import json
from typing import List
from pydantic import BaseModel
from litellm import completion
from prompts.generated_prompt import prompt_template
from tqdm.auto import tqdm

class Record(BaseModel):
    question: str
    answer: str
    
class Response(BaseModel):
    generated: List[Record]
    
def llm_call(data: str, num_records: int = 5) -> dict:
    stream = completion(
        model="ollama_chat/gpt-oss",
        messages=[
            {
                "role": "user", 
                "content": prompt_template(data=data, 
                                           num_records=num_records)
            }
        ],
        stream=True,
        options={"num_predict": 2000}, # Number of tokens
        format=Response.model_json_schema()
    )
    
    data = ""
    
    for x in stream:
        delta = None
        try:
            delta = x.get('choices', [])[0].get("delta", {}).get("content")
        except Exception:
            delta = None
        if delta is not None:
            print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="")
            data += delta

    return json.loads(data)


def main():
    converter = DocumentConverter()
    doc = converter.convert(r"documents/tm1_dg_dvlpr-10pages.pdf").document
    # print(doc)
    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=doc)
    
    datasets = {}
    for i, chunk in enumerate(chunks):
        print(Fore.YELLOW + f"Raw Text:\n{chunk.text[:300]}..." + Fore.RESET)
        
        enriched_text = chunker.contextualize(chunk=chunk)
        print(Fore.LIGHTMAGENTA_EX + f"Contextualised Text:\n{enriched_text[:300]}..." + Fore.RESET)
        print("\n\n")
        
        data = llm_call(enriched_text)
        datasets[i] = {"generated": data["generated"], "context": enriched_text}
        
    with open('tm1data.json', 'w') as f:
        json.dump(datasets, f)
    


if __name__ == "__main__":
    st = time.time()
    main()
    print("/n/n")
    print(Fore.RED + f"Total Execution Time: {round(time.time() - st, 3)} s")