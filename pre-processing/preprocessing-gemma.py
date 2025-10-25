import time
import json
import re
from typing import List
from pydantic import BaseModel
from colorama import Fore
from tqdm.auto import tqdm

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from litellm import completion
from generated_prompt import prompt_template


# -----------------------------
# Data models
# -----------------------------
class Record(BaseModel):
    question: str
    answer: str


class Response(BaseModel):
    generated: List[Record]


# -----------------------------
# LLM call with error handling
# -----------------------------
def llm_call(data: str, num_records: int = 5) -> dict:
    """
    Calls the LLM to generate QA pairs for a given text chunk.
    Safely parses streamed responses into JSON.
    """

    print(Fore.CYAN + "\n[LLM CALL] Generating records..." + Fore.RESET)
    stream = completion(
        model="ollama_chat/gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt_template(
                    data=data,
                    num_records=num_records
                )
            }
        ],
        stream=True,
        options={"num_predict": 2000},  # Number of tokens
        format=Response.model_json_schema()
    )

    buffer = ""

    for chunk in stream:
        try:
            delta = chunk.get("choices", [])[0].get("delta", {}).get("content")
        except Exception:
            delta = None

        if delta is not None:
            print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="")
            buffer += delta

    # -----------------------------
    # Handle empty or invalid responses
    # -----------------------------
    if not buffer.strip():
        raise ValueError("❌ Empty response from LLM. Check if the model produced any output.")

    try:
        # Try direct JSON load
        parsed = json.loads(buffer)
        return parsed
    except json.JSONDecodeError:
        # Attempt to extract JSON substring
        print(Fore.RED + "\n⚠️ Invalid JSON detected. Attempting to recover..." + Fore.RESET)
        try:
            match = re.search(r"\{.*\}", buffer, re.DOTALL)
            if match:
                recovered_json = match.group()
                return json.loads(recovered_json)
            else:
                raise ValueError("Could not extract JSON from LLM output.")
        except Exception as e:
            print(Fore.RED + "\nLLM Output (for debugging):\n" + buffer + Fore.RESET)
            raise ValueError(f"Failed to parse LLM output as JSON: {str(e)}")


# -----------------------------
# Main data generation pipeline
# -----------------------------
def main():
    converter = DocumentConverter()
    doc = converter.convert(r"documents/tm1_dg_dvlpr-10pages.pdf").document
    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=doc)

    datasets = {}

    for i, chunk in enumerate(chunks):
        print(Fore.YELLOW + f"\n[Chunk {i}] Raw Text:\n{chunk.text[:300]}..." + Fore.RESET)
        enriched_text = chunker.contextualize(chunk=chunk)
        print(Fore.LIGHTMAGENTA_EX + f"[Contextualized]\n{enriched_text[:300]}..." + Fore.RESET)

        try:
            data = llm_call(enriched_text)
            datasets[i] = {
                "generated": data["generated"],
                "context": enriched_text
            }
        except Exception as e:
            print(Fore.RED + f"\n❌ Error in chunk {i}: {str(e)}" + Fore.RESET)
            continue  # Skip this chunk and move on

    # Save all generated data
    with open("tm1data.json", "w") as f:
        json.dump(datasets, f, indent=2)

    print(Fore.GREEN + "\n✅ Dataset successfully written to tm1data.json" + Fore.RESET)


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    st = time.time()
    try:
        main()
    except Exception as e:
        print(Fore.RED + f"\nFatal Error: {e}" + Fore.RESET)
    finally:
        print(Fore.RED + f"\nTotal Execution Time: {round(time.time() - st, 3)} s" + Fore.RESET)