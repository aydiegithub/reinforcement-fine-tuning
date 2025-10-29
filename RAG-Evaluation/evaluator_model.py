import requests
from colorama import Fore, init
from datetime import datetime

init(autoreset=False)

class OllamaGemmaChat:
    """Chat interface using Ollama Gemma 3 4B model."""
    
    def __init__(
        self,
        model: str = "gemma3:4b",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the Ollama Gemma chat.
        
        Args:
            model: Ollama model name
            ollama_base_url: Ollama server URL
        """
        self.model = model
        self.ollama_base_url = ollama_base_url
        self._verify_model()
    
    def _verify_model(self) -> None:
        """Verify that the model is available in Ollama."""
        try:
            # print(Fore.CYAN + f"[INFO] Verifying model: {self.model}" + Fore.RESET)
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            available_models = [model["name"] for model in response.json().get("models", [])]
            
            model_base = self.model.split(":")[0]
            model_found = any(m.startswith(model_base) for m in available_models)
            
            if not model_found:
                # print(Fore.RED + f"[ERROR] Model {self.model} not found in Ollama" + Fore.RESET)
                raise ValueError(f"Model {self.model} not found")
            
            # print(Fore.GREEN + f"[SUCCESS] Model {self.model} is available" + Fore.RESET)
        except Exception as e:
            # print(Fore.RED + f"[ERROR] Failed to verify model: {str(e)}" + Fore.RESET)
            raise
    
    def generate_response(self, prompt: str = None, 
                          context: str = None,
                          expected_answer: str = None) -> str:
        """
        Generate response using Ollama.
        
        Args:
            prompt: User input prompt
            
        Returns:
            Generated response
        """
        try:
            if context:
                formatted_prompt = f"""You are a factual, concise answering assistant. Think step-by-step before answering.

                                    INSTRUCTIONS:
                                    - Use ONLY the provided context to answer the question
                                    - If context lacks sufficient information, respond with exactly: "Insufficient information."
                                    - Keep your answer to a maximum of 3 sentences
                                    - Do not assume or create information not in the context
                                    - If unsure, say "No" or "Insufficient information"

                                    CHAIN OF THOUGHT EXAMPLES:
                                    1. Question: "What is the capital of France?"
                                    Context: "France is a country in Europe. Paris is the largest city and capital."
                                    Thought: The context directly states Paris is the capital.
                                    Answer: Paris is the capital of France.

                                    2. Question: "What year was Python created?"
                                    Context: "Python is a programming language."
                                    Thought: The context doesn't mention the year, only that it's a programming language.
                                    Answer: Insufficient information.

                                    ### Context: 
                                    - {context}

                                    ### Expected Answer Should Include:
                                    - {expected_answer}
                                    
                                    ### NOW ANSWER THE QUESTION:
                                    Question: {prompt}

                                    CHAIN OF THOUGHT (reason through this step-by-step):
                                    [Think about what information is needed and whether context provides it]

                                    Answer (maximum 3 sentences, factual only):
                                    """
            else:
                formatted_prompt = f"""You are a factual, concise answering assistant. Think step-by-step before answering.

                                    INSTRUCTIONS:
                                    - Answer only if you have reliable knowledge
                                    - If unsure or information is not available, respond with exactly: "No" or "I don't know"
                                    - Keep your answer to a maximum of 3 sentences
                                    - Do not assume or create information
                                    - Only provide the answer — do not repeat the question

                                    FEW-SHOT EXAMPLES:
                                    1. Question: "What is 2+2?"
                                    Thought: This is basic arithmetic with a definitive answer.
                                    Answer: 2+2 equals 4.

                                    2. Question: "What happened on Mars last week?"
                                    Thought: I don't have access to current Mars events or real-time data.
                                    Answer: I don't know.

                                    ### Expected Answer Should Include:
                                    - {expected_answer}
                                    
                                    ### NOW ANSWER THE QUESTION:
                                    Question: {prompt}

                                    CHAIN OF THOUGHT (reason through this step-by-step):
                                    [Do I have reliable knowledge about this? Is this factual?]

                                    Answer (maximum 3 sentences):
                                    """
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": formatted_prompt,
                    "stream": False,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.8
                },
                timeout=300
            )
            
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("response", "")
            
            answer = generated_text.split("Answer:")[-1].strip()
            
            return answer
        except Exception as e:
            # print(Fore.RED + f"[ERROR] Failed to generate response: {str(e)}" + Fore.RESET)
            raise


def main():
    """Main chat function."""
    
    try:
        chat = OllamaGemmaChat(
            model="gemma3:4b",
            ollama_base_url="http://localhost:11434"
        )
        
        # print(Fore.CYAN + "[INFO] Type '/bye' to exit" + Fore.RESET)
        # print()
        
        while True:
            prompt = input(Fore.GREEN + "Message: " + Fore.RESET)
            
            if prompt == "/bye":
                # print(Fore.RED + "Exiting..." + Fore.RESET)
                break
            
            if not prompt.strip():
                continue
            
            response = chat.generate_response(prompt)
            return response
            # print(Fore.BLUE + response + Fore.RESET)
            # print()
    
    except Exception as e:
        # print(Fore.RED + f"[ERROR] Failed: {str(e)}" + Fore.RESET)
        raise e


# if __name__ == "__main__":
#     main() 