import requests
from colorama import Fore, init
from datetime import datetime

init(autoreset=False)

class OllamaQwenChat:
    """Chat interface using Ollama Qwen 2.5 0.5B model."""
    
    def __init__(
        self,
        model: str = "qwen2.5:0.5b",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the Ollama Qwen chat.
        
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
            raise e
    
    def generate_response(self, prompt: str = None, context: str = None) -> str:
        """
        Generate response using Ollama.
        
        Args:
            prompt: User input prompt
            
        Returns:
            Generated response
        """
        try:
            if context:
                formatted_prompt = f"""
                                    You are a factual and concise answering assistant.
                                    Use the information provided in the context to answer the question accurately.
                                    If the context does not contain enough information, respond with "Insufficient information."

                                    Question: {prompt}

                                    Context:
                                    {context}

                                    Answer (concise factual statement):
                                    """
            else:
                formatted_prompt = f"""
                                    Answer the following question directly and concisely.
                                    Only provide the answer — do not repeat the question or include explanations.

                                    Question:
                                    {prompt}

                                    Answer:
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
            raise e


def main():
    """Main chat function."""
    
    try:
        chat = OllamaQwenChat(
            model="qwen2.5:0.5b",
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