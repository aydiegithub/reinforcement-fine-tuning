import requests
from colorama import Fore, init

init(autoreset=True)

class OllamaQwenChat:
    """Chat interface using Ollama Qwen 2.5 0.5B model."""

    def __init__(
        self,
        model: str = "qwen2.5:0.5b",
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.ollama_base_url = ollama_base_url
        self._verify_model()

    def _verify_model(self) -> None:
        """Verify that the model is available in Ollama."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            available_models = [model["name"] for model in response.json().get("models", [])]
            model_base = self.model.split(":")[0]
            model_found = any(m.startswith(model_base) for m in available_models)
            if not model_found:
                raise ValueError(f"Model {self.model} not found")
        except Exception as e:
            raise e

    def generate_response(self, prompt: str, context: str = None) -> str:
        """Generate response using Ollama."""
        try:
            if context:
                # Simple prompt for baseline with context
                formatted_prompt = f"""Answer the question using the context below.
                                    Context: {context}
                                    Question: {prompt}
                                    Answer briefly in 2 sentences maximum.
                                    """
            else:
                # Simple prompt for baseline without context
                # formatted_prompt = f"""You are a factual, concise answering assistant. Think step-by-step before answering.
                #                         Question: {prompt}
                #                     """
                formatted_prompt = f"""Answer briefly in 1 sentences maximum, 
                ### IF YOU DON'T KNOW THE ANSWER RESPOND "I don't know", 
                don't repeat the question and don't make assumption. 
                Question: {prompt}"""

            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": formatted_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Higher temperature - less focused
                        "top_p": 0.95,
                        "top_k": 50,
                        "repeat_penalty": 0,
                        "num_predict": 80
                    }
                },
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("response", "").strip()

            # Minimal cleaning for baseline
            answer = generated_text
            
            return answer
            
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            
            # Basic sentence limiting
            sentences = [s.strip() + '.' for s in answer.replace('\n', ' ').split('.') if s.strip() and len(s.strip()) > 5]
            if sentences:
                answer = ' '.join(sentences[:3])
            else:
                answer = "I don't know."
            
            if len(answer) > 500:
                answer = answer[:500].rsplit('.', 1)[0] + '.'

            return answer

        except Exception as e:
            return f"Error: {str(e)}"

def main():
    """Main chat function."""
    try:
        chat = OllamaQwenChat()
        print(Fore.CYAN + "[INFO] Type '/bye' to exit" + Fore.RESET)
        print()
        while True:
            prompt = input(Fore.GREEN + "Message: " + Fore.RESET)
            if prompt.strip() == "/bye":
                print(Fore.RED + "Exiting..." + Fore.RESET)
                break
            if not prompt.strip():
                continue
            response = chat.generate_response(prompt)
            print(Fore.BLUE + response + Fore.RESET)
            print()
    except Exception as e:
        print(Fore.RED + f"[ERROR] Failed: {str(e)}" + Fore.RESET)
        raise e

if __name__ == "__main__":
    main()