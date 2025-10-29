import requests
from colorama import Fore, init
from datetime import datetime
import json

init(autoreset=False)

class OllamaFineTunedQwenChat:
    """Chat interface using Ollama Gemma 3 4B model."""

    def __init__(
        self,
        model: str = "qwen_aydie_3:0.5b",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.5
    ):
        """
        Initialize the Ollama Gemma chat.

        Args:
            model: Ollama model name
            ollama_base_url: Ollama server URL
            temperature: Controls randomness of the output (default: 0.5)
        """
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
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
            raise

    def generate_response(self, prompt: str = None, context: str = None) -> str:
        """
        Generate response using Ollama.

        Args:
            prompt: User input prompt
            context: Optional context for better accuracy

        Returns:
            Generated response
        """
        try:
            if context:
                formatted_prompt = f"""
                You are a factual, concise answering assistant. Think step-by-step before answering. Dont repeat the question.
                                    ### IF YOU DON'T KNOW THE ANSWER RESPOND "I don't know". Refer the context for more knowladge and answer according to context.
                                    
                                    Context: {context}
                                    
                                    Question: {prompt}
                """
            else:
                formatted_prompt = f"""You are a factual, concise answering assistant. Think step-by-step before answering. Dont repeat the question.
                                    ### IF YOU DON'T KNOW THE ANSWER RESPOND "I don't know"
                                    
                                    Question: {prompt}
                                    """

            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": formatted_prompt,
                    "options": {
                        "temperature": self.temperature
                    }
                },
                timeout=300,
                stream=True
            )

            response.raise_for_status()
            generated_text = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.strip():
                    continue
                try:
                    result = json.loads(line)
                    generated_text += result.get("response", "")
                except Exception:
                    # Skip lines that aren't valid JSON
                    continue

            answer = generated_text.split("Answer:")[-1].strip()
            return answer
        except Exception as e:
            raise

def main():
    """Main chat function."""

    try:
        chat = OllamaFineTunedQwenChat(
            model="qwen_aydie_2:0.5b",
            ollama_base_url="http://localhost:11434",
            temperature=0.5
        )

        while True:
            prompt = input(Fore.GREEN + "Message: " + Fore.RESET)

            if prompt == "/bye":
                break

            if not prompt.strip():
                continue

            response = chat.generate_response(prompt)
            print(Fore.BLUE + response + Fore.RESET)
            print()

    except Exception as e:
        raise e

# if __name__ == "__main__":
#     main()