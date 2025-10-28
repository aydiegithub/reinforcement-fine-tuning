import requests
from colorama import Fore, init

init(autoreset=False)

class OllamaFineTunedQwenChat:
    """Chat interface using Ollama fine-tuned Qwen model."""
    
    def __init__(
        self,
        model: str = "aydie-finetuned-qwen:latest",
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
            raise
    
    def generate_response(self, prompt: str = None, context: str = None) -> str:
        """Generate response using Ollama."""
        try:
            if context:
                # Ultra-simple RAG prompt
                formatted_prompt = f"{prompt}\n\nContext: {context}"
            else:
                # Just the question
                formatted_prompt = prompt
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": formatted_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "top_k": 40,
                        "repeat_penalty": 1.2,
                        "num_predict": 128,
                        "stop": ["\n\n\n", "Question:", "Context:"]
                    }
                },
                timeout=300
            )
            
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
            
            # Aggressive cleanup
            if not answer or len(answer) < 10:
                return "I don't know."
            
            # Remove prompt artifacts
            for stop_word in ["Question:", "Context:", "Answer:", "\n\nContext"]:
                if stop_word in answer:
                    answer = answer.split(stop_word)[0].strip()
            
            # Get first paragraph only
            paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
            if paragraphs:
                answer = paragraphs[0]
            
            # Limit to 3 sentences
            sentences = []
            for s in answer.replace('!', '.').replace('?', '.').split('.'):
                s = s.strip()
                if s and len(s) > 10:
                    sentences.append(s)
                if len(sentences) >= 3:
                    break
            
            if sentences:
                answer = '. '.join(sentences) + '.'
            else:
                return "I don't know."
            
            # Final safety check
            if len(answer) > 400:
                answer = answer[:400].rsplit('.', 1)[0] + '.'
            
            return answer
            
        except Exception as e:
            return "I don't know."


def main():
    """Main chat function."""
    try:
        chat = OllamaFineTunedQwenChat()
        
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


if __name__ == "__main__":
    main()