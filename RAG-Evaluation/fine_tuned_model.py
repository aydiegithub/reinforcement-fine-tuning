from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from colorama import init, Fore
import warnings
import torch

warnings.filterwarnings("ignore")
init(autoreset=True)

class FineTunedQwenChat:
    """Chat interface for a fine-tuned Qwen 2.5 0.5B model with PEFT adapters."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-0.5B",
        adapter_path: str = "adapter-checkpoints/20-epochs",
        max_new_tokens: int = 200,
        temperature: float = 0.1,
        top_p: float = 0.85,
        top_k: int = 30,
        repetition_penalty: float = 1.3
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Load the tokenizer and model with adapters applied."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model, 
            device_map="auto",
            torch_dtype=torch.float16  # Use FP16 for faster inference
        )
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()  # Set to evaluation mode

    def generate_response(self, prompt: str, context: str = None) -> str:
        """Generate response."""
        try:
            if context:
                # RAG mode - optimized for context-based answering
                formatted_prompt = f"""Based on the context provided, answer the question directly and concisely.

Context:
{context}

Question: {prompt}

Provide a clear, factual answer in 2-3 sentences based only on the context above."""
            else:
                # No context - direct question answering
                formatted_prompt = f"""{prompt}

Provide a clear, factual answer in 2-3 sentences."""

            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            # Use torch.no_grad() for inference
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                    no_repeat_ngram_size=3  # Prevent repetitive n-grams
                )
            
            # Extract only the generated tokens
            new_tokens = output[0][input_length:]
            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Clean up the response
            cleanup_markers = ["Answer:", "Response:", "Question:", "Context:", "Based on", "Provide a"]
            for marker in cleanup_markers:
                if marker in answer:
                    parts = answer.split(marker)
                    # Take the part after the marker if it's at the start, otherwise before
                    if answer.startswith(marker):
                        answer = parts[-1].strip()
                    else:
                        answer = parts[0].strip()

            # Remove lines that are prompt artifacts
            lines = answer.split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                if line and not any(marker.lower() in line.lower() for marker in ["question:", "context:", "provide a", "based on the context", "clear, factual"]):
                    clean_lines.append(line)
            
            answer = ' '.join(clean_lines)

            # Limit to 3 sentences max
            sentences = []
            for sent in answer.replace('!', '.').replace('?', '.').split('.'):
                sent = sent.strip()
                if sent and len(sent) > 5:  # Avoid tiny fragments
                    sentences.append(sent)
            
            if sentences:
                answer = '. '.join(sentences[:3])
                if not answer.endswith('.'):
                    answer += '.'
            else:
                answer = "I don't know."
            
            # Final length check
            if len(answer) > 500:
                answer = answer[:500].rsplit('.', 1)[0] + '.'

            return answer
            
        except Exception as e:
            return f"Error: {str(e)}"


def main():
    """Main chat function."""
    try:
        chat = FineTunedQwenChat(
            base_model="Qwen/Qwen2.5-0.5B",
            adapter_path="adapter-checkpoints/sft_1"
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

if __name__ == "__main__":
    main()