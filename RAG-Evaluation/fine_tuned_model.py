from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from colorama import Fore, init

init(autoreset=False)

class FineTunedQwenChat:
    """Chat interface using fine-tuned Qwen 2.5 0.5B model with adapters."""
    
    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-0.5B",
        adapter_path: str = "adapter-checkpoints/sft_1"
    ):
        """
        Initialize the fine-tuned Qwen chat.
        
        Args:
            base_model: Base HuggingFace model name
            adapter_path: Path to PEFT adapter checkpoints
        """
        self.base_model = base_model
        self.adapter_path = adapter_path
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize tokenizer and fine-tuned model with adapters."""
        try:
            # print(Fore.CYAN + f"[INFO] Loading base model: {self.base_model}" + Fore.RESET)
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model, device_map="auto")
            
            # print(Fore.CYAN + f"[INFO] Loading adapter: {self.adapter_path}" + Fore.RESET)
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            
            # print(Fore.GREEN + f"[SUCCESS] Fine-tuned model loaded successfully" + Fore.RESET)
        except Exception as e:
            # print(Fore.RED + f"[ERROR] Failed to load model: {str(e)}" + Fore.RESET)
            raise e
    
    def generate_response(self, prompt: str = None, context: str = None) -> str:
        """
        Generate response using fine-tuned model.
        
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
                                    
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.8,
                do_sample=True
            )
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = response.split("Answer:")[-1].strip()
            
            return response
        except Exception as e:
            # print(Fore.RED + f"[ERROR] Failed to generate response: {str(e)}" + Fore.RESET)
            raise e


def main():
    """Main chat function."""
    
    try:
        chat = FineTunedQwenChat(
            base_model="Qwen/Qwen2.5-0.5B",
            adapter_path="adapter-checkpoints/sft_1"
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