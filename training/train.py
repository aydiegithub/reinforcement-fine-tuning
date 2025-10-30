import os
from datasets import load_dataset
from colorama import Fore

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training



HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
BASE_MODEL = "meta-llama/Llama-3.2-1B" 
OUTPUT_DIR = "meta-llama/Llama-3.2-1B-SFT"


dataset = load_dataset("data", split="train")
print(Fore.YELLOW + str(dataset[2]) + Fore.RESET)


def format_chat_template(batch, tokenizer):
        system_prompt = """
        You are a helpful, honest, and cautious assistant, designed to help engineers think through each question logically and provide accurate answers. 
        Do not make things up, Do not fabricate information. 
        If a question is beyond your knowledge or scope, clearly inform the user that you are unable to answer.
        """

        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        
        samples = []
        questions = batch['question']
        answers = batch['answer']
        for i in range(len(questions)):
            row_json = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": questions[i]},
                {"role": "assistant", "content": answers[i]}
            ]
            
            text = tokenizer.apply_chat_template(row_json, tokenize=False)
            samples.append(text)
            
        return {
            "instruction": questions,
            "response": answers,
            "text": samples
        }

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_auth_token=HUGGING_FACE_ACCESS_TOKEN
    )


    train_dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer), 
        num_proc=8,
        batched=True,
        batch_size=10
    )

    print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=quant_config, # Experiment with quantised and non quantised layers
        token=HUGGING_FACE_ACCESS_TOKEN,
        cache_dir="./workspace",
    )

    print(Fore.CYAN + str(model) + Fore.RESET)
    print(Fore.LIGHTMAGENTA_EX + str(next(model.parameters()).device))

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model=model)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir=OUTPUT_DIR, 
            num_train_epochs=20
            # save_steps=1000,
        ),
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model('complete_checkpoint')
    trainer.model.save_pretrained("final_model")

if __name__ == "__main__":
    main()