import os
from datasets import load_dataset
from colorama import Fore

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training


HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")

BASE_MODEL = "HuggingFaceTB/SmolLM-135M"
OUTPUT_DIR = "HuggingFaceTB/SmolLM-135M-SFT"

dataset = load_dataset("data", split="train")
print(Fore.YELLOW + str(dataset[2]) + Fore.RESET)

def format_chat_template(batch, tokenizer):
    system_prompt = (
        "You are a helpful, honest, and cautious assistant designed to help engineers think logically. "
        "Do not make things up. If you do not know something, clearly say so."
    )

    samples = []
    questions = batch["question"]
    answers = batch["answer"]

    for i in range(len(questions)):
        text = (
            f"### System:\n{system_prompt}\n\n"
            f"### User:\n{questions[i]}\n\n"
            f"### Assistant:\n{answers[i]}"
        )
        samples.append(text)

    return {
        "instruction": questions,
        "response": answers,
        "text": samples
    }


def main():
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_auth_token=HUGGING_FACE_ACCESS_TOKEN
    )

    # Ensure pad token exists (SmolLM may not define one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset mapping
    train_dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        num_proc=8,
        batched=True,
        batch_size=10
    )

    print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET)

    # Quantization configuration (works fine with SmolLM)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=quant_config,
        token=HUGGING_FACE_ACCESS_TOKEN,
        cache_dir="./workspace",
    )

    print(Fore.CYAN + str(model) + Fore.RESET)
    print(Fore.LIGHTMAGENTA_EX + str(next(model.parameters()).device))

    # Gradient checkpointing and LoRA prep
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model=model)

    # LoRA config
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
            num_train_epochs=50
        ),
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model("complete_checkpoint")
    trainer.model.save_pretrained("final_model")


if __name__ == "__main__":
    main()