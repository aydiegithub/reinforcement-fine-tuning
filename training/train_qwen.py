import os
import time
import warnings
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training

warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# CONFIGURATION
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./Qwen2.5-0.5B-SFT"
HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN", None)

# Load dataset - USING ALL 6445 SAMPLES
dataset = load_dataset("data", split="train")
print(f"Using {len(dataset)} samples for training")


def format_chat_template(batch, tokenizer):
    system_prompt = (
        "You are a helpful, honest, and cautious assistant. "
        "Always provide logical, accurate, and factual answers. "
        "If you do not know something, state that clearly."
    )

    samples = []
    questions = batch["question"]
    answers = batch["answer"]

    for i in range(len(questions)):
        text = (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{questions[i]}<|end|>\n"
            f"<|assistant|>\n{answers[i]}<|end|>"
        )
        samples.append(text)

    return {"text": samples}


def main():
    print("="*70)
    print("FULL DATASET TRAINING: 6445 SAMPLES")
    print("Qwen2.5-0.5B | RTX 4050 8GB | BEST QUALITY")
    print("="*70)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=HUGGING_FACE_ACCESS_TOKEN
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Formatting dataset...")
    train_dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        batched=True,
        num_proc=1,
        batch_size=1
    )

    print(f"Dataset: {len(train_dataset)} samples (100% of data)")

    print("Loading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True,
        token=HUGGING_FACE_ACCESS_TOKEN,
        cache_dir="./workspace",
    )

    print("Model loaded!")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Model: {BASE_MODEL}")

    print("Preparing model for training...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    print("LoRA configuration applied")
    print("\nTRAINING CONFIGURATION:")
    print("  Samples: 6445 (100% - BEST QUALITY)")
    print("  Epochs: 1")
    print("  Batch size: 1 per device")
    print("  Gradient accumulation: 1")
    print("  Learning rate: 5e-4")
    print("  Optimizer: adafactor")
    print("  Estimated time: ~5-6 hours")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            save_strategy="no",
            logging_steps=100,
            learning_rate=5e-4,
            lr_scheduler_type="constant",
            warmup_ratio=0.0,
            fp16=True,
            optim="adafactor",
            seed=42,
            remove_unused_columns=True,
        ),
    )

    print("FULL DATASET TRAINING STARTING...")
    
    start_time = time.time()
    
    trainer.train()

    elapsed = time.time() - start_time
    hours = elapsed / 3600
    
    print("="*70)
    print(f"TRAINING COMPLETED IN {hours:.2f} HOURS!")
    print("Saving final model...")
    
    trainer.save_model("complete_checkpoint")
    trainer.model.save_pretrained("final_model")
    tokenizer.save_pretrained("final_model")

    print("Model saved successfully!")
    print("Model location: ./final_model")
    print("Checkpoint location: ./complete_checkpoint")


if __name__ == "__main__":
    st = time.time()
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_mem:.2f} GB")
        print(f"Expected usage: ~5.5-6GB (Safe!)\n")
    
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed = round(time.time() - st, 3)
        hours = elapsed / 3600
        print(f"TOTAL EXECUTION TIME: {hours:.2f} hours ({elapsed/60:.1f} minutes)")