import os
import time
import warnings
from datasets import load_dataset
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training

import mlflow
import mlflow.pytorch

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# CONFIGURATION
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN", None)
USER_NAME = "aydiegithub"
BASE_MODEL_NAME = "qwen2p5-0p5B"  # . replaced with p

# Load dataset once
dataset = load_dataset("data", split="train")
print(f"{Fore.GREEN}✓ Using {len(dataset)} samples for training\n{Style.RESET_ALL}")

# 5 DIFFERENT CONFIGURATIONS TO TEST
CONFIGS = [
    {
        "id": 1,
        "name": "aggressive_lr",
        "description": "High learning rate, high dropout - faster convergence",
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "learning_rate": 1e-3,
        "warmup_ratio": 0.05,
        "lr_scheduler": "cosine",
        "optim": "adamw_8bit",
    },
    {
        "id": 2,
        "name": "balanced",
        "description": "Balanced parameters - good for production",
        "lora_r": 3,
        "lora_alpha": 6,
        "lora_dropout": 0.075,
        "learning_rate": 5e-4,
        "warmup_ratio": 0.025,
        "lr_scheduler": "linear",
        "optim": "adafactor",
    },
    {
        "id": 3,
        "name": "conservative",
        "description": "Low learning rate, low dropout - stable training",
        "lora_r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.01,
        "lr_scheduler": "constant",
        "optim": "paged_adamw_8bit",
    },
    {
        "id": 4,
        "name": "high_rank",
        "description": "Higher LoRA rank - more parameters to learn",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "learning_rate": 3e-4,
        "warmup_ratio": 0.05,
        "lr_scheduler": "cosine",
        "optim": "adamw_8bit",
    },
    {
        "id": 5,
        "name": "low_dropout",
        "description": "Lower dropout for less regularization",
        "lora_r": 3,
        "lora_alpha": 6,
        "lora_dropout": 0.02,
        "learning_rate": 4e-4,
        "warmup_ratio": 0.03,
        "lr_scheduler": "linear",
        "optim": "adafactor",
    },
]


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


def train_with_config(config, train_dataset, tokenizer):
    """Train model with specific configuration and track with MLflow"""
    
    config_id = f"{config['id']:02d}"
    # Format: qwen2p5-0p5B_aydiegithub_config_01_aggressive_lr
    model_name = f"{BASE_MODEL_NAME}_{USER_NAME}_config_{config_id}_{config['name']}"
    output_dir = f"./models/{model_name}"
    
    print(f"\n{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🚀 CONFIG {config_id}: {config['name'].upper()}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Description: {config['description']}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Model will be saved as: {Fore.GREEN}{model_name}{Style.RESET_ALL}")
    
    # MLflow Experiment Setup
    experiment_name = f"{BASE_MODEL_NAME}_{USER_NAME}_finetuning"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"config_{config_id}_{config['name']}"):
        
        # Log parameters
        mlflow.log_params({
            "config_id": config_id,
            "config_name": config["name"],
            "lora_r": config["lora_r"],
            "lora_alpha": config["lora_alpha"],
            "lora_dropout": config["lora_dropout"],
            "learning_rate": config["learning_rate"],
            "warmup_ratio": config["warmup_ratio"],
            "lr_scheduler": config["lr_scheduler"],
            "optimizer": config["optim"],
            "dataset_size": len(train_dataset),
            "base_model": BASE_MODEL,
            "user": USER_NAME,
            "model_naming_convention": f"{BASE_MODEL_NAME}_{USER_NAME}_config_XX_name",
        })
        
        print(f"\n{Fore.BLUE}📊 Configuration Parameters:{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLUE_EX}  • LoRA Rank: {Fore.YELLOW}{config['lora_r']}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLUE_EX}  • LoRA Alpha: {Fore.YELLOW}{config['lora_alpha']}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLUE_EX}  • LoRA Dropout: {Fore.YELLOW}{config['lora_dropout']}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLUE_EX}  • Learning Rate: {Fore.YELLOW}{config['learning_rate']}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLUE_EX}  • Warmup Ratio: {Fore.YELLOW}{config['warmup_ratio']}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLUE_EX}  • LR Scheduler: {Fore.YELLOW}{config['lr_scheduler']}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLUE_EX}  • Optimizer: {Fore.YELLOW}{config['optim']}{Style.RESET_ALL}")
        
        # Load fresh model
        print(f"\n{Fore.CYAN}📦 Loading model...{Style.RESET_ALL}")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True,
            token=HUGGING_FACE_ACCESS_TOKEN,
            cache_dir="./workspace",
        )
        
        print(f"{Fore.GREEN}✓ Model loaded on {Fore.YELLOW}{next(model.parameters()).device}{Style.RESET_ALL}")
        
        # Prepare model
        print(f"{Fore.CYAN}Preparing model for training...{Style.RESET_ALL}")
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        print(f"{Fore.GREEN}✓ Model prepared{Style.RESET_ALL}")
        
        # LoRA Config
        peft_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )
        
        # Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            args=SFTConfig(
                output_dir=output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                gradient_checkpointing=True,
                save_strategy="no",
                logging_steps=100,
                learning_rate=config["learning_rate"],
                lr_scheduler_type=config["lr_scheduler"],
                warmup_ratio=config["warmup_ratio"],
                fp16=True,
                optim=config["optim"],
                seed=42,
                remove_unused_columns=True,
            ),
        )
        
        print(f"\n{Fore.GREEN}🎯 Training Configuration {config_id} starting...{Style.RESET_ALL}\n")
        start_time = time.time()
        
        # Train
        trainer.train()
        
        elapsed = time.time() - start_time
        hours = elapsed / 3600
        
        print(f"\n{Fore.GREEN}✓ Training completed in {Fore.YELLOW}{hours:.2f}{Fore.GREEN} hours!{Style.RESET_ALL}")
        
        # Log metrics
        mlflow.log_metrics({
            "training_time_hours": hours,
            "training_time_seconds": elapsed,
        })
        
        # Save model with proper naming
        final_model_dir = f"./final_models/{model_name}"
        os.makedirs(final_model_dir, exist_ok=True)
        
        print(f"{Fore.CYAN}💾 Saving model...{Style.RESET_ALL}")
        trainer.model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        # Create metadata file
        metadata = {
            "model_name": model_name,
            "config_id": config_id,
            "config_name": config["name"],
            "base_model": BASE_MODEL,
            "user": USER_NAME,
            "timestamp": datetime.utcnow().isoformat(),
            "training_time_hours": hours,
            "dataset_size": len(train_dataset),
            "lora_r": config["lora_r"],
            "lora_alpha": config["lora_alpha"],
            "lora_dropout": config["lora_dropout"],
            "learning_rate": config["learning_rate"],
            "warmup_ratio": config["warmup_ratio"],
            "lr_scheduler": config["lr_scheduler"],
            "optimizer": config["optim"],
        }
        
        # Save metadata as JSON
        import json
        metadata_path = os.path.join(final_model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Log model info
        mlflow.log_param("model_save_path", final_model_dir)
        mlflow.log_artifact(metadata_path)
        
        print(f"{Fore.GREEN}✓ Model saved to: {Fore.YELLOW}{final_model_dir}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Metadata saved: {Fore.YELLOW}{metadata_path}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ MLflow run tracked: {Fore.CYAN}{mlflow.active_run().info.run_name}{Style.RESET_ALL}")
        
        return {
            "config_id": config_id,
            "model_name": model_name,
            "model_dir": final_model_dir,
            "training_time": hours,
            "config": config,
        }


def main():
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🔬 MULTI-CONFIG PARAMETER TUNING WITH MLFLOW{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}User: {Fore.GREEN}{USER_NAME}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Base Model: {Fore.GREEN}{BASE_MODEL}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Base Model Name (for files): {Fore.GREEN}{BASE_MODEL_NAME}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Dataset Size: {Fore.GREEN}{len(dataset)} samples{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Total Configs: {Fore.GREEN}{len(CONFIGS)}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Estimated Total Time: {Fore.YELLOW}~{len(CONFIGS) * 1.75:.1f} hours{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Current Time: {Fore.YELLOW}{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    
    # Format dataset once
    print(f"\n{Fore.CYAN}📋 Formatting dataset...{Style.RESET_ALL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=HUGGING_FACE_ACCESS_TOKEN
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        batched=True,
        num_proc=1,
        batch_size=1
    )
    
    print(f"{Fore.GREEN}✓ Dataset formatted: {Fore.YELLOW}{len(train_dataset)} samples{Style.RESET_ALL}")
    
    # Train with each configuration
    results = []
    start_time_total = time.time()
    
    for config in CONFIGS:
        try:
            result = train_with_config(config, train_dataset, tokenizer)
            results.append(result)
        except Exception as e:
            print(f"\n{Fore.RED}❌ Error in config {config['id']}: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            continue
    
    total_elapsed = time.time() - start_time_total
    total_hours = total_elapsed / 3600
    
    # Summary
    print(f"\n{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✅ ALL TRAINING COMPLETED!{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}📊 TRAINING SUMMARY:\n{Style.RESET_ALL}")
    
    print(f"{Fore.LIGHTWHITE_EX}{'Config':<8} {'Model Name':<55} {'Time (h)':<10}{Style.RESET_ALL}")
    print(f"{Fore.LIGHTBLACK_EX}{'-' * 75}{Style.RESET_ALL}")
    
    for result in results:
        config_id = result["config_id"]
        model_name = result["model_name"]
        training_time = result["training_time"]
        print(f"{Fore.LIGHTGREEN_EX}{config_id:<8}{Style.RESET_ALL} {Fore.WHITE}{model_name:<55}{Style.RESET_ALL} {Fore.YELLOW}{training_time:<10.2f}{Style.RESET_ALL}")
    
    print(f"{Fore.LIGHTBLACK_EX}{'-' * 75}{Style.RESET_ALL}")
    print(f"{Fore.LIGHTGREEN_EX}{'TOTAL':<8}{Style.RESET_ALL} {Fore.WHITE}{len(results)} configs trained{'':<38}{Style.RESET_ALL} {Fore.YELLOW}{total_hours:.2f}{Style.RESET_ALL}")
    
    print(f"\n{Fore.MAGENTA}📁 Models saved to: {Fore.GREEN}./final_models/{Style.RESET_ALL}")
    print(f"\n{Fore.MAGENTA}📋 Model Naming Convention:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}   {BASE_MODEL_NAME}_{USER_NAME}_config_XX_<config_name>{Style.RESET_ALL}")
    print(f"\n{Fore.MAGENTA}   Examples:{Style.RESET_ALL}")
    for result in results[:3]:
        print(f"{Fore.LIGHTGREEN_EX}   - {result['model_name']}{Style.RESET_ALL}")
    if len(results) > 3:
        print(f"{Fore.LIGHTGREEN_EX}   - ... and {len(results) - 3} more{Style.RESET_ALL}")
    
    print(f"\n{Fore.BLUE}🎯 Best practices:{Style.RESET_ALL}")
    print(f"{Fore.LIGHTBLUE_EX}  1. Use MLflow UI to compare metrics: {Fore.YELLOW}mlflow ui{Style.RESET_ALL}")
    print(f"{Fore.LIGHTBLUE_EX}  2. Evaluate each model on validation set{Style.RESET_ALL}")
    print(f"{Fore.LIGHTBLUE_EX}  3. Check metadata.json in each model folder for config details{Style.RESET_ALL}")
    
    print(f"\n{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.RED}⏱️  TOTAL EXECUTION TIME: {Fore.YELLOW}{total_hours:.2f} hours ({total_elapsed/60:.1f} minutes){Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    
    return results


if __name__ == "__main__":
    print(f"{Fore.GREEN}CUDA available: {Fore.YELLOW}{torch.cuda.is_available()}{Style.RESET_ALL}")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"{Fore.GREEN}GPU Memory: {Fore.YELLOW}{gpu_mem:.2f} GB")
        print(f"{Fore.GREEN}Expected usage: {Fore.YELLOW}~5.5-6GB (Safe!){Style.RESET_ALL}\n")
    
    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./final_models", exist_ok=True)
    
    try:
        results = main()
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()