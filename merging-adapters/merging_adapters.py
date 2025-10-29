import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- Configuration ---
# Make sure these paths match your setup
BASE_MODEL_PATH = "Qwen/Qwen2.5-0.5B"
ADAPTER_PATH = "adapter-checkpoints/20-epochs" # !! IMPORTANT: Change this if you use a different adapter folder !!
EXPORT_PATH = "./merged_model_for_ollama"
# ---------------------

print(f"[*] Loading base model: {BASE_MODEL_PATH}")
# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency
    device_map="auto"
)

print(f"[*] Loading tokenizer: {BASE_MODEL_PATH}")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

print(f"[*] Loading PEFT adapters from: {ADAPTER_PATH}")
# Load the PEFT model (adapters)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("[*] Merging adapters into the base model...")
# Merge the adapters into the base model
model = model.merge_and_unload()
print("[+] Merge complete.")

print(f"[*] Saving merged model to: {EXPORT_PATH}")
# Create directory if it doesn't exist
os.makedirs(EXPORT_PATH, exist_ok=True)

# Save the merged model
model.save_pretrained(EXPORT_PATH)

# Save the tokenizer
tokenizer.save_pretrained(EXPORT_PATH)

print(f"[+] Successfully saved merged model and tokenizer to {EXPORT_PATH}")
print("\nNext steps:")
print("1. Make sure you have the 'Modelfile' in your current directory.")
print("2. Run the following command in your terminal:")
print(f"   ollama create my-finetuned-qwen -f ./Modelfile")