from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from colorama import Fore


base_model = "Qwen/Qwen2.5-0.5B"
adapter_path = "adapter-checkpoints/sft_1"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)

while True:
    print("\n")
    prompt = input(Fore.GREEN + "Message: " + Fore.RESET)
    if prompt == "/bye":
        print(Fore.RED + "Exiting..." + Fore.RESET)
        break
    prompt = f"""Answer the following question directly and concisely. Only provide the answer, nothing else.
Question: {prompt}
Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.8,
        do_sample=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.split("Answer:")[-1].strip()
    print(Fore.BLUE + response + Fore.RESET)