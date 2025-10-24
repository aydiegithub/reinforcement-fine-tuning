import os
from datasets import load_dataset
from colorama import Fore
from transformers import AutoTokenizer, AutoModelForCausalLM


HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")

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


base_model = "meta-llama/Llama-3.2-1B" 
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
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

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    token=HUGGING_FACE_ACCESS_TOKEN,
    cache_dir="./workspace",
)

print(Fore.CYAN + str(model) + Fore.RESET)
print(Fore.LIGHTMAGENTA_EX + str(next(model.parameters()).device))