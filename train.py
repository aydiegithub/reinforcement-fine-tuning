from datasets import load_dataset
from colorama import Fore

dataset = load_dataset("data" split="train")
print(Fore.YELLOW + str(dataset[2]) + Fore.RESET)