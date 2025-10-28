import json
import random
from colorama import Fore, init
from datetime import datetime

init(autoreset=False)

class JSONSampler:
    """Extract and randomly sample records from JSON files."""
    
    def __init__(self, input_file: str, output_file: str, sample_size: int = 1000):
        """
        Initialize JSON sampler.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file
            sample_size: Number of samples to randomly select
        """
        self.input_file = input_file
        self.output_file = output_file
        self.sample_size = sample_size
    
    def _load_json(self) -> list:
        """Load and parse JSON file."""
        try:
            print(Fore.CYAN + f"[INFO] Loading JSON file: {self.input_file}" + Fore.RESET)
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                # If it's a dictionary, try to extract list values
                print(Fore.YELLOW + f"[WARNING] JSON is a dictionary, looking for list values" + Fore.RESET)
                
                # Try common keys
                for key in ['data', 'records', 'items', 'samples', 'chunks']:
                    if key in data and isinstance(data[key], list):
                        print(Fore.CYAN + f"[INFO] Found data in key: '{key}'" + Fore.RESET)
                        return data[key]
                
                # If no common key found, convert values to list
                if isinstance(list(data.values())[0], list):
                    first_key = list(data.keys())[0]
                    print(Fore.CYAN + f"[INFO] Using data from key: '{first_key}'" + Fore.RESET)
                    return data[first_key]
                else:
                    print(Fore.YELLOW + f"[WARNING] No list found, treating dictionary items as records" + Fore.RESET)
                    return list(data.values())
            
            elif isinstance(data, list):
                print(Fore.GREEN + f"[SUCCESS] JSON is a list with {len(data)} records" + Fore.RESET)
                return data
            
            else:
                print(Fore.RED + f"[ERROR] Unsupported JSON structure: {type(data)}" + Fore.RESET)
                raise ValueError(f"JSON must be a list or dictionary, got {type(data)}")
        
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to load JSON: {str(e)}" + Fore.RESET)
            raise
    
    def _sample_data(self, data: list) -> list:
        """Randomly sample data."""
        try:
            total_records = len(data)
            print(Fore.CYAN + f"[INFO] Total records in file: {total_records}" + Fore.RESET)
            print(Fore.CYAN + f"[INFO] Requested sample size: {self.sample_size}" + Fore.RESET)
            
            # Sample min of requested size or available records
            actual_sample_size = min(self.sample_size, total_records)
            
            if actual_sample_size < self.sample_size:
                print(Fore.YELLOW + f"[WARNING] Only {total_records} records available, sampling all" + Fore.RESET)
            
            sampled_data = random.sample(data, actual_sample_size)
            print(Fore.GREEN + f"[SUCCESS] Randomly selected {len(sampled_data)} samples" + Fore.RESET)
            
            return sampled_data
        
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to sample data: {str(e)}" + Fore.RESET)
            raise
    
    def _export_json(self, data: list) -> None:
        """Export sampled data to JSON file."""
        try:
            print(Fore.CYAN + f"[INFO] Exporting to file: {self.output_file}" + Fore.RESET)
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(Fore.GREEN + f"[SUCCESS] Exported {len(data)} samples to {self.output_file}" + Fore.RESET)
        
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to export JSON: {str(e)}" + Fore.RESET)
            raise
    
    def extract(self) -> None:
        """Execute the complete extraction and sampling process."""
        print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
        print(Fore.MAGENTA + "JSON Sampler" + Fore.RESET)
        print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
        print()
        
        try:
            # Load
            data = self._load_json()
            
            print()
            
            # Sample
            sampled_data = self._sample_data(data)
            
            print()
            
            # Export
            self._export_json(sampled_data)
            
            print()
            print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
            print(Fore.GREEN + "[SUCCESS] Sampling completed successfully" + Fore.RESET)
            print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
        
        except Exception as e:
            print(Fore.RED + f"[ERROR] Extraction failed: {str(e)}" + Fore.RESET)
            raise


def main():
    """Main function."""
    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    current_user = "aydiegithub"
    
    print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
    print(Fore.MAGENTA + "JSON Sample Extractor" + Fore.RESET)
    print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
    print(Fore.CYAN + f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {current_time}" + Fore.RESET)
    print(Fore.CYAN + f"Current User's Login: {current_user}" + Fore.RESET)
    print(Fore.MAGENTA + "=" * 80 + Fore.RESET)
    print()
    
    try:
        sampler = JSONSampler(
            input_file="data/instructionquality.json",
            output_file="evaluation_data.json",
            sample_size=1000
        )
        
        sampler.extract()
    
    except Exception as e:
        print(Fore.RED + f"[ERROR] Failed: {str(e)}" + Fore.RESET)


if __name__ == "__main__":
    main()