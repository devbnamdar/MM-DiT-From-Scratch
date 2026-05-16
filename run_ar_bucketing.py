import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from ar_bucketing import create_ar_buckets

def main():
    print("==================================================")
    print(" AR BUCKET AUTO GENERATOR (Config Based) ")
    print("==================================================")
    
    config = Config()
    
    dataset_configs = getattr(config, 'datasets', [])
    if not dataset_configs:
        print("Warning: 'datasets' list not found in config.py!")
        return
            
    print(f"\nFound {len(dataset_configs)} dataset configurations in config file.")
            
    print("\nProcess is starting...")
    
    output_dir = getattr(config, 'output_path', '.')
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.abspath(os.path.join(output_dir, "ar_buckets.json"))
    
    create_ar_buckets(dataset_configs, output_name=out_file)
    
    print("\n✅ Process completed. 'ar_buckets.json' successfully created.")
    print("==================================================")

if __name__ == "__main__":
    main()
