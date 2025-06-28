import os
import gzip
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
from transformers import GPT2TokenizerFast

# Configuration
input_dir = "cs336-basics/final_output"     # Folder containing .final.gz files
output_dir = "cs336-basics/tokenized_output"  # Output directory for .bin files
output_prefix = "gpt2_data"
val_ratio = 0.1  # 10% for validation
chunksize = 100

# Create output directory
Path(output_dir).mkdir(exist_ok=True)

# Load GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def tokenize_line_and_add_eos(line):
    encoded = tokenizer.encode(
        line.strip(),
        add_special_tokens=False,
        max_length=511,
        truncation=True
    )
    if len(encoded) == 511:
        print(f"✂️ Line truncated to {len(encoded)} tokens")
    return encoded + [tokenizer.eos_token_id]

def read_gz_file(file_path):
    """Read lines from a .gz file"""
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return [line for line in f if line.strip()]

def process_file(file_path):
    """Read and tokenize all lines in a file without internal tqdm"""
    lines = read_gz_file(file_path)
    with Pool(cpu_count()) as pool:
        tokenized = pool.map(tokenize_line_and_add_eos, lines, chunksize=chunksize)
    # Flatten list of tokens
    return [token for sublist in tokenized for token in sublist]

if __name__ == "__main__":
    print("🔍 Reading input files...")
    input_files = list(Path(input_dir).glob("*.final.gz"))
    if not input_files:
        raise FileNotFoundError(f"No .final.gz files found in {input_dir}")

    print(f"🧠 Loading and tokenizing {len(input_files)} files...")
    all_ids = []

    # Show progress bar over files only
    for file in tqdm(input_files, desc="Files processed", total=len(input_files)):
        try:
            ids = process_file(file)
            all_ids.extend(ids)
        except Exception as e:
            print(f"\n⚠️ Error processing file {file}: {e}")
            continue

    print(f"🔢 Total tokens: {len(all_ids):,}")

    # Convert to NumPy array
    ids_array = np.array(all_ids, dtype=np.uint16)  # GPT-2 vocab size is ~50k → fits in uint16

    # Split into train and validation
    val_size = int(len(ids_array) * val_ratio)
    train_ids = ids_array[val_size:]
    val_ids = ids_array[:val_size]

    # Save to disk
    train_output = Path(output_dir) / f"{output_prefix}_train.bin"
    val_output = Path(output_dir) / f"{output_prefix}_val.bin"

    train_ids.tofile(train_output)
    val_ids.tofile(val_output)

    print(f"\n💾 Saved {len(train_ids):,} training tokens to {train_output}")
    print(f"💾 Saved {len(val_ids):,} validation tokens to {val_output}")