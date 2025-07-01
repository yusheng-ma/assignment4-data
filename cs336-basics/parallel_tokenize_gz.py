import os
import gzip
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
from transformers import GPT2TokenizerFast

# Configuration
input_dir = "cs336-basics/final_output"     # Folder containing .final.gz files
output_dir = "cs336-basics/tokenized_output/splits"  # Output directory for .bin files
output_prefix = "gpt2_data"
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

    split_size = 50

    for i, start_idx in enumerate(range(0, len(input_files), split_size)):
        group = input_files[start_idx : start_idx + split_size]
        print(f"ParallelGroup {i} → Files: {len(group)}")

        all_ids = []

        # Show progress bar over files only
        for file in tqdm(group, desc="Files processed", total=len(group)):
            try:
                ids = process_file(file)
                all_ids.extend(ids)
            except Exception as e:
                print(f"\n⚠️ Error processing file {file}: {e}")
                continue

        print(f"🔢 Total tokens: {len(all_ids):,}")

        # Convert to NumPy array
        ids_array = np.array(all_ids, dtype=np.uint16)  # GPT-2 vocab size is ~50k → fits in uint16

        # Save to disk
        train_output = Path(output_dir) / f"{output_prefix}_train_{i}.bin"

        ids_array.tofile(train_output)

        print(f"\n💾 Saved {len(ids_array):,} training tokens to {train_output}")