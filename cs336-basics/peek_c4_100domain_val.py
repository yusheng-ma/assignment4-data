import numpy as np

# Load the tokenized data
data = np.fromfile(
    # "cs336-basics/data/paloma/tokenized_paloma_c4_100_domains_validation.bin",
    "cs336-basics/tokenized_output/splits/merged_gpt2_data_train.bin",
    dtype=np.uint16
)

# Print number of tokens
print("Number of tokens:", data.size)

# Decode and print first 2000 tokens for inspection
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.decode(data[0:100]))