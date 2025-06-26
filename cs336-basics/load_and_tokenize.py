import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import os

def main():
    # 設定路徑（請根據你的實際路徑修改）
    output_path = "./data/paloma/tokenized_paloma_c4_100_domains_validation.bin"

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 初始化 GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 定義 tokenize 函數（每行加上 eos_token_id）
    def tokenize_line_and_add_eos(example):
        return tokenizer.encode(
            example["text"].strip(),
            max_length=1024,
            truncation=True
        ) + [tokenizer.eos_token_id]

    # 載入 val 分組
    dataset = load_dataset("allenai/paloma", "c4_100_domains", split="val")

    print(f"Tokenizing {len(dataset)} lines...")

    # 使用 map 函數批量 tokenize 數據集
    results = []
    for example in tqdm(dataset, desc="Tokenizing lines"):
        token_ids = tokenize_line_and_add_eos(example)
        results.append(token_ids)

    # 將所有 token ID 扁平化為一維陣列
    all_ids = [token_id for sublist in results for token_id in sublist]

    # 輸出 token 數量
    print(f"Tokenized and encoded into {len(all_ids)} tokens")

    # 轉成 numpy array 並儲存為 binary file
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_path)

    print(f"Saved tokenized data to: {output_path}")


if __name__ == "__main__":
    main()