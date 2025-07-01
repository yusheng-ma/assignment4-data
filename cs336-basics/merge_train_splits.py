import os

# 修改這裡：資料夾路徑
input_folder = "cs336-basics/tokenized_output/splits"
output_file = os.path.join(input_folder, "merged_gpt2_data_train.bin")

# 確保資料夾存在
os.makedirs(input_folder, exist_ok=True)

# 找出所有 gpt2_data_train_*.bin 檔案
filenames = [f for f in os.listdir(input_folder) if f.startswith("gpt2_data_train_") and f.endswith(".bin")]

# 從檔名中提取數字並排序 (例如 gpt2_data_train_12.bin -> 12)
filenames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

print("🔍 找到以下檔案並將進行合併：")
for f in filenames:
    print(" -", f)

# 開始合併
with open(output_file, "wb") as outfile:
    for filename in filenames:
        file_path = os.path.join(input_folder, filename)
        print(f"🔄 正在合併: {filename}")
        with open(file_path, "rb") as infile:
            outfile.write(infile.read())

print(f"✅ 合併完成！結果檔案位於: {output_file}")