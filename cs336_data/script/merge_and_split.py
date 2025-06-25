import random
from sklearn.model_selection import train_test_split

# 讀取所有數據
with open("output/cc_data.train", "r", encoding="utf-8") as f:
    cc_lines = f.readlines()

with open("output/wiki_data.train", "r", encoding="utf-8") as f:
    wiki_lines = f.readlines()

print(f"Read {len(cc_lines)} CC lines")
print(f"Read {len(wiki_lines)} Wiki lines")

# 添加 label（其實已經有了）
all_lines = cc_lines + wiki_lines

# 打亂數據
random.shuffle(all_lines)

# 切分 train / valid
train_lines, valid_lines = train_test_split(all_lines, test_size=0.1, random_state=42)

# 寫入文件
with open("output/train_combined.txt", "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open("output/valid_combined.txt", "w", encoding="utf-8") as f:
    f.writelines(valid_lines)

print(f"Saved {len(train_lines)} train samples")
print(f"Saved {len(valid_lines)} valid samples")