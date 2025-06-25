import fasttext
import os

# 設定訓練參數
model = fasttext.train_supervised(
    input="output/train_combined.txt",   # 輸入文件
    epoch=20,                            # 訓練 20 輪，讓模型學得更好
    lr=0.2,                              # 學習率提高一點
    wordNgrams=3,                        # 使用 up to tri-gram 特徵
    verbose=2,                           # 輸出詳細訓練信息
    dim=100                              # 詞向量維度（默認 100，也可調到 200）
)

# 保存模型
model.save_model("output/quality_classifier.bin")

# 在驗證集上測試
result = model.test("output/valid_combined.txt")

# 解析結果
num_samples = result[0]
precision_at_1 = result[1]
recall_at_1 = result[2]

# 打印人類可讀的結果
print("\n=== Model Evaluation ===")
print(f"Number of validation samples: {num_samples}")
print(f"Precision @1: {precision_at_1 * 100:.2f}%")
print(f"Recall @1: {recall_at_1 * 100:.2f}%")
print(f"Accuracy (approx.): {(precision_at_1 + recall_at_1) / 2 * 100:.2f}%")
print("========================")
