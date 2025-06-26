import os
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import threading

# 設定
WET_PATHS_FILE = "wet.paths.gz"        # 你的本機 .gz 檔
OUTPUT_DIR = "wet_files"               # 下載目標資料夾
TARGET_COUNT = 5000                     # 需要 100 個成功下載的 WET 檔
MAX_THREADS = 8                        # 同時下載數量
CHUNK_SIZE = 1024 * 1024               # 每次下載大小（1MB）

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 已存在的檔案
downloaded_files = set(os.listdir(OUTPUT_DIR))
success_count = len(downloaded_files)
print(f"📦 已存在 {success_count} 個檔案。目標：{TARGET_COUNT} 個")

# 共享變數與鎖
stop_flag = threading.Event()
lock = threading.Lock()

def download_file(path):
    if stop_flag.is_set():
        return False

    url = f"https://data.commoncrawl.org/{path}"
    filename = os.path.join(OUTPUT_DIR, os.path.basename(path))
    temp_file = filename + ".part"

    if os.path.basename(filename) in downloaded_files:
        return True  # 已存在

    try:
        headers = {}
        if os.path.exists(temp_file):
            downloaded_size = os.path.getsize(temp_file)
            headers["Range"] = f"bytes={downloaded_size}-"
        else:
            downloaded_size = 0

        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            if r.status_code not in (200, 206):
                print(f"\n❌ HTTP Error {r.status_code} for {url}")
                return False

            with open(temp_file, "ab") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

            os.rename(temp_file, filename)
            with lock:
                downloaded_files.add(os.path.basename(filename))
                global success_count
                success_count += 1
                if success_count >= TARGET_COUNT:
                    stop_flag.set()  # 觸發停止信號
            return True

    except Exception as e:
        print(f"\n❌ Error downloading {url}: {e}")
        return False

def generate_paths():
    """從本機的 wet.paths.gz 讀取每一行"""
    with gzip.open(WET_PATHS_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            yield line.strip()

def main():
    global success_count

    remaining = max(TARGET_COUNT - success_count, 0)
    print(f"⏳ 還需要下載 {remaining} 個檔案")

    with tqdm(total=remaining, desc="Downloading") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = []

            for path in generate_paths():
                if stop_flag.is_set():
                    break
                futures.append(executor.submit(download_file, path))

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    completed += 1
                    pbar.update(1)
                if stop_flag.is_set():
                    break

        print(f"\n✅ 成功下載 {min(success_count, TARGET_COUNT)} 個 WET 文件")

if __name__ == "__main__":
    main()