import concurrent.futures
import os
import gzip
import pathlib
import datetime
from tqdm import tqdm
from pathlib import Path
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tldextract import TLDExtract

# 初始化 TLDExtract（第一次使用會下載資料）
tld_extractor = TLDExtract()

def process_single_wet_file(input_path: str, output_path: str):
    """
    讀取 WET 檔案，只保留 .edu 或 .gov 的網頁內容，
    並將純文字內容寫入 output_path（以 .gz 壓縮格式）。
    """
    with gzip.open(output_path, 'wt', encoding='utf-8') as out_file:  # ✅ 改成 gzip.open
        with open(input_path, 'rb') as stream:
            for record in ArchiveIterator(stream):
                if record.record_type == WarcRecordType.conversion:
                    url = record.headers.get('WARC-Target-URI', '')
                    text_content = record.reader.read().decode('utf-8', errors='ignore')

                    # 使用 tldextract 抓 domain 資訊
                    extracted = tld_extractor(url)
                    suffix = extracted.suffix  # e.g., 'edu', 'gov'

                    # 只保留 edu 和 gov 網域
                    if suffix in ["edu", "gov"]:
                        # 寫入 URL 和內容（你可以根據需求調整格式）
                        out_file.write(f"URL: {url}\n")
                        out_file.write(f"{text_content.strip()}\n")
                        out_file.write("-" * 80 + "\n")

    return output_path


# Set up the executor
num_cpus = len(os.sched_getaffinity(0))
executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)

wet_filepaths = [
    "wet_files/CC-MAIN-20250417135010-20250417165010-00000.warc.wet.gz",
    "wet_files/CC-MAIN-20250417135010-20250417165010-00001.warc.wet.gz",
    "wet_files/CC-MAIN-20250417135010-20250417165010-00002.warc.wet.gz"
]

# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# output_directory_path = Path(f"processed_wet_files_{timestamp}")
# output_directory_path.mkdir(exist_ok=True)
# print(f"Output directory: {output_directory_path}")
output_directory_path = Path("processed_wet_files")
output_directory_path.mkdir(exist_ok=True)


futures = []
for wet_filepath in wet_filepaths:
    # 去掉 .warc.wet.gz，換成 .filtered.txt.gz
    base_name = pathlib.Path(wet_filepath).stem  # e.g. CC-MAIN-...-00000
    base_name = base_name.replace('.warc.wet', '')  # 移除可能殘留的 .warc.wet
    wet_filename = f"{base_name}.filtered.txt.gz"

    future = executor.submit(
        process_single_wet_file,
        wet_filepath,
        os.path.join(output_directory_path, wet_filename)
    )
    futures.append(future)

# 追蹤進度
for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(wet_filepaths),
):
    try:
        output_file = future.result()
        print(f"Output file written: {output_file}")
    except Exception as exc:
        print(f"Task generated an exception: {exc}")