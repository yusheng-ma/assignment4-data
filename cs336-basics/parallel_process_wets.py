import concurrent.futures
import os
import csv
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tldextract import TLDExtract
from cs336_data.utilities import identify_language
from cs336_data.utilities import exact_line_deduplication
from cs336_data.minhash_deduplication import minhash_deduplication
import re
import gzip

# Initialize TLDExtract
tld_extractor = TLDExtract()

# Load C4 domains and extract main domains
def load_c4_domains(file_path):
    domains = set()
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            full_subdomain = row['subdomain']
            clean_subdomain = '_'.join(full_subdomain.split('_')[1:])
            extracted = tld_extractor(clean_subdomain)
            main_domain = f"{extracted.domain}.{extracted.suffix}"
            domains.add(main_domain)
    return domains

def load_extracted_domains(file_path):
    domains = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                full_subdomain = row['subdomain']
                # 移除前面的編號部分（如 "123_"）
                clean_subdomain = '_'.join(full_subdomain.split('_')[1:])
                domains.add(clean_subdomain)
    except Exception as e:
        print(f"⚠️ 無法讀取 {file_path}：{e}")
    return domains

C4_DOMAINS = load_c4_domains("cs336-basics/sql-console-for-allenai-paloma.csv")
EXTRACTED_DOMAINS = load_extracted_domains("cs336-basics/results_100000domains.csv")

# Load bad words from file
def load_bad_words(file_path="cs336-basics/bad_words_en.txt"):
    bad_words_set = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                phrase = line.strip().lower()
                if phrase:
                    bad_words_set.add(phrase)
    except FileNotFoundError:
        print(f"⚠️ 警告：找不到 {file_path}，使用默認過濾規則或請確認檔案存在。")
    return bad_words_set

BAD_WORDS = load_bad_words()

# C4 heuristic functions
def ends_with_punctuation(line):
    return line.strip().endswith(('.', '!', '?', '"', "’", "”"))

def count_words(line):
    return len(line.strip().split())

def is_junk_line(line):
    return (
        'javascript' in line.lower() or
        'lorem ipsum' in line.lower() or
        '{' in line or
        '}' in line
    )

def contains_bad_word(text):
    return any(word in text.lower() for word in BAD_WORDS)

def count_sentences(text):
    return len(re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip()))

def count_lines_in_file(file_path):
    """Helper: Count number of non-empty lines in a file"""
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception as e:
        print(f"Error counting lines in {file_path}: {e}")
    return count

# Process WET file and write plain .txt output
def process_single_wet_file(input_path: str, output_dir: str):
    stats = defaultdict(int)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    base_name = Path(input_path).stem.replace('.warc.wet', '')
    temp_output_path = output_dir / f"{base_name}.cleaned.txt"
    
    with open(temp_output_path, 'w', encoding='utf-8') as out_file:
        with open(input_path, 'rb') as stream:
            for record in ArchiveIterator(stream):
                if record.record_type == WarcRecordType.conversion:
                    stats['total_records'] += 1
                    url = record.headers.get('WARC-Target-URI', '')
                    text_content = record.reader.read().decode('utf-8', errors='ignore')
                    
                    # Extract main domain
                    extracted = tld_extractor(url)
                    main_domain = f"{extracted.domain}.{extracted.suffix}"

                    # Check against C4 and Extracted domains
                    in_c4 = main_domain in C4_DOMAINS
                    in_extracted = main_domain in EXTRACTED_DOMAINS

                    if not in_c4 and not in_extracted:
                        stats['not_in_any_domains'] += 1
                        continue
                    else:
                        if not in_c4:
                            stats['not_in_c4_but_in_extracted'] += 1
                        elif not in_extracted:
                            stats['not_in_extracted_but_in_c4'] += 1
                        else:
                            stats['in_both_domains'] += 1
                    
                    # Language filtering
                    language_code, score = identify_language(text_content)
                    if not (language_code == "en" and score > 0.85):
                        stats['not_english'] += 1
                        continue
                    
                    # Line cleaning based on C4 heuristics
                    cleaned_lines = []
                    lines = text_content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        if not ends_with_punctuation(line):
                            continue
                        if count_words(line) < 3:
                            continue
                        if is_junk_line(line):
                            continue
                        cleaned_lines.append(line)
                    cleaned_text = '\n'.join(cleaned_lines)
                    
                    if count_sentences(cleaned_text) < 5:
                        stats['too_few_sentences'] += 1
                        continue
                    # if contains_bad_word(cleaned_text):
                    #     stats['bad_content'] += 1
                    #     continue
                    
                    out_file.write(f"{cleaned_text}\n")
    
    # Add final counts
    stats['output_lines'] = count_lines_in_file(temp_output_path)
    return temp_output_path, stats

# Compress final output to .gz
def compress_final_output(input_files, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    final_files = []
    for input_file in input_files:
        base_name = Path(input_file).stem
        output_path = output_dir / f"{base_name}.final.gz"
        with open(input_file, 'r', encoding='utf-8') as fin, \
             gzip.open(output_path, 'wt', encoding='utf-8') as fout:
            for line in fin:
                fout.write(line)
        final_files.append(output_path)
    return final_files

# Main execution
if __name__ == "__main__":
    # Setup logging file
    log_path = Path("processing_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
    
    # CPU setup
    num_cpus = len(os.sched_getaffinity(0))
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
    
    # Input WET files — get ALL .warc.wet.gz files
    wet_dir = Path("cs336-basics/wet_files")
    wet_filepaths = sorted(wet_dir.glob("*.warc.wet.gz"))
    if not wet_filepaths:
        raise FileNotFoundError(f"No WET files found in {wet_dir}")
    print(f"✅ Found and selected {len(wet_filepaths)} WET files for processing")
    
    # Step 1: Process WET files → Plain .txt
    temp_cleaned_dir = Path("cs336-basics/temp_cleaned")
    futures = []
    for wet_filepath in wet_filepaths:
        future = executor.submit(
            process_single_wet_file,
            wet_filepath,
            temp_cleaned_dir
        )
        futures.append(future)
    
    total_stats = defaultdict(int)
    temp_files = []
    log("📊 STEP 1: Raw Conversion Extraction")
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(wet_filepaths)):
        try:
            temp_file, stats = future.result()
            temp_files.append(temp_file)
            for key in stats:
                total_stats[key] += stats[key]
            log(f"📄 {temp_file.name}:")
            log(f"  Total conversion records: {stats['total_records']}")
            log(f"  Not in any domains (C4 & extracted): {stats['not_in_any_domains']}")
            log(f"  Not in C4 but in extracted: {stats['not_in_c4_but_in_extracted']}")
            log(f"  Not in extracted but in C4: {stats['not_in_extracted_but_in_c4']}")
            log(f"  In both domains: {stats['in_both_domains']}")
            log(f"  Not English: {stats['not_english']}")
            log(f"  Too few sentences: {stats['too_few_sentences']}")
            log(f"  Bad content: {stats['bad_content']}")
            log(f"  Final output lines: {stats['output_lines']}")
        except Exception as exc:
            log(f"Task generated an exception: {exc}")
    
    log("\n📊 總結（第一階段）：")
    log(f"總共處理的 conversion 數量: {total_stats['total_records']}")
    log(f"不屬於任何 domain (C4 & extracted) 的 conversion 數量: {total_stats['not_in_any_domains']}")
    log(f"屬於 extracted domain 但不屬於 C4 的 conversion 數量: {total_stats['not_in_c4_but_in_extracted']}")
    log(f"屬於 C4 domain 但不屬於 extracted 的 conversion 數量: {total_stats['not_in_extracted_but_in_c4']}")
    log(f"同時屬於兩者的 conversion 數量: {total_stats['in_both_domains']}")
    log(f"屬於 C4 或 extracted 但非英文的 conversion 數量: {total_stats['not_english']}")
    log(f"因句子太少被過濾的 conversion 數量: {total_stats['too_few_sentences']}")
    log(f"因包含不良字詞被過濾的 conversion 數量: {total_stats['bad_content']}")

    # Step 2: Exact line deduplication
    line_deduplicated_dir = Path("cs336-basics/line_deduplicated")
    exact_line_deduplication(temp_files, line_deduplicated_dir)
    log("\n📊 STEP 2: Line Deduplication Summary")
    line_deduplicated_files = list(line_deduplicated_dir.glob("*.cleaned.txt"))
    total_before = sum(count_lines_in_file(f) for f in temp_files)
    total_after = sum(count_lines_in_file(f) for f in line_deduplicated_files)
    log(f"Lines before line deduplication: {total_before}")
    log(f"Lines after line deduplication: {total_after}")
    log(f"Removed by line deduplication: {total_before - total_after} lines")
    
    # Delete first-stage files
    for f in temp_files:
        os.remove(f)
    log("✅ Deleted temporary cleaned files")

    # Step 3: Minhash deduplication
    final_output_dir = Path("cs336-basics/final_output")
    minhash_deduplication(
        input_files=[str(f) for f in line_deduplicated_files],
        num_hashes=128,
        num_bands=4,
        ngrams=5,
        jaccard_threshold=0.8,
        output_directory=final_output_dir
    )
    log("\n📊 STEP 3: MinHash Deduplication Summary")
    final_files = list(final_output_dir.glob("*.txt"))
    total_after_line = sum(count_lines_in_file(f) for f in line_deduplicated_files)
    total_after_minhash = sum(count_lines_in_file(f) for f in final_files)
    log(f"Lines before minhash deduplication: {total_after_line}")
    log(f"Lines after minhash deduplication: {total_after_minhash}")
    log(f"Removed by minhash deduplication: {total_after_line - total_after_minhash} lines")

    # Delete second-stage files
    for f in line_deduplicated_files:
        os.remove(f)
    log("✅ Deleted line-deduplicated files")

    # Step 4: Compress final deduplicated files
    compress_final_output(final_files, final_output_dir)

    # Delete uncompressed final files
    for f in final_files:
        os.remove(f)
    log("✅ Final output compressed and old files deleted")

    log("\n✅ 所有處理完成！")
    log(f"最終結果已寫入：{final_output_dir}")
    log_file.close()