import random
import subprocess
import os
from multiprocessing import Pool, Value
import argparse
from tqdm import tqdm
import time
import glob
import gzip
import uuid
from warcio.archiveiterator import ArchiveIterator


# 全局共享變量
counter = None
total_urls = None

def init_counter(init_val, total):
    global counter, total_urls
    counter = init_val
    total_urls = total

def subsample_urls(input_file, output_file, n):
    """Subsample N URLs from input_file and save to output_file."""
    try:
        with open(input_file, 'r') as f:
            urls = [url.strip() for url in f if url.strip()]
        
        if n > len(urls):
            print(f"Requested {n} URLs, but only {len(urls)} available. Using all URLs.")
            n = len(urls)
        
        sampled_urls = random.sample(urls, n)
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(sampled_urls) + '\n')
        
        print(f"Subsampled {n} URLs and saved to {output_file}")
        return n
    except Exception as e:
        print(f"Error during subsampling: {e}")
        raise

def download_url(args):
    """Download a single URL using wget into a unique WARC file."""
    global counter
    url = args

    try:
        url = url.strip()
        if not url:
            with counter.get_lock():
                counter.value += 1
            return 0

        temp_dir = "temp_warc_files"
        warc_file = os.path.join(temp_dir, f"subsampled_positive_urls_{uuid.uuid4()}.warc")

        cmd = [
            'wget',
            '--timeout=15',
            '--tries=1',
            '--no-warc-compression',
            f'--warc-file={warc_file}',
            '-O', '/dev/null',
            url
        ]

        start_time = time.time()
        with open('download_errors.log', 'a') as log:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            elapsed = time.time() - start_time
            if result.returncode == 0:
                with open('download_success.log', 'a') as success_log:
                    success_log.write(f"Success: {url} ({elapsed:.2f}s)\n")
                with counter.get_lock():
                    counter.value += 1
                return 1
            else:
                log.write(f"Failed: {url} (Error: {result.stderr}, {elapsed:.2f}s)\n")
                with counter.get_lock():
                    counter.value += 1
                return 0
    except subprocess.TimeoutExpired:
        with open('download_errors.log', 'a') as log:
            log.write(f"Timeout: {url} (System timeout after 20s)\n")
            with counter.get_lock():
                counter.value += 1
        return 0
    except Exception as e:
        with open('download_errors.log', 'a') as log:
            log.write(f"Error downloading {url}: {e}\n")
            with counter.get_lock():
                counter.value += 1
        return 0

def merge_warc_files(output_warc_gz):
    """Merge all WARC files into a single compressed WARC.gz file."""
    temp_dir = "temp_warc_files"
    warc_files = glob.glob(os.path.join(temp_dir, "subsampled_positive_urls_*.warc"))

    if not warc_files:
        print("No WARC files found to merge.")
        return

    with gzip.open(output_warc_gz, 'wb') as outfile:
        for warc_file in warc_files:
            try:
                with open(warc_file, 'rb') as infile:
                    chunk = infile.read(1024 * 1024)  # 1MB per chunk
                    while chunk:
                        outfile.write(chunk)
                        chunk = infile.read(1024 * 1024)
                os.remove(warc_file)
            except Exception as e:
                print(f"Error merging {warc_file}: {e}")

    print(f"Merged {len(warc_files)} WARC files into {output_warc_gz}")

def parallel_download(urls, num_processes):
    """Download URLs in parallel with progress bar."""
    try:
        total = len(urls)
        shared_counter = Value('i', 0)

        temp_dir = "temp_warc_files"
        os.makedirs(temp_dir, exist_ok=True)

        with Pool(processes=num_processes, initializer=init_counter, initargs=(shared_counter, total)) as pool:
            with tqdm(total=total, desc="Downloading URLs", unit="url") as pbar:
                result = pool.map_async(download_url, urls)

                while not result.ready():
                    with shared_counter.get_lock():
                        pbar.n = shared_counter.value
                        pbar.refresh()
                    time.sleep(0.1)

                with shared_counter.get_lock():
                    pbar.n = shared_counter.value
                    pbar.refresh()

                successes = sum(result.get())

        print(f"Parallel download completed: {successes}/{total} URLs had valid responses.")
        merge_warc_files("subsampled_positive_urls.warc.gz")

    except Exception as e:
        print(f"Error during parallel download: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Subsample URLs and download them in parallel.")
    parser.add_argument('--input_file', default='enwiki_urls.txt', help='Input file containing URLs')
    parser.add_argument('--output_file', default='subsampled_positive_urls.txt', help='Output file for subsampled URLs')
    parser.add_argument('--num_urls', type=int, default=100000, help='Number of URLs to subsample')
    parser.add_argument('--num_processes', type=int, default=16, help='Number of parallel processes')

    args = parser.parse_args()

    num_urls = subsample_urls(args.input_file, args.output_file, args.num_urls)

    with open(args.output_file, 'r') as f:
        urls = [url.strip() for url in f if url.strip()]

    parallel_download(urls, args.num_processes)

if __name__ == "__main__":
    main()