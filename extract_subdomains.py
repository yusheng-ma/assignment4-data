import csv
from urllib.parse import urlparse

def extract_subdomain(url):
    try:
        # 移除 URL 中的空白字符
        url = url.strip()
        # 解析 URL
        parsed_url = urlparse(url)
        # 獲取網域部分
        netloc = parsed_url.netloc
        if not netloc:
            return None
        # 去掉 www. 前綴（如果有的話）
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return None

input_file = "enwiki_urls.txt"
output_file = "results.csv"

seen_domains = set()
count = 1

with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["subdomain"])  # 寫入 header

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            subdomain = extract_subdomain(line)
            if subdomain and subdomain not in seen_domains:
                seen_domains.add(subdomain)
                writer.writerow([f"{count}_{subdomain}"])
                count += 1
                if count > 100000:
                    break

print(f"已成功寫出 {count - 1} 個唯一 subdomains 到 {output_file}")