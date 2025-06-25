import gzip
from tqdm import tqdm
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.utilities import extract_text_from_html_bytes, identify_language
from cs336_data.utilities import mask_emails, mask_phone_numbers, mask_ips
from cs336_data.utilities import classify_nsfw, classify_toxic_speech
from cs336_data.utilities import gopher_quality_filter

warc_file_path = "subsampled_positive_urls.warc.gz"
test_count = None
output_path = "output/wiki_data.train"

# gather response
def extract_response(warc_file_path):
    responses = []
    with gzip.open(warc_file_path, 'rb') as warc_file:
        iterator = ArchiveIterator(warc_file)
        for record in tqdm(iterator, desc="Processing WARC Records"):
            if record.headers.get('WARC-Type') == 'response':
                content = record.reader.read()
                responses.append(content)    
                if test_count is not None and len(responses) > test_count:
                    break
    print(f"Finished getting {len(responses)} responses")
    return responses

# warc html to text
def extract_text(responses):
    texts = []
    for response in tqdm(responses, desc="Processing Text Extraction"):
        text = extract_text_from_html_bytes(response) or ""
        texts.append(text)
    print(f"Finished extracting {len(texts)} texts")
    return texts

# remove non-english
def remove_nonenglish(texts):
    results = []
    for text in tqdm(texts, desc="Processing Non-English Removal"):
        language_code, confidence_score = identify_language(text) if text else ("unknown", 0.0)
        if language_code == "en" and confidence_score > 0.8:
            results.append(text)
    print(f"Finish removing non-english with {(len(results))} results left")
    return results

# mask pii
def mask_pii(texts):
    results = []
    count_emails, count_phones, count_ips = 0, 0, 0
    for text in tqdm(texts, desc="Processing PII Masking"):
        masked_text_emails, num_emails = mask_emails(text)
        masked_text_phones, num_phones = mask_phone_numbers(masked_text_emails)
        masked_text_ips, num_ips = mask_ips(masked_text_phones)
        
        count_emails += num_emails
        count_phones += num_phones
        count_ips += num_ips
        results.append(masked_text_ips)
    print(f"Finish masking {len(results)} texts, number of masked emails: {count_emails}, phones: {count_phones}, ips: {count_ips}")
    return results

# remove nsfw and toxic
def remove_harmful(texts):
    results = []
    for text in tqdm(texts, desc="Processing Harmful Removal"):
        nsfw_label, nsfw_score = classify_nsfw(text)
        toxic_label, toxic_score = classify_toxic_speech(text)
        is_harmful = (nsfw_label == "nsfw" and nsfw_score > 0.5) or (toxic_label == "toxic" and toxic_score > 0.5)
        if not is_harmful:
            results.append(text)
    print(f"Finished removing harmful text with {len(results)} results left")
    return results

# filter high gopher quiality
def filter_gopher(texts):
    results = []
    for text in tqdm(texts, desc="Processing Gopher Filter"):
        result_bool = gopher_quality_filter(text)
        if result_bool is True:
            results.append(text)
    print(f"Finished filtering gopher text with {len(results)} results left")
    return results

def save_to_fasttext_wiki_format(texts, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            cleaned_text = text.replace("\n", " ").strip()
            if cleaned_text:
                f.write(f"__label__wiki {cleaned_text}\n")

def main():
    responses = extract_response(warc_file_path)
    texts = extract_text(responses)
    all_english_texts = remove_nonenglish(texts)
    pii_masked_texts = mask_pii(all_english_texts)
    no_harmful_texts = remove_harmful(pii_masked_texts)
    filtered_texts = filter_gopher(no_harmful_texts)

    save_to_fasttext_wiki_format(filtered_texts, output_path)

if __name__ == "__main__":
    main()