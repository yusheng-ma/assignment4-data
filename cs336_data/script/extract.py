import re
import gzip
import pandas as pd
import random
from fastwarc.warc import ArchiveIterator

from cs336_data.utilities import extract_text_from_html_bytes, identify_language
from cs336_data.utilities import mask_emails, mask_phone_numbers, mask_ips
from cs336_data.utilities import classify_nsfw, classify_toxic_speech
from cs336_data.utilities import gopher_quality_filter

warc_file_path = "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

def extract_warc(warc_file_path: str) -> list:
    records = []
    with gzip.open(warc_file_path, 'rb') as warc_file:
        iterator = ArchiveIterator(warc_file)
        count = 0
        for record in iterator:
            if record.headers.get('WARC-Type') == 'response':
                headers = {header: value for header, value in record.headers}
                content = record.reader.read()
                text = extract_text_from_html_bytes(content) or ""

                # Identify language
                language_code, confidence_score = identify_language(text) if text else ("unknown", 0.0)

                # Mask PII
                masked_text_emails, num_emails = mask_emails(text)
                masked_text_phones, num_phones = mask_phone_numbers(masked_text_emails)
                masked_text_ips, num_ips = mask_ips(masked_text_phones)

                # Store record data
                record_data = headers.copy()
                record_data['original_text'] = text
                record_data['masked_text'] = masked_text_ips
                record_data['language_code'] = language_code
                record_data['confidence_score'] = confidence_score
                record_data['num_emails'] = num_emails
                record_data['num_phones'] = num_phones
                record_data['num_ips'] = num_ips

                records.append(record_data)

                count += 1
                if count >= 200:  # Limit to 200 records
                    break

    return records

# Run extraction
extracted_records = extract_warc(warc_file_path)

# Convert to DataFrame
df = pd.DataFrame(extracted_records)

# Function to find false positives and false negatives
def evaluate_pii_masking(df):
    false_positives = []
    false_negatives = []

    for index, row in df.iterrows():
        original_text = row['original_text']
        masked_text = row['masked_text']

        # Patterns for PII detection
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'

        # Check for false positives
        if any(mask in masked_text for mask in ["|||EMAIL_ADDRESS|||", "|||PHONE_NUMBER|||", "|||IP_ADDRESS|||"]):
            email_found = "@" in original_text
            phone_found = re.search(phone_pattern, original_text) is not None
            ip_found = re.search(ip_pattern, original_text) is not None

            if not (email_found or phone_found or ip_found):
                # Find the first replacement position
                first_replacement_pos = masked_text.find("|||EMAIL_ADDRESS|||")
                if first_replacement_pos == -1:
                    first_replacement_pos = masked_text.find("|||PHONE_NUMBER|||")
                if first_replacement_pos == -1:
                    first_replacement_pos = masked_text.find("|||IP_ADDRESS|||")

                # Extract text around the first replacement with more context
                start_pos = max(0, first_replacement_pos - 50)
                end_pos = min(len(masked_text), first_replacement_pos + 50)
                snippet = masked_text[start_pos:end_pos]

                false_positives.append((index, snippet))

        # Check for false negatives
        email_found = "@" in original_text
        phone_found = re.search(phone_pattern, original_text) is not None
        ip_found = re.search(ip_pattern, original_text) is not None

        if email_found or phone_found or ip_found:
            if not any(mask in masked_text for mask in ["|||EMAIL_ADDRESS|||", "|||PHONE_NUMBER|||", "|||IP_ADDRESS|||"]):
                # Find the first PII position
                first_pii_pos = -1
                if email_found:
                    first_pii_pos = original_text.find("@")
                elif phone_found:
                    first_pii_pos = original_text.find(re.search(phone_pattern, original_text).group())
                elif ip_found:
                    first_pii_pos = original_text.find(re.search(ip_pattern, original_text).group())

                # Extract text around the first PII with more context
                start_pos = max(0, first_pii_pos - 50)
                end_pos = min(len(original_text), first_pii_pos + 50)
                snippet = original_text[start_pos:end_pos]

                false_negatives.append((index, snippet))

    return false_positives, false_negatives

false_positives, false_negatives = evaluate_pii_masking(df)

# Randomly sample false positives and false negatives
sample_size = 5
sampled_false_positives = random.sample(false_positives, min(sample_size, len(false_positives)))
sampled_false_negatives = random.sample(false_negatives, min(sample_size, len(false_negatives)))

# Print examples of sampled false positives and false negatives
print("====================================Randomly Sampled False Positives:====================================")
for fp in sampled_false_positives:
    print(f"Index: {fp[0]}")
    print(f"Snippet: {fp[1]}")
    print()

print("====================================Randomly Sampled False Negatives:====================================")
for fn in sampled_false_negatives:
    print(f"Index: {fn[0]}")
    print(f"Snippet: {fn[1]}")
    print()

# Function to classify and evaluate harmful content
def evaluate_harmful_content(df):
    harmful_docs = 0
    sample_size = 20
    sampled_indices = random.sample(range(len(df)), min(sample_size, len(df)))

    print("====================================Harmful====================================")
    for index in sampled_indices:
        text = df.loc[index, 'original_text']

        # Classify the text
        nsfw_label, nsfw_score = classify_nsfw(text)
        toxic_label, toxic_score = classify_toxic_speech(text)

        # Determine if the document is harmful based on classifier labels and scores
        is_harmful = (nsfw_label == "nsfw" and nsfw_score > 0.5) or (toxic_label == "toxic" and toxic_score > 0.5)

        if is_harmful:
            harmful_docs += 1

            # Print results for manual evaluation
            print(f"Index: {index}")
            print(f"NSFW Label: {nsfw_label}, Score: {nsfw_score}")
            print(f"Toxic Label: {toxic_label}, Score: {toxic_score}")
            print(f"Text Snippet: {text[:10]}...")  # Print a snippet for brevity
            print("---")

    # Calculate the fraction of harmful documents
    fraction_harmful = harmful_docs / sample_size

    return fraction_harmful

# Evaluate harmful content
fraction_harmful = evaluate_harmful_content(df)

# Print the fraction of harmful documents
print(f"Fraction of harmful documents: {fraction_harmful:.2f}")

# Sample 20 random indices
sample_size = 20
sampled_indices = random.sample(range(len(df)), min(sample_size, len(df)))

# Store results for comparison
results = []

print("==================================== Gopher Quality Filter Evaluation ====================================")
for idx in sampled_indices:
    text = df.loc[idx, 'original_text']
    filter_result = gopher_quality_filter(text)

    # Store result for manual evaluation
    results.append({
        'index': idx,
        'filter_result': filter_result,
        'text_snippet': text[:100] + "..." if len(text) > 100 else text  # Truncate for display
    })

    # Print filter result and snippet
    print(f"Index: {idx}")
    print(f"Filter Result: {'Pass' if filter_result else 'Fail'}")
    print(f"Text Snippet: {results[-1]['text_snippet']}")
    print("---")