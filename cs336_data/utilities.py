import re
import fasttext
from typing import Any
from resiliparse.parse.encoding import detect_encoding, bytes_to_str
from resiliparse.extract.html2text import extract_plain_text

def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    decoded = bytes_to_str(html_bytes, detect_encoding(html_bytes))
    return extract_plain_text(decoded)

def identify_language(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("lid.176.bin")
    # Remove newlines by replacing them with spaces
    cleaned_text = text.replace('\n', ' ').strip()
    # Predict language, take first label and score
    predictions, scores = model.predict(cleaned_text)
    # Strip '__label__' from the predicted label and return it with the score
    predicted_language = predictions[0].replace('__label__', '')
    return predicted_language, scores[0]

def mask_emails(text: str) -> tuple[str, int]:
    # Regular expression pattern for matching email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    # Find all email addresses in the text
    emails = re.findall(email_pattern, text)
    # Replace each email address with the placeholder
    masked_text = re.sub(email_pattern, '|||EMAIL_ADDRESS|||', text)
    # Return the masked text and the count of masked emails
    return masked_text, len(emails)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    # Updated regular expression pattern for matching phone numbers
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    # Find all phone numbers in the text
    phone_numbers = re.findall(phone_pattern, text)
    # Replace each phone number with the placeholder
    masked_text = re.sub(phone_pattern, '|||PHONE_NUMBER|||', text)
    # Return the masked text and the count of masked phone numbers
    return masked_text, len(phone_numbers)

def mask_ips(text: str) -> tuple[str, int]:
    # Regular expression pattern for matching IPv4 addresses
    ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    # Find all IP addresses in the text
    ips = re.findall(ip_pattern, text)
    # Replace each IP address with the placeholder
    masked_text = re.sub(ip_pattern, '|||IP_ADDRESS|||', text)
    # Return the masked text and the count of masked IP addresses
    return masked_text, len(ips)

def classify_nsfw(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("jigsaw_fasttext_bigrams_nsfw_final.bin")
    cleaned_text = text.replace('\n', ' ').strip()
    predictions, scores = model.predict(cleaned_text)
    predicted_language = predictions[0].replace('__label__', '')
    return predicted_language, scores[0]

def classify_toxic_speech(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("jigsaw_fasttext_bigrams_hatespeech_final.bin")
    cleaned_text = text.replace('\n', ' ').strip()
    predictions, scores = model.predict(cleaned_text)
    predicted_language = predictions[0].replace('__label__', '')
    return predicted_language, scores[0]

def gopher_quality_filter(text: str) -> bool:
    words = text.split()
    # Check if the number of words is less than 50 or more than 100,000
    if len(words) < 50 or len(words) > 100000:
        return False

    # Check if the mean word length is outside the range of 3 to 10 characters
    mean_word_length = sum(map(len, words)) / len(words)
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # Check if more than 30% of lines end with an ellipsis
    lines = text.split('\n')
    ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith("..."))
    if ellipsis_lines / len(lines) > 0.3:
        return False

    # Check if less than 80% of words have at least one alphabetic character
    alphabetic_words = sum(1 for word in words if re.search('[a-zA-Z]', word))
    if alphabetic_words / len(words) < 0.8:
        return False

    return True

def classify_quality(text: str) -> tuple[Any, float]:
    return "cc", 69.