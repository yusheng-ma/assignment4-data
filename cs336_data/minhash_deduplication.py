import os
from pathlib import Path
from typing import List, Set, Dict, Tuple
from collections import defaultdict
from datasketch import MinHash, MinHashLSH
from unidecode import unidecode
import re


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, normalizing whitespace, and unidecode."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = unidecode(text)
    return text


def get_word_ngrams(text: str, n: int) -> Set[str]:
    """Extract word-level n-grams of length n from normalized text."""
    words = text.split()
    return set(
        " ".join(words[i:i + n])
        for i in range(len(words) - n + 1)
    )


def minhash_deduplication(
    input_files: List[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    assert num_hashes % num_bands == 0, "num_hashes must be divisible by num_bands"
    num_rows = num_hashes // num_bands

    # Step 1: Load all documents and normalize
    doc_id_to_path = {}
    original_texts = {}

    for idx, input_file in enumerate(input_files):
        input_path = Path(input_file)
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
        normalized_text = normalize_text(raw_text)
        if not normalized_text.strip():
            continue  # skip empty docs

        doc_id_to_path[idx] = input_path
        original_texts[idx] = raw_text

    # Step 2: Generate MinHash signatures
    signatures = {}
    for doc_id, raw_text in original_texts.items():
        normalized_text = normalize_text(raw_text)
        ngram_set = get_word_ngrams(normalized_text, ngrams)

        m = MinHash(num_perm=num_hashes)
        for ngram in ngram_set:
            m.update(ngram.encode('utf-8'))
        signatures[doc_id] = m

    # Step 3: Build LSH index and find candidate pairs
    lsh = MinHashLSH(
        threshold=0.0,
        num_perm=num_hashes,
        params=(num_bands, num_rows)
    )
    for doc_id, sig in signatures.items():
        lsh.insert(f"doc_{doc_id}", sig)

    candidate_pairs = set()
    for doc_id, sig in signatures.items():
        results = lsh.query(sig)
        for res in results:
            other_id = int(res.replace("doc_", ""))
            if doc_id < other_id:
                candidate_pairs.add((doc_id, other_id))

    # Step 4: Compute actual Jaccard and filter duplicates
    duplicate_pairs = []
    for a, b in candidate_pairs:
        text_a = normalize_text(original_texts[a])
        text_b = normalize_text(original_texts[b])
        set_a = get_word_ngrams(text_a, ngrams)
        set_b = get_word_ngrams(text_b, ngrams)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard = intersection / union if union else 0.0
        if jaccard >= jaccard_threshold:
            duplicate_pairs.append((a, b))

    # Step 5: Cluster using Union-Find
    parent = {}

    def find(x):
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for a, b in duplicate_pairs:
        union(a, b)

    clusters = defaultdict(list)
    for doc_id in doc_id_to_path:
        root = find(doc_id)
        clusters[root].append(doc_id)

    keep_ids = {min(group) for group in clusters.values()}

    # Step 6: Write output files
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    for doc_id, input_file in zip(doc_id_to_path.keys(), input_files):
        input_path = Path(input_file)
        output_path = output_dir / input_path.name
        if doc_id in keep_ids:
            with open(input_file, 'r', encoding='utf-8') as fin, \
                 open(output_path, 'w', encoding='utf-8') as fout:
                fout.write(fin.read())
