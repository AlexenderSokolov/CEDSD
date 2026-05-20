import hashlib
import os
import csv
import argparse
from pathlib import Path

from offline_cache import normalize_audio_path

def get_stem(audio_path):
    normalized = normalize_audio_path(audio_path)
    digest_hex = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]
    base_name = Path(normalized).stem
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base_name)
    return f"{safe_name}_{digest_hex}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the first CSV row whose normalized stem exists in cache.")
    parser.add_argument("--cache-dir", default="offline_cache_store/acoustic_cache")
    parser.add_argument("--csv", default="data/train/train.csv")
    args = parser.parse_args()

    cache_files = os.listdir(args.cache_dir)
    cache_stems = {f.replace(".pt", "") for f in cache_files if f.endswith(".pt")}

    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            file = row['file']
            stem = get_stem(file)
            if stem in cache_stems:
                print(f"MATCH FOUND! file: {file}, stem: {stem}")
                break
            if i < 3:
                print(f"No match for {file} (stem: {stem})")
        else:
            print("No matches found.")
