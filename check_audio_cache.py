import os
import pandas as pd
import glob
import argparse

from offline_cache import OfflineAcousticFeatureCache

def main():
    parser = argparse.ArgumentParser(description="Inspect the expected acoustic cache file for the first CSV row.")
    parser.add_argument("--csv", default="data/train/train.csv")
    parser.add_argument("--root", default="data/train")
    parser.add_argument("--cache-dir", default="offline_cache_store/acoustic_cache")
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    if df.empty:
        print("CSV is empty")
        return
    
    first_file = df.iloc[0]['file']
    full_audio_path = os.path.join(args.root, first_file)
    print(f"Full audio path: {full_audio_path}")

    # 2. Use OfflineAcousticFeatureCache._file_stem to get expected cache filename
    cache = OfflineAcousticFeatureCache(args.cache_dir)
    expected_stem = cache._file_stem(full_audio_path)
    expected_cache_path = os.path.join(args.cache_dir, f"{expected_stem}.pt")
    
    print(f"Expected stem: {expected_stem}")
    print(f"Expected cache path: {expected_cache_path}")
    print(f"Exists: {os.path.exists(expected_cache_path)}")

    # 3. Glob for files with the same basename prefix in the cache directory
    audio_basename = os.path.splitext(os.path.basename(full_audio_path))[0]
    print(f"Audio basename for glob: {audio_basename}")
    
    glob_pattern = os.path.join(args.cache_dir, f"{audio_basename}*.pt")
    found_files = glob.glob(glob_pattern)
    
    print(f"Found {len(found_files)} files matching pattern {glob_pattern}:")
    for f in found_files[:5]:
        print(f" - {f}")

if __name__ == "__main__":
    main()
