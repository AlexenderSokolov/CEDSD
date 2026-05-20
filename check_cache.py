import csv
import os
import random
import argparse
from pathlib import Path

from offline_cache import OfflineAcousticFeatureCache, normalize_audio_path

def check_split(name, csv_path, cache):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row['file'])
    
    hits = 0
    misses = []
    hit_paths = []
    
    # Try multiple path formats for each file
    for p in rows:
        found = False
        candidates = [p]
        # Try a normalized split-root candidate when the CSV stores relative names.
        prefix = os.environ.get("FAPI_DATA_ROOT", "data")
        if not p.startswith("/"):
            candidates.append(os.path.join(prefix, name, p))
            
        for cand in candidates:
            if cache.exists(cand):
                hits += 1
                hit_paths.append(cand)
                found = True
                break
        
        if not found:
            if len(misses) < 3:
                misses.append(p)
    
    hit_rate = hits / len(rows) if rows else 0
    
    # Structure check
    struct_pass = 0
    samples_to_check = random.sample(hit_paths, min(5, len(hit_paths)))
    expected_shapes = {
        'mfcc': (512, 120),
        'f0': (512, 1),
        'voiced_probs': (512, 1),
        'mask': (512, 1)
    }
    
    for p in samples_to_check:
        feat = cache.get(p)
        if feat is None:
            continue
        
        match = True
        for k, shape in expected_shapes.items():
            if k not in feat:
                match = False
                break
            val = feat[k]
            if hasattr(val, 'shape'):
                if tuple(val.shape) != shape:
                   match = False
                   break
            else:
                match = False
                break
        if match:
            struct_pass += 1
            
    return {
        'name': name,
        'total': len(rows),
        'hits': hits,
        'hit_rate': hit_rate,
        'misses': misses,
        'struct_pass': struct_pass,
        'checked': len(samples_to_check)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check offline acoustic cache coverage for dataset splits.")
    parser.add_argument("--cache-dir", default="offline_cache_store/acoustic_cache")
    parser.add_argument("--train-csv", default="data/train/train.csv")
    parser.add_argument("--val-csv", default="data/val/val.csv")
    parser.add_argument("--test-csv", default="data/test/test.csv")
    args = parser.parse_args()

    cache = OfflineAcousticFeatureCache(args.cache_dir, rank=0, world_size=1)
    splits = [
        ("train", args.train_csv),
        ("val", args.val_csv),
        ("test", args.test_csv),
    ]
    
    results = []
    total_hits = 0
    total_all = 0
    
    for name, path in splits:
        res = check_split(name, path, cache)
        results.append(res)
        total_hits += res['hits']
        total_all += res['total']
    
    overall_rate = total_hits / total_all if total_all else 0
    
    print(f"Overall Hit Rate: {overall_rate:.4f} ({total_hits}/{total_all})")
    for res in results:
        print(f"\nSplit: {res['name']}")
        print(f"  Hit Rate: {res['hit_rate']:.4f} ({res['hits']}/{res['total']})")
        print(f"  Structure Pass: {res['struct_pass']}/{res['checked']}")
        if res['misses']:
            print("  Missed Examples:")
            for m in res['misses']:
                print(f"    - {m}")
