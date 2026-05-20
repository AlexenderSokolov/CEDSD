'''
Using the finetuned emotion recognization model

rec_result contains {'feats', 'labels', 'scores'}
	extract_embedding=False: 9-class emotions with scores
	extract_embedding=True: 9-class emotions with scores, along with features

9-class emotions: 
iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large (May. 2024 release)
iic/emotion2vec_base_finetuned (Jan. 2024 release)
    0: angry
    1: disgusted
    2: fearful
    3: happy
    4: neutral
    5: other
    6: sad
    7: surprised
    8: unknown
'''

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

# Keep the repository root importable when this helper is run as a script.
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from Audio_united.Emotion2vec.e2v import Emotion2VecExtractor

def add_e2v_scores_to_csv(
    csv_path,
    wav_dir,
    out_csv_path=None,
    model_id="iic/emotion2vec_plus_large",
    device="cuda",
    batch_size=8,
):
    if out_csv_path is None:
        out_csv_path = csv_path.replace(".csv", "_with_e2v.csv")

    df = pd.read_csv(csv_path)
    if "file" not in df.columns:
        raise ValueError("csv must contain 'file' column")
    if "label" not in df.columns:
        raise ValueError("csv must contain 'label' column")

    # Nine emotion score columns returned by the finetuned emotion2vec model.
    emotion_cols = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]
    for c in emotion_cols:
        df[c] = 0.0

    extractor = Emotion2VecExtractor(model_id=model_id, device=device)

    file_list = df["file"].tolist()
    n = len(file_list)
    for start in tqdm(range(0, n, batch_size), desc="Emotion2Vec"):
        end = min(n, start + batch_size)
        batch_files = file_list[start:end]
        wav_paths = []

        # Resolve each CSV entry against the audio root unless it is already absolute.
        wav_paths = []
        for fn in batch_files:
            p = fn if os.path.isabs(fn) else os.path.join(wav_dir, fn)
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} not found")
            wav_paths.append(p)

        # Only scores are written to the CSV; feature tensors are discarded here.
        _, _, scores = extractor.extract(wav_paths, return_meta=False)
        scores = scores.cpu().numpy()

        for i, row_idx in enumerate(range(start, end)):
            for j in range(9):
                df.at[row_idx, emotion_cols[j]] = float(scores[i, j])

    df.to_csv(out_csv_path, index=False)
    print(f"Saved: {out_csv_path}")
    return out_csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append emotion2vec score columns to a dataset CSV.")
    parser.add_argument("--csv", default=os.environ.get("FAPI_TRAIN_CSV", os.path.join("data", "train", "train.csv")))
    parser.add_argument("--root", default=os.environ.get("FAPI_TRAIN_ROOT", os.path.join("data", "train")))
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--model-id", default=os.environ.get("FAPI_E2V_MODEL_ID", "iic/emotion2vec_plus_large"))
    parser.add_argument("--device", default=os.environ.get("FAPI_DEVICE", "cuda"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    add_e2v_scores_to_csv(
        args.csv,
        args.root,
        out_csv_path=args.out_csv,
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
    )
    # Legacy hard-coded local paths were replaced by CLI arguments above.
