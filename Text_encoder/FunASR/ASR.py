# Local SenseVoiceSmall ASR and optional CLAP consistency features.
from functools import partialmethod
from tqdm import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

import logging
import os
from pathlib import Path

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

_asr_model = None

def get_asr_model(device):
    global _asr_model
    device_str = str(device)
    default_model_dir = Path(__file__).resolve().parent / "local_models" / "SenseVoiceSmall"
    model_dir = Path(os.environ.get("FAPI_ASR_MODEL_DIR", str(default_model_dir))).expanduser()

    if _asr_model is None:
        print(f"--- Loading local ASR model: {model_dir} ---")
        if not model_dir.exists():
            raise FileNotFoundError(
                f"ASR model directory not found: {model_dir}. "
                "Set FAPI_ASR_MODEL_DIR or run sense_download.py first."
            )
            
        _asr_model = AutoModel(
            model=str(model_dir),
            device=device_str,
            vad_model=None,
            disable_update=True,
            disable_pbar=True,
            log_level="ERROR"
        )
    return _asr_model

def asr_infer(audio_path, device='cuda'):
    model = get_asr_model(device)

    if isinstance(audio_path, str):
        if "," in audio_path:
            path_list = [p.strip() for p in audio_path.split(",") if p.strip()]
        else:
            path_list = [audio_path]
    elif isinstance(audio_path, (list, tuple)):
        path_list = [str(p) for p in audio_path]
    else:
        path_list = [audio_path]

    final_texts = []
    for p in path_list:
        try:
            res = model.generate(
                input=p,
                cache={},
                language="zh",
                use_itn=True,
                batch_size_s=300,
                merge_vad=True,
                merge_length_s=15,
                disable_pbar=True
            )
            if res and len(res) > 0:
                raw_text = res[0].get("text", "")
                processed_text = rich_transcription_postprocess(raw_text).strip()
                final_texts.append(processed_text)
            else:
                final_texts.append("")
        except Exception as e:
            final_texts.append("")
    return final_texts


# Optional CLAP feature extraction.
import torch
try:
    import laion_clap
except Exception:
    laion_clap = None
import numpy as np
import random
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

logging.getLogger("funasr").setLevel(logging.ERROR)

clap_model = None

def get_clap_model():
    global clap_model
    if laion_clap is None:
        raise ImportError(
            "laion_clap is not installed, so CLAP features are unavailable. "
            "Install it with: pip install laion-clap"
        )
    if clap_model is None:
        clap_model = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel="HTSAT-base"
        )
        default_ckpt_path = Path(__file__).resolve().parent / "local_models" / "clap" / "music_speech_audioset_epoch_15_esc_89.98.pt"
        ckpt_path = Path(os.environ.get("FAPI_CLAP_CKPT", str(default_ckpt_path))).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"CLAP checkpoint not found: {ckpt_path}. Set FAPI_CLAP_CKPT to the checkpoint path."
            )
        clap_model.load_ckpt(ckpt_path)
        if torch.cuda.is_available():
            clap_model = clap_model.to("cuda")
    return clap_model

def load_data_list(txt_path):
    paths, labels = [], []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            paths.append(parts[0])
            labels.append(int(parts[1]))
    return paths, labels

def extract_features(audio_paths):
    features = []
    clap_model = get_clap_model()
    for path in audio_paths:
        try:
            text_list = asr_infer(path)
            text = text_list[0] if text_list else ""
            if not text:
                text = "[EMPTY]"
            audio_emb = clap_model.get_audio_embedding_from_filelist([path])
            text_emb = clap_model.get_text_embedding([text])
            score = torch.cosine_similarity(torch.tensor(audio_emb), torch.tensor(text_emb)).item()
            features.append([score])
        except:
            features.append([0.0])
    return np.array(features)

if __name__ == "__main__":
    base_dir = os.environ.get("FAPI_CLAP_DATA_DIR", os.path.join("data", "clap"))
    train_txt = os.path.join(base_dir, "train.txt")
    test_txt = os.path.join(base_dir, "test.txt")

    train_paths, y_train = load_data_list(train_txt)
    test_paths, y_test = load_data_list(test_txt)

    X_train = extract_features(train_paths)
    X_test = extract_features(test_paths)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=3000,
        early_stopping=True,
        random_state=42
    )
    mlp.fit(X_train, y_train)

    print(f"\nTest accuracy: {mlp.score(X_test, y_test):.4f}")
