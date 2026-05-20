import os
import hashlib
import torchcrepe
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import librosa

from offline_cache import OfflineAcousticFeatureCache, OfflineE2VFeatureCache, normalize_audio_path


def _sanitize_array(arr, name, fallback_shape=None, fallback_dtype=np.float32):
    """Convert an array to finite values, falling back to zeros when needed."""
    if arr is None:
        if fallback_shape is None:
            raise ValueError(f"{name} is empty and no fallback shape was provided")
        return np.zeros(fallback_shape, dtype=fallback_dtype)

    arr_np = np.asarray(arr)
    if not np.isfinite(arr_np).all():
        print(f"Detected non-finite values and sanitized: {name}")
    arr_np = np.nan_to_num(arr_np, nan=0.0, posinf=0.0, neginf=0.0)

    if fallback_shape is not None and arr_np.shape != fallback_shape:
        print(f"Unexpected shape, using fallback: {name} | got={arr_np.shape} | expect={fallback_shape}")
        return np.zeros(fallback_shape, dtype=fallback_dtype)

    return arr_np.astype(fallback_dtype, copy=False)

# ===================== 1. feature extraction helpers =====================
def extract_mfcc_with_deltas(waveform, sample_rate=16000, n_mfcc=40, max_len=300):
    """Extract MFCC, delta, and delta-delta features."""
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)
    # librosa.feature.delta defaults to width=9, which fails on very short clips.
    # Bound the width by the sequence length and keep it odd.
    n_frames = mfcc.shape[1]
    delta_width = min(9, n_frames)
    if delta_width % 2 == 0:
        delta_width = max(1, delta_width - 1)
    mfcc_delta = librosa.feature.delta(mfcc, width=delta_width)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=delta_width)
    mfcc_combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0).T
    
    # Normalize and align length.
    mfcc_combined = (mfcc_combined - np.mean(mfcc_combined)) / (np.std(mfcc_combined) + 1e-8)
    if len(mfcc_combined) > max_len:
        mfcc_combined = mfcc_combined[:max_len]
    else:
        mfcc_combined = np.pad(mfcc_combined, ((0, max_len - len(mfcc_combined)), (0, 0)), mode="constant")
    
    return _sanitize_array(mfcc_combined, name="mfcc_combined", fallback_shape=(max_len, n_mfcc * 3))

def extract_crepe_features(waveform, sample_rate=16000, max_len=512):
    """
    Extract F0 and voicing probability with torchcrepe instead of librosa.pyin.
    """
    # Convert to Tensor and add batch dimension [1, T].
    audio_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    
    # Extract F0 and periodicity (equivalent to voiced probability).
    # hop_length=160 at 16 kHz is 10 ms, matching standard acoustic features.
    f0_tensor, pd_tensor = torchcrepe.predict(
        audio_tensor,
        sample_rate,
        hop_length=160,
        fmin=65.41,   # Approximately C2.
        fmax=2093.00, # Approximately C7.
        model='full',
        return_periodicity=True,
        batch_size=1024,
        device='cpu'  # Dataset workers must use CPU to avoid CUDA crashes in multiprocessing.
    )
    
    # Convert back to numpy and remove the batch dimension.
    f0 = f0_tensor.squeeze(0).numpy()
    voiced_probs = pd_tensor.squeeze(0).numpy()
    
    # Process F0.
    f0 = np.nan_to_num(f0)
    # 1e-8 avoids division by zero.
    f0 = (f0 - np.mean(f0)) / (np.std(f0) + 1e-8)
    if len(f0) > max_len:
        f0 = f0[:max_len]
    else:
        f0 = np.pad(f0, (0, max_len - len(f0)), mode="constant")
    f0 = f0.reshape(-1, 1).astype(np.float32)
    
    # Process voiced probability.
    voiced_probs = np.nan_to_num(voiced_probs)
    voiced_probs = (voiced_probs - np.mean(voiced_probs)) / (np.std(voiced_probs) + 1e-8)
    if len(voiced_probs) > max_len:
        voiced_probs = voiced_probs[:max_len]
    else:
        voiced_probs = np.pad(voiced_probs, (0, max_len - len(voiced_probs)), mode="constant")
    voiced_probs = voiced_probs.reshape(-1, 1).astype(np.float32)
    
    # Build the audio mask. The F0-magnitude heuristic is retained for downstream compatibility.
    mask = np.where(np.abs(f0) > 1e-6, 1.0, 0.0).astype(np.float32)
    
    f0 = _sanitize_array(f0, name="f0", fallback_shape=(max_len, 1))
    voiced_probs = _sanitize_array(voiced_probs, name="voiced_probs", fallback_shape=(max_len, 1))
    mask = _sanitize_array(mask, name="mask", fallback_shape=(max_len, 1))
    return f0, voiced_probs, mask

# ===================== 2. unified multimodal dataset =====================
class UnifiedMultimodalDataset(Dataset):
    """
    Unified multimodal dataset.
    Each __getitem__ returns the materials needed by all branches:
    1. Raw waveform tensor for waveform-level models such as emotion2vec.
    2. Acoustic feature tensors such as MFCC and F0 for CNN/Transformer encoders.
    3. Absolute audio path for ASR-based text extraction.
    """
    def __init__(
        self,
        df,
        root_dir,
        sr=16000,
        raw_duration=5.0,
        acoustic_max_len=512,
        n_mfcc=40,
        cache_dir=None,
        e2v_cache_dir=None,
        e2v_cache_enable=False,
        acoustic_cache_dir=None,
        acoustic_cache_enable=False,
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.sr = sr
        
        # Length settings for waveform and acoustic branches.
        # sample_rate * duration gives sample count, e.g. 16000 * 5.0 = 80000.
        self.raw_target_samples = int(sr * raw_duration)
        self.acoustic_max_len = acoustic_max_len
        self.n_mfcc = n_mfcc
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.e2v_cache_enable = bool(e2v_cache_enable)
        self.e2v_cache = None
        if self.e2v_cache_enable:
            if not e2v_cache_dir:
                raise ValueError("e2v_cache_dir is required when e2v_cache_enable=True")
            self.e2v_cache = OfflineE2VFeatureCache(str(e2v_cache_dir), rank=0, world_size=1)

        self.acoustic_cache_enable = bool(acoustic_cache_enable)
        self.acoustic_cache = None
        if self.acoustic_cache_enable:
            if not acoustic_cache_dir:
                raise ValueError("acoustic_cache_dir is required when acoustic_cache_enable=True")
            self.acoustic_cache = OfflineAcousticFeatureCache(str(acoustic_cache_dir), rank=0, world_size=1)

    def _build_cache_path(self, audio_path):
        cache_key = f"{audio_path}|sr={self.sr}|raw={self.raw_target_samples}|amax={self.acoustic_max_len}|mfcc={self.n_mfcc}"
        cache_name = hashlib.md5(cache_key.encode("utf-8")).hexdigest() + ".npz"
        return os.path.join(self.cache_dir, cache_name)

    def _load_e2v_cache(self, audio_path):
        if not self.e2v_cache_enable:
            return None

        normalized_audio_path = normalize_audio_path(audio_path)
        payload = self.e2v_cache.get(normalized_audio_path)
        if payload is None:
            return None

        required_keys = ("e2v_feats", "e2v_mask", "e2v_scores")
        for key in required_keys:
            if key not in payload:
                return None

        return {
            "e2v_feats": payload["e2v_feats"],
            "e2v_mask": payload["e2v_mask"],
            "e2v_scores": payload["e2v_scores"],
        }

    def _load_acoustic_cache(self, audio_path):
        if not self.acoustic_cache_enable:
            return None

        normalized_audio_path = normalize_audio_path(audio_path)
        payload = self.acoustic_cache.get(normalized_audio_path)
        if payload is None:
            return None

        required_keys = ("mfcc", "f0", "voiced_probs", "mask")
        for key in required_keys:
            if key not in payload:
                return None

        expected_mfcc_shape = (self.acoustic_max_len, self.n_mfcc * 3)
        expected_seq_shape = (self.acoustic_max_len, 1)

        mfcc = np.asarray(payload["mfcc"])
        f0 = np.asarray(payload["f0"])
        voiced_probs = np.asarray(payload["voiced_probs"])
        mask = np.asarray(payload["mask"])

        if mfcc.shape != expected_mfcc_shape:
            return None
        if f0.shape != expected_seq_shape:
            return None
        if voiced_probs.shape != expected_seq_shape:
            return None
        if mask.shape != expected_seq_shape:
            return None

        return {
            "mfcc": mfcc,
            "f0": f0,
            "voiced_probs": voiced_probs,
            "mask": mask,
        }

    def _save_acoustic_cache(self, audio_path, mfcc, f0, voiced_probs, mask):
        if not self.acoustic_cache_enable:
            return
        payload = {
            "mfcc": np.asarray(mfcc, dtype=np.float32),
            "f0": np.asarray(f0, dtype=np.float32),
            "voiced_probs": np.asarray(voiced_probs, dtype=np.float32),
            "mask": np.asarray(mask, dtype=np.float32),
        }
        try:
            self.acoustic_cache.put(normalize_audio_path(audio_path), payload)
        except Exception as e:
            print(f"Offline acoustic cache write failed and was ignored: {audio_path} | {e}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Resolve the audio path used by both waveform loading and ASR.
        audio_path = os.path.join(self.root_dir, str(row["file"]))
        # Expected emotion score columns in the CSV.
        emo_cols = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]
        f_emo_vector = row[emo_cols].values.astype(np.float32)
        
        raw_waveform_np = None
        mfcc = None
        f0 = None
        voiced_probs = None
        mask = None

        expected_raw_shape = (self.raw_target_samples,)
        expected_mfcc_shape = (self.acoustic_max_len, self.n_mfcc * 3)
        expected_seq_shape = (self.acoustic_max_len, 1)

        cache_path = self._build_cache_path(audio_path) if self.cache_dir else None
        if cache_path and os.path.exists(cache_path):
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    raw_waveform_np = cached["raw_waveform"]
                    mfcc = cached["mfcc"]
                    f0 = cached["f0"]
                    voiced_probs = cached["voiced_probs"]
                    mask = cached["mask"]
            except Exception as e:
                print(f"Feature cache read failed; rebuilding in place: {cache_path} | {e}")
                raw_waveform_np = None
                mfcc = None
                f0 = None
                voiced_probs = None
                mask = None

        if raw_waveform_np is None or mfcc is None or f0 is None or voiced_probs is None or mask is None:
            # Core optimization: read the waveform from disk only once.
            try:
                # librosa returns a float32-like numpy array and resamples to sr.
                # duration=5.0 protects memory when audio is unexpectedly long.
                waveform_np, _ = librosa.load(audio_path, sr=self.sr, mono=True, duration=5.0)
            except Exception as e:
                print(f"Audio read failed {audio_path}: {e}")
                # On read failure, return zeros; downstream masks can suppress the sample.
                waveform_np = np.zeros((int(self.sr * self.raw_target_samples / self.sr),), dtype=np.float32)

            # -------------------- empty-audio handling start --------------------
            if waveform_np is None or len(waveform_np) == 0:
                print(f"Empty audio detected; using zero waveform: {audio_path}")
                waveform_np = np.zeros((self.raw_target_samples,), dtype=np.float32)

            waveform_np = _sanitize_array(
                waveform_np,
                name=f"waveform_np({audio_path})",
                fallback_shape=None,
            )
            if waveform_np.ndim != 1:
                waveform_np = waveform_np.reshape(-1)
            # -------------------- empty-audio handling end --------------------

            # Branch 1: raw waveform.
            # Add a channel dimension [1, T] before padding/cropping.
            waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0)

            if waveform_tensor.size(1) > self.raw_target_samples:
                waveform_tensor = waveform_tensor[:, :self.raw_target_samples] # Crop.
            else:
                pad_amount = self.raw_target_samples - waveform_tensor.size(1)
                waveform_tensor = F.pad(waveform_tensor, (0, pad_amount)) # Pad.

            raw_waveform_np = waveform_tensor.squeeze(0).numpy().astype(np.float32) # Back to [T].

            # Branch 2: acoustic features.
            if mfcc is None or f0 is None or voiced_probs is None or mask is None:
                acoustic_cache_pack = self._load_acoustic_cache(audio_path)
                if acoustic_cache_pack is not None:
                    mfcc = acoustic_cache_pack["mfcc"]
                    f0 = acoustic_cache_pack["f0"]
                    voiced_probs = acoustic_cache_pack["voiced_probs"]
                    mask = acoustic_cache_pack["mask"]
                else:
                    mfcc = extract_mfcc_with_deltas(waveform_np, self.sr, self.n_mfcc, self.acoustic_max_len)
                    f0, voiced_probs, mask = extract_crepe_features(waveform_np, self.sr, self.acoustic_max_len)
                    # librosa/torchcrepe extraction is CPU-only here and has no trainable state.
                    self._save_acoustic_cache(audio_path, mfcc, f0, voiced_probs, mask)

            if cache_path:
                try:
                    np.savez_compressed(
                        cache_path,
                        raw_waveform=raw_waveform_np,
                        mfcc=mfcc,
                        f0=f0,
                        voiced_probs=voiced_probs,
                        mask=mask,
                    )
                except Exception as e:
                    print(f"Feature cache write failed and was ignored: {cache_path} | {e}")

        raw_waveform_np = _sanitize_array(raw_waveform_np, name=f"raw_waveform({audio_path})", fallback_shape=expected_raw_shape)
        mfcc = _sanitize_array(mfcc, name=f"mfcc({audio_path})", fallback_shape=expected_mfcc_shape)
        f0 = _sanitize_array(f0, name=f"f0({audio_path})", fallback_shape=expected_seq_shape)
        voiced_probs = _sanitize_array(voiced_probs, name=f"voiced_probs({audio_path})", fallback_shape=expected_seq_shape)
        mask = _sanitize_array(mask, name=f"mask({audio_path})", fallback_shape=expected_seq_shape)

        # To keep DataLoader collation stable when only some samples have E2V cache,
        # the dataset returns placeholders; the model performs batch-level cache-first loading.
        e2v_cache_pack = None
        if e2v_cache_pack is None:
            e2v_feats = torch.empty(0, dtype=torch.float32)
            e2v_mask = torch.empty(0, dtype=torch.bool)
            e2v_scores = torch.empty(0, dtype=torch.float32)
        else:
            e2v_feats = e2v_cache_pack["e2v_feats"]
            e2v_mask = e2v_cache_pack["e2v_mask"]
            e2v_scores = e2v_cache_pack["e2v_scores"]

        raw_waveform = torch.from_numpy(raw_waveform_np.astype(np.float32))

        # Map string labels to numeric targets: bonafide/real -> 0.0, spoof/fake -> 1.0.
        label_map = {
        "bonafide": 0.0, "spoof": 1.0, 
        "real": 0.0, "fake": 1.0,
        "0": 0.0, "1": 1.0, 
        "0.0": 0.0, "1.0": 1.0
        }
        label_value = label_map.get(str(row["label"]).lower(), 0.0)  # Default to 0.0 for unknown labels.

        # Package the multimodal sample.
        return {
            # --- Branch 1: raw audio ---
            "raw_waveform": raw_waveform,                     # [T_samples] Tensor
            
            # --- Branch 2: acoustic features ---
            "mfcc": torch.from_numpy(mfcc),                   # [max_len, 120] Tensor
            "f0": torch.from_numpy(f0),                       # [max_len, 1] Tensor
            "voiced_probs": torch.from_numpy(voiced_probs),   # [max_len, 1] Tensor
            "mask": torch.from_numpy(mask),                   # [max_len, 1] Tensor
            
            # --- Branch 3: text / ASR path ---
            "audio_path": audio_path,                         # PyTorch collates strings into List[str].

            # --- Branch 4: Offline Emotion2Vec ---
            "e2v_feats": e2v_feats,
            "e2v_mask": e2v_mask,
            "e2v_scores": e2v_scores,
            
            # --- General Info ---
            "labels": torch.tensor(label_value, dtype=torch.float32),
            "F_emo": torch.tensor(f_emo_vector, dtype=torch.float32)
        }
