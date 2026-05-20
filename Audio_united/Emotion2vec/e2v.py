from funasr import AutoModel
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class Emotion2VecExtractor:
    def __init__(self, model_id="iic/emotion2vec_plus_large", device="cuda", frame_rate=50.0):
        self.model = AutoModel(
            model=model_id,
            hub="hf",# Currently uses Hugging Face Hub models; local loading can be added later.
            device=str(device),
            disable_update=True
        )
        # Freeze/mode guard: the project uses emotion2vec only as an embedding extractor.
        self.embedding_only = True
        self._warned_embedding_only = False
        # Force eval mode to avoid Dropout/BN randomness.
        if hasattr(self.model, "model") and hasattr(self.model.model, "eval"):
            self.model.model.eval()
        else:
            eval_fn = getattr(self.model, "eval", None)
            if callable(eval_fn):
                eval_fn()
        # emotion2vec frame outputs are usually about 50 Hz, used to map frames to time.
        self.frame_rate = float(frame_rate)



    def extract(self, wav_list, target_num_frames=None, pad_value=0.0, return_meta=False):
        """Extract emotion2vec frame features and alignment masks.

        Args:
            wav_list: Audio paths or waveform objects supported by the model.
            target_num_frames: Optional fixed frame count for direct fusion with other branches.
            pad_value: Padding value for variable-length batches.
            return_meta: If True, also return lengths and frame_time.

        Returns:
            feats: [B, T, 768] batched features.
            mask: [B, T], True for valid frames and False for padding.
            scores: [B, 9] emotion-class scores.
            When return_meta=False (default):
                feats, mask, scores
            When return_meta=True:
                feats, mask, scores, lengths, frame_time
        """
        # Embedding-only mode with explicit output validation.
        if self.embedding_only and (not self._warned_embedding_only):
            print("[INFO] Emotion2VecExtractor is running in embedding-only mode; decoder missing-key warnings are acceptable.")
            self._warned_embedding_only = True

        try:
            rec_result = self.model.generate(
                wav_list,
                granularity="frame",
                extract_embedding=True
            )
        except Exception as e:
            raise RuntimeError(f"Emotion2Vec generate failed: {e}") from e

        if rec_result is None or len(rec_result) == 0:
            raise RuntimeError("Emotion2Vec returned an empty result")

        # Collect batched results.
        all_scores = []
        all_feats = []
        lengths = []

        for idx_res, res in enumerate(rec_result):
            if "feats" not in res or "scores" not in res:
                raise KeyError(f"Emotion2Vec result is missing feats/scores, sample_index={idx_res}")
            feat = torch.as_tensor(res["feats"], dtype=torch.float32)   # [Ti, 768]
            if feat.ndim != 2 or feat.shape[0] == 0:
                raise ValueError(f"Invalid Emotion2Vec feats shape, sample_index={idx_res}, shape={tuple(feat.shape)}")
            all_feats.append(feat)
            lengths.append(feat.shape[0])

            all_scores.append(torch.as_tensor(res["scores"], dtype=torch.float32))

        lengths = torch.as_tensor(lengths, dtype=torch.long)

        # Optionally align to a fixed frame count before building the mask.
        if target_num_frames is not None:
            target_num_frames = int(target_num_frames)
            if target_num_frames <= 0:
                raise ValueError("target_num_frames must be > 0")

            feats_aligned = [
                self._resample_single_feat(feat, target_num_frames)
                for feat in all_feats
            ]
            feats = torch.stack(feats_aligned, dim=0)  # [B, T_target, 768]
            mask = torch.ones(feats.shape[0], feats.shape[1], dtype=torch.bool)
            frame_time = torch.arange(target_num_frames, dtype=torch.float32) / self.frame_rate
        else:
            # Variable-length batches use right padding plus an explicit mask.
            feats = pad_sequence(all_feats, batch_first=True, padding_value=float(pad_value))  # [B, Tmax, 768]
            B, Tmax, _ = feats.shape
            idx = torch.arange(Tmax).unsqueeze(0)
            mask = idx < lengths.unsqueeze(1)  # [B, Tmax], bool mask usable by attention.
            frame_time = torch.arange(Tmax, dtype=torch.float32) / self.frame_rate

        # Emotion scores.
        scores = torch.stack(all_scores)   # [B, 9]
        if return_meta:
            return feats, mask, scores, lengths, frame_time
        return feats, mask, scores

    def _resample_single_feat(self, feat, target_num_frames):
        """Resample one [Ti, D] feature matrix to [target_num_frames, D]."""
        ti, dim = feat.shape
        if ti == target_num_frames:
            return feat
        if ti <= 1:
            # Degenerate very-short input: repeat instead of interpolating unstable values.
            return feat.repeat(target_num_frames, 1)

        # F.interpolate expects [N, C, L], so treat D as the channel dimension.
        x = feat.transpose(0, 1).unsqueeze(0)  # [1, D, Ti]
        x = F.interpolate(x, size=target_num_frames, mode="linear", align_corners=False) # Linear interpolation.
        return x.squeeze(0).transpose(0, 1)    # [T_target, D]

    @staticmethod
    def build_attention_bias(mask):
        """Convert a bool mask to an attention-bias tensor.

        Output shape is [B, 1, 1, T]; invalid frames are -inf and can be added to attention logits.
        """
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        bias = torch.zeros_like(mask, dtype=torch.float32)
        bias = bias.masked_fill(~mask, float("-inf"))
        return bias[:, None, None, :]

'''
print("feats:", feats.shape)   # [B, T, 768]
print("mask:", mask.shape)           # [B, T]
print("scores:", scores.shape)
'''
