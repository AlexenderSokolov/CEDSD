import torch
import torch.nn as nn
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from Audio_united.Emotion2vec.e2v import Emotion2VecExtractor
from Audio_united.MFCASTDA.MFCASTDA import Stage1_Dual_Stream
from Text_encoder.BERT import TextEncoder
from Text_encoder.FunASR.ASR import asr_infer
from spectrum.Acoustic_models import CNN_MFCC, Transformer_F0, Fusion_MLP
from offline_cache import OfflineAsrTextCache, OfflineE2VFeatureCache, OfflineFapiStatsCache, normalize_audio_path

# Local model blocks and utilities.
from FinalPart import FAPI,FAPILMScorer,IACAGate,temporal_align,compute_cross_entropy_from_att,MultiTaskHead
from config import TrainConfig
from inverse_attention_runtime import InverseAttentionRuntime

class MultimodalDeepfakeDetector(nn.Module):
    def __init__(self, cfg: TrainConfig, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = str(device) if not isinstance(device, str) else device
        
        # ==========================================
        # Acoustic feature encoders (MFCC + F0) and fusion.
        # ==========================================
        self.cnn_module = CNN_MFCC(max_len=512)
        self.transformer_module = Transformer_F0(max_len=512)
        self.acoustic_fusion_model = Fusion_MLP(self.cnn_module, self.transformer_module)

        # ==========================================
        # Frozen Stage1 stream for time-aligned acoustic features.
        # ==========================================
        self.stage1 = Stage1_Dual_Stream(
            sample_rate=16000, n_mels=80, duration=5.0,
            cutoff_freq=4000, split_mode="mel_mask", transition_bins=2
        )
        # Freeze Stage1 to keep the acoustic backbone fixed.
        for param in self.stage1.parameters():
            param.requires_grad = False
        self.stage1.eval()
        
        # Emotion2Vec feature extractor for affective cues.
        self.extractor = Emotion2VecExtractor(device=self.device)

        # ==========================================
        # Text encoder for ASR transcripts.
        # ==========================================
        self.text_encoder = TextEncoder()
        # Unfreeze a small top portion for task adaptation.
        self.text_encoder.unfreeze_top_k_layers(k=2, unfreeze_pooler=False)

        # Freeze fusion MLP to avoid DDP sync for unused params.
        for p in self.acoustic_fusion_model.mlp.parameters():
            p.requires_grad_(False)

        # ==========================================
        # Language-model scorer used by FAPI.
        # ==========================================
        lm_device_cfg = str(getattr(cfg, "lm_device", "cpu")).strip().lower()
        lm_precision_cfg = str(getattr(cfg, "lm_precision", "fp16")).strip().lower()
        if lm_device_cfg not in {"cpu", "cuda"} and not lm_device_cfg.startswith("cuda:"):
            lm_device_cfg = "cpu"
        if lm_device_cfg == "cuda" and torch.cuda.is_available():
            # Use rank-local CUDA device under DDP.
            lm_device_cfg = f"cuda:{torch.cuda.current_device()}"
        self.lm_scorer = FAPILMScorer(cfg.lm_name, torch.device(lm_device_cfg), precision=lm_precision_cfg)
        stage1_audio_dim = self._infer_stage1_audio_dim(self.stage1)
        e2v_feature_dim = int(getattr(cfg, "e2v_feature_dim", 768))
        # Cross-modal inverse attention block.
        self.inverse_attention = InverseAttentionRuntime(
            audio_dim=stage1_audio_dim + e2v_feature_dim,
            text_dim=768,
        )
        # IACA gate for audio confidence vs. cross-attention entropy.
        self.IACA=IACAGate()
        # Multi-task head for spoof and emotion predictions.
        self.muli_head = MultiTaskHead(d_model=512, n_emotions=cfg.n_emotions)
        self._g_debug_step = 0
        self._g_debug_logger = logging.getLogger("fapi_train")
        self._empty_text_path_counts = defaultdict(int)
        self._empty_text_batch_warn_count = 0
        self._e2v_source_counts = defaultdict(int)
        self._e2v_offline_last_fail = ""
        # FAPI scoring and fusion step.
        self._last_l_fapi_raw = None
        self._last_fapi_grad_rms_mean = None
        self._last_fapi_grads_is_none = True
        # Placeholders to keep tokenizer/FAPI stable when ASR returns blank.
        self._empty_text_placeholders = list(getattr(cfg, "empty_text_placeholder_candidates", ["嗯"]))
        if len(self._empty_text_placeholders) == 0:
            self._empty_text_placeholders = ["嗯"]
        placeholder_seed = int(getattr(cfg, "empty_text_placeholder_seed", 42))
        self._placeholder_rng = random.Random(placeholder_seed)

        # Optional offline caches (rank-aware) for ASR/FAPI/E2V.
        self.asr_text_cache = None
        self.fapi_stats_cache = None
        self.e2v_feature_cache = None
        try:
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except Exception:
            rank = 0
            world_size = 1

        cache_root = Path(__file__).resolve().parent
        cache_root.mkdir(parents=True, exist_ok=True)

        if bool(getattr(cfg, "asr_offline_cache_enable", True)):
            asr_cache_file = str(getattr(cfg, "asr_offline_cache_file", "offline_asr_cache.csv")).strip()
            asr_cache_path = Path(asr_cache_file)
            if not asr_cache_path.is_absolute():
                asr_cache_path = cache_root / asr_cache_path
            self.asr_text_cache = OfflineAsrTextCache(str(asr_cache_path), rank=rank, world_size=world_size)

        if bool(getattr(cfg, "fapi_offline_cache_enable", True)):
            fapi_cache_file = str(getattr(cfg, "fapi_offline_cache_file", "offline_fapi_stats_cache.csv")).strip()
            fapi_cache_path = Path(fapi_cache_file)
            if not fapi_cache_path.is_absolute():
                fapi_cache_path = cache_root / fapi_cache_path
            self.fapi_stats_cache = OfflineFapiStatsCache(str(fapi_cache_path), rank=rank, world_size=world_size)

        if bool(getattr(cfg, "e2v_offline_cache_enable", True)):
            e2v_cache_dir = str(getattr(cfg, "e2v_offline_cache_dir", "offline_cache_store/e2v_cache")).strip()
            e2v_cache_path = Path(e2v_cache_dir)
            if not e2v_cache_path.is_absolute():
                e2v_cache_path = cache_root / e2v_cache_path
            self.e2v_feature_cache = OfflineE2VFeatureCache(str(e2v_cache_path), rank=rank, world_size=world_size)

        self._forward_context = {}
        self._last_forward_error_info = None

    @staticmethod
    def _infer_stage1_audio_dim(stage1_module) -> int:
        final_compression = getattr(stage1_module, "final_compression", None)
        if isinstance(final_compression, nn.Sequential):
            for layer in reversed(final_compression):
                if isinstance(layer, nn.Conv1d):
                    return int(layer.out_channels)
        return 128

    def set_forward_context(self, batch_id=None, epoch_id=None, step_id=None, rank_id=None, audio_paths=None):
        """Inject batch context from the training loop to aid failure diagnosis."""
        paths = []
        if isinstance(audio_paths, (list, tuple)):
            paths = [str(p) for p in audio_paths]
        elif audio_paths is not None:
            paths = [str(audio_paths)]
        self._forward_context = {
            "batch_id": batch_id,
            "epoch_id": epoch_id,
            "step_id": step_id,
            "rank": rank_id,
            "audio_paths": paths,
            "sample_idx": list(range(len(paths))),
        }

    def _infer_forward_root_cause_code(self, err_msg: str) -> str:
        msg = str(err_msg or "")
        if "ASR 推理失败" in msg:
            return "E_TEXT_ASR_FAIL"
        if "ASR 输出为空" in msg:
            return "E_TEXT_ASR_EMPTY"
        if "空文本批次占比过高" in msg or "重复空路径" in msg:
            return "E_TEXT_EMPTY_REPEAT"
        if "Stage1 特征提取失败" in msg:
            return "E_AUDIO_STAGE1_FAIL"
        if "Emotion2Vec 提取失败" in msg:
            return "E_AUDIO_E2V_FAIL"
        if "非有限" in msg or "包含非有限值" in msg:
            return "E_NUMERIC_NONFINITE"
        if "为空" in msg:
            return "E_EMPTY_VALUE"
        return "E_FORWARD_UNKNOWN"

    def get_last_forward_error_info(self):
        return self._last_forward_error_info

    def _assert_finite_tensor(self, tensor, name, paths=None):
        """Assert tensor finiteness at critical points to catch the first bad batch."""
        if tensor is None:
            raise ValueError(f"{name} 为空")
        if not torch.is_tensor(tensor):
            raise ValueError(f"{name} 不是 Tensor")
        if not torch.isfinite(tensor).all():
            bad_count = int((~torch.isfinite(tensor)).sum().item())
            sample_paths = []
            if isinstance(paths, (list, tuple)):
                sample_paths = [str(p) for p in paths[:3]]
            raise ValueError(
                f"{name} 包含非有限值 | bad_count={bad_count} | sample_paths={sample_paths}"
            )

    def _sample_empty_text_placeholder(self) -> str:
        """Sample a neutral placeholder when ASR returns empty text."""
        if len(self._empty_text_placeholders) == 1:
            return str(self._empty_text_placeholders[0])
        return str(self._placeholder_rng.choice(self._empty_text_placeholders))

    def _validate_and_align_e2v_payload(self, payload, t_spec):
        """Validate a cached E2V sample and align time length.
        Returns (feat[T,D], mask[T], score[C], err).
        """
        if payload is None:
            return None, None, None, "payload_none"
        if not all(k in payload for k in ("e2v_feats", "e2v_mask", "e2v_scores")):
            return None, None, None, "payload_missing_keys"

        feat = torch.as_tensor(payload["e2v_feats"], dtype=torch.float32)
        mask = torch.as_tensor(payload["e2v_mask"], dtype=torch.bool)
        score = torch.as_tensor(payload["e2v_scores"], dtype=torch.float32)

        if feat.ndim != 2 or mask.ndim != 1 or score.ndim != 1:
            return None, None, None, f"payload_bad_ndim:feat={feat.ndim},mask={mask.ndim},score={score.ndim}"
        if feat.size(0) != mask.size(0):
            return None, None, None, f"payload_time_mismatch:feat_t={feat.size(0)},mask_t={mask.size(0)}"

        if feat.size(0) > t_spec:
            feat = feat[:t_spec, :]
            mask = mask[:t_spec]
        elif feat.size(0) < t_spec:
            pad_t = t_spec - feat.size(0)
            feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_t))
            mask = torch.nn.functional.pad(mask, (0, pad_t), value=False)

        return feat, mask, score, ""


    def _extract_audio_stream(self, batch_audio, batch_paths=None, batch_e2v_feats=None, batch_e2v_mask=None, batch_e2v_scores=None):
        """Internal helper: process Stream 2 (Stage1 + E2V)."""
        # Validate raw waveform batch.
        if batch_audio is None or (not torch.is_tensor(batch_audio)):
            raise ValueError("batch_audio 不能为空且必须是 Tensor")
        if batch_audio.ndim != 2:
            raise ValueError(f"batch_audio 期望形状 [B, T]，实际为 {tuple(batch_audio.shape)}")
        if batch_audio.size(0) == 0 or batch_audio.size(1) == 0:
            raise ValueError("batch_audio 的 batch 维或时间维为空")

        try:
            # Stage1 feature extraction (frozen backbone).
            stage1_feats = self.stage1(batch_audio)
        except Exception as e:
            raise RuntimeError(f"Stage1 特征提取失败: {e}") from e

        if stage1_feats is None or stage1_feats.ndim != 3 or stage1_feats.size(-1) == 0:
            raise RuntimeError("Stage1 输出为空或形状非法，期望 [B, C, T]")
        # Align E2V timeline to Stage1 frame length.
        t_spec = stage1_feats.size(-1)
        cached_e2v_available = (
            batch_e2v_feats is not None
            and batch_e2v_mask is not None
            and batch_e2v_scores is not None
            and torch.is_tensor(batch_e2v_feats)
            and torch.is_tensor(batch_e2v_mask)
            and torch.is_tensor(batch_e2v_scores)
            and batch_e2v_feats.ndim == 3
            and batch_e2v_mask.ndim == 2
            and batch_e2v_scores.ndim == 2
            and batch_e2v_feats.size(0) == batch_audio.size(0)
            and batch_e2v_mask.size(0) == batch_audio.size(0)
            and batch_e2v_scores.size(0) == batch_audio.size(0)
        )

        if cached_e2v_available:
            e2v_source = "batch_input"
            e2v_feats = batch_e2v_feats.to(stage1_feats.device)
            e2v_mask = batch_e2v_mask.to(stage1_feats.device)
            e2v_scores = batch_e2v_scores.to(stage1_feats.device)

            if e2v_feats.ndim != 3:
                raise RuntimeError(f"缓存 E2V feats 维度异常，期望 [B, T, D]，实际为 {tuple(e2v_feats.shape)}")
            if e2v_mask.ndim != 2:
                raise RuntimeError(f"缓存 E2V mask 维度异常，期望 [B, T]，实际为 {tuple(e2v_mask.shape)}")
            if e2v_scores.ndim != 2:
                raise RuntimeError(f"缓存 E2V scores 维度异常，期望 [B, C]，实际为 {tuple(e2v_scores.shape)}")

            if e2v_feats.size(1) != t_spec:
                if e2v_feats.size(1) > t_spec:
                    e2v_feats = e2v_feats[:, :t_spec, :]
                    e2v_mask = e2v_mask[:, :t_spec]
                else:
                    pad_t = t_spec - e2v_feats.size(1)
                    e2v_feats = torch.nn.functional.pad(e2v_feats, (0, 0, 0, pad_t))
                    e2v_mask = torch.nn.functional.pad(e2v_mask, (0, pad_t))
        else:
            normalized_paths = [normalize_audio_path(p) for p in batch_paths] if batch_paths is not None else []
            payloads = None
            if self.e2v_feature_cache is not None and len(normalized_paths) == batch_audio.size(0):
                payloads = self.e2v_feature_cache.get_many(normalized_paths)

            # Collect per-sample E2V features (cache first, then online).
            sample_feats = []
            sample_masks = []
            sample_scores = []
            offline_hit_count = 0
            online_count = 0
            miss_reasons = []

            for i in range(batch_audio.size(0)):
                used_offline = False
                if payloads is not None and i < len(payloads):
                    feat2d, mask1d, score1d, cache_err = self._validate_and_align_e2v_payload(payloads[i], t_spec)
                    if cache_err == "":
                        sample_feats.append(feat2d.unsqueeze(0))
                        sample_masks.append(mask1d.unsqueeze(0))
                        sample_scores.append(score1d.unsqueeze(0))
                        offline_hit_count += 1
                        used_offline = True
                    else:
                        miss_path = normalized_paths[i] if i < len(normalized_paths) else "unknown"
                        miss_reasons.append(f"{i}:{cache_err}:{miss_path}")
                elif self.e2v_feature_cache is None:
                    miss_reasons.append("cache_disabled_or_not_initialized")
                elif batch_paths is None:
                    miss_reasons.append("batch_paths_none")
                else:
                    miss_reasons.append(f"payload_missing_at_index:{i}")

                if used_offline:
                    continue

                single_audio = batch_audio[i:i+1]
                try:
                    with torch.no_grad():
                        result = self.extractor.extract(
                            single_audio, target_num_frames=t_spec, return_meta=True
                        )
                except Exception as e:
                    raise RuntimeError(f"Emotion2Vec 提取失败，batch_index={i}: {e}") from e

                if not isinstance(result, tuple) or len(result) != 5:
                    raise RuntimeError(f"Emotion2Vec 返回结果数量异常，batch_index={i}")

                f = result[0]
                m = result[1]
                s = result[2]
                l = result[3]

                if f is None or m is None or s is None or l is None:
                    raise RuntimeError(f"Emotion2Vec 返回空结果，batch_index={i}")

                f = torch.as_tensor(f, dtype=torch.float32)
                m = torch.as_tensor(m, dtype=torch.bool)
                s = torch.as_tensor(s, dtype=torch.float32)

                if f.ndim == 2:
                    f = f.unsqueeze(0)
                if m.ndim == 1:
                    m = m.unsqueeze(0)
                if s.ndim == 1:
                    s = s.unsqueeze(0)

                if f.ndim != 3 or m.ndim != 2 or s.ndim != 2 or f.size(0) != 1 or m.size(0) != 1 or s.size(0) != 1:
                    raise RuntimeError(
                        f"Emotion2Vec 输出维度异常，batch_index={i}, "
                        f"f={tuple(f.shape)}, m={tuple(m.shape)}, s={tuple(s.shape)}"
                    )

                if f.size(1) > t_spec:
                    f = f[:, :t_spec, :]
                    m = m[:, :t_spec]
                elif f.size(1) < t_spec:
                    pad_t = t_spec - f.size(1)
                    f = torch.nn.functional.pad(f, (0, 0, 0, pad_t))
                    m = torch.nn.functional.pad(m, (0, pad_t), value=False)

                sample_feats.append(f)
                sample_masks.append(m)
                sample_scores.append(s)
                online_count += 1

                if self.e2v_feature_cache is not None and i < len(normalized_paths):
                    try:
                        self.e2v_feature_cache.put(
                            normalized_paths[i],
                            {
                                "e2v_feats": f.squeeze(0).detach().cpu(),
                                "e2v_mask": m.squeeze(0).detach().cpu(),
                                "e2v_scores": s.squeeze(0).detach().cpu(),
                            },
                        )
                    except Exception:
                        pass

            if len(sample_feats) != batch_audio.size(0):
                raise RuntimeError(
                    f"E2V 特征样本数与 batch 不一致: got={len(sample_feats)}, expect={batch_audio.size(0)}"
                )

            e2v_feats = torch.cat(sample_feats, dim=0).to(self.device)
            e2v_mask = torch.cat(sample_masks, dim=0).to(self.device)
            e2v_scores = torch.cat(sample_scores, dim=0).to(self.device)

            if offline_hit_count == batch_audio.size(0):
                e2v_source = "offline_cache"
                self._e2v_offline_last_fail = ""
            elif online_count == batch_audio.size(0):
                e2v_source = "online_extract"
                self._e2v_offline_last_fail = " | ".join(miss_reasons[:3])
            else:
                e2v_source = "mixed_cache_online"
                self._e2v_offline_last_fail = f"offline_hit={offline_hit_count}/{batch_audio.size(0)} | online={online_count}"

        e2v_feats = e2v_feats.to(stage1_feats.device).transpose(1, 2)
        
        # Concatenate Stage1 and E2V channels, then switch to [B, T, C].
        fused_audio = torch.cat([stage1_feats, e2v_feats], dim=1)
        F_AE = fused_audio.transpose(1, 2).contiguous()  # [B, T_a, C]
        
        return F_AE, e2v_mask, e2v_scores, e2v_source

    def _extract_text_stream(self, paths):
        """Internal helper: process Stream 3 (ASR + BERT)."""
        logger = logging.getLogger("fapi_train")
        # Validate ASR input paths.
        if paths is None:
            raise ValueError("batch_data 不能为空")
        if isinstance(paths, (list, tuple)) and len(paths) == 0:
            raise ValueError("batch_data 为空列表，无法执行 ASR")
        
        if isinstance(paths, tuple):
            paths = list(paths)
        elif not isinstance(paths, list):
            paths = [paths]

        # Initialize with cache hits first.
        texts = [None for _ in range(len(paths))]
        if self.asr_text_cache is not None:
            cached_texts = self.asr_text_cache.get_many(paths)
            for i, cached in enumerate(cached_texts):
                if cached is not None:
                    texts[i] = cached

        miss_indices = [i for i, t in enumerate(texts) if t is None]
        if len(miss_indices) > 0:
            miss_paths = [paths[i] for i in miss_indices]
            try:
                with torch.no_grad():
                    miss_texts = asr_infer(miss_paths, device=self.device)
            except Exception as e:
                raise RuntimeError(f"ASR 推理失败: {e}") from e

            if miss_texts is None or len(miss_texts) == 0:
                raise RuntimeError("ASR 输出为空")

            if not isinstance(miss_texts, list):
                miss_texts = list(miss_texts)

            if len(miss_texts) != len(miss_indices):
                logger.warning(
                    f"ASR miss 返回长度异常 | miss={len(miss_indices)} | got={len(miss_texts)}"
                )

            for idx, miss_idx in enumerate(miss_indices):
                text_val = miss_texts[idx] if idx < len(miss_texts) else ""
                texts[miss_idx] = text_val

            if self.asr_text_cache is not None:
                self.asr_text_cache.put_many(miss_paths, miss_texts)

        texts = ["" if t is None else t for t in texts]

        if len(texts) != len(paths):
            logger.warning(
                f"ASR 输出长度与输入路径长度不一致 | num_texts={len(texts)} | num_paths={len(paths)}"
            )

        sanitized_texts = []
        empty_text_paths = []
        repeated_empty_paths = []
        repeated_warn_records = []
        repeat_log_every = 20
        batch_log_every = 20
        for idx, text in enumerate(texts):
            if text is None:
                bad_path = paths[idx] if idx < len(paths) else "<out_of_range>"
                logger.warning(f"ASR 返回 None 文本，已替换为空串 | idx={idx} | path={bad_path}")
                text = ""
            elif not isinstance(text, str):
                bad_path = paths[idx] if idx < len(paths) else "<out_of_range>"
                logger.warning(f"ASR 返回非字符串文本，已强制转换 | idx={idx} | path={bad_path} | type={type(text)}")
                text = str(text)

            text = text.strip()
            if len(text) == 0:
                bad_path = paths[idx] if idx < len(paths) else "<out_of_range>"
                self._empty_text_path_counts[bad_path] += 1
                cur_count = int(self._empty_text_path_counts[bad_path])
                empty_text_paths.append(bad_path)
                if cur_count >= int(getattr(self.cfg, "text_empty_path_patience", 3)):
                    repeated_empty_paths.append(bad_path)
                    patience = int(getattr(self.cfg, "text_empty_path_patience", 3))
                    if cur_count == patience or ((cur_count - patience) % max(1, repeat_log_every) == 0):
                        repeated_warn_records.append((idx, bad_path, cur_count))
                text = self._sample_empty_text_placeholder()
            sanitized_texts.append(text)

        texts = sanitized_texts
        empty_text_count = len(empty_text_paths)
        total_text_count = max(1, len(texts))
        empty_text_ratio = float(empty_text_count) / float(total_text_count)

        if empty_text_count > 0:
            self._empty_text_batch_warn_count += 1
            need_batch_log = (
                self._empty_text_batch_warn_count == 1
                or (self._empty_text_batch_warn_count % max(1, batch_log_every) == 0)
            )
            if need_batch_log:
                sample_paths = ", ".join(empty_text_paths[:3])
                logger.warning(
                    f"ASR 返回空文本，已使用占位符避免 tokenizer/FAPI 崩溃 | "
                    f"batch_empty={empty_text_count}/{total_text_count}({empty_text_ratio:.2%}) | "
                    f"sample_paths={sample_paths}"
                )

        for idx, bad_path, cur_count in repeated_warn_records:
            logger.warning(
                f"ASR 返回空文本且重复出现 | idx={idx} | path={bad_path} | count={cur_count}"
            )

        text_batch_warn = empty_text_ratio >= float(getattr(self.cfg, "text_empty_batch_warn_ratio", 0.25))
        repeated_empty_count = len(repeated_empty_paths)
        repeated_empty_ratio = float(repeated_empty_count) / float(total_text_count)
        repeat_hard_skip_enable = bool(getattr(self.cfg, "text_repeat_path_hard_skip_enable", False))
        repeat_hard_skip_min_count = int(max(1, getattr(self.cfg, "text_repeat_path_hard_skip_min_count", 2)))
        repeat_hard_skip_ratio = float(getattr(self.cfg, "text_repeat_path_hard_skip_ratio", 0.5))
        repeat_hard_skip = bool(
            repeat_hard_skip_enable
            and repeated_empty_count >= repeat_hard_skip_min_count
            and repeated_empty_ratio >= repeat_hard_skip_ratio
        )
        text_batch_warn = bool(text_batch_warn or repeated_empty_count > 0)
        text_batch_invalid = bool(
            empty_text_ratio >= float(getattr(self.cfg, "text_empty_batch_skip_ratio", 0.5))
            or repeat_hard_skip
        )
        # Tokenize and encode text for the transformer.
        encoded = self.text_encoder.tokenizer(
            texts, padding=True, truncation=True, max_length=32, return_tensors="pt"
        ).to(self.device)
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        F_text, text_mask = self.text_encoder(encoded["input_ids"], encoded["attention_mask"])
        return F_text, text_mask, texts, {
            "empty_text_count": empty_text_count,
            "empty_text_ratio": empty_text_ratio,
            "empty_text_paths": empty_text_paths,
            "repeated_empty_text_paths": repeated_empty_paths,
            "repeated_empty_count": repeated_empty_count,
            "repeated_empty_ratio": repeated_empty_ratio,
            "repeat_hard_skip": repeat_hard_skip,
            "text_batch_warn": text_batch_warn,
            "text_batch_invalid": text_batch_invalid,
        }


    def forward(self, batch_data, is_train=None, labels=None, compute_fapi_penalty=True):
        """
        Full forward pass.
        Input: batch_data dictionary from the DataLoader.
        Output: logits, masks, and diagnostics.
        """
        # Resolve the training mode for optional behaviors.
        current_train_mode = is_train if is_train is not None else self.training
        # Validate batch input.
        if batch_data is None:
            raise ValueError("batch 不能为空")
        if len(batch_data) == 0:
            print("[WARN] batch 为空，已跳过推理")
            empty_ret = {
                "all_audio_outputs": [],
                "outputs_text": {"F_text": None, "text_mask": None, "texts": []}
            }
        # Reset last error info for this forward pass.
        self._last_forward_error_info = None

        try:    
            # Load batch tensors and check numeric sanity.
            mfcc = batch_data["mfcc"].to(self.device)
            f0 = batch_data["f0"].to(self.device)
            batch_audio = batch_data['raw_waveform'].to(self.device)
            paths = batch_data["audio_path"]
            U_audio = batch_data["voiced_probs"].to(self.device)
            self._assert_finite_tensor(mfcc, "mfcc", paths)
            self._assert_finite_tensor(f0, "f0", paths)
            self._assert_finite_tensor(batch_audio, "raw_waveform", paths)
            self._assert_finite_tensor(U_audio, "voiced_probs", paths)
            # Reduce voiced probability to a per-sample confidence.
            if U_audio.dim() == 3:
                U_audio = U_audio.squeeze(-1).mean(dim=1)
            elif U_audio.dim() == 2:
                U_audio = U_audio.mean(dim=1)
            elif U_audio.dim() == 1:
                pass
            else:
                raise ValueError(f"U_audio 期望维度为 1/2/3，实际为 {U_audio.dim()}D")
            self._assert_finite_tensor(U_audio, "U_audio_reduced", paths)
            
            # Acoustic fusion branch (MFCC + F0).
            F_A = self.acoustic_fusion_model(mfcc, f0)  # [B, 512]
            self._assert_finite_tensor(F_A, "F_A", paths)
            
            # Stage1 + Emotion2Vec stream.
            F_AE, audio_mask, F_emo, e2v_source = self._extract_audio_stream(
                batch_audio,
                paths,
                batch_data.get("e2v_feats"),
                batch_data.get("e2v_mask"),
                batch_data.get("e2v_scores"),
            )
            self._assert_finite_tensor(F_AE, "F_AE", paths)
            self._assert_finite_tensor(audio_mask, "audio_mask", paths)
            self._assert_finite_tensor(F_emo, "F_emo", paths)
            
            # ASR + text encoder stream.
            F_text, text_mask, texts, text_quality = self._extract_text_stream(paths)
            self._assert_finite_tensor(F_text, "F_text", paths)
            self._assert_finite_tensor(text_mask, "text_mask", paths)
            if text_quality["text_batch_invalid"]:
                raise ValueError(
                    f"空文本批次占比过高或存在重复空路径 | empty_ratio={text_quality['empty_text_ratio']:.2f} | "
                    f"empty_count={text_quality['empty_text_count']} | repeat_paths={len(text_quality['repeated_empty_text_paths'])} | "
                    f"repeat_ratio={float(text_quality.get('repeated_empty_ratio', 0.0)):.2f} | "
                    f"repeat_hard_skip={bool(text_quality.get('repeat_hard_skip', False))}"
                )
            if text_quality["text_batch_warn"]:
                self._g_debug_logger.warning(
                    f"[TextQuality] empty_ratio={text_quality['empty_text_ratio']:.2f} | "
                    f"empty_count={text_quality['empty_text_count']} | repeat_paths={len(text_quality['repeated_empty_text_paths'])} | "
                    f"repeat_ratio={float(text_quality.get('repeated_empty_ratio', 0.0)):.2f}"
                )
            
            # FAPI scoring and penalty terms.
            Fapi_out=FAPI(
                F_A,
                F_AE,
                F_text,
                audio_mask,
                text_mask,
                texts,
                U_audio,
                F_emo,
                self.lm_scorer,
                self.cfg,
                audio_paths=[normalize_audio_path(p) for p in paths],
                fapi_stats_cache=self.fapi_stats_cache,
            )
            ppl=Fapi_out["ppl"]
            topk_mean=Fapi_out["topk_mean"]
            gamma_t=Fapi_out["gamma_t"]
            lambda_penalty=Fapi_out["lambda_penalty"]
            self._assert_finite_tensor(ppl, "ppl", paths)
            self._assert_finite_tensor(topk_mean, "topk_mean", paths)
            self._assert_finite_tensor(gamma_t, "gamma_t", paths)
            self._assert_finite_tensor(lambda_penalty, "lambda_penalty", paths)
            
            # Inverse attention for cross-modal alignment.
            iam_out=self.inverse_attention.run(F_AE,F_text,text_mask,audio_mask,is_train=current_train_mode)
            att_cross_weights = iam_out["att_cross_weights"]
            F_AT = iam_out["F_AT"]
            self._assert_finite_tensor(att_cross_weights, "att_cross_weights", paths)
            self._assert_finite_tensor(F_AT, "F_AT", paths)
            
            # Align acoustic tokens to text length when needed.
            if F_A.dim() == 2:
                F_A = F_A.unsqueeze(1).expand(-1, F_AT.size(1), -1)
            elif F_A.size(1) != F_AT.size(1):
                # Temporal align when sequence lengths differ.
                F_A = temporal_align(F_A, F_AT.size(1)) 
            
            # Cross-attention entropy for the IACA gate.
            h_cross=compute_cross_entropy_from_att(att_cross_weights)
            self._assert_finite_tensor(h_cross, "h_cross", paths)
            
            # IACA gate for audio/text fusion.
            g = self.IACA(U_audio, h_cross).unsqueeze(-1).unsqueeze(-1)   # [B,1,1]
            self._assert_finite_tensor(g, "g", paths)

            if current_train_mode:
                self._g_debug_step += 1
                self._e2v_source_counts[e2v_source] += 1
                if self._g_debug_step % max(1, int(getattr(self.cfg, "print_every_n_steps", 10))) == 0:
                    if str(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == "0":
                        total_cnt = max(1, sum(self._e2v_source_counts.values()))
                        batch_in_cnt = int(self._e2v_source_counts.get("batch_input", 0))
                        offline_cnt = int(self._e2v_source_counts.get("offline_cache", 0))
                        mixed_cnt = int(self._e2v_source_counts.get("mixed_cache_online", 0))
                        online_cnt = int(self._e2v_source_counts.get("online_extract", 0))
                        self._g_debug_logger.info(
                            f"[E2VSource] step={self._g_debug_step} | current_batch={e2v_source} | "
                            f"batch_input={batch_in_cnt}/{total_cnt}={batch_in_cnt/total_cnt:.2%} | "
                            f"offline_cache={offline_cnt}/{total_cnt}={offline_cnt/total_cnt:.2%} | "
                            f"mixed_cache_online={mixed_cnt}/{total_cnt}={mixed_cnt/total_cnt:.2%} | "
                            f"online_extract={online_cnt}/{total_cnt}={online_cnt/total_cnt:.2%}"
                        )
                        if e2v_source == "online_extract" and len(str(self._e2v_offline_last_fail).strip()) > 0:
                            self._g_debug_logger.warning(
                                f"[E2VSourceFallback] reason={self._e2v_offline_last_fail}"
                            )
                        self._g_debug_logger.info(f"[Gate] step={self._g_debug_step} | g_mean={g.mean().item():.4f}")
            
            device_cur=g.device
            # Clamp lambda to a stable range.
            lam_scalar = torch.clamp(
                lambda_penalty.to(device_cur),
                min=float(getattr(self.cfg, "fapi_lambda_floor", 0.15)),
                max=float(getattr(self.cfg, "fapi_lambda_ceiling", 0.85)),
            )
            self._assert_finite_tensor(lam_scalar, "lambda_penalty_clamped", paths)
            lam = lam_scalar.unsqueeze(-1).unsqueeze(-1)
           
            # Fuse acoustic and text-attended features.
            f_fused = F_A * g + ((1.0 - lam) * F_AT) * (1.0 - g)
            self._assert_finite_tensor(f_fused, "f_fused", paths)
            
            # Multi-task outputs for spoof and emotion.
            fake_logit, emo_logit=self.muli_head(f_fused)
            self._assert_finite_tensor(fake_logit, "fake_logit", paths)
            self._assert_finite_tensor(emo_logit, "emo_logit", paths)
            
            # DDP-safe defaults when gradients are skipped.
            l_fapi_raw = torch.zeros((), device=fake_logit.device)
            fapi_grad_rms_mean = torch.zeros((), device=fake_logit.device)
            fapi_grads_is_none = True

            # Compute FAPI penalty from text gradients when enabled.
            if labels is not None and F_text.requires_grad and bool(compute_fapi_penalty):
                # Ensure labels are on the same device/dtype.
                labels_t = labels.to(device=fake_logit.device, dtype=fake_logit.dtype)
                
                # Base loss used to derive gradient norms.
                l_ce_tmp = torch.nn.functional.binary_cross_entropy_with_logits(fake_logit, labels_t)
                
                # DDP-safe gradient extraction.
                grads = torch.autograd.grad(
                    outputs=l_ce_tmp,
                    inputs=F_text,
                    create_graph=True,   # Enable higher-order grads for penalty.
                    retain_graph=True,
                    only_inputs=True,
                    allow_unused=True,
                )[0]
                
                if grads is not None:
                    fapi_grads_is_none = False
                    # RMS gradient magnitude per sample.
                    grad_rms = grads.reshape(grads.size(0), -1).pow(2).mean(dim=1).sqrt()
                    fapi_grad_rms_mean = grad_rms.mean()
                    self._assert_finite_tensor(fapi_grad_rms_mean, "fapi_grad_rms_mean", paths)
                    
                    # Lambda-weighted penalty.
                    lam_s = lam_scalar.view_as(grad_rms) 
                    l_fapi_raw = (lam_s * grad_rms).mean()
                    self._assert_finite_tensor(l_fapi_raw, "l_fapi_raw", paths)
                # Cache penalty metrics for optional reuse.
                self._last_l_fapi_raw = l_fapi_raw.detach()
                self._last_fapi_grad_rms_mean = fapi_grad_rms_mean.detach()
                self._last_fapi_grads_is_none = bool(fapi_grads_is_none)
            elif labels is not None and F_text.requires_grad and bool(getattr(self.cfg, "fapi_penalty_reuse_enable", False)):
                # Reuse cached penalty when gradients are skipped.
                if self._last_l_fapi_raw is not None:
                    l_fapi_raw = self._last_l_fapi_raw.to(fake_logit.device, dtype=fake_logit.dtype)
                if self._last_fapi_grad_rms_mean is not None:
                    fapi_grad_rms_mean = self._last_fapi_grad_rms_mean.to(fake_logit.device, dtype=fake_logit.dtype)
                fapi_grads_is_none = bool(self._last_fapi_grads_is_none)
            # =======================================================================
            
            return {
            "fake_logit": fake_logit,
            "emo_logit": emo_logit,
            "F_A": F_A,
            "F_AE": F_AE,
            "F_AT": F_AT,
            "F_text": F_text,
            "audio_mask": audio_mask,
            "text_mask": text_mask,
            "F_emo": F_emo,
            "audio_paths": [normalize_audio_path(p) for p in paths],
            "empty_text_count": torch.tensor(float(text_quality["empty_text_count"]), device=self.device),
            "empty_text_ratio": torch.tensor(float(text_quality["empty_text_ratio"]), device=self.device),
            "text_batch_warn": torch.tensor(1.0 if text_quality["text_batch_warn"] else 0.0, device=self.device),
            "text_batch_invalid": torch.tensor(1.0 if text_quality["text_batch_invalid"] else 0.0, device=self.device),
            "ppl": ppl,
            "topk_mean": topk_mean,
            "gamma_t": gamma_t,
            "lambda_penalty": lambda_penalty,
            "lambda_penalty_clamped": lam_scalar,
            "g": g.squeeze(-1).squeeze(-1),
            "h_cross": h_cross,
            "u_audio": U_audio,
            "att_cross_weights": att_cross_weights,
            "att_inversed_weights": iam_out["att_inversed_weights"],
            "beta": iam_out["beta"],
            # FAPI penalty diagnostics.
            "l_fapi_raw": l_fapi_raw,
            "fapi_grad_rms_mean": fapi_grad_rms_mean,
            "fapi_grads_is_none": fapi_grads_is_none,
        }
            
        except Exception as e:
            ctx = dict(self._forward_context) if isinstance(self._forward_context, dict) else {}
            root_cause_code = self._infer_forward_root_cause_code(str(e))
            diag = {
                "root_cause_code": root_cause_code,
                "batch_id": ctx.get("batch_id"),
                "epoch_id": ctx.get("epoch_id"),
                "step_id": ctx.get("step_id"),
                "rank": ctx.get("rank"),
                "sample_idx": ctx.get("sample_idx", []),
                "sample_paths": ctx.get("audio_paths", [])[:3],
                "detail": str(e),
            }
            self._last_forward_error_info = diag
            self._g_debug_logger.exception(
                f"[ForwardDiag] root_cause_code={diag['root_cause_code']} | "
                f"batch_id={diag['batch_id']} | epoch_id={diag['epoch_id']} | step_id={diag['step_id']} | "
                f"rank={diag['rank']} | sample_idx={diag['sample_idx'][:8]} | sample_paths={diag['sample_paths']} | "
                f"detail={diag['detail']}"
            )
            raise RuntimeError(
                f"Forward failed | code={diag['root_cause_code']} | "
                f"batch={diag['batch_id']} | rank={diag['rank']} | detail={diag['detail']}"
            ) from e
                
        
    

