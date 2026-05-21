import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import math
from typing import cast

"""Stage-1 dual-stream audio feature extraction.

Pipeline:
1) split the raw waveform into low- and high-frequency branches
2) low-frequency branch: STDA temporal-difference attention
3) high-frequency branch: MFCA plus DCT frequency compression
4) concatenate features and compress channels with a 1x1 Conv
"""


def hz_to_mel(hz):
    """Convert Hz to the Mel scale."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    """Convert Mel values back to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

class DCT2D(nn.Module):
    """2D DCT module using precomputed transform matrices.

    Left/right multiplication avoids rebuilding the DCT basis every forward pass.
    """

    def __init__(self, h, w):
        super().__init__()
        self.register_buffer('weight_h', self._get_dct_matrix(h))
        self.register_buffer('weight_w', self._get_dct_matrix(w))

    def _get_dct_matrix(self, N):
        dct_m = torch.empty(N, N)
        for k in range(N):
            for n in range(N):
                alpha = math.sqrt(1/N) if k == 0 else math.sqrt(2/N)
                dct_m[k, n] = alpha * math.cos(math.pi * (2*n + 1) * k / (2 * N))
        return dct_m

    def forward(self, x):
        # Buffers are registered as tensors but need casts for static checkers.
        weight_h = cast(torch.Tensor, self.weight_h)
        weight_w = cast(torch.Tensor, self.weight_w)
        out = torch.matmul(weight_h, x)
        out = torch.matmul(out, weight_w.t())
        return out

class MFCA_Branch(nn.Module):
    """High-frequency branch: DCT features plus channel-weight estimation.

    Input:
        high_mel: [B, C, F, T]
    Output:
        [B, C, T], averaged across the weighted frequency dimension.
    """

    def __init__(
        self,
        channels,
        n_mels,
        time_steps,
        k_low=8,
        k_high=8,
        sample_rate=16000,
        cutoff_freq=4000,
        omega_low=None,
        omega_high=None,
    ):
        super().__init__()
        self.dct2d = DCT2D(n_mels, time_steps)
        self.k = k_low + k_high
        self.k_low = k_low
        self.k_high = k_high
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq

        # Build the low/high frequency index sets once and keep them as buffers.
        low_idx, high_idx = self._build_omega_indices(omega_low, omega_high)
        self.register_buffer("omega_low_idx", low_idx)
        self.register_buffer("omega_high_idx", high_idx)

        mid_channels = max(1, channels // 4)
        self.mlp = nn.Sequential(
            nn.Linear(self.k, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, 1)
        )

    def _sample_indices(self, start, end, count):
        """Uniformly sample a fixed number of indices from [start, end).

        Args:
            start: Inclusive interval start.
            end: Exclusive interval end.
            count: Number of indices to sample.

        Returns:
            LongTensor[count] for downstream index_select calls.

        Notes:
            Repeated samples are allowed when the interval is too narrow so the
            output length always matches the MLP input width.
        """
        if count <= 0:
            return torch.empty(0, dtype=torch.long)
        if end <= start:
            return torch.full((count,), max(0, min(self.n_mels - 1, start)), dtype=torch.long)

        idx = torch.linspace(start, end - 1, steps=count)
        idx = torch.round(idx).to(torch.long)
        idx = torch.clamp(idx, 0, self.n_mels - 1)
        return idx

    def _build_omega_indices(self, omega_low, omega_high):
        """Build the MFCA low/high frequency index sets, Omega_low and Omega_high.

        Args:
            omega_low: Optional manually provided low-frequency indices.
            omega_high: Optional manually provided high-frequency indices.

        Returns:
            Two LongTensors used to gather low- and high-frequency components.

        Rules:
            1) use manually provided indices after length/bounds validation
            2) otherwise derive indices from sample_rate and cutoff_freq
        """
        if (omega_low is None) ^ (omega_high is None):
            raise ValueError("omega_low and omega_high must be provided together")

        if omega_low is not None and omega_high is not None:
            low_idx = torch.tensor(omega_low, dtype=torch.long)
            high_idx = torch.tensor(omega_high, dtype=torch.long)
            if low_idx.numel() != self.k_low or high_idx.numel() != self.k_high:
                raise ValueError("omega_low / omega_high length must match k_low / k_high")
            low_idx = torch.clamp(low_idx, 0, self.n_mels - 1)
            high_idx = torch.clamp(high_idx, 0, self.n_mels - 1)
            return low_idx, high_idx

        f_min, f_max = 0.0, self.sample_rate / 2.0
        mel_min, mel_max = hz_to_mel(f_min), hz_to_mel(f_max)
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        mel_centers_hz = mel_to_hz(mel_points[1:-1])
        # Pick the Mel bin closest to the desired frequency cutoff.
        boundary = int(torch.argmin(torch.abs(mel_centers_hz - self.cutoff_freq)).item())

        low_idx = self._sample_indices(0, boundary + 1, self.k_low)
        high_idx = self._sample_indices(boundary, self.n_mels, self.k_high)
        return low_idx, high_idx

    def forward(self, high_mel):
        """Run the MFCA branch forward pass.

        Args:
            high_mel: High-frequency Mel features with shape [B, C, F, T].

        Returns:
            Tensor with shape [B, C, T] for Stage-1 concatenation.
        """
        B, C, F, T = high_mel.size()
        dct_feat = self.dct2d(high_mel)
        # Use the center time slice as a compact frequency descriptor.
        t_ref = T // 2
        dct_freq = dct_feat[:, :, :, t_ref]  # [B, C, F]
        freq_low = dct_freq.index_select(dim=-1, index=self.omega_low_idx)
        freq_high = dct_freq.index_select(dim=-1, index=self.omega_high_idx)
        freq_c = torch.cat([freq_low, freq_high], dim=-1)
        w = torch.sigmoid(self.mlp(freq_c)).view(B, C, 1, 1)
        x_out = high_mel * w
        return x_out.mean(dim=2)

class STDA_Branch(nn.Module):
    """Low-frequency branch using frame-difference self-attention.

    Input:
        low_mel: [B, F, T]
    Output:
        [B, d_v, T]
    """

    def __init__(
        self,
        n_mels,
        d_model=64,
        d_k=32,
        d_v=64,
        use_local_mask=True,
        local_window_size=9,
    ):
        super().__init__()
        self.freq_project = nn.Linear(n_mels, d_model)
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        self.d_k = d_k
        self.use_local_mask = use_local_mask
        self.local_window_size = int(local_window_size)

        if self.local_window_size <= 0:
            raise ValueError("local_window_size must be > 0")
        if self.local_window_size % 2 == 0:
            # Keep the local window symmetric around the current frame.
            self.local_window_size += 1

    def _build_local_attn_mask(self, seq_len, device):
        """Build a [T, T] mask that blocks attention outside the local window."""
        if not self.use_local_mask:
            return None

        radius = self.local_window_size // 2
        pos = torch.arange(seq_len, device=device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        return dist > radius

    def forward(self, low_mel):
        # Project [B, F, T] Mel features into a temporal sequence.
        x = low_mel.transpose(1, 2)
        A_low = self.freq_project(x)
        # Attention is driven by frame-to-frame differences.
        delta = A_low[:, 1:, :] - A_low[:, :-1, :]
        delta = F.pad(delta, (0, 0, 1, 0))

        Q = self.W_q(delta)
        K = self.W_k(delta)
        V = self.W_v(delta)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        local_mask = self._build_local_attn_mask(delta.size(1), delta.device)
        if local_mask is not None:
            scores = scores.masked_fill(local_mask.unsqueeze(0), -1e9)

        att_diff = F.softmax(scores, dim=-1)
        F_stda = torch.matmul(att_diff, V)
        return F_stda.transpose(1, 2)

class Stage1_Dual_Stream(nn.Module):
    """Top-level Stage-1 interface for waveform-to-feature extraction.

    This module expects raw waveform tensors and emits fused time-frequency
    features for downstream multimodal fusion.

    Args:
        sample_rate: Target sample rate.
        n_mels: Number of Mel bands.
        duration: Unified audio duration in seconds.
        cutoff_freq: Low/high frequency split point in Hz.
        split_mode: Frequency split mode, either "biquad" or "mel_mask".
        transition_bins: Boundary transition width for mel_mask mode.
        stda_use_local_mask: Whether STDA uses a local attention mask.
        stda_local_window_size: STDA local window size in frames.

    Notes:
        biquad mode: time-domain low/high-pass filtering followed by dual Mel.
        mel_mask mode: one full-band Mel transform followed by soft frequency masks.

    Forward Input:
        raw_audio: Tensor[B, T_samples], batched raw waveform.
    Forward Output:
        [B, 128, T_frames]
    """

    def __init__(
        self,
        sample_rate=16000,
        n_mels=80,
        duration=5.0,
        cutoff_freq=4000,
        split_mode="biquad",
        transition_bins=2,
        stda_use_local_mask=True,
        stda_local_window_size=9,
    ):
        super().__init__()
        self.sr = sample_rate
        self.duration = duration
        self.target_samples = int(sample_rate * duration)
        self.cutoff_freq = cutoff_freq
        self.split_mode = split_mode
        self.transition_bins = max(0, int(transition_bins))

        if self.split_mode not in {"biquad", "mel_mask"}:
            raise ValueError("split_mode must be 'biquad' or 'mel_mask'")
        
        self.mel_trans = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            mel_scale="htk",
            )
        # Infer Mel frame count once so the DCT branch has a fixed time basis.
        dummy_input = torch.zeros(1, self.target_samples)
        self.time_steps = self.mel_trans(dummy_input).size(-1)
        
        self.mfca_branch = MFCA_Branch(
            channels=1,
            n_mels=n_mels,
            time_steps=self.time_steps,
            k_low=8,
            k_high=8,
            sample_rate=sample_rate,
            cutoff_freq=cutoff_freq,
        )
        self.stda_branch = STDA_Branch(
            n_mels=n_mels,
            d_v=64,
            use_local_mask=stda_use_local_mask,
            local_window_size=stda_local_window_size,
        )
        self.final_compression = nn.Sequential(
            nn.Conv1d(1 + 64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )

        # Precompute soft Mel masks for mel_mask splitting.
        low_mask, high_mask = self._build_mel_band_masks(n_mels, sample_rate, cutoff_freq, self.transition_bins)
        self.register_buffer("low_mel_mask", low_mask.view(1, n_mels, 1))
        self.register_buffer("high_mel_mask", high_mask.view(1, n_mels, 1))

    def _build_mel_band_masks(self, n_mels, sample_rate, cutoff_freq, transition_bins):
        """Build low/high soft masks along the Mel-frequency axis."""
        f_min, f_max = 0.0, sample_rate / 2.0
        mel_min, mel_max = hz_to_mel(f_min), hz_to_mel(f_max)

        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        mel_centers_hz = mel_to_hz(mel_points[1:-1])
        boundary = int(torch.argmin(torch.abs(mel_centers_hz - cutoff_freq)).item())

        low_mask = torch.zeros(n_mels)
        high_mask = torch.zeros(n_mels)

        start = max(0, boundary - transition_bins)
        end = min(n_mels - 1, boundary + transition_bins)

        if start > 0:
            low_mask[:start] = 1.0
        if end < n_mels - 1:
            high_mask[end + 1:] = 1.0

        if end >= start:
            # Cross-fade around the cutoff to avoid a hard frequency discontinuity.
            width = max(1, end - start)
            for i in range(start, end + 1):
                t = (i - start) / width
                low_mask[i] = 1.0 - t
                high_mask[i] = t

        return low_mask, high_mask

    def _split_to_mels(self, raw_audio):
        """Split raw audio into low/high Mel representations.

        Args:
            raw_audio: Raw waveform tensor with shape [B, T_samples].

        Returns:
            low_mel: Low-frequency STDA input, shape [B, F, T].
            high_mel: High-frequency MFCA input, shape [B, 1, F, T].

        Notes:
            - biquad: low/high-pass filter in the time domain, then compute Mel.
            - mel_mask: compute full-band Mel once, then split with frequency masks.
        """
        if self.split_mode == "biquad":
            low_audio = torchaudio.functional.lowpass_biquad(raw_audio, self.sr, self.cutoff_freq)
            high_audio = torchaudio.functional.highpass_biquad(raw_audio, self.sr, self.cutoff_freq)
            low_mel = self.mel_trans(low_audio)
            high_mel = self.mel_trans(high_audio).unsqueeze(1)
            return low_mel, high_mel

        # Soft-split a single full-band Mel spectrogram.
        full_mel = self.mel_trans(raw_audio)
        low_mel = full_mel * self.low_mel_mask
        high_mel = (full_mel * self.high_mel_mask).unsqueeze(1)
        return low_mel, high_mel

    

    def forward(self, raw_audio):
        """Run Stage 1 from raw waveform tensors to fused time-frequency features.

        Args:
            raw_audio: Tensor[B, T_samples], raw waveform to process.

        Returns:
            out: [B, 128, T_frames], consumed by downstream modules.
        """
        if raw_audio is None or (not torch.is_tensor(raw_audio)):
            raise ValueError("raw_audio 不能为空且必须是 Tensor")
        if raw_audio.ndim != 2:
            raise ValueError(f"raw_audio 期望形状 [B, T]，实际为 {tuple(raw_audio.shape)}")
        if raw_audio.size(0) == 0 or raw_audio.size(1) == 0:
            raise ValueError("raw_audio 的 batch 维或时间维为空")
        
        low_mel, high_mel = self._split_to_mels(raw_audio)
        
        f_mfca = self.mfca_branch(high_mel)
        f_stda = self.stda_branch(low_mel)
        
        # f_mfca: [B, 1, T], f_stda: [B, 64, T]
        stage1_matrix = torch.cat([f_mfca, f_stda], dim=1)
        out = self.final_compression(stage1_matrix)
        
        return out
    
'''
# ==========================================
# Example run with a direct path list.
# ==========================================
if __name__ == "__main__":
    # 1. Prepare an audio path list.
    # Replace these example paths with local files.
    my_audio_paths = [
        "examples/audio/sample_01.wav",
        "examples/audio/sample_02.wav",
        "examples/audio/sample_03.wav"
    ]
    

    # 2. Initialize Stage 1.
    # duration=1.0 normalizes input audio to one second.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1 = Stage1_Dual_Stream(
        sample_rate=16000,
        n_mels=80,
        duration=1.0,
        cutoff_freq=4000,
        split_mode="mel_mask",   # "biquad" | "mel_mask"
        transition_bins=2,
    ).to(device)

    # 3. Pass the list directly and obtain the result matrix.
    # Expected shape: [3, 128, 32] -> [Batch, Channels, Time_frames].
    try:
        result_matrix = stage1(my_audio_paths)
        print(f"成功处理！输出矩阵形状: {result_matrix.shape}")
        # The result can then be concatenated with emotion2vec features.
    except Exception as e:
        print(f"处理失败，请检查路径或音频格式: {e}")
'''
