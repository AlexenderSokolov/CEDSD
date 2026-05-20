import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import math
from typing import cast

"""
Stage1 双流音频特征提取模块。

总体流程:
1) 原始波形 -> 低频/高频分支分频
2) 低频分支: STDA 时序差分注意力
3) 高频分支: MFCA + DCT 频域压缩
4) 特征拼接后用 1x1 Conv 压缩到统一通道数
"""


def hz_to_mel(hz):
    """Hz -> Mel 变换。"""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    """Mel -> Hz 变换。"""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

# ==========================================
# Implementation detail.
# ==========================================
class DCT2D(nn.Module):
    """二维 DCT 变换模块。

    通过左右乘预计算 DCT 矩阵实现，避免每次前向重复构造基。
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
        # Implementation detail.
        weight_h = cast(torch.Tensor, self.weight_h)
        weight_w = cast(torch.Tensor, self.weight_w)
        out = torch.matmul(weight_h, x)
        out = torch.matmul(out, weight_w.t())
        return out

class MFCA_Branch(nn.Module):
    """高频分支: DCT + 通道权重估计。

    Input:
        high_mel: [B, C, F, T]
    Output:
        [B, C, T]，对频率维做加权后平均。
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

        # Implementation detail.
        low_idx, high_idx = self._build_omega_indices(omega_low, omega_high)
        # Implementation detail.
        # Implementation detail.
        self.register_buffer("omega_low_idx", low_idx)
        self.register_buffer("omega_high_idx", high_idx)

        mid_channels = max(1, channels // 4)
        # Implementation detail.
        self.mlp = nn.Sequential(
            # Implementation detail.
            nn.Linear(self.k, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, 1)
        )

    def _sample_indices(self, start, end, count):
        """在 [start, end) 区间内均匀采样固定数量索引。

        Args:
            start: 区间起点（包含）。
            end: 区间终点（不包含）。
            count: 采样数量。

        Returns:
            LongTensor[count]，用于后续 index_select。

        Notes:
            当可用区间过窄时允许重复采样，保证返回长度恒定，
            从而与 MLP 输入维度 self.k 严格对齐。
        """
        if count <= 0:
            # Implementation detail.
            return torch.empty(0, dtype=torch.long)
        if end <= start:
            # Implementation detail.
            return torch.full((count,), max(0, min(self.n_mels - 1, start)), dtype=torch.long)

        # Implementation detail.
        idx = torch.linspace(start, end - 1, steps=count)
        # Implementation detail.
        idx = torch.round(idx).to(torch.long)
        # Implementation detail.
        idx = torch.clamp(idx, 0, self.n_mels - 1)
        return idx

    def _build_omega_indices(self, omega_low, omega_high):
        """构造 MFCA 的低/高频索引集合 Ω_low 与 Ω_high。

        Args:
            omega_low: 手动指定的低频索引列表，长度应为 k_low。
            omega_high: 手动指定的高频索引列表，长度应为 k_high。

        Returns:
            (low_idx, high_idx): 两个 LongTensor，分别用于抽取低频和高频分量。

        规则:
            1) 若手动给定，则直接使用（并做长度与边界校验）。
            2) 若不手动给定，则基于采样率与 cutoff_freq 自动构造。
        """
        if (omega_low is None) ^ (omega_high is None):
            # Implementation detail.
            raise ValueError("omega_low and omega_high must be provided together")

        if omega_low is not None and omega_high is not None:
            # Implementation detail.
            # Implementation detail.
            low_idx = torch.tensor(omega_low, dtype=torch.long)
            high_idx = torch.tensor(omega_high, dtype=torch.long)
            # Implementation detail.
            if low_idx.numel() != self.k_low or high_idx.numel() != self.k_high:
                raise ValueError("omega_low / omega_high length must match k_low / k_high")
            low_idx = torch.clamp(low_idx, 0, self.n_mels - 1)
            high_idx = torch.clamp(high_idx, 0, self.n_mels - 1)
            return low_idx, high_idx

        f_min, f_max = 0.0, self.sample_rate / 2.0
        mel_min, mel_max = hz_to_mel(f_min), hz_to_mel(f_max)
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        mel_centers_hz = mel_to_hz(mel_points[1:-1])
        # Implementation detail.
        # Implementation detail.
        # Implementation detail.
        boundary = int(torch.argmin(torch.abs(mel_centers_hz - self.cutoff_freq)).item())

        # Implementation detail.
        low_idx = self._sample_indices(0, boundary + 1, self.k_low)
        high_idx = self._sample_indices(boundary, self.n_mels, self.k_high)
        return low_idx, high_idx

    def forward(self, high_mel):
        """执行 MFCA 分支前向计算。

        Args:
            high_mel: 高频 Mel 特征，shape [B, C, F, T]。

        Returns:
            shape [B, C, T]，作为高频分支输出供 Stage1 拼接。
        """
        B, C, F, T = high_mel.size()
        # Implementation detail.
        dct_feat = self.dct2d(high_mel)
        # Implementation detail.
        t_ref = T // 2
        dct_freq = dct_feat[:, :, :, t_ref]  # [B, C, F]
        # Implementation detail.
        freq_low = dct_freq.index_select(dim=-1, index=self.omega_low_idx)
        freq_high = dct_freq.index_select(dim=-1, index=self.omega_high_idx)
        freq_c = torch.cat([freq_low, freq_high], dim=-1)
        # Implementation detail.
        w = torch.sigmoid(self.mlp(freq_c)).view(B, C, 1, 1)
        x_out = high_mel * w
        # Implementation detail.
        return x_out.mean(dim=2)

# ==========================================
# Implementation detail.
# ==========================================
class STDA_Branch(nn.Module):
    """低频分支: 基于帧间差分的自注意力。

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
            # Implementation detail.
            self.local_window_size += 1

    def _build_local_attn_mask(self, seq_len, device):
        """构造 [T, T] 局部窗口掩码: 超出窗口的时刻对不参与注意力计算。"""
        if not self.use_local_mask:
            return None

        radius = self.local_window_size // 2
        pos = torch.arange(seq_len, device=device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        # Implementation detail.
        return dist > radius

    def forward(self, low_mel):
        # Implementation detail.
        x = low_mel.transpose(1, 2)
        # Implementation detail.
        A_low = self.freq_project(x)
        # Implementation detail.
        delta = A_low[:, 1:, :] - A_low[:, :-1, :]
        delta = F.pad(delta, (0, 0, 1, 0))

        # Implementation detail.
        Q = self.W_q(delta)
        K = self.W_k(delta)
        V = self.W_v(delta)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        local_mask = self._build_local_attn_mask(delta.size(1), delta.device)
        if local_mask is not None:
            # Implementation detail.
            scores = scores.masked_fill(local_mask.unsqueeze(0), -1e9)

        att_diff = F.softmax(scores, dim=-1)
        F_stda = torch.matmul(att_diff, V)
        # Implementation detail.
        return F_stda.transpose(1, 2)

# ==========================================
# Implementation detail.
# ==========================================
class Stage1_Dual_Stream(nn.Module):
    """Stage 1 顶层接口。

    这是一个“路径列表 -> 特征张量”的端到端接口。

    Args:
        sample_rate: 目标采样率。
        n_mels: Mel 频带数。
        duration: 每条音频统一时长(秒)。
        cutoff_freq: 低/高频分界点(Hz)。
        split_mode: 分频模式, "biquad" 或 "mel_mask"。
        transition_bins: mel_mask 模式下的边界过渡 bin 数。
        stda_use_local_mask: 是否启用 STDA 局部窗口注意力掩码。
        stda_local_window_size: STDA 局部窗口大小(帧)。

    Notes:
        biquad 模式: 时域分频 -> 双路 Mel。
        mel_mask 模式: 一次全频 Mel -> 频轴软掩码分离。

    Forward Input:
        raw_audio: Tensor[B, T_samples]，批量原始波形。
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
        
        # Implementation detail.
        self.mel_trans = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            mel_scale="htk" # Implementation detail.
            )
        # Implementation detail.
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
            # Implementation detail.
            nn.Conv1d(1 + 64, 128, kernel_size=1),
            nn.BatchNorm1d(128),  # Implementation detail.
            nn.SiLU()             # Implementation detail.
        )

        # Implementation detail.
        low_mask, high_mask = self._build_mel_band_masks(n_mels, sample_rate, cutoff_freq, self.transition_bins)
        self.register_buffer("low_mel_mask", low_mask.view(1, n_mels, 1))
        self.register_buffer("high_mel_mask", high_mask.view(1, n_mels, 1))

    def _build_mel_band_masks(self, n_mels, sample_rate, cutoff_freq, transition_bins):
        """构造低频/高频软掩码，掩码沿 mel 频率轴变化。"""
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
            # Implementation detail.
            width = max(1, end - start)
            for i in range(start, end + 1):
                t = (i - start) / width
                low_mask[i] = 1.0 - t
                high_mask[i] = t

        return low_mask, high_mask

    def _split_to_mels(self, raw_audio):
        """统一分频入口。

        Args:
            raw_audio: 原始波形张量，shape [B, T_samples]。

        Returns:
            low_mel: 低频输入给 STDA，shape [B, F, T]
            high_mel: 高频输入给 MFCA，shape [B, 1, F, T]

        Notes:
            - biquad: 时域低/高通后分别提 Mel。
            - mel_mask: 一次全频 Mel 后按频轴掩码分离。
        """
        if self.split_mode == "biquad":
            # Implementation detail.
            low_audio = torchaudio.functional.lowpass_biquad(raw_audio, self.sr, self.cutoff_freq)
            high_audio = torchaudio.functional.highpass_biquad(raw_audio, self.sr, self.cutoff_freq)
            low_mel = self.mel_trans(low_audio)
            high_mel = self.mel_trans(high_audio).unsqueeze(1)
            return low_mel, high_mel

        # Implementation detail.
        full_mel = self.mel_trans(raw_audio)
        low_mel = full_mel * self.low_mel_mask
        high_mel = (full_mel * self.high_mel_mask).unsqueeze(1)
        return low_mel, high_mel

    

    def forward(self, raw_audio):
        """
        Stage1 主入口：从原始波形张量直接得到融合时频特征。

        Args:
            raw_audio: Tensor[B, T_samples]，待处理的原始波形。

        Returns:
            out: [B, 128, T_frames]，供后续模块继续处理。
        """
        if raw_audio is None or (not torch.is_tensor(raw_audio)):
            raise ValueError("raw_audio 不能为空且必须是 Tensor")
        if raw_audio.ndim != 2:
            raise ValueError(f"raw_audio 期望形状 [B, T]，实际为 {tuple(raw_audio.shape)}")
        if raw_audio.size(0) == 0 or raw_audio.size(1) == 0:
            raise ValueError("raw_audio 的 batch 维或时间维为空")
        
        # Implementation detail.
        low_mel, high_mel = self._split_to_mels(raw_audio)
        
        # Implementation detail.
        f_mfca = self.mfca_branch(high_mel)
        f_stda = self.stda_branch(low_mel)
        
        # Implementation detail.
        # f_mfca: [B, 1, T], f_stda: [B, 64, T]
        stage1_matrix = torch.cat([f_mfca, f_stda], dim=1)
        out = self.final_compression(stage1_matrix)
        
        return out
    
'''
    该部分未更新，请先忽略
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
