import numpy as np
import torch
from scipy import signal
import mne
import eeg_cuda

# ==================== FIR 函数（保持不变） ====================
def create_fir_weights(sfreq: float, low_freq: float = 1.0, high_freq: float = 40.0):
    weights_np = signal.firwin(
        256, [low_freq, high_freq], pass_zero=False, fs=sfreq, window='hamming'
    ).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.from_numpy(weights_np).to(device)
    return weights, device


def apply_fir_filter_cuda(data: np.ndarray, weights: torch.Tensor, device: torch.device):
    if data.ndim != 2:
        raise ValueError(f"数据维度错误，期望 (n_channels, n_times)，实际为 {data.shape}")
    x = torch.from_numpy(data).float().unsqueeze(0).to(device)
    filtered = eeg_cuda.fir_filter(x, weights)
    if filtered.ndim == 3:
        filtered = filtered.squeeze(0)
    return filtered.cpu().numpy()


# ==================== IIR 函数（支持 matrix 和 norm 两种版本） ====================
def apply_iir_filter_cuda(data: np.ndarray, sfre_tensor: torch.Tensor,
                         cfre_tensor: torch.Tensor, pass_type: str,
                         bandstop: bool = False, version: str = "norm"):
    """支持 norm 和 matrix 两种 IIR 版本"""
    if data.ndim != 2:
        raise ValueError(f"数据维度错误，期望 (n_channels, n_times)，实际为 {data.shape}")

    eeg_tensor = torch.from_numpy(data).float().unsqueeze(0).to(sfre_tensor.device)

    if version == "norm":
        filtered = eeg_cuda.iir_filtfilt_norm(
            eeg_tensor, sfre_tensor, cfre_tensor, pass_type, bandstop
        )
    elif version == "matrix":
        filtered = eeg_cuda.iir_filtfilt_matrix(
            eeg_tensor, sfre_tensor, cfre_tensor, pass_type, bandstop
        )
    else:
        raise ValueError(f"不支持的 IIR 版本: {version}，请使用 'norm' 或 'matrix'")

    if filtered.ndim == 3:
        filtered = filtered.squeeze(0)

    return filtered.cpu().numpy()


# ==================== FastICA 函数 ====================
def apply_fastica_cuda(data: np.ndarray, max_iter: int = 200, tol: float = 1e-4):
    if data.ndim != 2:
        raise ValueError(f"数据维度错误，期望 (n_channels, n_times)，实际为 {data.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(data).float().unsqueeze(0).to(device)

    # 1. 去中心化
    x_centered = eeg_cuda.centering(x)

    # 2. ZCA 白化
    x_whitened, w_whiten = eeg_cuda.whitening(x_centered)

    # 3. ICA 迭代分析 (加入 max_iter 和 tol)
    ica_signals, w_ica = eeg_cuda.fastica_iter(x_whitened, max_iter=max_iter, tol=tol)

    # 降维处理
    ica_signals_np = ica_signals.squeeze(0).cpu().numpy() if ica_signals.ndim == 3 else ica_signals.cpu().numpy()
    w_whiten_np = w_whiten.squeeze(0).cpu().numpy() if w_whiten.ndim == 3 else w_whiten.cpu().numpy()
    w_ica_np = w_ica.squeeze(0).cpu().numpy() if w_ica.ndim == 3 else w_ica.cpu().numpy()

    return ica_signals_np, w_whiten_np, w_ica_np

# ==================== 公共函数 ====================
def get_data_segment(raw: mne.io.BaseRaw, picks: list, start_sec: float, duration_sec: float):
    sfreq = raw.info['sfreq']
    start_sample = int(round(start_sec * sfreq))
    n_samples = int(round(duration_sec * sfreq))
    max_samples = len(raw.times) - start_sample
    n_samples = min(n_samples, max(1, max_samples))
    data, times = raw[picks, start_sample : start_sample + n_samples]
    return data, times


def create_filtered_raw(raw: mne.io.BaseRaw, filtered_data: np.ndarray, picks: list):
    info = mne.create_info(
        ch_names=[raw.ch_names[i] for i in range(len(raw.ch_names)) if raw.ch_names[i] in picks],
        sfreq=raw.info['sfreq'],
        ch_types=['eeg'] * len(picks)
    )
    if raw.info.get('dig') is not None:
        info['dig'] = raw.info['dig']
    return mne.io.RawArray(filtered_data, info, verbose=False)