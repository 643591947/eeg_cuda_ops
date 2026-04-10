import torch
import mne
import numpy as np
import scipy.signal
import os

import eeg_cuda
import torch.nn.functional as F

# ==========================================
# 1. 数据加载与预处理
# ==========================================
input_dir = r""
print(f"正在加载数据: {input_dir}")

# 防御性路径校验
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"找不到数据文件，请检查路径: {input_dir}")

raw_data = mne.io.read_raw_bdf(input_dir, preload=True)
data = raw_data.get_data().astype(np.float32)  # (C, T)

eeg_tensor = torch.from_numpy(data).unsqueeze(0).cuda()  # (1, C, T)
BATCH_SIZE = 10
eeg_tensor = eeg_tensor.expand(BATCH_SIZE, -1, -1)  # (B, C, T) 测试 Batch

print(f"输入形状: {eeg_tensor.shape} (Batch × Channels × Time)")

# ==========================================
# 2. 生成 FIR 滤波器系数
# ==========================================
fs = raw_data.info['sfreq']
print(f"采样率: {fs} Hz")

# 设计带通滤波器（示例：1-40Hz），长度 256 taps
weights_np = scipy.signal.firwin(256, [1, 40], pass_zero=False, fs=fs, window='hamming').astype(np.float32)
weights_tensor = torch.from_numpy(weights_np).cuda()  # (numtaps,)

kernel_size = len(weights_np)
print(f"FIR 滤波器 taps: {kernel_size}")

# ==========================================
# 3. CUDA FIR 滤波执行
# ==========================================
print("\nExecuting CUDA FIR Filter...")
with torch.no_grad():
    cuda_filtered = eeg_cuda.fir_filter(eeg_tensor, weights_tensor)

print(f"CUDA 输出形状: {cuda_filtered.shape}, dtype: {cuda_filtered.dtype}")

# ==========================================
# 4. SciPy lfilter 基准（因果 FIR）
# ==========================================
print("\nComputing SciPy lfilter baseline (causal FIR)...")
with torch.no_grad():
    # lfilter 需要 numpy，逐 batch 处理（或转 numpy 后向量化）
    eeg_np = eeg_tensor.cpu().numpy()  # (B, C, T)
    b = weights_np  # (numtaps,)
    a = np.array([1.0], dtype=np.float32)  # FIR: a=[1]

    scipy_filtered_list = []
    for b_idx in range(eeg_np.shape[0]):
        filtered = scipy.signal.lfilter(b, a, eeg_np[b_idx], axis=-1)  # (C, T)
        scipy_filtered_list.append(filtered)

    scipy_filtered_np = np.stack(scipy_filtered_list, axis=0)  # (B, C, T)
    scipy_filtered = torch.from_numpy(scipy_filtered_np).cuda()

# ==========================================
# 5. PyTorch conv1d 基准（causal padding 模拟因果 FIR）
# ==========================================
print("Computing PyTorch conv1d baseline (causal padding)...")
with torch.no_grad():
    # causal padding: 在时间维度左侧填充 (kernel_size-1, 0)
    pad = (kernel_size - 1, 0)
    x_padded = F.pad(eeg_tensor, pad, mode='constant', value=0.0)  # (B, C, T + kernel_size-1)

    # weights 需要 reshape 为 conv1d 格式: (out_channels=1, in_channels=1, kernel_size)
    weights_reshaped = weights_tensor.view(1, 1, kernel_size).repeat(eeg_tensor.shape[1], 1, 1)  # (C, 1, K)

    pytorch_filtered = F.conv1d(
        x_padded,
        weight=weights_reshaped,
        groups=eeg_tensor.shape[1],
        padding=0,
        stride=1
    )  # 输出形状应与输入相同 (B, C, T)

print(f"PyTorch 输出形状: {pytorch_filtered.shape}")

# ==========================================
# 6. 数值验证报告（全 Batch）
# ==========================================
print("\n" + "=" * 70)
print("       FIR Filter Numerical Verification Report (Causal)")
print("=" * 70)

cuda_np = cuda_filtered.cpu().numpy()
scipy_np = scipy_filtered.cpu().numpy()
pytorch_np = pytorch_filtered.cpu().numpy()

# --- 与 SciPy 对比 ---
diff_scipy = np.abs(cuda_np - scipy_np)
max_abs_scipy = np.max(diff_scipy)
mean_abs_scipy = np.mean(diff_scipy)

rel_scipy = diff_scipy / (np.abs(scipy_np) + 1e-10)
max_rel_scipy = np.max(rel_scipy)
mean_rel_scipy = np.mean(rel_scipy)

# --- 与 PyTorch 对比 ---
diff_pytorch = np.abs(cuda_np - pytorch_np)
max_abs_pytorch = np.max(diff_pytorch)
mean_abs_pytorch = np.mean(diff_pytorch)

rel_pytorch = diff_pytorch / (np.abs(pytorch_np) + 1e-10)
max_rel_pytorch = np.max(rel_pytorch)
mean_rel_pytorch = np.mean(rel_pytorch)

print(f"[Vs SciPy lfilter] Max Abs Error: {max_abs_scipy:.8e} | Mean Abs: {mean_abs_scipy:.8e}")
print(f"                    Max Rel Error: {max_rel_scipy:.8e} | Mean Rel: {mean_rel_scipy:.8e}")

print(f"\n[Vs PyTorch conv1d causal] Max Abs Error: {max_abs_pytorch:.8e} | Mean Abs: {mean_abs_pytorch:.8e}")
print(f"                           Max Rel Error: {max_rel_pytorch:.8e} | Mean Rel: {mean_rel_pytorch:.8e}")

# 阈值判断（可根据实际精度调整）
threshold_abs = 1e-4
threshold_rel = 1e-3

if max_abs_scipy < threshold_abs and max_abs_pytorch < threshold_abs:
    print("\n✅ PASS: CUDA FIR is numerically equivalent to both SciPy lfilter and PyTorch causal conv1d.")
else:
    print("\n⚠️  FAIL: Significant deviation detected. Check kernel implementation, constant memory, or precision.")

# 额外检查：输出是否合理（非 NaN、无极端值）
print(f"\nExtra Check - CUDA output range: [{cuda_np.min():.4f}, {cuda_np.max():.4f}]")
print(f"              SciPy output range: [{scipy_np.min():.4f}, {scipy_np.max():.4f}]")

print("\n" + "-" * 70)
print("FINAL CONCLUSION:")
if max_abs_scipy < threshold_abs and max_abs_pytorch < threshold_abs:
    print("✅ CUDA FIR Filter is fully aligned with standard causal FIR implementations.")
else:
    print("⚠️  Further debugging recommended (e.g., kernel launch, shared memory, or float32 accumulation).")
print("-" * 70 + "\n")

# ==========================================
# 7. 保存验证报告
# ==========================================
report_dir = r"E:\bizq\eeg_cuda_ops_history\eeg_cuda_ops\verification"
report_path = os.path.join(report_dir, "fir_filter_verification.log")

os.makedirs(report_dir, exist_ok=True)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("EEG CUDA FIR Filter Verification Report (Causal)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Data shape: Batch {BATCH_SIZE} × Channels {eeg_tensor.shape[1]} × Time {eeg_tensor.shape[2]}\n")
    f.write(f"FIR taps: {kernel_size}, Cutoff: 1-40Hz\n\n")

    f.write(f"[Vs SciPy] Max Abs: {max_abs_scipy:.8e}  Mean Abs: {mean_abs_scipy:.8e}\n")
    f.write(f"[Vs SciPy] Max Rel: {max_rel_scipy:.8e}  Mean Rel: {mean_rel_scipy:.8e}\n\n")

    f.write(f"[Vs PyTorch] Max Abs: {max_abs_pytorch:.8e}  Mean Abs: {mean_abs_pytorch:.8e}\n")
    f.write(f"[Vs PyTorch] Max Rel: {max_rel_pytorch:.8e}  Mean Rel: {mean_rel_pytorch:.8e}\n\n")

    f.write(f"Conclusion: {'PASS' if max_abs_scipy < threshold_abs else 'FAIL'}\n")

print(f"✅ Verification report saved to: {report_path}")