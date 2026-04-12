import torch
import mne
import numpy as np
import scipy.signal
import math
import os

import eeg_cuda


# ==========================================
# 0. 辅助函数：生成与 C++ 端完全一致的 a, b 系数
# ==========================================
def get_biquad_coeffs(sfre, cfre, pass_type="high", bandstop=False):
    """Python 端计算 IIR 滤波器系数，用于 SciPy 基准对齐"""

    def Omega(sf, cf):
        return math.tan(math.pi * cf / sf)

    if isinstance(cfre, (list, tuple, np.ndarray)) and len(cfre) == 2:
        f1, f2 = min(cfre), max(cfre)
        low_omega = Omega(sfre, f1)
        high_omega = Omega(sfre, f2)
        BW = high_omega - low_omega
        omega_2 = low_omega * high_omega
        a0_inv = 1.0 / (1.0 + BW + omega_2)

        a = [1.0, 2.0 * (omega_2 - 1.0) * a0_inv, (1.0 - BW + omega_2) * a0_inv]
        if bandstop:
            b = [BW * a0_inv, 0.0, -BW * a0_inv]
        else:
            b = [(1.0 + omega_2) * a0_inv, 2.0 * (omega_2 - 1.0) * a0_inv, (1.0 + omega_2) * a0_inv]
    else:
        fc = cfre[0] if isinstance(cfre, (list, tuple, np.ndarray)) else cfre
        omega = Omega(sfre, fc)
        sqrt2 = math.sqrt(2.0)
        omega_sq = omega * omega
        a0_inv = 1.0 / (1.0 + sqrt2 * omega + omega_sq)

        a = [1.0, 2.0 * (omega_sq - 1.0) * a0_inv, (1.0 - sqrt2 * omega + omega_sq) * a0_inv]
        if pass_type == "low":
            b = [omega_sq * a0_inv, 2.0 * omega_sq * a0_inv, omega_sq * a0_inv]
        elif pass_type == "high":
            b = [1.0 * a0_inv, -2.0 * a0_inv, 1.0 * a0_inv]

    return np.array(b, dtype=np.float32), np.array(a, dtype=np.float32)


# ==========================================
# 1. 数据加载与预处理
# ==========================================
input_dir = r""
print(f"正在加载数据: {input_dir}")

if not os.path.exists(input_dir) and input_dir != "":
    raise FileNotFoundError(f"找不到数据文件，请检查路径: {input_dir}")
elif input_dir == "":
    # 如果没填路径，生成随机 EEG 测试数据以供快速验证
    print("未提供路径，使用随机合成数据测试...")
    data = np.random.randn(64, 1000).astype(np.float32)
    fs = 250
else:
    raw_data = mne.io.read_raw_bdf(input_dir, preload=True)
    data = raw_data.get_data().astype(np.float32)  # (C, T)
    fs = raw_data.info['sfreq']

eeg_tensor = torch.from_numpy(data).unsqueeze(0).cuda()  # (1, C, T)
BATCH_SIZE = 10
eeg_tensor = eeg_tensor.expand(BATCH_SIZE, -1, -1)  # (B, C, T) 测试 Batch

print(f"输入形状: {eeg_tensor.shape} (Batch × Channels × Time)")

# ==========================================
# 2. 滤波器参数配置
# ==========================================
print(f"采样率: {fs} Hz")

# 测试配置：1Hz 高通滤波器
cfre_val = [1]
pass_type = "high"
bandstop = False

sfre_tensor = torch.tensor(int(fs), dtype=torch.int32)
cfre_tensor = torch.tensor(cfre_val, dtype=torch.float32)

# 获取对照系数
b_np, a_np = get_biquad_coeffs(fs, cfre_val, pass_type, bandstop)
print(f"IIR 系数: b={b_np}, a={a_np}")

# ==========================================
# 3. CUDA IIR filtfilt 执行
# ==========================================
print("\nExecuting CUDA IIR filtfilt...")
with torch.no_grad():
    cuda_filtered = eeg_cuda.iir_filtfilt(
        eeg_tensor,
        sfre_tensor,
        cfre_tensor,
        pass_type,
        bandstop
    )

print(f"CUDA 输出形状: {cuda_filtered.shape}, dtype: {cuda_filtered.dtype}")

# ==========================================
# 4. SciPy 基准测试 (严格零初始状态模拟)
# ==========================================
print("\nComputing SciPy exact equivalent baseline (Forward-Reverse-Forward-Reverse)...")
eeg_np = eeg_tensor.cpu().numpy()  # (B, C, T)
scipy_strict_list = []

# 为了验证 CUDA 数学的绝对正确性，我们不用带边界延拓的 filtfilt
# 而是手动实现双向 lfilter 零初始状态
for b_idx in range(eeg_np.shape[0]):
    # 1. 正向
    y_fw = scipy.signal.lfilter(b_np, a_np, eeg_np[b_idx], axis=-1)
    # 2. 翻转
    y_fw_rev = np.flip(y_fw, axis=-1)
    # 3. 反向
    y_bw = scipy.signal.lfilter(b_np, a_np, y_fw_rev, axis=-1)
    # 4. 再次翻转恢复
    y_final = np.flip(y_bw, axis=-1)
    scipy_strict_list.append(y_final)

scipy_strict_filtered_np = np.stack(scipy_strict_list, axis=0)

# ==========================================
# 5. 标准 SciPy filtfilt 参考对比 (带 Padding)
# ==========================================
print("Computing Standard SciPy filtfilt (with padding/zi for reference)...")
# 注意：这与 CUDA 的结果会有边界差异，仅作为合理性参考
scipy_std_list = []
for b_idx in range(eeg_np.shape[0]):
    # scipy 默认使用 odd padding
    y_std = scipy.signal.filtfilt(b_np, a_np, eeg_np[b_idx], axis=-1)
    scipy_std_list.append(y_std)

scipy_std_filtered_np = np.stack(scipy_std_list, axis=0)

# ==========================================
# 6. 数值验证报告（全 Batch）
# ==========================================
print("\n" + "=" * 70)
print("       IIR Zero-Phase Filter Numerical Verification Report")
print("=" * 70)

cuda_np = cuda_filtered.cpu().numpy()

# --- 与严格等效基准对比 (核心测试) ---
diff_strict = np.abs(cuda_np - scipy_strict_filtered_np)
max_abs_strict = np.max(diff_strict)
mean_abs_strict = np.mean(diff_strict)

# --- 与标准 filtfilt 对比 (仅参考) ---
diff_std = np.abs(cuda_np - scipy_std_filtered_np)
max_abs_std = np.max(diff_std)
mean_abs_std = np.mean(diff_std)

print(f"[Vs Exact Scipy LFilter Math] Max Abs Error: {max_abs_strict:.8e} | Mean Abs: {mean_abs_strict:.8e}")
print(f"[Vs Standard Scipy filtfilt ] Max Abs Error: {max_abs_std:.8e} | Mean Abs: {mean_abs_std:.8e}")
print("  *(Note: Standard filtfilt uses edge padding, so boundary deviation is mathematically expected)*")

# 阈值判断：IIR因为有自回归累加，FP32精度下误差会比FIR大一点，通常1e-4到1e-5算通过
threshold_abs = 1e-4

if max_abs_strict < threshold_abs:
    print("\n✅ PASS: CUDA IIR matrix scan is mathematically equivalent to SciPy lfilter zero-state forward/backward.")
else:
    print("\n⚠️  FAIL: Significant deviation detected in matrix multiplication or scan operation.")

print(f"\nExtra Check - CUDA output range: [{cuda_np.min():.4f}, {cuda_np.max():.4f}]")

# ==========================================
# 7. 保存验证报告
# ==========================================
report_dir = r"E:\bizq\eeg_cuda_ops_history\eeg_cuda_ops\verification"
report_path = os.path.join(report_dir, "iir_filtfilt_verification.log")

try:
    os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("EEG CUDA IIR filtfilt Verification Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Data shape: Batch {BATCH_SIZE} × Channels {eeg_tensor.shape[1]} × Time {eeg_tensor.shape[2]}\n")
        f.write(f"Filter type: {pass_type}, Cutoff: {cfre_val}Hz\n\n")

        f.write(f"[Vs Exact Equivalent] Max Abs Error: {max_abs_strict:.8e}\n")
        f.write(f"[Vs Exact Equivalent] Mean Abs Error: {mean_abs_strict:.8e}\n\n")

        f.write(f"[Vs Standard filtfilt] Max Abs Error (Boundary Effects Expected): {max_abs_std:.8e}\n\n")

        f.write(f"Conclusion: {'PASS' if max_abs_strict < threshold_abs else 'FAIL'}\n")

    print(f"\n✅ Verification report saved to: {report_path}")
except Exception as e:
    print(f"\n无法保存日志文件，可能路径不存在。错误信息: {e}")