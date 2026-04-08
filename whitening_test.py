import torch
import mne
import numpy as np
import scipy.signal

# 假设你的 eeg_cuda 已经绑定了 run_whitening
import eeg_cuda

# ==========================================
# 1. 数据加载与预处理
# ==========================================
input_dir = r"E:\bizq\脑电分析\EEG测试数据\Copy_20260130175721005\赵婷\赵婷_20260130_173432\1\1\data.bdf"
raw_data = mne.io.read_raw_bdf(input_dir, preload=True)
data = raw_data.get_data().astype(np.float32)

eeg_tensor = torch.from_numpy(data).unsqueeze(0).cuda()
eeg_tensor = eeg_tensor.expand(10, -1, -1)

# ==========================================
# 2. CUDA 加速计算流水线
# ==========================================
print("正在执行 CUDA 算子流水线...")
fs = raw_data.info['sfreq']
weights_np = scipy.signal.firwin(256, [1, 40], pass_zero=False, fs=fs).astype(np.float32)
weights_tensor = torch.from_numpy(weights_np).cuda()

# 1. 滤波 -> 2. 中心化
filtered_eeg = eeg_cuda.fir_filter(eeg_tensor, weights_tensor)
centered_eeg = eeg_cuda.centering(filtered_eeg)

# 3. 执行白化 (Whitening)
print("白化输入数据精度：", centered_eeg.dtype)
whitened_eeg, p = eeg_cuda.whitening(centered_eeg)
print("白化输出数据精度：", whitened_eeg.dtype)

# ==========================================
# 3. 基准计算与变量隔离
# ==========================================
print("正在执行严格验证...")

# 这保证了 NumPy 裁判的计算精度与我们 C++ 内部的计算精度完全在同一级别
X_tensor = centered_eeg[0]
X_np = X_tensor.cpu().numpy().astype(np.float64)

n_channels, n_times = X_np.shape

# NumPy 实现 ZCA 白化基准 (此时所有计算都在极高精度的 Float64 下进行)
cov_np = np.dot(X_np, X_np.T) / (n_times - 1)
L, V = np.linalg.eigh(cov_np)
max_eig = np.max(np.abs(L))
eps = max_eig * 1e-6
D_inv_sqrt = 1.0 / np.sqrt(np.clip(np.abs(L), a_min=eps, a_max=None))

P_np = V @ np.diag(D_inv_sqrt) @ V.T
expected_white = np.dot(P_np, X_np)

# ==========================================
# 纯净对齐验证
# ==========================================
print("\n" + "="*40)
print("       CUDA 白化算子对齐验证")
print("="*40)

# 提取 CUDA 白化后的数据 (Float32)
cuda_out_tensor = whitened_eeg[0]
cuda_white_np = cuda_out_tensor.cpu().numpy()

# 在 GPU 上验证协方差时，必须临时升到 Float64！
# 否则 PyTorch 会默认启动 TF32 硬件加速
cuda_out_f64 = cuda_out_tensor.to(torch.float64)
cuda_cov_tensor = (cuda_out_f64 @ cuda_out_f64.T) / (n_times - 1)
cuda_cov_np = cuda_cov_tensor.cpu().numpy()

# 计算 NumPy 预期输出的真实协方差
expected_cov = np.dot(expected_white, expected_white.T) / (n_times - 1)

# --- 验证 A: 协方差矩阵对齐度 ---
cov_diff = np.abs(cuda_cov_np - expected_cov)
max_cov_diff = np.max(cov_diff)
mean_cov_diff = np.mean(cov_diff)

print(f"[验证 A] 协方差矩阵最大偏差: {max_cov_diff:.8e}")
print(f"         协方差矩阵平均偏差: {mean_cov_diff:.8e}")

if max_cov_diff < 1e-4:
    print("  ✅ 成功：CUDA 的统计分布与 NumPy 在数学上完美等价。")
else:
    print("  ❌ 失败：协方差分布不一致。")

# --- 验证 B: 输出数据矩阵对齐度 ---
diff = np.abs(cuda_white_np - expected_white)
max_val_error = np.max(diff)
mean_val_error = np.mean(diff)

print(f"\n[验证 B] 数据点最大绝对误差: {max_val_error:.8e}")
print(f"         数据点平均绝对误差: {mean_val_error:.8e}")

print("\n" + "-"*40)
if max_val_error < 1e-4:
    print("🎉 终极结论: 恭喜！你的 CUDA 算子与理论数学模型完美对齐，毫无破绽！")
    print("下一步：请放心地把它接入到 ICA 模块中去吧！")
else:
    print("⚠️ 结论: 算法仍需排查。")
print("-" * 40 + "\n")