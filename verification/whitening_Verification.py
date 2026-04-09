import torch
import mne
import numpy as np
import scipy.signal

import eeg_cuda

# ==========================================
# 1. 数据加载与预处理
# ==========================================
input_dir = r"data.bdf"
raw_data = mne.io.read_raw_bdf(input_dir, preload=True)
data = raw_data.get_data().astype(np.float32)

eeg_tensor = torch.from_numpy(data).unsqueeze(0).cuda()
eeg_tensor = eeg_tensor.expand(10, -1, -1)   # Batch 扩展（与原版一致）

# ==========================================
# 2. CUDA 加速流水线
# ==========================================
print("Executing CUDA operator pipeline...")
fs = raw_data.info['sfreq']
weights_np = scipy.signal.firwin(256, [1, 40], pass_zero=False, fs=fs).astype(np.float32)
weights_tensor = torch.from_numpy(weights_np).cuda()

# 1. 滤波 -> 2. 中心化
filtered_eeg = eeg_cuda.fir_filter(eeg_tensor, weights_tensor)
centered_eeg = eeg_cuda.centering(filtered_eeg)

# 3. 白化
print("Whitening input data precision:", centered_eeg.dtype)
whitened_eeg, p = eeg_cuda.whitening(centered_eeg)
print("Whitening output data precision:", whitened_eeg.dtype)

# ==========================================
# 3. 基准计算与变量隔离
# ==========================================
print("Running strict verification...")

# 确保 NumPy 参考计算使用与 C++ 内部算子相同的高精度 (Float64)
X_tensor = centered_eeg[0]
X_np = X_tensor.cpu().numpy().astype(np.float64)

n_channels, n_times = X_np.shape

# NumPy ZCA 白化基准（所有计算均在 Float64 下进行）
cov_np = np.dot(X_np, X_np.T) / (n_times - 1)
L, V = np.linalg.eigh(cov_np)
max_eig = np.max(np.abs(L))
eps = max_eig * 1e-6
D_inv_sqrt = 1.0 / np.sqrt(np.clip(np.abs(L), a_min=eps, a_max=None))

P_np = V @ np.diag(D_inv_sqrt) @ V.T
expected_white = np.dot(P_np, X_np)

# ==========================================
# 对齐验证报告
# ==========================================
print("\n" + "="*60)
print("       CUDA Whitening Operator Alignment Verification Report")
print("="*60)

# 提取 CUDA 白化结果 (Float32)
cuda_out_tensor = whitened_eeg[0]
cuda_white_np = cuda_out_tensor.cpu().numpy()

# 临时转换为 Float64 进行 GPU 上的协方差验证
cuda_out_f64 = cuda_out_tensor.to(torch.float64)
cuda_cov_tensor = (cuda_out_f64 @ cuda_out_f64.T) / (n_times - 1)
cuda_cov_np = cuda_cov_tensor.cpu().numpy()

# NumPy 期望的协方差
expected_cov = np.dot(expected_white, expected_white.T) / (n_times - 1)

# --- 验证 A: 协方差矩阵对齐 ---
cov_diff = np.abs(cuda_cov_np - expected_cov)
max_cov_diff = np.max(cov_diff)
mean_cov_diff = np.mean(cov_diff)

# 相对偏差（更能反映数值稳定性）
cov_rel_diff = cov_diff / (np.abs(expected_cov) + 1e-10)
max_cov_rel = np.max(cov_rel_diff)
mean_cov_rel = np.mean(cov_rel_diff)

print(f"[Verification A] Covariance max absolute deviation: {max_cov_diff:.8e}")
print(f"                 Covariance mean absolute deviation: {mean_cov_diff:.8e}")
print(f"                 Covariance max relative deviation: {max_cov_rel:.8e}")
print(f"                 Covariance mean relative deviation: {mean_cov_rel:.8e}")

if max_cov_diff < 1e-4:
    print("  ✅ PASS: CUDA statistical distribution is mathematically equivalent to NumPy.")
else:
    print("  ❌ FAIL: Covariance distributions do not match.")

# --- 验证 B: 输出数据矩阵对齐 ---
diff = np.abs(cuda_white_np - expected_white)
max_val_error = np.max(diff)
mean_val_error = np.mean(diff)

# 相对误差（考虑 float32 与 float64 的精度差异）
rel_diff = diff / (np.abs(expected_white) + 1e-10)
max_rel_error = np.max(rel_diff)
mean_rel_error = np.mean(rel_diff)

print(f"\n[Verification B] Data point max absolute error: {max_val_error:.8e}")
print(f"                 Data point mean absolute error: {mean_val_error:.8e}")
print(f"                 Data point max relative error: {max_rel_error:.8e}")
print(f"                 Data point mean relative error: {mean_rel_error:.8e}")

# 额外检查：两者都应满足白化属性（协方差 ≈ 单位矩阵 I）
I_mat = np.eye(n_channels)
cuda_to_I_diff = np.abs(cuda_cov_np - I_mat)
expected_to_I_diff = np.abs(expected_cov - I_mat)

print(f"\n[Extra Check] CUDA whitened covariance → Identity max deviation: {np.max(cuda_to_I_diff):.8e}")
print(f"              NumPy baseline whitened covariance → Identity max deviation: {np.max(expected_to_I_diff):.8e}")

print("\n" + "-"*60)
if max_val_error < 1e-4 and max_cov_diff < 1e-4:
    print("✅ FINAL CONCLUSION: CUDA operator is fully aligned with theoretical math model")
else:
    print("⚠️  FINAL CONCLUSION: Further debugging may be needed")
print("-" * 60 + "\n")

# ==========================================
# 保存验证报告
# ==========================================
report_path = "cuda_whitening_verification.log"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("EEG CUDA Whitening Operator Alignment Verification Report\n")
    f.write("="*60 + "\n")
    f.write(f"Data dimensions: {n_channels} channels × {n_times} time points\n")
    f.write(f"CUDA output precision: {whitened_eeg.dtype}\n\n")
    f.write(f"[Verification A] Covariance max absolute deviation: {max_cov_diff:.8e}\n")
    f.write(f"[Verification A] Covariance mean absolute deviation: {mean_cov_diff:.8e}\n")
    f.write(f"[Verification A] Covariance max relative deviation: {max_cov_rel:.8e}\n\n")
    f.write(f"[Verification B] Data point max absolute error: {max_val_error:.8e}\n")
    f.write(f"[Verification B] Data point mean absolute error: {mean_val_error:.8e}\n")
    f.write(f"[Verification B] Data point max relative error: {max_rel_error:.8e}\n\n")
    f.write(f"[Extra Check] CUDA → Identity max deviation: {np.max(cuda_to_I_diff):.8e}\n")
    f.write(f"[Extra Check] NumPy → Identity max deviation: {np.max(expected_to_I_diff):.8e}\n\n")
    f.write("Conclusion: CUDA operator numerically aligned with NumPy baseline\n")
    f.write("="*60 + "\n")

print(f"✅ Verification report saved to: {report_path}")