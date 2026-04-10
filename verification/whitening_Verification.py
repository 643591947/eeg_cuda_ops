import torch
import mne
import numpy as np
import scipy.signal
import os

import eeg_cuda

# ==========================================
# 1. 数据加载与预处理
# ==========================================
input_dir = r"data.bdf"
print(f"正在加载数据: {input_dir}")

# 【修改点 1】增加防御性路径校验
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"找不到数据文件，请检查盘符或路径: {input_dir}")

raw_data = mne.io.read_raw_bdf(input_dir, preload=True)
data = raw_data.get_data().astype(np.float32)

eeg_tensor = torch.from_numpy(data).unsqueeze(0).cuda()
BATCH_SIZE = 10
eeg_tensor = eeg_tensor.expand(BATCH_SIZE, -1, -1) # 扩展以测试 CUDA Batch 处理

# ==========================================
# 2. CUDA 加速流水线
# ==========================================
print("\nExecuting CUDA operator pipeline...")
fs = raw_data.info['sfreq']
weights_np = scipy.signal.firwin(256, [1, 40], pass_zero=False, fs=fs).astype(np.float32)
weights_tensor = torch.from_numpy(weights_np).cuda()

# 【修改点 2】加入 no_grad() 避免显存浪费和反向图构建开销
with torch.no_grad():
    # 1. 滤波 -> 2. 中心化
    filtered_eeg = eeg_cuda.fir_filter(eeg_tensor, weights_tensor)
    centered_eeg = eeg_cuda.centering(filtered_eeg)

    # 3. 白化
    print("Whitening input data precision:", centered_eeg.dtype)
    whitened_eeg, p_cuda = eeg_cuda.whitening(centered_eeg)
    print("Whitening output data precision:", whitened_eeg.dtype)

# ==========================================
# 3. 基准计算与变量隔离 (纯 PyTorch CUDA + 全程 Batched)
# ==========================================
print("\nRunning strict verification...")

with torch.no_grad():
    X_batch = centered_eeg.clone()
    B, C, T = X_batch.shape

    # 原生 PyTorch CUDA 白化基准 (全程 Batched + FP32)
    cov_batch = torch.bmm(X_batch, X_batch.transpose(-2, -1)) / (T - 1)
    L, V = torch.linalg.eigh(cov_batch)

    max_eig = torch.max(torch.abs(L), dim=-1, keepdim=True)[0]
    eps = max_eig * 1e-6
    D_inv_sqrt = 1.0 / torch.sqrt(torch.maximum(torch.abs(L), eps))

    P_batch = torch.bmm(V, D_inv_sqrt.unsqueeze(-1) * V.transpose(-2, -1))
    expected_white_batch = torch.bmm(P_batch, X_batch)

# ==========================================
# 4. 提取全 Batch 并进行对齐验证报告计算
# ==========================================
print("\n" + "="*60)
print("       CUDA Whitening Operator Alignment Verification Report")
print("="*60)

expected_white_np = expected_white_batch.cpu().numpy()
cuda_white_np = whitened_eeg.cpu().numpy()

# 在 FP64 下计算全 Batch 的统计协方差矩阵 [B, C, C]
cuda_out_f64 = whitened_eeg.to(torch.float64)
cuda_cov_tensor = torch.bmm(cuda_out_f64, cuda_out_f64.transpose(-2, -1)) / (T - 1)
cuda_cov_np = cuda_cov_tensor.cpu().numpy()

expected_out_f64 = expected_white_batch.to(torch.float64)
expected_cov_f64_tensor = torch.bmm(expected_out_f64, expected_out_f64.transpose(-2, -1)) / (T - 1)
expected_cov_f64_np = expected_cov_f64_tensor.cpu().numpy()

# --- 验证 A: 白化后协方差矩阵对齐 (全 Batch 计算) ---
cov_diff = np.abs(cuda_cov_np - expected_cov_f64_np)
max_cov_diff = np.max(cov_diff)
mean_cov_diff = np.mean(cov_diff)

# 相对偏差
cov_rel_diff = cov_diff / (np.abs(expected_cov_f64_np) + 1e-10)
max_cov_rel = np.max(cov_rel_diff)
mean_cov_rel = np.mean(cov_rel_diff)

print(f"[Verification A] Covariance max absolute deviation: {max_cov_diff:.8e}")
print(f"                 Covariance mean absolute deviation: {mean_cov_diff:.8e}")
print(f"                 Covariance max relative deviation: {max_cov_rel:.8e}")
print(f"                 Covariance mean relative deviation: {mean_cov_rel:.8e}")

if max_cov_diff < 1e-4:
    print("  ✅ PASS: CUDA statistical distribution is mathematically equivalent to PyTorch baseline.")
else:
    print("  ❌ FAIL: Covariance distributions do not match.")

# --- 验证 B: 输出数据点绝对对齐 (全 Batch 计算) ---
diff = np.abs(cuda_white_np - expected_white_np)
max_val_error = np.max(diff)
mean_val_error = np.mean(diff)

# 相对误差
rel_diff = diff / (np.abs(expected_white_np) + 1e-10)
max_rel_error = np.max(rel_diff)
mean_rel_error = np.mean(rel_diff)

print(f"\n[Verification B] Data point max absolute error: {max_val_error:.8e}")
print(f"                 Data point mean absolute error: {mean_val_error:.8e}")
print(f"                 Data point max relative error: {max_rel_error:.8e}")
print(f"                 Data point mean relative error: {mean_rel_error:.8e}")

# 额外检查：两者是否真正完成了白化（协方差 ≈ 单位矩阵 I）
I_mat = np.eye(C)
I_mat_batch = np.tile(I_mat, (B, 1, 1))

cuda_to_I_diff = np.abs(cuda_cov_np - I_mat_batch)
expected_to_I_diff = np.abs(expected_cov_f64_np - I_mat_batch)

print(f"\n[Extra Check] CUDA whitened covariance → Identity max deviation: {np.max(cuda_to_I_diff):.8e}")
print(f"              PyTorch baseline whitened cov → Identity max deviation: {np.max(expected_to_I_diff):.8e}")

print("\n" + "-"*60)
if max_val_error < 1e-4 and max_cov_diff < 1e-4:
    print("✅ FINAL CONCLUSION: CUDA operator is fully aligned with theoretical math model across ALL batches")
else:
    print("⚠️  FINAL CONCLUSION: Further debugging may be needed")
print("-" * 60 + "\n")

# ==========================================
# 5. 保存验证报告
# ==========================================
report_path = r"E:\bizq\eeg_cuda_ops\V0_1_0verification\cuda_whitening_verification.log"

os.makedirs(os.path.dirname(report_path), exist_ok=True)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("EEG CUDA Whitening Operator Alignment Verification Report\n")
    f.write("="*60 + "\n")
    f.write(f"Data dimensions: Batch {B} × {C} channels × {T} time points\n")
    f.write(f"CUDA output precision: {whitened_eeg.dtype}\n\n")
    f.write(f"[Verification A] Covariance max absolute deviation: {max_cov_diff:.8e}\n")
    f.write(f"[Verification A] Covariance mean absolute deviation: {mean_cov_diff:.8e}\n")
    f.write(f"[Verification A] Covariance max relative deviation: {max_cov_rel:.8e}\n\n")
    f.write(f"[Verification B] Data point max absolute error: {max_val_error:.8e}\n")
    f.write(f"[Verification B] Data point mean absolute error: {mean_val_error:.8e}\n")
    f.write(f"[Verification B] Data point max relative error: {max_rel_error:.8e}\n\n")
    f.write(f"[Extra Check] CUDA → Identity max deviation: {np.max(cuda_to_I_diff):.8e}\n")
    f.write(f"[Extra Check] PyTorch → Identity max deviation: {np.max(expected_to_I_diff):.8e}\n\n")


print(f"✅ Verification report saved to: {report_path}")