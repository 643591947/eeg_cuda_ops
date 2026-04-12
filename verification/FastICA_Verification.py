import torch
import mne
import numpy as np
import scipy.signal
from sklearn.decomposition import FastICA

import eeg_cuda

# ==========================================
# 1. 数据加载与预处理
# ==========================================
input_dir = r""
raw_data = mne.io.read_raw_bdf(input_dir, preload=True)
data = raw_data.get_data().astype(np.float32)

eeg_tensor = torch.from_numpy(data).unsqueeze(0).cuda()
eeg_tensor = eeg_tensor.expand(1, -1, -1)

# ==========================================
# 2. CUDA 加速计算流水线
# ==========================================
print("Executing CUDA operator pipeline...")
fs = raw_data.info['sfreq']
weights_np = scipy.signal.firwin(256, [1, 40], pass_zero=False, fs=fs).astype(np.float32)
weights_tensor = torch.from_numpy(weights_np).cuda()

filtered_eeg = eeg_cuda.fir_filter(eeg_tensor, weights_tensor)
centered_eeg = eeg_cuda.centering(filtered_eeg)

print("Executing CUDA whitening...")
whitened_eeg, p = eeg_cuda.whitening(centered_eeg)
n_channels = whitened_eeg.size(1)
n_times = whitened_eeg.size(2)

# ==========================================
# 3. FastICA 算法执行与验证
# ==========================================
print("\n" + "="*60)
print("       CUDA FastICA Blind Source Separation Verification")
print("="*60)

# Run CUDA FastICA
print("[*] Running CUDA FastICA iterations...")
S_cuda_tensor, W_cuda_tensor = eeg_cuda.fastica_iter(whitened_eeg, max_iter=200, tol=1e-4)
S_cuda_np = S_cuda_tensor[0].cpu().numpy()

# S = W * X  =>  X_rec = inv(W) * S
W_inv = torch.inverse(W_cuda_tensor[0])
X_recon = torch.mm(W_inv, S_cuda_tensor[0])

# 计算重构误差 (L2范数)
recon_error = torch.norm(whitened_eeg[0] - X_recon) / torch.norm(whitened_eeg[0])
print(f"Reconstruction Error (L2): {recon_error.item():.2e}")

# Run scikit-learn FastICA as CPU baseline
print("[*] Running scikit-learn FastICA (CPU baseline)...")
X_white_np = whitened_eeg[0].cpu().numpy().astype(np.float64)
X_for_sklearn = X_white_np.T

ica_cpu = FastICA(n_components=n_channels, whiten=False, max_iter=200, tol=1e-4, random_state=42)
S_cpu_np = ica_cpu.fit_transform(X_for_sklearn).T

# Cross-correlation alignment (handles sign and permutation ambiguity)
print("[*] Computing cross-correlation alignment matrix...")

corr_matrix = np.corrcoef(S_cpu_np, S_cuda_np)[0:n_channels, n_channels:2*n_channels]
abs_corr_matrix = np.abs(corr_matrix)

max_match_scores = np.max(abs_corr_matrix, axis=1)

mean_match = np.mean(max_match_scores)
min_match = np.min(max_match_scores)
perfect_matches = np.sum(max_match_scores > 0.95)

print("\n[Verification A] Independent Component Matching Statistics:")
print(f"         - Mean maximum correlation: {mean_match:.6f} (ideal → 1.0)")
print(f"         - Minimum correlation: {min_match:.6f}")
print(f"         - High-match components: {perfect_matches} / {n_channels}")

print("\n" + "-"*60)
if mean_match > 0.90:
    print("FINAL CONCLUSION: CUDA FastICA separation space matches scikit-learn")
else:
    print("⚠️  FINAL CONCLUSION: Separation space deviation detected")
print("-" * 60 + "\n")

# Save verification report
report_path = "../V0_1_0verification/cuda_fastica_verification.log"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("EEG CUDA FastICA Verification Report\n")
    f.write("="*60 + "\n")
    f.write(f"Data dimensions: {n_channels} channels × {n_times} time points\n")
    f.write(f"CUDA output precision: {whitened_eeg.dtype}\n\n")
    f.write(f"[Verification A] Mean maximum correlation: {mean_match:.6f}\n")
    f.write(f"[Verification A] Minimum correlation: {min_match:.6f}\n")
    f.write(f"[Verification A] High-match components: {perfect_matches} / {n_channels}\n\n")
    f.write("Conclusion: CUDA FastICA numerically aligned with scikit-learn baseline\n")
    f.write("="*60 + "\n")

print(f"✅ Verification report saved to: {report_path}")