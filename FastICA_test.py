import torch
import mne
import numpy as np
import scipy.signal
from sklearn.decomposition import FastICA

# 假设你的 eeg_cuda 已经绑定了 run_whitening 和 run_fastica_iter
import eeg_cuda

# ==========================================
# 1. 数据加载与预处理 (保持原样)
# ==========================================
input_dir = r"E:\bizq\脑电分析\EEG测试数据\Copy_20260130175721005\赵婷\赵婷_20260130_173432\1\1\data.bdf"
raw_data = mne.io.read_raw_bdf(input_dir, preload=True)
data = raw_data.get_data().astype(np.float32)

eeg_tensor = torch.from_numpy(data).unsqueeze(0).cuda()
# 注意：为了让 ICA 收敛更稳定，测试阶段建议只用单 batch 或真实时长的维度
eeg_tensor = eeg_tensor.expand(1, -1, -1)

# ==========================================
# 2. CUDA 加速计算流水线 (前置)
# ==========================================
print("正在执行 CUDA 算子流水线...")
fs = raw_data.info['sfreq']
weights_np = scipy.signal.firwin(256, [1, 40], pass_zero=False, fs=fs).astype(np.float32)
weights_tensor = torch.from_numpy(weights_np).cuda()

filtered_eeg = eeg_cuda.fir_filter(eeg_tensor, weights_tensor)
centered_eeg = eeg_cuda.centering(filtered_eeg)

print("正在执行 CUDA 白化...")
whitened_eeg, p = eeg_cuda.whitening(centered_eeg)
n_channels = whitened_eeg.size(1)
n_times = whitened_eeg.size(2)

# ==========================================
# 3. FastICA 算法执行与验证
# ==========================================
print("\n" + "="*40)
print("       CUDA FastICA 算子盲源分离验证")
print("="*40)

# --- 步骤 1: 运行你的 CUDA FastICA ---
print("[*] 正在执行 CUDA FastICA 迭代...")
# 输出 S_cuda_tensor 形状为 [Batch=1, Channels, Time]
S_cuda_tensor, W_cuda_tensor = eeg_cuda.fastica_iter(whitened_eeg, max_iter=200, tol=1e-4)
S_cuda_np = S_cuda_tensor[0].cpu().numpy() # 提取 numpy 数组 [Channels, Time]

# --- 步骤 2: 运行 Scikit-Learn FastICA 作为 CPU 基准 ---
print("[*] 正在执行 Scikit-Learn FastICA (CPU 基准)...")
# 提取白化后的真实数据喂给 sklearn
X_white_np = whitened_eeg[0].cpu().numpy().astype(np.float64)

# sklearn 的输入要求是 [n_samples, n_features]，我们需要转置
X_for_sklearn = X_white_np.T

# 核心设置：关闭 sklearn 的自带白化 (whiten=False)，因为我们已经严格白化过了
ica_cpu = FastICA(n_components=n_channels, whiten=False, max_iter=200, tol=1e-4, random_state=42)
# 拟合出独立成分并转回 [Channels, Time]
S_cpu_np = ica_cpu.fit_transform(X_for_sklearn).T

# --- 步骤 3: 互相关对齐验证 (解决符号与排列模糊性) ---
print("[*] 正在计算跨设备互相关矩阵对齐度...")

# 计算 CUDA 结果和 CPU 结果的皮尔逊相关系数矩阵
# np.corrcoef 会把两个矩阵拼起来算，我们只需要右上角的 [n_channels, n_channels] 交叉块
corr_matrix = np.corrcoef(S_cpu_np, S_cuda_np)[0:n_channels, n_channels:2*n_channels]

# 因为 ICA 允许正负号翻转，我们取绝对值
abs_corr_matrix = np.abs(corr_matrix)

# 寻找每个 CPU 提取的独立成分，在 CUDA 提取的成分中的最大匹配度
# 理论上，每一个 CPU 成分都应该在 CUDA 里找到一个相关度接近 1.0 的“灵魂伴侣”
max_match_scores = np.max(abs_corr_matrix, axis=1)

# 统计指标
mean_match = np.mean(max_match_scores)
min_match = np.min(max_match_scores)
perfect_matches = np.sum(max_match_scores > 0.95)

print("\n[验证 A] 独立成分匹配度统计:")
print(f"         - 平均最大相关系数: {mean_match:.6f} (理想趋近 1.0)")
print(f"         - 最低匹配相关系数: {min_match:.6f}")
print(f"         - 成功高匹配成分数: {perfect_matches} / {n_channels} 个")

print("\n" + "-"*40)
# 只要平均匹配度极高，说明核心寻优逻辑等价
if mean_match > 0.90:
    print("🎉 结论: 恭喜！CUDA FastICA 的分离空间与 Scikit-Learn 高度一致！")
    print("提示: 少数成分未完全达到 0.99 属于正常现象，这是由于不同硬件的随机初始化点和迭代收敛路径不同导致的物理不可避免差异。")
else:
    print("⚠️ 结论: 分离空间偏差较大，请检查 g(u) 导数计算或迭代正交化逻辑。")
print("-" * 40 + "\n")