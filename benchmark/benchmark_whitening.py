import torch
import time
import numpy as np
import pandas as pd
import eeg_cuda

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")
print(f"PyTorch 版本: {torch.__version__}\n")


def time_func(fn, *args, warmup=5, runs=20):
    """精确计时（warmup + 同步）"""
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs


def benchmark_whitening(batch, channels, time_steps):
    # 随机生成 EEG 数据
    x = torch.randn(batch, channels, time_steps, device=device, dtype=torch.float32)

    torch.cuda.synchronize()
    x_centered = eeg_cuda.centering(x)
    torch.cuda.synchronize()

    # 1. 你的 CUDA 实现
    def cuda_whitening():
        return eeg_cuda.whitening(x_centered)

    t_cuda = time_func(cuda_whitening)

    # 2. PyTorch 原生 ZCA 白化 baseline (对齐你 C++ 里的数学逻辑)
    def torch_whitening():
        cov = torch.bmm(x_centered, x_centered.transpose(-2, -1)) / (time_steps - 1)

        # 特征值分解
        eigvals, eigvecs = torch.linalg.eigh(cov)

        # ZCA scaling
        max_eig = eigvals.abs().max(dim=-1, keepdim=True)[0]
        eps = max_eig * 1e-6
        D_inv_sqrt = torch.maximum(eigvals.abs(), eps).pow(-0.5)

        # Whitening matrix W = V @ D^{-1/2} @ V^T
        W = torch.bmm(eigvecs, D_inv_sqrt.unsqueeze(-1) * eigvecs.transpose(-2, -1))

        # Apply ZCA
        return torch.bmm(W, x_centered)

    t_torch = time_func(torch_whitening)

    # 3. NumPy CPU baseline（单 batch 公平对比）
    x_c_cpu = x_centered[0].cpu().numpy()  # shape (C, T)

    def numpy_whitening():
        C, T = x_c_cpu.shape
        # 直接使用 x_c_cpu 算协方差
        cov = (x_c_cpu @ x_c_cpu.T) / (T - 1)

        eigvals, eigvecs = np.linalg.eigh(cov)

        # 同样对齐数值逻辑
        max_eig = np.max(np.abs(eigvals))
        eps = max_eig * 1e-6
        D_inv_sqrt = 1.0 / np.sqrt(np.maximum(np.abs(eigvals), eps))

        W = eigvecs @ (D_inv_sqrt[:, None] * eigvecs.T)
        return W @ x_c_cpu

    t_numpy = time_func(numpy_whitening, warmup=2, runs=5)

    # 计算加速比
    speedup_vs_torch = t_torch / t_cuda
    speedup_vs_numpy = t_numpy / t_cuda

    return {
        "batch": batch,
        "channels": channels,
        "time_steps": time_steps,
        "CUDA (ms)": f"{t_cuda * 1000:.2f}",
        "PyTorch (ms)": f"{t_torch * 1000:.2f}",
        "NumPy (ms)": f"{t_numpy * 1000:.2f}",
        "vs PyTorch": f"{speedup_vs_torch:.1f}x",
        "vs NumPy": f"{speedup_vs_numpy:.1f}x"
    }


# ==================== 测试配置 ====================
configs = [
    (1, 32, 10000),  # 小 batch，长序列
    (1, 64, 50000),  # 标准 64 通道，长序列
    (4, 64, 20000),
    (8, 128, 10000),
    (16, 64, 50000),  # 大 batch
]

print("🔥 开始 ZCA 白化 (Whitening) 性能测试（预计 40-80 秒）...\n")
results = []

for b, c, t in configs:
    print(f"正在测试 → batch={b} | channels={c} | time={t:,}")
    res = benchmark_whitening(b, c, t)
    results.append(res)

# ==================== 输出结果 ====================
df = pd.DataFrame(results)
print("\n" + "=" * 100)
print("📊 ZCA 白化 (Whitening) 性能对比结果")
print("=" * 100)
print(df.to_markdown(index=False))

# 自动保存
df.to_markdown("benchmark_whitening_results.md", index=False)
print("\n✅ 测试完成！结果已保存到 benchmark_whitening_results.md")