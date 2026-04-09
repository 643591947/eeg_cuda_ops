import torch
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
import eeg_cuda

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")
print(f"PyTorch 版本: {torch.__version__}\n")


def time_func(fn, *args, warmup=3, runs=10):
    """精确计时（FastICA 迭代较重，warmup 和 runs 适当减少）"""
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs


def benchmark_fastica(batch, channels, time_steps):
    # 随机生成 EEG 数据
    x = torch.randn(batch, channels, time_steps, device=device, dtype=torch.float32)

    # 1. CUDA 实现
    def cuda_fastica():
        return eeg_cuda.fastica_iter(x)

    t_cuda = time_func(cuda_fastica)

    # 2. sklearn FastICA CPU baseline（标准 EEG 使用方式）
    x_cpu = x[0].cpu().numpy().T

    def sklearn_fastica():
        ica = FastICA(n_components=channels, random_state=42, max_iter=200, tol=1e-4)
        return ica.fit_transform(x_cpu)  # 只做单 batch 公平对比

    t_sklearn = time_func(sklearn_fastica, warmup=1, runs=3)  # sklearn 较慢，减少 runs

    # 计算加速比
    speedup_vs_sklearn = t_sklearn / t_cuda

    return {
        "batch": batch,
        "channels": channels,
        "time_steps": time_steps,
        "CUDA (ms)": f"{t_cuda * 1000:.2f}",
        "sklearn (ms)": f"{t_sklearn * 1000:.2f}",
        "vs sklearn": f"{speedup_vs_sklearn:.1f}x"
    }


# ==================== 测试配置 ====================
configs = [
    (1, 32, 10000),  # 小 batch，长序列
    (1, 64, 50000),  # 标准 64 通道，长序列
    (4, 64, 20000),
    (8, 128, 10000),
    (16, 64, 50000),  # 大 batch
]

print("🔥 开始 FastICA 性能测试（预计 1-3 分钟，迭代较重）...\n")
results = []

for b, c, t in configs:
    print(f"正在测试 → batch={b} | channels={c} | time={t:,}")
    res = benchmark_fastica(b, c, t)
    results.append(res)

# ==================== 输出结果 ====================
df = pd.DataFrame(results)
print("\n" + "=" * 90)
print("📊 FastICA 性能对比结果")
print("=" * 90)
print(df.to_markdown(index=False))

# 自动保存
df.to_markdown("benchmark_fastica_results.md", index=False)
print("\n✅ 测试完成！结果已保存到 benchmark_fastica_results.md")