import torch
import time
import numpy as np
import pandas as pd
from scipy import signal
import eeg_cuda

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")
print(f"PyTorch 版本: {torch.__version__}\n")


def time_func(fn, *args, warmup=5, runs=20):
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs


def benchmark_fir(batch, channels, time_steps):
    # 随机生成 EEG 数据
    x = torch.randn(batch, channels, time_steps, device=device, dtype=torch.float32)

    # 256 tap FIR 滤波器
    weights_np = signal.firwin(256, [1, 40], pass_zero=False, fs=250).astype(np.float32)
    weights = torch.from_numpy(weights_np).to(device)

    # 1.  CUDA 实现
    def cuda_fir():
        return eeg_cuda.fir_filter(x, weights)

    t_cuda = time_func(cuda_fir)

    # 2. PyTorch conv1d 基准（模拟因果卷积）
    def torch_conv():
        pad = weights.shape[0] - 1
        x_pad = torch.nn.functional.pad(x, (pad, 0))
        weight = weights.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1)  # [channels, 1, 256]
        return torch.nn.functional.conv1d(x_pad, weight, groups=channels)

    t_torch = time_func(torch_conv)

    # 3. SciPy lfilter（CPU 单 batch 基准）
    x_cpu = x[0].cpu().numpy()  # 只取第一个 batch 做公平对比
    w_cpu = weights_np

    def scipy_fir():
        return signal.lfilter(w_cpu, [1.0], x_cpu, axis=-1)

    t_scipy = time_func(scipy_fir, warmup=2, runs=5)  # SciPy 较慢，少跑几次

    # 计算加速比
    speedup_vs_torch = t_torch / t_cuda
    speedup_vs_scipy = t_scipy / t_cuda

    return {
        "batch": batch,
        "channels": channels,
        "time_steps": time_steps,
        "CUDA (ms)": f"{t_cuda * 1000:.2f}",
        "PyTorch (ms)": f"{t_torch * 1000:.2f}",
        "SciPy (ms)": f"{t_scipy * 1000:.2f}",
        "vs PyTorch": f"{speedup_vs_torch:.1f}x",
        "vs SciPy": f"{speedup_vs_scipy:.1f}x"
    }


# ==================== 测试配置（典型 EEG 规模）====================
configs = [
    (1, 32, 10000),  # 小 batch，长序列
    (1, 64, 50000),  # 标准 64 通道，长序列
    (4, 64, 20000),
    (8, 128, 10000),
    (16, 64, 50000),  # 大 batch
]

print("🔥 开始 FIR Filter 性能测试（预计 30-60 秒）...\n")
results = []

for b, c, t in configs:
    print(f"正在测试 → batch={b} | channels={c} | time={t:,}")
    res = benchmark_fir(b, c, t)
    results.append(res)

# ==================== 输出结果 ====================
df = pd.DataFrame(results)
print("\n" + "=" * 80)
print("📊 FIR Filter 性能对比结果")
print("=" * 80)
print(df.to_markdown(index=False))

# 自动保存为 Markdown（直接复制到 README）
df.to_markdown("benchmark_fir_results.md", index=False)
print("\n✅ 测试完成！结果已保存到 benchmark_fir_results.md")