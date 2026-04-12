import torch
import time
import numpy as np
import pandas as pd
from scipy import signal
import math
import torchaudio
import eeg_cuda

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"torchaudio 版本: {torchaudio.__version__}\n")


def time_func(fn, *args, warmup=5, runs=20):
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs


# ==========================================
# 辅助函数：生成与 C++ 端完全一致的 a, b 系数
# ==========================================
def get_biquad_coeffs(sfre, cfre, pass_type="high", bandstop=False):
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


def benchmark_iir(batch, channels, time_steps):
    # 随机生成 EEG 数据
    x = torch.randn(batch, channels, time_steps, device=device, dtype=torch.float32)

    # ====================== IIR 滤波器配置 ======================
    fs = 250.0
    cfre_val = [1]          # 1Hz 高通滤波器
    pass_type = "high"
    bandstop = False

    b_np, a_np = get_biquad_coeffs(fs, cfre_val, pass_type, bandstop)
    print(f"   IIR 配置 → {pass_type}pass @ {cfre_val} Hz (fs={fs}Hz) | Batch={batch} | Channels={channels} | Time={time_steps:,}")

    # PyTorch torchaudio 需要 a_coeffs（分母）在前，b_coeffs（分子）在后
    a_torch = torch.from_numpy(a_np).to(device)   # denom
    b_torch = torch.from_numpy(b_np).to(device)   # num

    # 1. CUDA Matrix 版本
    def cuda_matrix():
        return eeg_cuda.iir_filtfilt_matrix(
            x,
            torch.tensor(int(fs), dtype=torch.int32, device=device),
            torch.tensor(cfre_val, dtype=torch.float32, device=device),
            pass_type,
            bandstop
        )

    t_matrix = time_func(cuda_matrix)

    # 2. CUDA Norm 版本
    def cuda_norm():
        return eeg_cuda.iir_filtfilt_norm(
            x,
            torch.tensor(int(fs), dtype=torch.int32, device=device),
            torch.tensor(cfre_val, dtype=torch.float32, device=device),
            pass_type,
            bandstop
        )

    t_norm = time_func(cuda_norm)

    # 数值一致性检查
    y_matrix = cuda_matrix()
    y_norm = cuda_norm()
    max_diff = torch.max(torch.abs(y_matrix - y_norm)).item()
    print(f"   数值差异 (Matrix vs Norm): {max_diff:.2e} （越接近 0 越好）")

    # 3. PyTorch torchaudio filtfilt
    def torch_iir():
        return torchaudio.functional.filtfilt(x, a_torch, b_torch)

    t_torch = time_func(torch_iir)

    # 4. SciPy filtfilt 基准（完整 batch，CPU）
    x_cpu = x.cpu().numpy()

    def scipy_iir():
        y = np.empty_like(x_cpu)
        for b_idx in range(batch):
            y[b_idx] = signal.filtfilt(b_np, a_np, x_cpu[b_idx], axis=-1)
        return y

    t_scipy = time_func(scipy_iir, warmup=2, runs=3)

    # 计算加速比（以 PyTorch 和 SciPy 为基准）
    speedup_matrix_vs_torch = t_torch / t_matrix
    speedup_norm_vs_torch = t_torch / t_norm
    speedup_matrix_vs_scipy = t_scipy / t_matrix
    speedup_norm_vs_scipy = t_scipy / t_norm

    return {
        "batch": batch,
        "channels": channels,
        "time_steps": time_steps,
        "CUDA Matrix (ms)": f"{t_matrix * 1000:.2f}",
        "CUDA Norm (ms)": f"{t_norm * 1000:.2f}",
        "PyTorch (ms)": f"{t_torch * 1000:.2f}",
        "SciPy (ms)": f"{t_scipy * 1000:.2f}",
        "Matrix vs PyTorch": f"{speedup_matrix_vs_torch:.1f}x",
        "Norm vs PyTorch": f"{speedup_norm_vs_torch:.1f}x",
        "Matrix vs SciPy": f"{speedup_matrix_vs_scipy:.1f}x",
        "Norm vs SciPy": f"{speedup_norm_vs_scipy:.1f}x",
        "Matrix-Norm diff": f"{max_diff:.2e}",
    }


# ==================== 测试配置 ====================
configs = [
    (1, 32, 10000),
    (1, 64, 50000),
    (4, 64, 20000),
    (8, 128, 10000),
    (16, 64, 50000),
]

print("🔥 开始 IIR Filter 性能测试（CUDA Matrix vs CUDA Norm vs PyTorch-torchaudio vs SciPy）...\n")
results = []

for b, c, t in configs:
    print(f"正在测试 → batch={b} | channels={c} | time={t:,}")
    res = benchmark_iir(b, c, t)
    results.append(res)

# ==================== 输出结果 ====================
df = pd.DataFrame(results)
print("\n" + "=" * 100)
print("📊 IIR Filter 性能对比结果（两个 CUDA 版本完整对比）")
print("=" * 100)
print(df.to_markdown(index=False))

df.to_markdown("benchmark_iir_matrix_vs_norm_results.md", index=False)
print("\n✅ 测试完成！结果已保存到 benchmark_iir_matrix_vs_norm_results.md")
print("   （两个版本的数值差异已打印，如果 diff > 1e-5 请检查实现是否一致）")