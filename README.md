# EEG_CUDA: High-Performance EEG Signal Processing Accelerators

[ English | [Chinese](./README_zh.md) ]

EEG_CUDA is a high-performance library designed for Electroencephalogram (EEG) signal processing. By leveraging NVIDIA CUDA, this project accelerates critical algorithms for long-sequence and multi-channel data while maintaining high numerical stability.

---

## 🚀 Key Algorithms

### 1. FIR Filter
* **Purpose:** Frequency domain filtering (e.g., 1-40Hz bandpass) to remove powerline interference and baseline drift.
* **Implementation:** * Implements discrete-time causal convolution: $$y[n] = \sum_{k=0}^{N} h[k] \cdot x[n-k]$$
    * One GPU thread per sample point for massive parallelism.
    * Built-in zero-padding for memory safety.

### 2. Channel Centering (Baseline Correction)
* **Purpose:** Removes DC offset by centering each channel's mean to zero.
* **Four-Stage Optimization:**
    1.  **Local Accumulation:** Grid-stride loops for efficient memory reading.
    2.  **In-block Reduction:** Tree-based summation in **Shared Memory**.
    3.  **Mean Broadcasting:** Final mean calculation written to shared variables.
    4.  **Parallel Subtraction:** Element-wise subtraction across all samples.
* **Efficiency:** Fuses "Sum" and "Subtract" into a single Kernel to minimize VRAM I/O.

### 3. ZCA Whitening
* **Purpose:** Essential preprocessing for ICA. It de-correlates channels and normalizes variance while preserving the topological structure of EEG sensors.
* **Mathematical Flow:**
    1.  Covariance Matrix: $$\Sigma = \frac{1}{T-1}XX^T$$
    2.  Eigendecomposition: $$\Sigma = VDV^T$$
    3.  Whitening Matrix: $P = VD^{-1/2}V^T$
* **Precision Strategy:** Uses `Float64` on GPU to bypass TF32 effects, ensuring results align with NumPy's double-precision standards.

### 4. FastICA (Independent Component Analysis)
* **Purpose:** Blind Source Separation (BSS). Extracts independent brain sources and removes artifacts like EOG and ECG.
* **Core Logic:** Maximizes non-Gaussianity using the Fixed-point iteration (Fixed-point algorithm).
* **Orthogonalization:** Symmetric orthogonalization ($W = (WW^T)^{-1/2}W$) to prevent component convergence overlap.

---

## 📊 Validation & Benchmarks

We use a "Pure Variable Isolation" strategy to verify CUDA kernels against standard CPU implementations (NumPy/Scikit-learn).

### Precision Results (ZCA)
| Metric | Max Absolute Error | Status |
| :--- | :--- | :--- |
| **Covariance Alignment** | $< 10^{-7}$ | ✅ Passed |
| **Data Tensor Alignment** | $< 10^{-6}$ | ✅ Passed |

### Separation Fidelity (FastICA)
Due to the inherent scale and sign ambiguity of ICA, we use **Cross-Correlation Matrix** analysis:
* **Mean Max Correlation:** **0.909** (Highly consistent with Scikit-learn).
* **Matched Components:** **41 / 62** channels achieved high-fidelity matching.
* *Note: Minor variances are expected due to hardware-specific floating-point truncation and random initialization.*

---
## 📊 Performance Benchmarks

We conducted comprehensive benchmarks on typical EEG data scales using the following configurations:

```python
configs = [
    (1, 32, 10000),   # small batch, long sequence
    (1, 64, 50000),   # standard 64-channel, long sequence
    (4, 64, 20000),
    (8, 128, 10000),
    (16, 64, 50000),  # larger batch
]

### FIR Filter
```markdown
|   batch |   channels |   time_steps |   CUDA (ms) |   PyTorch (ms) |   SciPy (ms) | vs PyTorch   | vs SciPy   |
|--------:|-----------:|-------------:|------------:|---------------:|-------------:|:-------------|:-----------|
|       1 |         32 |        10000 |        0.3  |           0.55 |        12.43 | 1.8x         | 41.7x      |
|       1 |         64 |        50000 |        2.84 |           4.84 |       116.87 | 1.7x         | 41.2x      |
|       4 |         64 |        20000 |        3.61 |           5.94 |        46.96 | 1.6x         | 13.0x      |
|       8 |        128 |        10000 |        7.4  |          11.88 |        46.63 | 1.6x         | 6.3x       |
|      16 |         64 |        50000 |       37.45 |          59.65 |       114.92 | 1.6x         | 3.1x       |

### Centering
```markdown
|   batch |   channels |   time_steps |   CUDA (ms) |   PyTorch (ms) |   NumPy (ms) | vs PyTorch   | vs NumPy   |
|--------:|-----------:|-------------:|------------:|---------------:|-------------:|:-------------|:-----------|
|       1 |         32 |        10000 |        0.04 |           0.04 |         0.48 | 0.8x         | 10.6x      |
|       1 |         64 |        50000 |        0.39 |           0.22 |         5.97 | 0.6x         | 15.2x      |
|       4 |         64 |        20000 |        0.57 |           0.35 |         2.99 | 0.6x         | 5.2x       |
|       8 |        128 |        10000 |        1.17 |           0.68 |         2.76 | 0.6x         | 2.4x       |
|      16 |         64 |        50000 |        5.65 |           3.31 |         8.5  | 0.6x         | 1.5x       |

### Whitening
```markdown
|   batch |   channels |   time_steps |   CUDA (ms) |   PyTorch (ms) |   NumPy (ms) | vs PyTorch   | vs NumPy   |
|--------:|-----------:|-------------:|------------:|---------------:|-------------:|:-------------|:-----------|
|       1 |         32 |        10000 |        3.46 |           2.43 |        12.49 | 0.7x         | 3.6x       |
|       1 |         64 |        50000 |       14.64 |           3.93 |       138.35 | 0.3x         | 9.5x       |
|       4 |         64 |        20000 |       32.51 |          10.86 |        80.74 | 0.3x         | 2.5x       |
|       8 |        128 |        10000 |      120.39 |          29.15 |        21.65 | 0.2x         | 0.2x       |
|      16 |         64 |        50000 |      171.5  |          29.08 |        17.72 | 0.2x         | 0.1x       |

### FastICA
```markdown
|   batch |   channels |   time_steps |   CUDA (ms) |   sklearn (ms) | vs sklearn   |
|--------:|-----------:|-------------:|------------:|---------------:|:-------------|
|       1 |         32 |        10000 |      300.15 |         612.03 | 2.0x         |
|       1 |         64 |        50000 |      548.11 |        4925.6  | 9.0x         |
|       4 |         64 |        20000 |     1798.81 |        2031.58 | 1.1x         |
|       8 |        128 |        10000 |     4701.17 |        4386.61 | 0.9x         |
|      16 |         64 |        50000 |     7883.38 |        4424.04 | 0.6x         |

```
---

## Requirements

- Python ≥ 3.8
- PyTorch 2.6.0+cu124 (with CUDA 12.4)
- CUDA Toolkit 12.4 (matching the PyTorch build)
- NVIDIA GPU with compute capability ≥ 7.0 (recommended)

This extension was built and tested with:
- PyTorch 2.6.0+cu124
- Built using: `pip install -e . --no-build-isolation`

## Installation

### From Source (Recommended for latest features)

```bash
git clone https://github.com/643591947/eeg_cuda_ops.git
cd eeg_cuda_ops

# Install in editable mode (recommended for development)
pip install -e . --no-build-isolation
```
