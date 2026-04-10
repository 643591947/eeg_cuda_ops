![EEG CUDA Ops](images/the_banner.png)
# EEG_CUDA: High-Performance EEG Signal Processing Accelerators

[ English | [Chinese](./README_zh.md) ]

EEG_CUDA is a high-performance library designed for Electroencephalogram (EEG) signal processing. By leveraging NVIDIA CUDA, this project accelerates critical algorithms for long-sequence and multi-channel data while maintaining high numerical stability.

---
## Numerical Verification

### EEG CUDA Whitening Operator Alignment Verification Report
Data dimensions: Batch 10 × 62 channels × 111000 time points
CUDA output precision: torch.float32

[Verification A] Covariance max absolute deviation: 5.03196036e-06
[Verification A] Covariance mean absolute deviation: 4.51245319e-07
[Verification A] Covariance max relative deviation: 1.13689825e-01

[Verification B] Data point max absolute error: 9.59396362e-04
[Verification B] Data point mean absolute error: 1.41664771e-06
[Verification B] Data point max relative error: 7.35398026e+01

[Extra Check] CUDA → Identity max deviation: 1.81416576e+00
[Extra Check] PyTorch → Identity max deviation: 1.81416596e+00


### EEG CUDA FastICA Verification Report
Data dimensions: 62 channels × 111000 time points
CUDA output precision: torch.float32

- [Verification A] Mean maximum correlation: 0.900836
- [Verification A] Minimum correlation: 0.499077
- [Verification A] High-match components: 40 / 62

Conclusion: CUDA FastICA numerically aligned with scikit-learn baseline
* *Note: Minor variances are expected due to hardware-specific floating-point truncation and random initialization.*

---
## 📊 Performance Benchmarks

We conducted comprehensive benchmarks on typical EEG data scales using the following configurations:

```python

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
|       1 |         32 |        10000 |        0.02 |           0.04 |         0.41 | 2.1x         | 21.4x      |
|       1 |         64 |        50000 |        0.21 |           0.22 |         5.82 | 1.1x         | 28.2x      |
|       4 |         64 |        20000 |        0.32 |           0.34 |         2.64 | 1.1x         | 8.3x       |
|       8 |        128 |        10000 |        0.45 |           0.68 |         2.92 | 1.5x         | 6.5x       |
|      16 |         64 |        50000 |        3.19 |           3.3  |         5.96 | 1.0x         | 1.9x       |

### Whitening
```markdown
|   batch |   channels |   time_steps |   CUDA (ms) |   PyTorch (ms) |   NumPy (ms) | vs PyTorch   | vs NumPy   |
|--------:|-----------:|-------------:|------------:|---------------:|-------------:|:-------------|:-----------|
|       1 |         32 |        10000 |        1.04 |           1.19 |         1.6  | 1.1x         | 1.5x       |
|       1 |         64 |        50000 |        1.75 |           1.86 |        11.53 | 1.1x         | 6.6x       |
|       4 |         64 |        20000 |        8.21 |           7.07 |         6.61 | 0.9x         | 0.8x       |
|       8 |        128 |        10000 |       18.67 |          18.85 |        11.2  | 1.0x         | 0.6x       |
|      16 |         64 |        50000 |       24.69 |          24.92 |        11.44 | 1.0x         | 0.5x       |

### FastICA
```markdown
|   batch |   channels |   time_steps |   CUDA (ms) |   sklearn (ms) | vs sklearn   |
|--------:|-----------:|-------------:|------------:|---------------:|:-------------|
|       1 |         32 |        10000 |      279.02 |         647.97 | 2.3x         |
|       1 |         64 |        50000 |      521.46 |        4178.58 | 8.0x         |
|       4 |         64 |        20000 |     1745.98 |        1933.9  | 1.1x         |
|       8 |        128 |        10000 |     4676.83 |        4372.91 | 0.9x         |
|      16 |         64 |        50000 |     7782.68 |        4311.99 | 0.6x         |

```
---

## Requirements

- Python 3.11
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
