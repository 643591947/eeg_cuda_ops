![EEG CUDA Ops](images/the_banner.png)
# EEG_CUDA: High-Performance EEG Signal Processing Accelerators
0010
[ English | [Chinese](./README_zh.md) ]

EEG_CUDA is a high-performance library designed for Electroencephalogram (EEG) signal processing. By leveraging NVIDIA CUDA, this project accelerates critical algorithms for long-sequence and multi-channel data while maintaining high numerical stability.

See `docs/math.md` for mathematical details.
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

### IIR Filtfilt Filter
```markdown
Note : Full Batch Fair Comparison
|   batch |   channels |   time_steps |   CUDA Matrix (ms) |   CUDA Norm (ms) |   PyTorch (ms) |   SciPy (ms) | Matrix vs PyTorch   | Norm vs PyTorch   | Matrix vs SciPy   | Norm vs SciPy   |   Matrix-Norm diff |
|--------:|-----------:|-------------:|-------------------:|-----------------:|---------------:|-------------:|:--------------------|:------------------|:------------------|:----------------|-------------------:|
|       1 |         32 |        10000 |               7.54 |             1.79 |           4.48 |         4.86 | 0.6x                | 2.5x              | 0.6x              | 2.7x            |           0.000129 |
|       1 |         64 |        50000 |              39.93 |             8.28 |          21.64 |        40.6  | 0.5x                | 2.6x              | 1.0x              | 4.9x            |           0.000131 |
|       4 |         64 |        20000 |              62.41 |            15.71 |          21.47 |        69.52 | 0.3x                | 1.4x              | 1.1x              | 4.4x            |           0.000139 |
|       8 |        128 |        10000 |             123.94 |             7.57 |          13.55 |       142.12 | 0.1x                | 1.8x              | 1.1x              | 18.8x           |           0.000159 |
|      16 |         64 |        50000 |             606.51 |            36.46 |          70    |       663.91 | 0.1x                | 1.9x              | 1.1x              | 18.2x           |           0.000145 |

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
See `docs/numerical_verification.md` for verification reports.
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
