![EEG CUDA Ops](images/the_banner.png)
# EEG_CUDA: High-Performance EEG Signal Processing Accelerators

[ English | [Chinese](./README_zh.md) ]

EEG_CUDA is a high-performance library designed for Electroencephalogram (EEG) signal processing. By leveraging NVIDIA CUDA, this project accelerates critical algorithms for long-sequence and multi-channel data while maintaining high numerical stability.

See [math.md](docs/math.md) for mathematical details.
---
## 📊 Performance Benchmarks

We conducted comprehensive benchmarks on typical EEG data scales using the following configurations:

```markdown
### FIR Filter
|   batch |   channels |   time_steps |   CUDA (ms) |   PyTorch (ms) |   SciPy (ms) | vs PyTorch   | vs SciPy   |
|--------:|-----------:|-------------:|------------:|---------------:|-------------:|:-------------|:-----------|
|       1 |         32 |        10000 |        0.3  |           0.55 |        12.43 | 1.8x         | 41.7x      |
|       1 |         64 |        50000 |        2.84 |           4.84 |       116.87 | 1.7x         | 41.2x      |
|       4 |         64 |        20000 |        3.61 |           5.94 |        46.96 | 1.6x         | 13.0x      |
|       8 |        128 |        10000 |        7.4  |          11.88 |        46.63 | 1.6x         | 6.3x       |
|      16 |         64 |        50000 |       37.45 |          59.65 |       114.92 | 1.6x         | 3.1x       |

### IIR Filtfilt Filter
Note : Full Batch Fair Comparison
|   batch |   channels |   time_steps |   CUDA Matrix (ms) |   CUDA Norm (ms) |   PyTorch (ms) |   SciPy (ms) | Matrix vs PyTorch   | Norm vs PyTorch   | Matrix vs SciPy   | Norm vs SciPy   |   Matrix-Norm diff |
|--------:|-----------:|-------------:|-------------------:|-----------------:|---------------:|-------------:|:--------------------|:------------------|:------------------|:----------------|-------------------:|
|       1 |         32 |        10000 |              12    |             1.62 |           4.03 |         4.75 | 0.3x                | 2.5x              | 0.4x              | 2.9x            |           0.000129 |
|       1 |         64 |        50000 |              63.46 |             8.34 |          21.78 |        41.42 | 0.3x                | 2.6x              | 0.7x              | 5.0x            |           0.000131 |
|       4 |         64 |        20000 |              85.57 |            15.88 |          21.58 |        70.12 | 0.3x                | 1.4x              | 0.8x              | 4.4x            |           0.000139 |
|       8 |        128 |        10000 |              43.91 |             7.61 |          13.6  |       141.2  | 0.3x                | 1.8x              | 3.2x              | 18.6x           |           0.000159 |
|      16 |         64 |        50000 |             218.4  |            36.43 |          70.05 |       660.02 | 0.3x                | 1.9x              | 3.0x              | 18.1x           |           0.000145 |
### Centering
|   batch |   channels |   time_steps |   CUDA (ms) |   PyTorch (ms) |   NumPy (ms) | vs PyTorch   | vs NumPy   |
|--------:|-----------:|-------------:|------------:|---------------:|-------------:|:-------------|:-----------|
|       1 |         32 |        10000 |        0.02 |           0.04 |         0.41 | 2.1x         | 21.4x      |
|       1 |         64 |        50000 |        0.21 |           0.22 |         5.82 | 1.1x         | 28.2x      |
|       4 |         64 |        20000 |        0.32 |           0.34 |         2.64 | 1.1x         | 8.3x       |
|       8 |        128 |        10000 |        0.45 |           0.68 |         2.92 | 1.5x         | 6.5x       |
|      16 |         64 |        50000 |        3.19 |           3.3  |         5.96 | 1.0x         | 1.9x       |

### Whitening
|   batch |   channels |   time_steps |   CUDA (ms) |   PyTorch (ms) |   NumPy (ms) | vs PyTorch   | vs NumPy   |
|--------:|-----------:|-------------:|------------:|---------------:|-------------:|:-------------|:-----------|
|       1 |         32 |        10000 |        1.04 |           1.19 |         1.6  | 1.1x         | 1.5x       |
|       1 |         64 |        50000 |        1.75 |           1.86 |        11.53 | 1.1x         | 6.6x       |
|       4 |         64 |        20000 |        8.21 |           7.07 |         6.61 | 0.9x         | 0.8x       |
|       8 |        128 |        10000 |       18.67 |          18.85 |        11.2  | 1.0x         | 0.6x       |
|      16 |         64 |        50000 |       24.69 |          24.92 |        11.44 | 1.0x         | 0.5x       |

### FastICA
|   batch |   channels |   time_steps |   CUDA (ms) |   sklearn (ms) | vs sklearn   |
|--------:|-----------:|-------------:|------------:|---------------:|:-------------|
|       1 |         32 |        10000 |      279.02 |         647.97 | 2.3x         |
|       1 |         64 |        50000 |      521.46 |        4178.58 | 8.0x         |
|       4 |         64 |        20000 |     1745.98 |        1933.9  | 1.1x         |
|       8 |        128 |        10000 |     4676.83 |        4372.91 | 0.9x         |
|      16 |         64 |        50000 |     7782.68 |        4311.99 | 0.6x         |

```
See [numerical_verification.md](docs/numerical_verification.md) for verification reports.
---

## Web UI demo
![EEG CUDA Demo](assets/demo.gif)

## Requirements

- Python 3.11
- PyTorch 2.6.0+cu124 (with CUDA 12.4)
- CUDA Toolkit 12.4 (matching the PyTorch build)
- NVIDIA GPU with compute capability ≥ 7.0 (recommended)

## 🛠 Installation & Setup
Follow these steps to set up the environment and build the CUDA extensions.

1. Clone the Repository
```bash
git clone https://github.com/643591947/eeg_cuda_ops.git
cd eeg_cuda_ops
```

2. Install Dependencies
It is recommended to use a virtual environment (venv or conda).
```bash
pip install -r requirements.txt
```

3. Build CUDA Extensions
```bash
pip install -e . --no-build-isolation
```

4. Running the Visualization UI
```bash
cd web_UI
streamlit run main.py
```
