# EEG_CUDA: High-Performance EEG Signal Processing Accelerators

[English] | [简体中文](./README_zh.md)

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

## 🛠 Prerequisites
* CUDA Toolkit 11.0+
* C++17 Compiler
* Eigen3 / cuBLAS (Optional: depending on your dependencies)

## 🏗 Installation
```bash
git clone [https://github.com/YourUsername/EEG_CUDA.git](https://github.com/YourUsername/EEG_CUDA.git)
mkdir build && cd build
cmake ..
make
