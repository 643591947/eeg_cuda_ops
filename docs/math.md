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
