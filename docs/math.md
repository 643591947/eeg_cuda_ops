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

### 5. Mathematical Principles of IIR Filters

The IIR filter is based on a **second-order Butterworth prototype** and is designed using the bilinear transform to create digital biquad filters, supporting low-pass, high-pass, band-pass, and band-stop filtering. The core computation is implemented using two equivalent approaches: the traditional Direct Form II and a state-space matrix-based parallel prefix product method.

#### Analog Prototype and Digital Filter Design

Starting from the analog second-order Butterworth low-pass prototype:

$$
H(s) = \frac{1}{s^2 + \sqrt{2}\ s + 1}
$$

1. **Pre-warping**  
   $$
   \Omega = \tan\left(\pi \cdot \frac{f_c}{f_s}\right)
   $$

2. **Bilinear Transform**  
   Convert the analog transfer function to the digital domain to obtain the biquad coefficients $(b_0, b_1, b_2, a_1, a_2)$ (with $a_0 = 1$).

High-pass, band-pass, and band-stop filters are obtained by first applying analog frequency transformations to the low-pass prototype and then applying the bilinear transform to obtain the corresponding coefficients. This process is equivalent to SciPy's `signal.butter` + `bilinear`.

#### Traditional Recursive Form (Direct Form II)

Once the coefficients are determined, the filter can be expressed by the following difference equations:

$$
\begin{aligned}
w[n] &= x[n] - a_1 w[n-1] - a_2 w[n-2] \\
y[n] &= b_0 w[n] + b_1 w[n-1] + b_2 w[n-2]
\end{aligned}
$$

In the CUDA kernel, each thread computes this recursion sequentially over time, suitable for moderate-scale parallelism.

#### State-Space Matrix Representation (Parallelization Scheme)

To achieve efficient GPU parallelism, the recursion is converted into a **state-space form** using an **augmented matrix trick**:

Define the state vector:

$$    
S_{n-1} = \begin{bmatrix}
y_{n-1} \\
y_{n-2} \\
1
\end{bmatrix}
$$

Construct the time-varying state transition matrix $M_n$ (with the input term embedded):

$$
M_n = \begin{bmatrix}
-\frac{a_1}{a_0} & -\frac{a_2}{a_0} & u_n \\
1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

where $u_n$ is the normalized input term:

$$
u_n = \frac{1}{a_0}(b_0 x_n + b_1 x_{n-1} + b_2 x_{n-2})
$$

The state update simplifies to matrix multiplication:

$$
S_n = M_n \ S_{n-1}
$$

#### Parallel Prefix Product

The recursion is fully expanded as:

$$
S_n = M_n \cdot M_{n-1} \cdots M_0 \cdot S_{-1}
$$

Define the prefix product matrix:

$$
P_n = \prod_{k=0}^{n} M_k = M_n \cdot M_{n-1} \cdots M_0
$$

Since matrix multiplication is **associative**, the **Parallel Prefix Scan** algorithm can be used to compute all $P_n$ in $O(\log N)$ time.

Under zero initial conditions ($y_{-1} = y_{-2} = 0$), the output $y_n$ can be directly extracted from the first row, third column element of $P_n$, enabling fully parallel computation.

#### Zero-Phase Filtering (filtfilt)

The method described above implements **causal forward filtering**. To achieve zero-phase filtering (consistent with `scipy.signal.filtfilt`), the following steps are performed:

1. Forward filtering: compute $y_{\text{forward}}$
2. Time reversal
3. Reverse filtering (apply prefix product again to the reversed signal)
4. Time reversal again

The resulting magnitude response is the square of the original filter's magnitude, and the phase is completely canceled to zero.

#### Comparison of the Two Implementations

- **Direct Form II**: Simple to implement, numerically stable, suitable for real-time or medium-length sequences.
- **State-Space + Prefix Product**: Converts recursion into matrix multiplication, leveraging GPU parallel scan capabilities with a theoretical time complexity of $O(\log N)$, particularly suitable for high-throughput parallel processing of very long sequences.

The two methods are mathematically equivalent and can be flexibly selected or combined based on specific requirements.
