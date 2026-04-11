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

### 5. IIR Filtfilt
Infinite Impulse Response (IIR) filters are traditionally computed sequentially, making them difficult to parallelize on GPUs. This project parallelizes the IIR filter computation by converting the standard difference equation into a **State-Space Matrix Representation** and leveraging **Parallel Prefix Product (Scan)** algorithms.

#### Standard Difference Equation

For a 2nd-order IIR filter (Biquad), the time-domain difference equation is given by:

$$a_0 y_n + a_1 y_{n-1} + a_2 y_{n-2} = b_0 x_n + b_1 x_{n-1} + b_2 x_{n-2}$$

Rearranging to solve for the current output $y_n$:

$$y_n = -\frac{a_1}{a_0} y_{n-1} - \frac{a_2}{a_0} y_{n-2} + \frac{1}{a_0} (b_0 x_n + b_1 x_{n-1} + b_2 x_{n-2})$$

Let $u_n$ represent the aggregated input term at time $n$:

$$u_n = \frac{1}{a_0} (b_0 x_n + b_1 x_{n-1} + b_2 x_{n-2})$$

Thus, the equation simplifies to:

$$y_n = -\frac{a_1}{a_0} y_{n-1} - \frac{a_2}{a_0} y_{n-2} + u_n$$

#### State-Space Matrix Representation

To parallelize the recursive equation, we convert it into a linear matrix operation. We define a state vector $S_n$ and a state-transition matrix $M_n$. 

To incorporate the dynamic input $u_n$ using pure matrix multiplication, we use an **augmented matrix trick** by appending a constant `1` to our state vector.

Let the state vector be:

$$S_{n-1} = \begin{bmatrix} y_{n-1} \\ y_{n-2} \\ 1 \end{bmatrix}$$

We can represent the transition from $S_{n-1}$ to $S_n$ as $S_n = M_n S_{n-1}$, where $M_n$ is:

$$M_n = \begin{bmatrix} -\frac{a_1}{a_0} & -\frac{a_2}{a_0} & u_n \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

Multiplying $M_n$ by $S_{n-1}$ perfectly recovers our difference equation:
* Row 1: $y_n = (-\frac{a_1}{a_0})y_{n-1} + (-\frac{a_2}{a_0})y_{n-2} + u_n \cdot 1$
* Row 2: $y_{n-1} = 1 \cdot y_{n-1} + 0 \cdot y_{n-2} + 0 \cdot 1$
* Row 3: $1 = 0 \cdot y_{n-1} + 0 \cdot y_{n-2} + 1 \cdot 1$

#### Parallelization via Prefix Product

By unrolling the recursion, the state at time $n$ depends only on the initial state $S_{-1}$ and the product of all intermediate matrices:

$$S_n = M_n \cdot M_{n-1} \cdot M_{n-2} \dots M_0 \cdot S_{-1}$$

Let $P_n$ be the accumulated prefix product of the matrices up to time $n$:

$$P_n = \prod_{k=0}^{n} M_k = M_n \cdot M_{n-1} \dots M_0$$

Since matrix multiplication is **associative** ($(A \cdot B) \cdot C = A \cdot (B \cdot C)$), we can compute $P_n$ for all $n$ simultaneously using a **Parallel Prefix Scan** algorithm (e.g., via NVIDIA Thrust). 

Once $P_n$ is computed for all time steps, the final output $y_n$ can be extracted independently in parallel:

$$S_n = P_n \cdot \begin{bmatrix} y_{-1} \\ y_{-2} \\ 1 \end{bmatrix}$$

Assuming zero initial conditions ($y_{-1} = 0, y_{-2} = 0$), $y_n$ is simply the element at the first row and third column of $P_n$.

#### Zero-Phase Filtering (filtfilt)

The matrix scan approach computes the **causal (forward) filter**. To achieve zero-phase distortion (identical to SciPy's `scipy.signal.filtfilt`), the filter is applied twice:
1.  **Forward pass:** Compute $y_{forward}$ using the parallel matrix scan.
2.  **Time reversal:** Reverse the time axis of $y_{forward}$.
3.  **Backward pass:** Apply the same parallel filter on the reversed signal.
4.  **Time reversal:** Reverse the output back to original chronological order.

This approach achieves complete $O(N)$ sequential IIR filtering in highly parallel $O(\log N)$ GPU time bounds.