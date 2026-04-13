#include <torch/extension.h>
#include <tuple>

/**
 * Fused ZCA Whitening Kernel（融合 ZCA 白化缩放核函数）
 *
 * 该 kernel 为 ZCA 白化（Zero-phase Component Analysis）中的关键步骤进行融合加速：
 *   1. 找到每个 batch 中特征值 absolute 的最大值（用于数值稳定）
 *   2. 计算 D^{-1/2}（特征值倒数平方根，加入 eps 保护）
 *   3. 计算 scaled_eigvecs_T = D^{-1/2} * V^T   （转置后的缩放特征向量）
 *
 * 输出 scaled_eigvecs_T 用于后续构造 ZCA 白化矩阵。
 */
template <typename scalar_t>
__global__ void fused_zca_scaling_kernel(
    const scalar_t* __restrict__ eigenvalues,   // [B, C]  特征值
    const scalar_t* __restrict__ eigenvectors,  // [B, C, C]  特征向量 (列正交)
    scalar_t* __restrict__ scaled_eigvecs_T,    // [B, C, C]  输出：D^{-1/2} * V^T
    int C                                       // num_channels
) {
    const int bid = blockIdx.x;      // 当前 batch 索引
    const int tid = threadIdx.x;     // 线程索引（用于归约和计算）

    // 动态共享内存，用于 block 内求最大特征值
    extern __shared__ char shared_mem[];
    scalar_t* max_vals = reinterpret_cast<scalar_t*>(shared_mem);

    // ==================== Step 1: 把特征值载入共享内存 ====================
    scalar_t eig = 0.0;
    if (tid < C) {
        eig = abs(eigenvalues[bid * C + tid]);
        max_vals[tid] = eig;
    } else {
        max_vals[tid] = 0.0;
    }
    __syncthreads();

    // ==================== Step 2: Block 内并行求最大值归约 ====================
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (max_vals[tid + stride] > max_vals[tid]) {
                max_vals[tid] = max_vals[tid + stride];
            }
        }
        __syncthreads();
    }

    scalar_t batch_max_eig = max_vals[0];                    // 整个 batch 的最大 |eigenvalue|
    scalar_t eps = batch_max_eig * 1e-6f;                    // 自适应数值稳定 eps

    // ==================== Step 3: 计算 D^{-1/2} 并构造 scaled V^T ====================
    if (tid < C) {
        // 对特征值加入 eps 保护后取倒数平方根
        scalar_t val = (eig > eps) ? eig : eps;
        scalar_t d_inv_sqrt = rsqrt(val);                    // 1 / sqrt(val)，CUDA 内置快速函数

        // scaled_eigvecs_T[tid, :] = d_inv_sqrt * eigenvectors[:, tid]  （即 D^{-1/2} * V^T 的第 tid 行）
        for (int i = 0; i < C; ++i) {
            // eigenvectors[bid, i, tid] 即第 tid 列的第 i 行元素
            scaled_eigvecs_T[bid * C * C + tid * C + i] =
                d_inv_sqrt * eigenvectors[bid * C * C + i * C + tid];
        }
    }
}


/**
 * PyTorch 绑定函数：ZCA 白化（Zero-Phase Component Analysis Whitening）
 *
 * 输入：
 *   centered_data : [batch_size, num_channels, time_steps]，已经中心化（mean=0）的数据
 *
 * 输出：std::tuple<whitened_data, whitening_matrix>
 *   - whitened_data : 白化后的数据，形状与输入相同
 *   - W             : ZCA 白化矩阵 [batch_size, num_channels, num_channels]
 *
 * ZCA 白化公式：W = V * D^{-1/2} * V^T
 * 特点：白化后数据协方差接近单位阵，且与 PCA 白化相比旋转误差最小（保留原始数据方向感）
 */
std::tuple<torch::Tensor, torch::Tensor> run_whitening(torch::Tensor centered_data) {
    // ====================== 1. 输入合法性检查 ======================
    TORCH_CHECK(centered_data.is_cuda(), "Whitening input must be a CUDA tensor");
    TORCH_CHECK(centered_data.scalar_type() == torch::kFloat32,
                "Input must be float32 (current implementation only supports float32)");
    TORCH_CHECK(centered_data.dim() == 3,
                "Input must be 3D: [batch_size, num_channels, time_steps]");

    auto X = centered_data.contiguous();
    const int batch_size = X.size(0);
    const int num_channels = X.size(1);
    const int time_steps = X.size(2);

    // ====================== 2. 计算协方差矩阵 ======================
    // cov = (X @ X^T) / (T - 1)   注意：这里使用无偏估计
    auto X_T = X.transpose(1, 2);                                   // [B, T, C]
    auto cov = torch::bmm(X, X_T).div_(time_steps - 1);            // [B, C, C]

    // ====================== 3. 特征值分解 (Eigendecomposition) ======================
    // 返回 eigenvalues [B, C]（从大到小排序），eigenvectors [B, C, C]
    auto eigh_result = torch::linalg_eigh(cov);
    auto eigenvalues  = std::get<0>(eigh_result).contiguous();   // [B, C]
    auto eigenvectors = std::get<1>(eigh_result).contiguous();   // [B, C, C]

    // ====================== 4. Fused Kernel：计算 D^{-1/2} * V^T ======================
    int B = batch_size;
    int C = num_channels;

    auto scaled_eigvecs_T = torch::empty({B, C, C}, eigenvectors.options());

    // 线程数取 >= C 的最小 2 的幂（便于 reduction）
    int threads = 1;
    while (threads < C) threads *= 2;

    size_t shared_mem_size = threads * sizeof(float);

    // 使用 AT_DISPATCH 支持未来扩展其他浮点类型（当前只 float32）
    AT_DISPATCH_FLOATING_TYPES(eigenvalues.scalar_type(), "fused_zca_scaling", ([&] {
        fused_zca_scaling_kernel<scalar_t><<<B, threads, shared_mem_size>>>(
            eigenvalues.data_ptr<scalar_t>(),
            eigenvectors.data_ptr<scalar_t>(),
            scaled_eigvecs_T.data_ptr<scalar_t>(),
            C
        );
    }));

    // ====================== 5. 构造 ZCA 白化矩阵 W = V * (D^{-1/2} * V^T) ======================
    auto W = torch::bmm(eigenvectors, scaled_eigvecs_T);   // [B, C, C]

    // ====================== 6. 应用白化变换 ======================
    auto whitened_data = torch::bmm(W, X);                 // [B, C, T]

    return std::make_tuple(whitened_data, W);
}