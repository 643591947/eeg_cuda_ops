#include <torch/extension.h>
#include <tuple>

template <typename scalar_t>

__global__ void fused_zca_scaling_kernel(
    const scalar_t* __restrict__ eigenvalues,  // [B, C]
    const scalar_t* __restrict__ eigenvectors, // [B, C, C]
    scalar_t* __restrict__ scaled_eigvecs_T,   // [B, C, C]
    int C
){
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    scalar_t* max_vals = (scalar_t*)shared_mem;

    // step1: data to shared_mem
    scalar_t eig = 0.0;
    if (tid < C) {
        eig = abs(eigenvalues[bid * C + tid]);
        max_vals[tid] = eig;
    }else {
        max_vals[tid] = 0.0;
    }
    __syncthreads();

    // step2: fond max reduction [,,,,....] vs [....,,,,]
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (max_vals[tid + stride] > max_vals[tid]) {
                max_vals[tid] = max_vals[tid + stride];
            }
        }
        __syncthreads();
    }
    scalar_t batch_max_eig = max_vals[0];
    scalar_t eps = batch_max_eig * 1e-6;

    // step3: use eps
    if (tid < C) {
        scalar_t val = eig > eps ? eig : eps;
        scalar_t d_inv_sqrt = rsqrt(val);

        for (int i = 0; i < C; ++i) {
            scaled_eigvecs_T[bid * C * C + tid * C + i] = d_inv_sqrt * eigenvectors[bid * C * C + i * C + tid];
        }
    }

}

std::tuple<torch::Tensor, torch::Tensor> run_whitening(torch::Tensor centered_data) {
    // 数据合法性校验
    TORCH_CHECK(centered_data.is_cuda(), "Whitening inputs must be CUDA tensors");
    TORCH_CHECK(centered_data.scalar_type() == torch::kFloat32, "Input must be Float32");

    auto X = centered_data.contiguous();
    int time_steps = X.size(2);

    // 1. 计算协方差
    auto X_T = X.transpose(1, 2);
    auto cov = torch::bmm(X, X_T).div_(time_steps - 1);

    // 2. 特征值分解
    auto eigh_result = torch::linalg_eigh(cov);
    auto eigenvalues = std::get<0>(eigh_result).contiguous();
    auto eigenvectors = std::get<1>(eigh_result).contiguous();

    //3. 融合算子
    int B = eigenvalues.size(0);
    int C = eigenvalues.size(1); // channel number
    auto scaled_eigvecs_T = torch::empty({B, C, C}, eigenvectors.options());

    int threads = 1;
    while (threads < C) threads *= 2;

    size_t shared_mem_size = threads * sizeof(float);

    // 启动 Fused Kernel
    AT_DISPATCH_FLOATING_TYPES(eigenvalues.scalar_type(), "fused_zca_scaling", ([&] {
        fused_zca_scaling_kernel<scalar_t><<<B, threads, shared_mem_size>>>(
            eigenvalues.data_ptr<scalar_t>(),
            eigenvectors.data_ptr<scalar_t>(),
            scaled_eigvecs_T.data_ptr<scalar_t>(),
            C
        );
    }));

    // 构建 ZCA 白化矩阵 W = V * (D^-1/2 * V^T)
    auto W = torch::bmm(eigenvectors, scaled_eigvecs_T);
    // 应用白化矩阵
    auto whitened_data = torch::bmm(W, X);

    return std::make_tuple(whitened_data, W);
}