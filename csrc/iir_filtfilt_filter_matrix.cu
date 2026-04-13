#include <torch/extension.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "filter_utils.cuh"

#define THREADS_PER_BLOCK 256

/**
 * @brief 二阶 IIR 滤波器状态转移矩阵结构体
 * * 用于将差分方程转换为线性代数形式。
 * 状态向量定义为：S_n = [y[n], y[n-1], 1]^T
 */
struct IIRMatrix {
    double m00, m01, m02;
    double m10, m11, m12;

    /**
     * @brief 执行矩阵乘法 C = A * B
     * 针对 IIR 状态转移矩阵的稀疏性进行了手动展开优化
     */
    __device__ __host__ static IIRMatrix multiply(const IIRMatrix& A, const IIRMatrix& B) {
        IIRMatrix C;
        // 第一行计算
        C.m00 = A.m00 * B.m00 + A.m01 * B.m10;
        C.m01 = A.m00 * B.m01 + A.m01 * B.m11;
        C.m02 = A.m00 * B.m02 + A.m01 * B.m12 + A.m02;

        // 第二行计算
        C.m10 = A.m10 * B.m00 + A.m11 * B.m10;
        C.m11 = A.m10 * B.m01 + A.m11 * B.m11;
        C.m12 = A.m10 * B.m02 + A.m11 * B.m12 + A.m12;

        return C;
    }
};

// ====================== 核心 CUDA Kernel ======================
/**
 * @brief IIR 前向滤波 Kernel（基于寄存器累积）
 * * 每个线程独立处理一个 (Batch, Channel) 序列。
 * 通过在寄存器中维护状态转移矩阵 P 的累积乘积，实现对时间序列的线性变换。
 */
__global__ void iir_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    double b0, double b1, double b2,
    double a0, double a1, double a2,
    int total_sequences,
    int time_steps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_sequences) return;

    // 指向当前序列的起始位置
    const float* in_seq = input + idx * time_steps;
    float* out_seq = output + idx * time_steps;

    // 初始化累积转移矩阵为单位矩阵
    IIRMatrix P;
    P.m00 = 1.0; P.m01 = 0.0; P.m02 = 0.0;
    P.m10 = 0.0; P.m11 = 1.0; P.m12 = 0.0;

    // 初始状态假设 y[-1] = 0, y[-2] = 0
    const double y_prev1 = 0.0;
    const double y_prev2 = 0.0;

    for (int t = 0; t < time_steps; ++t) {
        double x_n   = in_seq[t];
        double x_n_1 = (t >= 1) ? in_seq[t - 1] : 0.0;
        double x_n_2 = (t >= 2) ? in_seq[t - 2] : 0.0;

        // 构造当前步的转移矩阵 M_n
        IIRMatrix M;
        M.m00 = -a1 / a0;
        M.m01 = -a2 / a0;
        M.m02 = (b0 * x_n + b1 * x_n_1 + b2 * x_n_2) / a0;
        M.m10 = 1.0;
        M.m11 = 0.0;
        M.m12 = 0.0;

        // 更新累积矩阵 P = M_n * P
        P = IIRMatrix::multiply(M, P);

        // 计算当前输出：y[n] 为状态向量 S_n 的第一个分量
        // S_n = P * S_initial，其中 S_initial = [y[-1], y[-2], 1]^T
        out_seq[t] = static_cast<float>(P.m00 * y_prev1 + P.m01 * y_prev2 + P.m02);
    }
}

/**
 * @brief 执行单向 IIR 滤波
 * * 采用状态空间表示法，利用矩阵连乘计算输出序列。
 */
static torch::Tensor run_iir_filter_forward(
    torch::Tensor input,
    torch::Tensor sfre,
    torch::Tensor cfre,
    const std::string& pass,
    bool bandstop
) {
    // ====================== 输入合法性检查 ======================
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be (Batch, Channels, TimeSteps)");

    // ====================== 计算滤波器系数 ======================
    auto [b_vec, a_vec] = a_b(sfre, cfre, pass, bandstop);

    float b0 = b_vec[0], b1 = b_vec[1], b2 = b_vec[2];
    float a0 = 1.0,      a1 = a_vec[1], a2 = a_vec[2];   // a[0] 恒为 1.0

    auto input_c = input.contiguous();
    auto output = torch::empty_like(input_c);

    const int batch_size     = input_c.size(0);
    const int num_channels   = input_c.size(1);
    const int time_steps     = input_c.size(2);
    const int total_sequences = batch_size * num_channels;

    // ====================== 启动 Kernel ======================
    const int blockSize = THREADS_PER_BLOCK;
    const int numBlocks = (total_sequences + blockSize - 1) / blockSize;

    iir_forward_kernel<<<numBlocks, blockSize>>>(
        input_c.data_ptr<float>(),
        output.data_ptr<float>(),
        b0, b1, b2,
        a0, a1, a2,
        total_sequences,
        time_steps
    );

    return output;
}

/**
 * @brief 零相位 IIR 滤波 (Filt-Filt)
 * * 通过正向和反向两次滤波消除相位失真。
 */
torch::Tensor run_iir_filtfilt_matrix(
    torch::Tensor input,
    torch::Tensor sfre,
    torch::Tensor cfre,
    const std::string& pass = "high",
    bool bandstop = false
) {
    // 1. 正向滤波
    auto y_forward = run_iir_filter_forward(input, sfre, cfre, pass, bandstop);

    // 2. 时间反转 (维度 2 为 TimeSteps)
    auto y_reversed = torch::flip(y_forward, {2});

    // 3. 反向滤波 (对反转后的信号再次滤波)
    auto y_backward = run_iir_filter_forward(y_reversed, sfre, cfre, pass, bandstop);

    // 4. 再次反转还原时间轴
    return torch::flip(y_backward, {2});
}