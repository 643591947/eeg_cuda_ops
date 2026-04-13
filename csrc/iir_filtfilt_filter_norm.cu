#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <utility>
#include <cmath>

#include "filter_utils.cuh"

#define THREADS_PER_BLOCK 256

// ====================== 核心 CUDA Kernel ======================
/**
 * 二阶 IIR 双二阶滤波器 Kernel（Biquad Filter）
 *
 * 使用 Direct Form II 结构实现单通道因果 IIR 滤波。
 * 每个线程独立处理一个 (batch, channel) 的完整时间序列。
 *
 * 滤波公式（Direct Form II）：
 *   w[n]   = x[n] - a1*w[n-1] - a2*w[n-2]
 *   y[n]   = b0*w[n] + b1*w[n-1] + b2*w[n-2]
 */
__global__ void iir_biquad_kernel(
    const float* __restrict__ input,     // [B*C, T] 展平后的输入信号
    float* __restrict__ output,          // [B*C, T] 输出信号
    float b0, float b1, float b2,        // 分子系数
    float a1, float a2,                  // 分母系数（a0 固定为 1.0）
    int batch,
    int channels,
    int time_steps
) {
    // 全局流索引：每个 (batch, channel) 对应一个独立信号流
    int stream_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_streams = batch * channels;

    if (stream_idx >= num_streams) return;

    // 该线程负责处理的输入和输出信号指针
    const float* x = input + stream_idx * time_steps;
    float* y = output + stream_idx * time_steps;

    // 状态变量（Direct Form II 中的中间变量 w）
    float w_prev1 = 0.0f;   // w[n-1]
    float w_prev2 = 0.0f;   // w[n-2]

    // ==================== 按时间顺序因果滤波 ====================
    for (int t = 0; t < time_steps; ++t) {
        float x_t = x[t];

        // Direct Form II 核心递推公式
        float w = x_t - a1 * w_prev1 - a2 * w_prev2;
        float y_t = b0 * w + b1 * w_prev1 + b2 * w_prev2;

        y[t] = y_t;

        // 更新状态
        w_prev2 = w_prev1;
        w_prev1 = w;
    }
}

// ====================== 单向 IIR 滤波（等价于 scipy.signal.lfilter） ======================
/**
 * 单向因果 IIR 滤波（Forward Pass）
 *
 * 使用预先计算好的 Butterworth 双二阶系数对输入信号进行因果滤波。
 * 内部会调用 a_b() 函数计算滤波器系数。
 */
static torch::Tensor run_iir_filter_forward(
    torch::Tensor input,
    torch::Tensor sfre,                    // 采样频率（标量 tensor）
    torch::Tensor cfre,                    // 截止频率（标量或含 2 个元素的 tensor）
    const std::string& pass = "high",      // "low" 或 "high"（仅用于单截止频率）
    bool bandstop = false                  // 双截止频率时是否为带阻
) {
    // ====================== 输入检查 ======================
    TORCH_CHECK(input.is_cuda() && input.dim() == 3,
                "Input must be a CUDA tensor of shape (B, C, T)");

    // ====================== 计算滤波器系数 ======================
    // 调用 filter_utils.cuh 中的工具函数
    auto [b_vec, a_vec] = a_b(sfre, cfre, pass, bandstop);

    float b0 = b_vec[0], b1 = b_vec[1], b2 = b_vec[2];
    float a1 = a_vec[1], a2 = a_vec[2];   // a[0] 恒为 1.0

    // ====================== 准备数据并启动 Kernel ======================
    auto input_c = input.contiguous();
    auto output = torch::empty_like(input_c);

    const int batch      = input_c.size(0);
    const int channels   = input_c.size(1);
    const int time_steps = input_c.size(2);

    const int num_streams = batch * channels;
    const int block_size  = THREADS_PER_BLOCK;
    const int grid_size   = (num_streams + block_size - 1) / block_size;

    // 每个线程处理一个完整的 (batch, channel) 信号
    iir_biquad_kernel<<<grid_size, block_size>>>(
        input_c.data_ptr<float>(),
        output.data_ptr<float>(),
        b0, b1, b2,
        a1, a2,
        batch, channels, time_steps
    );

    // 等待 kernel 执行完成（实际项目中建议换成 stream 同步以提升性能）
    cudaDeviceSynchronize();

    return output;
}

// ====================== 零相位滤波（filtfilt） ======================
/**
 * 零相位 IIR 滤波（Zero-Phase Filtering）
 *
 * 使用 filtfilt 技术实现零相位滤波，避免因果滤波带来的相位失真。
 * 算法流程：
 *   1. 正向滤波（forward）
 *   2. 将结果翻转
 *   3. 反向滤波（backward）
 *   4. 再次翻转得到最终输出
 *
 * 特点：相位响应为零，幅频响应为单向滤波的平方，适合需要保相位的信号处理场景。
 */
torch::Tensor run_iir_filtfilt_norm(
    torch::Tensor input,
    torch::Tensor sfre,
    torch::Tensor cfre,
    const std::string& pass = "high",
    bool bandstop = false
) {
    // Step 1: 正向因果滤波
    auto y_forward = run_iir_filter_forward(input, sfre, cfre, pass, bandstop);

    // Step 2: 时间维度翻转
    auto y_reversed = torch::flip(y_forward, {2});

    // Step 3: 对翻转后的信号再次进行因果滤波（等价于反向滤波）
    auto y_backward = run_iir_filter_forward(y_reversed, sfre, cfre, pass, bandstop);

    // Step 4: 再次翻转，得到零相位滤波结果
    return torch::flip(y_backward, {2});
}