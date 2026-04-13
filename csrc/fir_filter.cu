#include <torch/extension.h>
#include <cuda_runtime.h>

// ====================== 常量定义 ======================
#define MAX_FILTER_LEN 256

// 使用常量内存存储滤波器系数（对所有线程只读，访问速度极快）
__constant__ float d_filter_weights[MAX_FILTER_LEN];

/**
 * FIR 滤波器 CUDA Kernel（因果滤波 / Causal FIR Filter）
 *
 * 对输入信号 [batch_size, num_channels, time_steps] 的每个通道独立进行 FIR 滤波。
 *
 * 实现特点：
 *   - 因果滤波（causal）：只使用当前时刻及过去时刻的数据，不看未来
 *   - 使用常量内存加速滤波器系数访问
 *   - 每个线程负责计算输出序列中的一个时间点
 *   - 支持 batch 和多通道并行
 */
__global__ void fir_filter_kernel(
    const float* __restrict__ input,     // [batch*channel, time_steps] 展平后的输入
    float* __restrict__ output,          // 同上，输出
    int batch_size,
    int num_channels,
    int time_steps,
    int filter_len
) {
    // blockIdx.y 负责 (batch, channel) 维度
    // blockIdx.x + threadIdx.x 负责时间维度
    const int bc_idx = blockIdx.y;                    // batch × channel 全局索引
    const int t_idx  = threadIdx.x + blockIdx.x * blockDim.x;   // 当前时间索引

    if (bc_idx >= batch_size * num_channels || t_idx >= time_steps) {
        return;
    }

    // 该通道在展平后的一维起始位置
    const int start_idx = bc_idx * time_steps;

    // ==================== 计算当前输出的 FIR 加权和 ====================
    float sum = 0.0f;

    // 因果卷积：只累加过去 filter_len 个采样点（含当前点）
    // i=0 对应当前时刻，i=filter_len-1 对应最早使用的过去时刻
    for (int i = 0; i < filter_len; ++i) {
        int data_t = t_idx - i;                     // 查找过去第 i 个时刻
        if (data_t >= 0) {
            sum += input[start_idx + data_t] * d_filter_weights[i];
        }
        // data_t < 0 时隐式补零（不做任何操作）
    }

    output[start_idx + t_idx] = sum;
}


/**
 * PyTorch 绑定函数：运行因果 FIR 滤波器
 *
 * 输入：
 *   input   : [batch_size, num_channels, time_steps]，float32，CUDA tensor
 *   weights : [filter_len]，滤波器系数（ impulse response ），float32，CUDA tensor
 *
 * 输出：
 *   输出张量形状与 input 完全一致，同样为因果滤波结果
 *
 * 注意：
 *   - 滤波器长度不能超过 MAX_FILTER_LEN (256)
 *   - 内部会自动把 weights 拷贝到 GPU 常量内存以加速访问
 */
torch::Tensor run_fir_filter(torch::Tensor input, torch::Tensor weights) {
    // ====================== 1. 输入合法性检查 ======================
    TORCH_CHECK(input.is_cuda() && weights.is_cuda(),
                "Both input and weights must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "Input must be float32 type");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32,
                "Weights must be float32 type");
    TORCH_CHECK(input.dim() == 3,
                "Input must be 3D tensor [batch_size, num_channels, time_steps]");
    TORCH_CHECK(weights.dim() == 1,
                "Weights must be 1D tensor [filter_len]");
    TORCH_CHECK(weights.size(0) <= MAX_FILTER_LEN,
                "Filter length exceeds maximum allowed length (256)");

    // ====================== 2. 准备数据 ======================
    auto input_c   = input.contiguous();
    auto weights_c = weights.contiguous();
    auto output    = torch::empty_like(input_c);     // 输出与输入形状相同

    const int batch_size  = input_c.size(0);
    const int num_channels = input_c.size(1);
    const int time_steps   = input_c.size(2);
    const int filter_len   = weights_c.size(0);

    // ====================== 3. 将滤波器系数拷贝到常量内存 ======================
    // 常量内存对所有线程广播只读，访问延迟极低，非常适合滤波器这类固定系数
    cudaMemcpyToSymbol(d_filter_weights,
                      weights_c.data_ptr<float>(),
                      filter_len * sizeof(float),
                      0,                           // offset
                      cudaMemcpyDeviceToDevice);

    // ====================== 4. 配置并启动 Kernel ======================
    // 时间维度并行度：每个线程计算一个时间点
    const int threads_x = 256;                                   // 经验值，较好的 occupancy
    const int blocks_x  = (time_steps + threads_x - 1) / threads_x;

    // batch × channel 维度使用 y 方向并行（避免过多 block）
    const int blocks_y = batch_size * num_channels;

    dim3 threads(threads_x, 1, 1);
    dim3 blocks(blocks_x, blocks_y, 1);

    fir_filter_kernel<<<blocks, threads>>>(
        input_c.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        time_steps,
        filter_len
    );

    // 添加错误检查
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "FIR filter kernel launch failed: ", cudaGetErrorString(err));

    return output;
}