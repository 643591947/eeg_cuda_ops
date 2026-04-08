#include <torch/extension.h>
#include <cuda_runtime.h>

// -------------------------------------------------------------------------
// 核心核函数
// -------------------------------------------------------------------------
#define MAX_FILTER_LEN 256
__constant__ float d_filter_weights[MAX_FILTER_LEN];
__global__ void fir_filter_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int num_channels,
    int time_steps,
    int filter_len
) {
    int bc_idx = blockIdx.y;
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (bc_idx < (batch_size * num_channels) && t_idx < time_steps) {
        float sum = 0.0f;
        int start_idx = bc_idx * time_steps;

        // 因果性，当前线程计算的是前几项加权之和
        for (int i = 0; i < filter_len; ++i) {
            int data_t = t_idx - i;
            if (data_t >= 0) {
                sum += input[start_idx + data_t] * d_filter_weights[i];
            }
        }
        output[start_idx + t_idx] = sum;
    }
}



// -------------------------------------------------------------------------
// C++ 宿主函数
// -------------------------------------------------------------------------
torch::Tensor run_fir_filter(torch::Tensor input, torch::Tensor weights) {
    // 1. 张量合法性校验
    TORCH_CHECK(input.is_cuda() && weights.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weights.size(0) <= MAX_FILTER_LEN, "Filter length exceeds 256");

    // 2. 确保内存连续并分配输出空间
    auto input_c = input.contiguous();
    auto weights_c = weights.contiguous();
    auto output = torch::empty_like(input_c);

    int batch_size = input_c.size(0);
    int num_channels = input_c.size(1);
    int time_steps = input_c.size(2);
    int filter_len = weights_c.size(0);

    // 3. 拷贝权重到 GPU 常量内存
    cudaMemcpyToSymbol(d_filter_weights, weights_c.data_ptr<float>(), filter_len * sizeof(float));

    // 4. 配置并启动 Kernel
    int threads_x = 256;
    int blocks_x = (time_steps + threads_x - 1) / threads_x;
    int blocks_y = batch_size * num_channels;

    dim3 threads(threads_x, 1, 1);
    dim3 blocks(blocks_x, blocks_y, 1);

    fir_filter_kernel<<<blocks, threads>>>(
        input_c.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, num_channels, time_steps, filter_len);

    return output;
}