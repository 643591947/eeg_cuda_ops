#include <torch/extension.h>
#include <cuda_runtime.h>

// 定义每个 Block 的线程数
#define THREADS_PER_BLOCK 256

__global__ void centering_kernel_optimized(
    float* __restrict__ data,
    int batch_size,
    int num_channels,
    int time_steps
) {
    int bc_idx = blockIdx.x;
    int tid = threadIdx.x;

    // 安全检查，防止越界
    if (bc_idx >= batch_size * num_channels) return;

    // 定位到当前通道的数据起始指针
    float* channel_data = data + bc_idx * time_steps;

    // 申请共享内存，用于 Block 内部的规约求和
    __shared__ float sdata[THREADS_PER_BLOCK];
    // 用于广播最终的均值
    __shared__ float s_mean;

    // 阶段 1：多线程并行读取数据，计算局部和 (网格跨步)
    float local_sum = 0.0f;
    for (int t = tid; t < time_steps; t += blockDim.x) {
        local_sum += channel_data[t];
    }
    sdata[tid] = local_sum;
    __syncthreads(); // 等待所有线程把局部和写入共享内存

    // 阶段 2：共享内存内的树状并行规约 (求出该通道总和)
    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads(); // 每一层加法后同步
    }

    // 阶段 3：计算均值并由 0 号线程广播
    if (tid == 0) {
        s_mean = sdata[0] / time_steps;
    }
    __syncthreads(); // 确保所有线程都能读到算好的均值

    // 阶段 4：多线程并行减去均值 (网格跨步)
    for (int t = tid; t < time_steps; t += blockDim.x) {
        channel_data[t] -= s_mean;
    }
}

torch::Tensor run_centering(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto input_ = input.clone();
    auto input_c = input_.contiguous();

    int batch_size = input_c.size(0);
    int num_channels = input_c.size(1);
    int time_steps = input_c.size(2);

    // Block 数量等于 (Batch * Channel)，每个 Block 负责一个通道
    int blocks = batch_size * num_channels;
    // 每个 Block 固定 256 个线程协作
    int threads = THREADS_PER_BLOCK;
    
    centering_kernel_optimized<<<blocks, threads>>>(
        input_c.data_ptr<float>(), batch_size, num_channels, time_steps);
    
    return input_c;
}