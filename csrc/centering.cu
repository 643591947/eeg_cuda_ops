#include <torch/extension.h>
#include <cuda_runtime.h>

// 每个 block 处理一个 (batch, channel) 的时间序列
#define THREADS_PER_BLOCK 1024

/**
 * 优化的 Centering Kernel（均值归零）
 *
 * 对输入张量 [batch_size, num_channels, time_steps] 的每个 channel 独立进行：
 *   output = input - mean(input, dim=2)
 *
 * 优化点：
 *   1. 使用 float4 向量化加载/存储（每次处理 4 个 float）
 *   2. 两阶段并行归约（warp shuffle + shared memory）
 *   3. 支持 padding 以保证时间步能被 4 整除
 */
__global__ void centering_kernel_optimized(
    const float* __restrict__ input_data,   // [batch*channel, padded_time_steps]
    float* __restrict__ output_data,        // 同上
    int batch_size,
    int num_channels,
    int padded_time_steps,   // 必须是 4 的倍数
    int orig_time_steps      // 原始未 padding 的长度（用于计算均值）
) {
    const int bid = blockIdx.x;                    // 全局 channel id: bid = batch_idx * num_channels + channel_idx
    const int tid = threadIdx.x;                   // 线程在线程块内的 id

    if (bid >= batch_size * num_channels) return;

    // 将指针转换为 float4* 以实现向量化访问
    const float4* in_channel_vec  = reinterpret_cast<const float4*>(input_data + bid * padded_time_steps);
    float4*       out_channel_vec = reinterpret_cast<float4*>(output_data + bid * padded_time_steps);

    const int time_steps_vec = padded_time_steps / 4;   // 向量化后的长度

    // ==================== Step 1: 每个线程局部求和 ====================
    float local_sum = 0.0f;
    for (int t = tid; t < time_steps_vec; t += blockDim.x) {
        float4 val = in_channel_vec[t];
        local_sum += val.x + val.y + val.z + val.w;     // 一次加载 4 个值
    }

    // ==================== Step 2: Block 内并行归约求总和并计算 mean ====================
    float sum = local_sum;

    // 第一阶段：warp 内使用 shuffle 快速归约（效率极高）
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    // 把每个 warp 的归约结果写到 shared memory（每个 warp 一个值）
    extern __shared__ float sdata[];
    __shared__ float s_mean;                     // 最终的 mean 值，所有线程可见

    if ((tid % 32) == 0) {
        sdata[tid / 32] = sum;                   // warp leader 写入
    }
    __syncthreads();

    // 第二阶段：只用前 32 个线程对 warp 结果继续归约（最多 32 个 warp）
    if (tid < 32) {
        float final_sum = (tid < (blockDim.x / 32)) ? sdata[tid] : 0.0f;

        final_sum += __shfl_down_sync(0xffffffff, final_sum, 16);
        final_sum += __shfl_down_sync(0xffffffff, final_sum, 8);
        final_sum += __shfl_down_sync(0xffffffff, final_sum, 4);
        final_sum += __shfl_down_sync(0xffffffff, final_sum, 2);
        final_sum += __shfl_down_sync(0xffffffff, final_sum, 1);

        if (tid == 0) {
            s_mean = final_sum / orig_time_steps;   // 注意：用原始长度计算均值（padding 部分不参与）
        }
    }
    __syncthreads();   // 确保所有线程都能读到 s_mean

    // ==================== Step 3: 减去均值并写回（同样向量化） ====================
    for (int t = tid; t < time_steps_vec; t += blockDim.x) {
        float4 val = in_channel_vec[t];
        float4 out_val;

        out_val.x = val.x - s_mean;
        out_val.y = val.y - s_mean;
        out_val.z = val.z - s_mean;
        out_val.w = val.w - s_mean;

        out_channel_vec[t] = out_val;
    }
}


/**
 * PyTorch 绑定函数：对输入张量执行 centering 操作
 *
 * 输入要求：
 *   - CUDA tensor
 *   - float32 类型
 *   - time_steps 维度能被 4 整除（内部会自动 padding）
 */
torch::Tensor run_centering(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D tensor [B, C, T]");

    auto input_c = input.contiguous();
    auto output = torch::empty_like(input_c);

    const int batch_size     = input_c.size(0);
    const int num_channels   = input_c.size(1);
    const int orig_time_steps = input_c.size(2);

    // ====================== Padding 处理（保证 float4 对齐） ======================
    int pad_size = (4 - (orig_time_steps % 4)) % 4;
    if (pad_size > 0) {
        namespace F = torch::nn::functional;
        input_c = F::pad(input_c, F::PadFuncOptions({0, pad_size})).contiguous();
    }
    const int padded_time_steps = input_c.size(2);

    // ====================== Kernel 启动参数 ======================
    const int blocks = batch_size * num_channels;           // 每个 (batch, channel) 一个 block
    const int threads = THREADS_PER_BLOCK;

    // shared memory 大小：每个 warp 一个 float（最多 threads/32 个）
    const int shared_mem_bytes = (threads / 32) * sizeof(float);

    centering_kernel_optimized<<<blocks, threads, shared_mem_bytes>>>(
        input_c.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        padded_time_steps,
        orig_time_steps          // 注意参数顺序：padded 在前，orig 在后
    );

    // ====================== 去掉 padding ======================
    if (pad_size > 0) {
        output = output.slice(2, 0, orig_time_steps).contiguous();
    }

    return output;
}