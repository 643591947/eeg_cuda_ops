#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024

__global__ void centering_kernel_optimized(
    const float* __restrict__ input_data,
    float* __restrict__ output_data,
    int batch_size,
    int num_channels,
    int padded_time_steps,
    int orig_time_steps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    if (bid >= batch_size * num_channels) return;

    // Padding offset
    const float4* in_channel_vec = reinterpret_cast<const float4*>(input_data + bid * padded_time_steps);
    float4* out_channel_vec = reinterpret_cast<float4*>(output_data + bid * padded_time_steps);

    int time_steps_vec = padded_time_steps / 4;

    // Step 1
    float local_sum = 0.0f;
    for (int t = tid; t < time_steps_vec; t += blockDim.x) {
        float4 val = in_channel_vec[t];
        local_sum += val.x + val.y + val.z + val.w;
    }

    // Step 2
    float sum = local_sum;
    extern __shared__ float sdata[];
    __shared__ float s_mean;

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    if ((tid % 32) == 0) sdata[tid / 32] = sum;
    __syncthreads();

    if (tid < 32) {
        float final_sum = (tid < (blockDim.x / 32)) ? sdata[tid] : 0.0f;
        final_sum += __shfl_down_sync(0xffffffff, final_sum, 16);
        final_sum += __shfl_down_sync(0xffffffff, final_sum, 8);
        final_sum += __shfl_down_sync(0xffffffff, final_sum, 4);
        final_sum += __shfl_down_sync(0xffffffff, final_sum, 2);
        final_sum += __shfl_down_sync(0xffffffff, final_sum, 1);

        if (tid == 0) {
            s_mean = final_sum / orig_time_steps;
        }
    }
    __syncthreads();

    // Step 3
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

torch::Tensor run_centering(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.size(2) % 4 == 0, "time_steps must be a multiple of 4 for float4 vectorization");

    auto input_c = input.contiguous();
    auto output = torch::empty_like(input_c);

    int batch_size = input_c.size(0);
    int num_channels = input_c.size(1);
    int orig_time_steps = input_c.size(2);

    // Padding
    int pad_size = (4 - (orig_time_steps % 4)) % 4;
    if (pad_size > 0) {
        namespace F = torch::nn::functional;
        input_c = F::pad(input_c, F::PadFuncOptions({0, pad_size})).contiguous();
    }
    int padded_time_steps = input_c.size(2);

    int blocks = batch_size * num_channels;
    int threads = THREADS_PER_BLOCK;

    int shared_mem_bytes = (threads / 32) * sizeof(float);

    centering_kernel_optimized<<<blocks, threads, shared_mem_bytes>>>(
        input_c.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        orig_time_steps,
        padded_time_steps
    );

    if (pad_size > 0) {
        output = output.slice(2, 0, orig_time_steps);
    }
    
    return output;
}