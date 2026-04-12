#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <utility>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define THREADS_PER_BLOCK 256

// ====================== 系数计算 ======================
static float Omega(float sfre, float cfre) {
    return tanf(M_PI * cfre / sfre);
}

static std::pair<std::vector<float>, std::vector<float>> a_b(
    torch::Tensor sfre,
    torch::Tensor cfre,
    const std::string& pass,
    bool bandstop
) {
    std::vector<float> b(3, 0.0f);
    std::vector<float> a(3, 0.0f);

    float sfre_val = sfre.item<float>();

    if (cfre.numel() == 2) {
        float f1 = cfre[0].item<float>();
        float f2 = cfre[1].item<float>();
        if (f1 > f2) std::swap(f1, f2);

        float low_omega = Omega(sfre_val, f1);
        float high_omega = Omega(sfre_val, f2);
        float BW = high_omega - low_omega;
        float omega_2 = low_omega * high_omega;
        float a0_inv = 1.0f / (1.0f + BW + omega_2);

        a[0] = 1.0f;
        a[1] = 2.0f * (omega_2 - 1.0f) * a0_inv;
        a[2] = (1.0f - BW + omega_2) * a0_inv;
        if (bandstop) {
            b[0] = BW * a0_inv;
            b[1] = 0.0f;
            b[2] = -BW * a0_inv;
        } else {
            b[0] = (1.0f + omega_2) * a0_inv;
            b[1] = 2.0f * (omega_2 - 1.0f) * a0_inv;
            b[2] = (1.0f + omega_2) * a0_inv;
        }
    } else if (cfre.numel() == 1) {
        float omega = Omega(sfre_val, cfre[0].item<float>());
        float sqrt2 = std::sqrt(2.0f);
        float omega_sq = omega * omega;
        float a0_inv = 1.0f / (1.0f + sqrt2 * omega + omega_sq);

        a[0] = 1.0f;
        a[1] = 2.0f * (omega_sq - 1.0f) * a0_inv;
        a[2] = (1.0f - sqrt2 * omega + omega_sq) * a0_inv;
        if (pass == "low") {
            b[0] = omega_sq * a0_inv;
            b[1] = 2.0f * omega_sq * a0_inv;
            b[2] = omega_sq * a0_inv;
        } else if (pass == "high") {
            b[0] = 1.0f * a0_inv;
            b[1] = -2.0f * a0_inv;
            b[2] = 1.0f * a0_inv;
        }
    } else {
        TORCH_CHECK(false, "Cutoff frequency must be either a single number or an array of 2 numbers");
    }

    return {b, a};
}

// ====================== 核心 CUDA Kernel ======================
__global__ void iir_biquad_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float b0, float b1, float b2,
    float a1, float a2,          // a[0] 永远是 1.0
    int batch,
    int channels,
    int time_steps
) {
    int stream_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_streams = batch * channels;
    if (stream_idx >= num_streams) return;

    // 该线程负责的信号起始地址
    const float* x = input + stream_idx * time_steps;
    float* y = output + stream_idx * time_steps;

    float w_prev1 = 0.0f;   // w[n-1]
    float w_prev2 = 0.0f;   // w[n-2]

    for (int t = 0; t < time_steps; ++t) {
        float x_t = x[t];

        // Direct Form II
        float w = x_t - a1 * w_prev1 - a2 * w_prev2;
        float y_t = b0 * w + b1 * w_prev1 + b2 * w_prev2;

        y[t] = y_t;

        w_prev2 = w_prev1;
        w_prev1 = w;
    }
}

// ====================== 单向 IIR（lfilter）======================
static torch::Tensor run_iir_filter_forward(
    torch::Tensor input,
    torch::Tensor sfre,
    torch::Tensor cfre,
    const std::string& pass = "high",
    bool bandstop = false
) {
    TORCH_CHECK(input.is_cuda() && input.dim() == 3, "Input must be CUDA tensor (B, C, T)");

    auto [b_vec, a_vec] = a_b(sfre, cfre, pass, bandstop);

    float b0 = b_vec[0], b1 = b_vec[1], b2 = b_vec[2];
    float a1 = a_vec[1], a2 = a_vec[2];   // a[0] == 1.0

    auto input_c = input.contiguous();
    auto output = torch::empty_like(input_c);

    int batch = input_c.size(0);
    int channels = input_c.size(1);
    int time_steps = input_c.size(2);

    int num_streams = batch * channels;
    int block_size = THREADS_PER_BLOCK;
    int grid_size = (num_streams + block_size - 1) / block_size;

    iir_biquad_kernel<<<grid_size, block_size>>>(
        input_c.data_ptr<float>(),
        output.data_ptr<float>(),
        b0, b1, b2,
        a1, a2,
        batch, channels, time_steps
    );

    cudaDeviceSynchronize();
    return output;
}

// ====================== 零相位 filtfilt ======================
torch::Tensor run_iir_filtfilt_norm(
    torch::Tensor input,
    torch::Tensor sfre,
    torch::Tensor cfre,
    const std::string& pass = "high",
    bool bandstop = false
) {
    // Forward
    auto y_forward = run_iir_filter_forward(input, sfre, cfre, pass, bandstop);

    // Reverse
    auto y_reversed = torch::flip(y_forward, {2});

    // Backward
    auto y_backward = run_iir_filter_forward(y_reversed, sfre, cfre, pass, bandstop);

    // Reverse again
    return torch::flip(y_backward, {2});
}