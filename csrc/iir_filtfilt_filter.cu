#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include <string>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define THREADS_PER_BLOCK 256
// Omega
float Omega(float sfre, float cfre) {
    return tanf(M_PI * cfre / sfre);
}

// a b computation
std::pair<std::vector<float>, std::vector<float>> a_b(
    torch::Tensor sfre,
    torch::Tensor cfre,
    const std::string& pass,   // 'low' or 'high'
    bool bandstop
) {
    std::vector<float> b(3, 0.0f);
    std::vector<float> a(3, 0.0f);

    float sfre_val = sfre.item<float>();

    // a、b computation
    if (cfre.numel() == 2) {
        TORCH_CHECK(cfre.sizes().size() == 1, "Cutoff frequency array must be 1-dimensional");
        TORCH_CHECK(cfre.size(0) == 2, "Cutoff frequency array must have exactly 2 elements");

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
            // bandstop
            b[0] = BW * a0_inv;
            b[1] = 0.0f;
            b[2] = -BW * a0_inv;
        }else {
            // no bandstop
            b[0] = (1.0f + omega_2) * a0_inv;
            b[1] = 2.0f * (omega_2 - 1.0f) * a0_inv;
            b[2] = (1.0f + omega_2) * a0_inv;
        }
    }else if (cfre.numel() == 1) {
        float omega = Omega(sfre_val, cfre[0].item<float>());
        float sqrt2 = std::sqrt(2.0f);
        float omega_sq = omega * omega;
        float a0_inv = 1.0f / (1.0f + sqrt2 * omega + omega_sq);

        a[0] = 1.0f;
        a[1] = 2.0f * (omega_sq - 1.0f) * a0_inv;
        a[2] = (1.0f - sqrt2 * omega + omega_sq) * a0_inv;
        if (pass == "low") {
            // lowpass
            b[0] = omega_sq * a0_inv;
            b[1] = 2.0f * omega_sq * a0_inv;
            b[2] = omega_sq * a0_inv;
        }else if (pass == "high") {
            // highpass
            b[0] = 1.0f * a0_inv;
            b[1] = -2.0f * a0_inv;
            b[2] = 1.0f * a0_inv;
        }
    }else {
        TORCH_CHECK(false, "Cutoff frequency must be either a single number or an array of 2 numbers, got ", cfre.numel(), " elements");
    }

    return {b, a};
}


struct IIRMatrix {
    double m00, m01, m02;
    double m10, m11, m12;

    __device__ __host__ static IIRMatrix multiply(const IIRMatrix& A, const IIRMatrix& B) {
        IIRMatrix C;

        C.m00 = A.m00 * B.m00 + A.m01 * B.m10;
        C.m01 = A.m00 * B.m01 + A.m01 * B.m11;
        C.m02 = A.m00 * B.m02 + A.m01 * B.m12 + A.m02;

        C.m10 = A.m10 * B.m00 + A.m11 * B.m10;
        C.m11 = A.m10 * B.m01 + A.m11 * B.m11;
        C.m12 = A.m10 * B.m02 + A.m11 * B.m12 + A.m12;

        return C;
    }
};

struct IIRMatrixScanOp {
    __device__ __host__
    IIRMatrix operator()(const IIRMatrix& A, const IIRMatrix& B) const {
        return IIRMatrix::multiply(B, A);
    }
};

struct SegmentKeyOp {
    int T;
    SegmentKeyOp(int T) : T(T) {}
    __host__ __device__
    int operator()(const int& i) const {
        return i / T;
    }
};

// Step 1 : Generate the initial state transition matrix Mn based on the input.
__global__ void iir_generate_matrices_kernel(
    const float* __restrict__ input,
    IIRMatrix* __restrict__ matrices,
    double b0, double b1, double b2,
    double a0, double a1, double a2,
    int total_elements,
    int time_steps
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < total_elements) {
        int t = n % time_steps;

        double x_n = input[n];
        double x_n_1 = (t >= 1) ? input[n - 1] : 0.0f;
        double x_n_2 = (t >= 2) ? input[n - 2] : 0.0f;

        IIRMatrix mat;
        //[m00, m01, inputiterm]
        //[ 1 ,  0 , 0         ]
        mat.m00 = -a1 / a0;
        mat.m01 = -a2 / a0;
        mat.m02 = (b0 * x_n + b1 * x_n_1 + b2 * x_n_2) / a0;

        mat.m10 = 1.0f;
        mat.m11 = 0.0f;
        mat.m12 = 0.0f;

        matrices[n] = mat;
    }
}

// step 2 : Calculate the final y[n] based on the result of the prefix product.
__global__ void iir_extract_output_kernel(
    const IIRMatrix* __restrict__ scanned_matrices,
    float* __restrict__ output,
    int total_elements,
    double y_minus_1 = 0.0f,
    double y_minus_2 = 0.0f
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < total_elements) {
        IIRMatrix P = scanned_matrices[n];

        // S_n = P_n * S_{n-1}
        // S_{-1} = [y_minus_1, y_minus_2, 1]^T
        output[n] = static_cast<float>(P.m00 * y_minus_1 + P.m01 * y_minus_2 + P.m02 * 1.0);
    }
}

// forward
torch::Tensor run_iir_filter_forward(
    torch::Tensor input,
    torch::Tensor sfre,
    torch::Tensor cfre,
    const std::string& pass = "high",
    bool bandstop = false
) {
    // check data
    TORCH_CHECK(input.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(sfre.defined(), "Sampling frequency tensor must be specified");
    TORCH_CHECK(sfre.dtype() == torch::kInt, "Sampling frequency must be int type");
    TORCH_CHECK(cfre.defined(), "Cutoff frequency tensor must be specified");
    TORCH_CHECK(cfre.is_floating_point() || cfre.dtype() == torch::kInt, "Cutoff frequency must be either float or int type");

    // a、b
    auto result = a_b(sfre, cfre, pass, bandstop);
    std::vector<float> b = result.first;
    std::vector<float> a = result.second;

    auto input_c = input.contiguous();
    auto output = torch::empty_like(input_c);

    TORCH_CHECK(input_c.dim() == 3, "Input must be a 3D tensor (Batch, Channels, TimeSteps)");
    int batch_size = input_c.size(0);
    int num_channels = input_c.size(1);
    int time_steps = input_c.size(2);
    int total_elements = batch_size * num_channels * time_steps;

    // define kernel parameter
    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;

    // Allocate GPU memory for storing the matrix.
    IIRMatrix* d_matrices;
    cudaMalloc(&d_matrices, total_elements * sizeof(IIRMatrix));

    // Generate the initial matrix
    iir_generate_matrices_kernel<<<numBlocks, blockSize>>>(
        input_c.data_ptr<float>(),
        d_matrices,
        b[0], b[1], b[2],
        a[0], a[1], a[2],
        total_elements,
        time_steps
    );
    cudaDeviceSynchronize();

    // Perform Inclusive Scan using Thrust.
    thrust::device_ptr<IIRMatrix> dev_ptr_matrices(d_matrices);

    thrust::counting_iterator<int> count_iter(0);
    thrust::transform_iterator<SegmentKeyOp, thrust::counting_iterator<int>> keys_iter(count_iter, SegmentKeyOp(time_steps));

    thrust::inclusive_scan_by_key(
        keys_iter,
        keys_iter + total_elements,
        dev_ptr_matrices,
        dev_ptr_matrices,
        thrust::equal_to<int>(),
        IIRMatrixScanOp()
    );

    // Extract the output y_n.
    iir_extract_output_kernel<<<numBlocks, blockSize>>>(
        d_matrices,
        output.data_ptr<float>(),
        total_elements,
        0.0f, 0.0f
    );
    cudaDeviceSynchronize();

    cudaFree(d_matrices);

    return output;
}

// main
torch::Tensor run_iir_filtfilt(
    torch::Tensor input,
    torch::Tensor sfre,
    torch::Tensor cfre,
    const std::string& pass = "high",
    bool bandstop = false
) {
    // Forward
    auto y_forward = run_iir_filter_forward(input, sfre, cfre, pass, bandstop);

    // Reversed
    auto y_reversed = torch::flip(y_forward, /*dims=*/{2});

    // Backward
    auto y_backward = run_iir_filter_forward(y_reversed, sfre, cfre, pass, bandstop);

    // Reversed
    auto final_output = torch::flip(y_backward, /*dims=*/{2});

    return final_output;
}

