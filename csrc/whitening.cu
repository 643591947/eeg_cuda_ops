#include <torch/torch.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> run_whitening(torch::Tensor centered_data) {
    // 数据合法性校验
    TORCH_CHECK(centered_data.is_cuda(), "Whitening inputs must be CUDA tensors");

    auto X_f32 = centered_data.contiguous();
    int batch_size = X_f32.size(0);
    int num_channels = X_f32.size(1);
    int time_steps = X_f32.size(2);

    // ================= 极高精度计算区 =================
    // 强制 PyTorch 禁用 TF32 硬件加速，确保协方差矩阵的绝对精度
    auto X = X_f32.to(torch::kFloat64);

    // 1. 计算批次协方差矩阵: X * X^T (此时为 F64 级别乘法)
    auto X_T = X.transpose(1, 2);
    auto cov = torch::bmm(X, X_T) / (time_steps - 1);

    // 2. 特征值分解
    auto eigh_result = torch::linalg_eigh(cov);
    auto eigenvalues = std::get<0>(eigh_result);   // [Batch, Channels], F64
    auto eigenvectors = std::get<1>(eigh_result);  // [Batch, Channels, Channels], F64

    // 3. 构建白化矩阵
    auto max_eig = std::get<0>(eigenvalues.abs().max(-1, true));
    auto eps = max_eig * 1e-6;

    auto D_inv_sqrt = torch::maximum(eigenvalues.abs(), eps).pow(-0.5);
    auto D_mat = torch::diag_embed(D_inv_sqrt);

    auto P = torch::bmm(eigenvectors, torch::bmm(D_mat, eigenvectors.transpose(1, 2)));

    // 4. 应用白化矩阵: X_white = P * X (依旧在 F64 下进行，保证最后的乘法不被污染)
    auto whitened_data_f64 = torch::bmm(P, X);

    // ==============================================
    // 5. 安全降级回 Float32 返回给后续网络
    return std::make_tuple(whitened_data_f64.to(torch::kFloat32), P.to(torch::kFloat32));

}