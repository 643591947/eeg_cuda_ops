#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> run_fastica_iter(torch::Tensor whitened_data, int max_iter = 200, float tol = 1e-4) {
    // 数据合法性校验
    TORCH_CHECK(whitened_data.is_cuda(), "FastICA inputs must be CUDA tensors");

    auto X = whitened_data.contiguous();
    int batch_size = X.size(0);
    int num_channels = X.size(1);
    int time_steps = X.size(2);

    // 1. 随机初始化解混矩阵 W [Batch, Channels, Channels]
    auto options = torch::TensorOptions().dtype(X.dtype()).device(X.device());
    auto W = torch::randn({batch_size, num_channels, num_channels}, options);

    // 初始对称正交化 W = (W * W^T)^{-1/2} * W
    auto W_WT = torch::bmm(W, W.transpose(1, 2));
    auto eigh_res_init = torch::linalg_eigh(W_WT);
    auto D_inv_init = torch::diag_embed(std::get<0>(eigh_res_init).clamp_min(1e-5).pow(-0.5));
    auto E_init = std::get<1>(eigh_res_init);
    auto sym_ortho_init = torch::bmm(torch::bmm(E_init, D_inv_init), E_init.transpose(1, 2));
    W = torch::bmm(sym_ortho_init, W);

    // 2. 开始不动点迭代
    for (int i = 0; i < max_iter; ++i) {
        auto W_old = W.clone();

        // U = W * X  (形状: [Batch, Channels, Time])
        auto U = torch::bmm(W, X);

        // g(U) = tanh(U)
        auto g_U = torch::tanh(U);

        // g'(U) = 1 - tanh(U)^2
        auto g_prime_U = 1.0f - g_U.pow(2);

        // 第一部分: (1/T) * g(U) * X^T
        auto part1 = torch::bmm(g_U, X.transpose(1, 2)) / time_steps;

        // 第二部分: mean(g'(U)) * W
        // 沿时间维度求均值，形状变为 [Batch, Channels, 1]
        auto g_prime_mean = g_prime_U.mean(/*dim=*/2, /*keepdim=*/true);
        auto part2 = g_prime_mean * W;

        // 更新 W
        W = part1 - part2;

        // 对称正交化
        auto W_WT_new = torch::bmm(W, W.transpose(1, 2));
        auto eigh_res = torch::linalg_eigh(W_WT_new);
        auto D_inv = torch::diag_embed(std::get<0>(eigh_res).clamp_min(1e-5).pow(-0.5));
        auto E = std::get<1>(eigh_res);
        auto sym_ortho = torch::bmm(torch::bmm(E, D_inv), E.transpose(1, 2));
        W = torch::bmm(sym_ortho, W);

        // 收敛检查：计算 W 和 W_old 的对角线乘积的绝对值 (看是否接近 1)
        auto cos_theta = torch::bmm(W, W_old.transpose(1, 2));

        // 取对角线元素的绝对值
        auto diag_cos = cos_theta.diagonal(0, 1, 2).abs();

        // 如果 1 - cos_theta 的最大值小于 tol，则认为收敛
        auto diff = 1.0f - diag_cos;
        float max_diff = diff.max().item<float>();

        if (max_diff < tol) {
            std::cout << "FastICA converged at iteration " << i << std::endl;
            break;
        }
    }

    // 返回分离出的独立成分 (Independent Components): S = W * X
    auto S = torch::bmm(W, X);
    return std::make_tuple(S, W);
}