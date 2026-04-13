#include <torch/extension.h>
#include <tuple>

/**
 * FastICA 单批次不动点迭代（PyTorch C++ 实现）
 *
 * 输入：
 *   whitened_data: [batch_size, num_channels, time_steps]，已经白化（whitened）后的数据
 *
 * 输出：
 *   std::tuple<S, W>
 *     - S: 分离出的独立成分 (Independent Components)，形状 [batch_size, num_channels, time_steps]
 *     - W: 最终学到的解混矩阵 (unmixing matrix)，形状 [batch_size, num_channels, num_channels]
 *
 * 算法基于经典 FastICA 不动点迭代（tanh 非线性），每 batch 独立计算。
 * 每次迭代后都进行对称正交化，保证 W 保持正交性。
 */
std::tuple<torch::Tensor, torch::Tensor> run_fastica_iter(
    torch::Tensor whitened_data,
    int max_iter = 200,
    float tol = 1e-4
) {
    // ====================== 输入合法性检查 ======================
    TORCH_CHECK(whitened_data.is_cuda(), "FastICA inputs must be CUDA tensors");
    TORCH_CHECK(whitened_data.dim() == 3, "whitened_data must be 3D: [batch_size, num_channels, time_steps]");

    auto X = whitened_data.contiguous();        // 确保内存连续，提升后续性能
    const int batch_size  = X.size(0);
    const int num_channels = X.size(1);
    const int time_steps   = X.size(2);

    // ====================== 1. 随机初始化解混矩阵 W ======================
    // W 的形状: [batch_size, num_channels, num_channels]
    auto options = torch::TensorOptions().dtype(X.dtype()).device(X.device());
    auto W = torch::randn({batch_size, num_channels, num_channels}, options);

    // 初始对称正交化：W ← (W W^T)^{-1/2} W
    // 这一步确保初始 W 是正交矩阵，避免后续迭代数值不稳定
    {
        auto W_WT = torch::bmm(W, W.transpose(1, 2));
        auto eigh_res = torch::linalg_eigh(W_WT);           // 返回 (eigenvalues, eigenvectors)
        auto D = std::get<0>(eigh_res).clamp_min(1e-5);    // 防止除零
        auto D_inv_sqrt = torch::diag_embed(D.pow(-0.5f));
        auto E = std::get<1>(eigh_res);

        auto sym_ortho = torch::bmm(torch::bmm(E, D_inv_sqrt), E.transpose(1, 2));
        W = torch::bmm(sym_ortho, W);
    }

    // ====================== 2. 不动点迭代 (Fixed-Point Iteration) ======================
    for (int i = 0; i < max_iter; ++i) {
        auto W_old = W.clone();     // 保存旧 W 用于收敛判断

        // ---------- Step 1: 计算当前估计的独立成分 U = W X ----------
        auto U = torch::bmm(W, X);                    // [B, C, T]

        // ---------- Step 2: 计算非线性函数及其导数 ----------
        auto g_U       = torch::tanh(U);              // g(u) = tanh(u)
        auto g_prime_U = 1.0f - g_U.pow(2);           // g'(u) = 1 - tanh(u)^2

        // ---------- Step 3: 计算更新方向 ----------
        // part1 = (1/T) * g(U) * X^T
        auto part1 = torch::bmm(g_U, X.transpose(1, 2)) / time_steps;

        // part2 = mean(g'(U), dim=2) * W   （沿时间维度取均值）
        auto g_prime_mean = g_prime_U.mean(/*dim=*/2, /*keepdim=*/true);  // [B, C, 1]
        auto part2 = g_prime_mean * W;                                     // 广播乘法

        // 更新 W（这是 FastICA 的核心不动点公式）
        W = part1 - part2;

        // ---------- Step 4: 对称正交化（Symmetrical Orthogonalization）----------
        {
            auto W_WT = torch::bmm(W, W.transpose(1, 2));
            auto eigh_res = torch::linalg_eigh(W_WT);
            auto D = std::get<0>(eigh_res).clamp_min(1e-5);
            auto D_inv_sqrt = torch::diag_embed(D.pow(-0.5f));
            auto E = std::get<1>(eigh_res);

            auto sym_ortho = torch::bmm(torch::bmm(E, D_inv_sqrt), E.transpose(1, 2));
            W = torch::bmm(sym_ortho, W);
        }

        // ---------- Step 5: 收敛判断 ----------
        // 计算 W 和 W_old 的“角度相似度”（对角线元素的 cos θ）
        auto cos_theta = torch::bmm(W, W_old.transpose(1, 2));
        auto diag_cos  = cos_theta.diagonal(0, 1, 2).abs();   // 取对角线绝对值

        auto diff = 1.0f - diag_cos;
        float max_diff = diff.max().item<float>();

        if (max_diff < tol) {
            std::cout << "FastICA converged at iteration " << i << std::endl;
            break;
        }
    }

    // ====================== 3. 计算最终独立成分 ======================
    // S = W * X  （分离出的独立源信号）
    auto S = torch::bmm(W, X);

    return std::make_tuple(S, W);
}