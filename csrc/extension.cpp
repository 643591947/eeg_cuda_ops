#include <torch/extension.h>

// 声明
extern torch::Tensor run_fir_filter(torch::Tensor input, torch::Tensor weights);
extern torch::Tensor run_centering(torch::Tensor input);
extern std::tuple<torch::Tensor, torch::Tensor> run_whitening(torch::Tensor centered_data);
extern std::tuple<torch::Tensor, torch::Tensor> run_fastica_iter(torch::Tensor whitened_data, int max_iter, float tol);

// 直接绑定到底层函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fir_filter", &run_fir_filter, "Simple FIR Filter (CUDA)");
    m.def("centering", &run_centering, "Channel-wise Centering");
    m.def("whitening", &run_whitening, "Data Whitening via ATen (CUDA)");
    m.def("fastica_iter", &run_fastica_iter, "FastICA Fixed-point Iteration (CUDA)",
          pybind11::arg("whitened_data"), pybind11::arg("max_iter") = 200, pybind11::arg("tol") = 1e-4);
}