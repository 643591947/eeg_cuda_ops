#include <torch/extension.h>
#include <string>

extern torch::Tensor run_fir_filter(torch::Tensor input, torch::Tensor weights);
extern torch::Tensor run_centering(torch::Tensor input);
extern std::tuple<torch::Tensor, torch::Tensor> run_whitening(torch::Tensor centered_data);
extern std::tuple<torch::Tensor, torch::Tensor> run_fastica_iter(torch::Tensor whitened_data, int max_iter = 200, float tol = 1e-4);

extern torch::Tensor run_iir_filtfilt_matrix(
    torch::Tensor input,
    torch::Tensor sfre,
    torch::Tensor cfre,
    const std::string& pass = "high",
    bool bandstop = false
);

extern torch::Tensor run_iir_filtfilt_norm(
    torch::Tensor input,
    torch::Tensor sfre,
    torch::Tensor cfre,
    const std::string& pass = "high",
    bool bandstop = false
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fir_filter", &run_fir_filter, "Simple FIR Filter (CUDA)");

    m.def("centering", &run_centering, "Channel-wise Centering");

    m.def("whitening", &run_whitening, "Data Whitening via ATen (CUDA)");

    m.def("fastica_iter", &run_fastica_iter, "FastICA Fixed-point Iteration (CUDA)",
          pybind11::arg("whitened_data"),
          pybind11::arg("max_iter") = 200,
          pybind11::arg("tol") = 1e-4);

    m.def("iir_filtfilt_matrix", &run_iir_filtfilt_matrix, "IIR Zero-phase Filter_matirx (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("sfre"),
          pybind11::arg("cfre"),
          pybind11::arg("pass") = "high",
          pybind11::arg("bandstop") = false);

    m.def("iir_filtfilt_norm", &run_iir_filtfilt_norm, "IIR Zero-phase Filter_norm (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("sfre"),
          pybind11::arg("cfre"),
          pybind11::arg("pass") = "high",
          pybind11::arg("bandstop") = false);
}