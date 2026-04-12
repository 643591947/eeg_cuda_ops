from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# pip install -e . --no-build-isolation

setup(
    name='eeg_cuda',
    version='0.1.1',
    ext_modules=[
        CUDAExtension(
            name='eeg_cuda',
            sources=[
                'csrc/extension.cpp',
                'csrc/fir_filter.cu',
                'csrc/centering.cu',
                'csrc/whitening.cu',
                'csrc/fastica.cu',
                'csrc/iir_filtfilt_filter_matrix.cu',
                'csrc/iir_filtfilt_filter_norm.cu'
            ]
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)