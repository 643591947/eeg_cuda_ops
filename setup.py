from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# pip install -e . --no-build-isolation

setup(
    name='eeg_cuda',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='eeg_cuda',
            sources=[
                'csrc/extension.cpp',
                'csrc/filter_kernel.cu',
                'csrc/centering_kernel.cu',
                'csrc/whitening.cu',
                'csrc/fastica.cu'
            ]
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)