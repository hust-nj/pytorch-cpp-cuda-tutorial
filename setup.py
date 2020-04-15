from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['./models/ops/lltm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(
    name='lltm_cuda',
    ext_modules=[
        cpp_extension.CUDAExtension('lltm_cuda', [
            './models/ops/lltm_cuda.cpp',
            './models/ops/lltm_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    })
