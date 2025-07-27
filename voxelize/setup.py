from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='udf_ext',
    ext_modules=[
        CUDAExtension('udf_ext', [
            'src/udf_kernel.cu',
            'src/udf_cuda.cpp'
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

