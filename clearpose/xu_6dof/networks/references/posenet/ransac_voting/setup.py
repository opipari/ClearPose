from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ransac_voting_3d',
    ext_modules=[
        CUDAExtension('ransac_voting_3d', [
            './src/ransac_voting.cpp',
            './src/ransac_voting_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
