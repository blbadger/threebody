from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup(
#     name='threebody',
#     ext_modules=[
#         CUDAExtension('lltm_cuda', [
#             'threebody_divergence.cpp',
#             'divergence_kernel.cu',
#         ])
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })

from torch.utils.cpp_extension import load

lltm = load(name='divergence', sources=['divergence.cu'])