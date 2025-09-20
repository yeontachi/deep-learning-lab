'''
This file compiles the C++/CUDA Codes to
a python package my_quant_lib
'''
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name = 'my_quant_lib',
    ext_modules=[
        CUDAExtension(
            'my_quant_lib', # 라이브러리 이름
            [
                'quant/binding.cpp',
                'quant/kernel.cu',
            ]
        ),
    ],
    cmdclass = {
        'build_ext': BuildExtension
    }
)