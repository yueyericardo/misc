from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob


def cuda_extension(build_all=False):
    import torch
    from torch.utils.cpp_extension import CUDAExtension
    SMs = None
    if not build_all:
        SMs = []
        devices = torch.cuda.device_count()
        print('FAST_BUILD: ON')
        print('This build will only support the following devices or the devices with same cuda capability: ')
        for i in range(devices):
            d = 'cuda:{}'.format(i)
            sm = torch.cuda.get_device_capability(i)
            sm = int(f'{sm[0]}{sm[1]}')
            if sm >= 50:
                print('{}: {}'.format(i, torch.cuda.get_device_name(d)))
                print('   {}'.format(torch.cuda.get_device_properties(i)))
            if sm not in SMs and sm >= 50:
                SMs.append(sm)

    nvcc_args = ["-Xptxas=-v", "-lineinfo"]
    if SMs:
        for sm in SMs:
            nvcc_args.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
    else:
        nvcc_args.append("-gencode=arch=compute_60,code=sm_60")
        nvcc_args.append("-gencode=arch=compute_61,code=sm_61")
        nvcc_args.append("-gencode=arch=compute_70,code=sm_70")
        cuda_version = float(torch.version.cuda)
        if cuda_version >= 10:
            nvcc_args.append("-gencode=arch=compute_75,code=sm_75")
        if cuda_version >= 11:
            nvcc_args.append("-gencode=arch=compute_80,code=sm_80")
        if cuda_version >= 11.1:
            nvcc_args.append("-gencode=arch=compute_86,code=sm_86")
    print("nvcc_args: ", nvcc_args)
    return CUDAExtension(
        name='mbtopk',
        pkg='mbtopk',
        sources=['topk.cu'],
        include_dirs=[],
        extra_compile_args={'cxx': ['-std=c++14'], 'nvcc': nvcc_args})

setup(
    name='mbtopk',
    ext_modules=[
        cuda_extension(),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
