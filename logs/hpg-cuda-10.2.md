# Torchani install on hipergator
https://github.com/aiqm/torchani/tree/master/torchani/cuaev


### Hipergator
summary: The problem comes from hpg does not have cuda/10.2 installed, The folloing is done when a non-matched cuda/10.0 version is used. Fatail error message is
```
nvcc fatal   : Unknown option '-generate-dependencies-with-compile'
```
#### To Repeat
path: /blue/roitberg/jinzexue/dev/test
nvcc info: 

```
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
```
```py
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly
python -c 'from torch.utils.collect_env import get_pretty_env_info; print(get_pretty_env_info())'
```
pytorch info:
```py
PyTorch version: 1.8.0.dev20210120
Is debug build: False
CUDA used to build PyTorch: 10.2
ROCM used to build PyTorch: N/A

OS: Red Hat Enterprise Linux Server release 7.7 (Maipo) (x86_64)
GCC version: (GCC) 4.8.5 20150623 (Red Hat 4.8.5-39)
Clang version: Could not collect
CMake version: version 3.18.2

Python version: 3.7 (64-bit runtime)
Is CUDA available: True
CUDA runtime version: Could not collect
GPU models and configuration:
GPU 0: GeForce RTX 2080 Ti
GPU 1: GeForce RTX 2080 Ti
GPU 2: GeForce RTX 2080 Ti
GPU 3: GeForce RTX 2080 Ti
GPU 4: GeForce RTX 2080 Ti
GPU 5: GeForce RTX 2080 Ti
GPU 6: GeForce RTX 2080 Ti
GPU 7: GeForce RTX 2080 Ti

Nvidia driver version: 440.64
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A

Versions of relevant libraries:
[pip] numpy==1.18.5
[pip] pytorch-memlab==0.2.1
[pip] torch==1.8.0.dev20210120
[pip] torchaudio==0.8.0a0+46171b9
[pip] torchvision==0.9.0.dev20210120
[conda] _pytorch_select           0.2                       gpu_0
[conda] blas                      1.0                         mkl
[conda] cudatoolkit               10.2.89              hfd86e86_1
[conda] magma-cuda100             2.5.2                         1    pytorch
[conda] mkl                       2020.2                      256
[conda] mkl-include               2020.2                      256
[conda] mkl-service               2.3.0            py37he904b0f_0
[conda] mkl_fft                   1.0.15           py37ha843d7b_0
[conda] mkl_random                1.1.1            py37h0573a6f_0
[conda] numpy                     1.18.5                   pypi_0    pypi
[conda] numpy-base                1.18.1           py37hde5b4d6_1
[conda] pytorch                   1.8.0.dev20210120 py3.7_cuda10.2.89_cudnn7.6.5_0    pytorch-nightly
[conda] pytorch-memlab            0.2.1                    pypi_0    pypi
[conda] torchani                  2.3.dev41+g9e79c5d.d20210114          pypi_0    pypi
[conda] torchaudio                0.8.0.dev20210120            py37    pytorch-nightly
[conda] torchsnooper              0.8                      pypi_0    pypi
[conda] torchvision               0.9.0.dev20210120      py37_cu102    pytorch-nightly
```
install
```bash
git clone git@github.com:aiqm/torchani.git
cd torchani/
module load cuda/10.0.130
module load gcc/7.3.0
python setup.py install --cuaev
```
error message
```
building 'torchani.cuaev' extension
creating /blue/roitberg/jinzexue/dev/test/torchani/build/temp.linux-x86_64-3.7
creating /blue/roitberg/jinzexue/dev/test/torchani/build/temp.linux-x86_64-3.7/torchani
creating /blue/roitberg/jinzexue/dev/test/torchani/build/temp.linux-x86_64-3.7/torchani/cuaev
Emitting ninja build file /blue/roitberg/jinzexue/dev/test/torchani/build/temp.linux-x86_64-3.7/build.ninja...
Compiling objects...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/1] /apps/compilers/cuda/10.0.130/bin/nvcc --generate-dependencies-with-compile --dependency-output /blue/roitberg/jinzexue/dev/test/torchani/build/temp.linux-x86_64-3.7/torchani/cuaev/aev.o.d -I/blue/roitberg/jinzexue/dev/test/torchani/include -I/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/torch/include -I/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -I/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/torch/include/TH -I/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/torch/include/THC -I/apps/compilers/cuda/10.0.130/include -I/ufrc/roitberg/jinzexue/program/anaconda3/include/python3.7m -c -c /blue/roitberg/jinzexue/dev/test/torchani/torchani/cuaev/aev.cu -o /blue/roitberg/jinzexue/dev/test/torchani/build/temp.linux-x86_64-3.7/torchani/cuaev/aev.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -Xptxas=-v --expt-extended-lambda -use_fast_math -gencode=arch=compute_75,code=sm_75 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=cuaev -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin gcc -std=c++14
FAILED: /blue/roitberg/jinzexue/dev/test/torchani/build/temp.linux-x86_64-3.7/torchani/cuaev/aev.o
/apps/compilers/cuda/10.0.130/bin/nvcc --generate-dependencies-with-compile --dependency-output /blue/roitberg/jinzexue/dev/test/torchani/build/temp.linux-x86_64-3.7/torchani/cuaev/aev.o.d -I/blue/roitberg/jinzexue/dev/test/torchani/include -I/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/torch/include -I/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -I/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/torch/include/TH -I/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/torch/include/THC -I/apps/compilers/cuda/10.0.130/include -I/ufrc/roitberg/jinzexue/program/anaconda3/include/python3.7m -c -c /blue/roitberg/jinzexue/dev/test/torchani/torchani/cuaev/aev.cu -o /blue/roitberg/jinzexue/dev/test/torchani/build/temp.linux-x86_64-3.7/torchani/cuaev/aev.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -Xptxas=-v --expt-extended-lambda -use_fast_math -gencode=arch=compute_75,code=sm_75 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=cuaev -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin gcc -std=c++14
nvcc fatal   : Unknown option '-generate-dependencies-with-compile'
ninja: build stopped: subcommand failed.
Traceback (most recent call last):
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/torch/utils/cpp_extension.py", line 1655, in _run_ninja_build
    env=env)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/subprocess.py", line 512, in run
    output=stdout, stderr=stderr)
subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "setup.py", line 106, in <module>
    **cuaev_kwargs()
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/setuptools/__init__.py", line 153, in setup
    return distutils.core.setup(**attrs)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/core.py", line 148, in setup
    dist.run_commands()
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/dist.py", line 966, in run_commands
    self.run_command(cmd)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/dist.py", line 985, in run_command
    cmd_obj.run()
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/setuptools/command/install.py", line 67, in run
    self.do_egg_install()
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/setuptools/command/install.py", line 109, in do_egg_install
    self.run_command('bdist_egg')
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/cmd.py", line 313, in run_command
    self.distribution.run_command(command)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/dist.py", line 985, in run_command
    cmd_obj.run()
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/setuptools/command/bdist_egg.py", line 167, in run
    cmd = self.call_command('install_lib', warn_dir=0)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/setuptools/command/bdist_egg.py", line 153, in call_command
    self.run_command(cmdname)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/cmd.py", line 313, in run_command
    self.distribution.run_command(command)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/dist.py", line 985, in run_command
    cmd_obj.run()
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/setuptools/command/install_lib.py", line 11, in run
    self.build()
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/command/install_lib.py", line 107, in build
    self.run_command('build_ext')
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/cmd.py", line 313, in run_command
    self.distribution.run_command(command)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/dist.py", line 985, in run_command
    cmd_obj.run()
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/setuptools/command/build_ext.py", line 79, in run
    _build_ext.run(self)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/command/build_ext.py", line 340, in run
    self.build_extensions()
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/torch/utils/cpp_extension.py", line 704, in build_extensions
    build_ext.build_extensions(self)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/command/build_ext.py", line 449, in build_extensions
    self._build_extensions_serial()
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/command/build_ext.py", line 474, in _build_extensions_serial
    self.build_extension(ext)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/setuptools/command/build_ext.py", line 196, in build_extension
    _build_ext.build_extension(self, ext)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/distutils/command/build_ext.py", line 534, in build_extension
    depends=ext.depends)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/torch/utils/cpp_extension.py", line 534, in unix_wrap_ninja_compile
    with_cuda=with_cuda)
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/torch/utils/cpp_extension.py", line 1351, in _write_ninja_file_and_compile_objects
    error_prefix='Error compiling objects for extension')
  File "/ufrc/roitberg/jinzexue/program/anaconda3/lib/python3.7/site-packages/torch/utils/cpp_extension.py", line 1665, in _run_ninja_build
    raise RuntimeError(message) from e
RuntimeError: Error compiling objects for extension
```

### Local desktop
nvcc info

```
nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_19:24:38_PDT_2019
Cuda compilation tools, release 10.2, V10.2.89
```

```py
python -c 'from torch.utils.collect_env import get_pretty_env_info; print(get_pretty_env_info())'
```


pytorch info:

```py
PyTorch version: 1.8.0.dev20210120
Is debug build: False
CUDA used to build PyTorch: 10.2
ROCM used to build PyTorch: N/A

OS: Ubuntu 18.04.4 LTS (x86_64)
GCC version: (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
Clang version: Could not collect
CMake version: version 3.14.0

Python version: 3.6 (64-bit runtime)
Is CUDA available: True
CUDA runtime version: 10.2.89
GPU models and configuration:
GPU 0: GeForce GTX 1080
GPU 1: GeForce GT 710

Nvidia driver version: 440.33.01
cuDNN version: /usr/lib/x86_64-linux-gnu/libcudnn.so.7.6.5
HIP runtime version: N/A
MIOpen runtime version: N/A

Versions of relevant libraries:
[pip3] numpy==1.18.1
[pip3] numpydoc==0.9.2
[pip3] pytorch-memlab==0.2.1
[pip3] torch==1.8.0.dev20210120
[pip3] torchani==2.2
[pip3] torchaudio==0.8.0a0+46171b9
[pip3] torchvision==0.9.0.dev20210120
[conda] blas                      1.0                         mkl
[conda] cudatoolkit               10.2.89              hfd86e86_1
[conda] magma-cuda101             2.5.2                         1    pytorch
[conda] mkl                       2020.0                      166
[conda] mkl-include               2020.0                      166
[conda] mkl-service               2.3.0            py36he904b0f_0
[conda] mkl_fft                   1.0.15           py36ha843d7b_0
[conda] mkl_random                1.1.0            py36hd6b4f25_0
[conda] numpy                     1.18.1           py36h4f9e942_0
[conda] numpy-base                1.18.1           py36hde5b4d6_1
[conda] numpydoc                  0.9.2                      py_0
[conda] pytorch                   1.8.0.dev20210120 py3.6_cuda10.2.89_cudnn7.6.5_0    pytorch-nightly
[conda] pytorch-memlab            0.2.1                    pypi_0    pypi
[conda] torchani                  2.2                      pypi_0    pypi
[conda] torchaudio                0.8.0.dev20210120            py36    pytorch-nightly
[conda] torchvision               0.9.0.dev20210120      py36_cu102    pytorch-nightly
```
```
/home/richard/program/anaconda3/envs/ml/lib/python3.6/distutils/extension.py:131: UserWarning: Unknown Extension options: 'pkg'
  warnings.warn(msg)
running install
running bdist_egg
running egg_info
writing torchani.egg-info/PKG-INFO
writing dependency_links to torchani.egg-info/dependency_links.txt
writing requirements to torchani.egg-info/requires.txt
writing top-level names to torchani.egg-info/top_level.txt
reading manifest template 'MANIFEST.in'
writing manifest file 'torchani.egg-info/SOURCES.txt'
installing library code to build/bdist.linux-x86_64/egg
running install_lib
running build_py
running build_ext
building 'torchani.cuaev' extension
Emitting ninja build file /home/richard/dev/test/torchani/build/temp.linux-x86_64-3.6/build.ninja...
Compiling objects...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/1] /usr/local/cuda-10.2/bin/nvcc --generate-dependencies-with-compile --dependency-output /home/richard/dev/test/torchani/build/temp.linux-x86_64-3.6/torchani/cuaev/aev.o.d -I/home/richard/dev/test/torchani/include -I/home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages/torch/include -I/home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -I/home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages/torch/include/TH -I/home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages/torch/include/THC -I/usr/local/cuda-10.2/include -I/home/richard/program/anaconda3/envs/ml/include/python3.6m -c -c /home/richard/dev/test/torchani/torchani/cuaev/aev.cu -o /home/richard/dev/test/torchani/build/temp.linux-x86_64-3.6/torchani/cuaev/aev.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -Xptxas=-v --expt-extended-lambda -use_fast_math -gencode=arch=compute_75,code=sm_75 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=cuaev -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/richard/dev/test/torchani/torchani/cuaev/aev.cu(225): warning: variable "aev_offset" was declared but never referenced
          detected during:
            instantiation of "void cuAngularAEVs(at::PackedTensorAccessor32<SpeciesT, 2UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 3UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 1UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 1UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 1UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 1UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 3UL, at::RestrictPtrTraits>, PairDist<DataT> *, PairDist<DataT> *, int *, int *, AEVScalarParams<DataT, IndexT>, int, int, int) [with SpeciesT=int, DataT=float, IndexT=int, TILEX=8, TILEY=4]"
(575): here
            instantiation of "at::Tensor cuComputeAEV(at::Tensor, at::Tensor, double, double, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t) [with ScalarRealT=float]"
(581): here

/home/richard/dev/test/torchani/torchani/cuaev/aev.cu(225): warning: variable "aev_offset" was declared but never referenced
          detected during:
            instantiation of "void cuAngularAEVs(at::PackedTensorAccessor32<SpeciesT, 2UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 3UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 1UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 1UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 1UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 1UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 3UL, at::RestrictPtrTraits>, PairDist<DataT> *, PairDist<DataT> *, int *, int *, AEVScalarParams<DataT, IndexT>, int, int, int) [with SpeciesT=int, DataT=float, IndexT=int, TILEX=8, TILEY=4]"
(575): here
            instantiation of "at::Tensor cuComputeAEV(at::Tensor, at::Tensor, double, double, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t) [with ScalarRealT=float]"
(581): here


    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, 388 bytes cmem[0]
/home/richard/dev/test/torchani/torchani/cuaev/aev.cu(225): warning: variable "aev_offset" was declared but never referenced
          detected during:
            instantiation of "void cuAngularAEVs(at::PackedTensorAccessor32<SpeciesT, 2UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 3UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 1UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 1UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 1UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 1UL, at::RestrictPtrTraits>, at::PackedTensorAccessor32<DataT, 3UL, at::RestrictPtrTraits>, PairDist<DataT> *, PairDist<DataT> *, int *, int *, AEVScalarParams<DataT, IndexT>, int, int, int) [with SpeciesT=int, DataT=float, IndexT=int, TILEX=8, TILEY=4]"
(575): here
            instantiation of "at::Tensor cuComputeAEV(at::Tensor, at::Tensor, double, double, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t) [with ScalarRealT=float]"
(581): here

g++ -pthread -shared -B /home/richard/program/anaconda3/envs/ml/compiler_compat -L/home/richard/program/anaconda3/envs/ml/lib -Wl,-rpath=/home/richard/program/anaconda3/envs/ml/lib -Wl,--no-as-needed -Wl,--sysroot=/ /home/richard/dev/test/torchani/build/temp.linux-x86_64-3.6/torchani/cuaev/aev.o -L/home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages/torch/lib -L/usr/local/cuda-10.2/lib64 -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda -o build/lib.linux-x86_64-3.6/torchani/cuaev.cpython-36m-x86_64-linux-gnu.so
creating build/bdist.linux-x86_64
creating build/bdist.linux-x86_64/egg
creating build/bdist.linux-x86_64/egg/torchani
copying build/lib.linux-x86_64-3.6/torchani/cuaev.cpython-36m-x86_64-linux-gnu.so -> build/bdist.linux
byte-compiling build/bdist.linux-x86_64/egg/torchani/models.py to models.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torchani/units.py to units.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torchani/nn.py to nn.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torchani/aev.py to aev.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torchani/testing.py to testing.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torchani/utils.py to utils.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torchani/neurochem/parse_resources.py to parse_resources.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torchani/neurochem/trainer.py to trainer.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torchani/neurochem/__init__.py to __init__.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torchani/ase.py to ase.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torchani/__init__.py to __init__.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torchani/data/_pyanitools.py to _pyanitools.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torchani/data/__init__.py to __init__.cpython-36.pyc
creating stub loader for torchani/cuaev.cpython-36m-x86_64-linux-gnu.so
byte-compiling build/bdist.linux-x86_64/egg/torchani/cuaev.py to cuaev.cpython-36.pyc
creating build/bdist.linux-x86_64/egg/EGG-INFO
copying torchani.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
copying torchani.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying torchani.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying torchani.egg-info/requires.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying torchani.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
writing build/bdist.linux-x86_64/egg/EGG-INFO/native_libs.txt
zip_safe flag not set; analyzing archive contents...
torchani.__pycache__.cuaev.cpython-36: module references __file__
torchani.neurochem.__pycache__.parse_resources.cpython-36: module references __file__
creating dist
creating 'dist/torchani-2.3.dev41+g9e79c5d-py3.6-linux-x86_64.egg' and adding 'build/bdist.linux-x86_64/egg' to it
removing 'build/bdist.linux-x86_64/egg' (and everything under it)
Processing torchani-2.3.dev41+g9e79c5d-py3.6-linux-x86_64.egg
creating /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages/torchani-2.3.dev41+g9e79c5d-py3.6-linux-x86_64.egg
Extracting torchani-2.3.dev41+g9e79c5d-py3.6-linux-x86_64.egg to /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages
Adding torchani 2.3.dev41+g9e79c5d to easy-install.pth file

Installed /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages/torchani-2.3.dev41+g9e79c5d-py3.6-linux-x86_64.egg
Processing dependencies for torchani==2.3.dev41+g9e79c5d
Searching for importlib-metadata==1.5.0
Best match: importlib-metadata 1.5.0
Adding importlib-metadata 1.5.0 to easy-install.pth file

Using /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages
Searching for requests==2.23.0
Best match: requests 2.23.0
Adding requests 2.23.0 to easy-install.pth file

Using /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages
Searching for lark-parser==0.8.5
Best match: lark-parser 0.8.5
Adding lark-parser 0.8.5 to easy-install.pth file

Using /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages
Searching for torch==1.8.0.dev20210120
Best match: torch 1.8.0.dev20210120
Adding torch 1.8.0.dev20210120 to easy-install.pth file
Installing convert-caffe2-to-onnx script to /home/richard/program/anaconda3/envs/ml/bin
Installing convert-onnx-to-caffe2 script to /home/richard/program/anaconda3/envs/ml/bin

Using /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages
Searching for zipp==2.2.0
Best match: zipp 2.2.0
Adding zipp 2.2.0 to easy-install.pth file

Using /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages
Searching for chardet==3.0.4
Best match: chardet 3.0.4
Adding chardet 3.0.4 to easy-install.pth file
Installing chardetect script to /home/richard/program/anaconda3/envs/ml/bin

Using /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages
Searching for urllib3==1.25.8
Best match: urllib3 1.25.8
Adding urllib3 1.25.8 to easy-install.pth file

Using /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages
Searching for certifi==2020.12.5
Best match: certifi 2020.12.5
Adding certifi 2020.12.5 to easy-install.pth file

Using /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages
Searching for idna==2.9
Best match: idna 2.9
Adding idna 2.9 to easy-install.pth file

Using /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages
Searching for dataclasses==0.7
Best match: dataclasses 0.7
Adding dataclasses 0.7 to easy-install.pth file

Using /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages
Searching for numpy==1.18.1
Best match: numpy 1.18.1
Adding numpy 1.18.1 to easy-install.pth file
Installing f2py script to /home/richard/program/anaconda3/envs/ml/bin
Installing f2py3 script to /home/richard/program/anaconda3/envs/ml/bin
Installing f2py3.6 script to /home/richard/program/anaconda3/envs/ml/bin

Using /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages
Searching for typing-extensions==3.7.4.1
Best match: typing-extensions 3.7.4.1
Adding typing-extensions 3.7.4.1 to easy-install.pth file

Using /home/richard/program/anaconda3/envs/ml/lib/python3.6/site-packages
Finished processing dependencies for torchani==2.3.dev41+g9e79c5d
```
