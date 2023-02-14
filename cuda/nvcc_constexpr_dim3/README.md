```
(cu113) cu richard@anbai~/dev/cuda_misc/nvcc_constexpr_dim3> loadcuda 11.3
Found cuda, CUDA_HOME set to be /usr/local/cuda-11.3
----------------------------------------
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Mar_21_19:15:46_PDT_2021
Cuda compilation tools, release 11.3, V11.3.58
Build cuda_11.3.r11.3/compiler.29745058_0
(cu113) cu richard@anbai~/dev/cuda_misc/nvcc_constexpr_dim3> nvcc -std=c++11 -o test test.cu -run
test.cu(10): error: no instance of function template "foo" matches the argument list

1 error detected in the compilation of "test.cu".
(cu113) cu richard@anbai~/dev/cuda_misc/nvcc_constexpr_dim3> loadcuda 11.2
Found cuda, CUDA_HOME set to be /usr/local/cuda-11.2
----------------------------------------
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Feb_14_21:12:58_PST_2021
Cuda compilation tools, release 11.2, V11.2.152
Build cuda_11.2.r11.2/compiler.29618528_0
(cu113) cu richard@anbai~/dev/cuda_misc/nvcc_constexpr_dim3> nvcc -std=c++11 -o test test.cu -run
hello
```

