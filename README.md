# cuBLASTest

## Build:
```
  mkdir build
  cd build
  cmake ..
  make -j4
```

## RUN the Test:

GPU environment: Nvidia Tesla T4/16GB, Driver Version: 418.181.07, CUDA Version: 10.1

```
  nvprof ./cublastest
```

## Test Results:
```
root@c0dca262005a:~/cuBLASTest/build# nvprof ./cublastest
==7890== NVPROF is profiling process 7890, command: ./cublastest
===== start to test HGEMM, M=768, N=786432, K=128, test iterations:16 =====
FP16, total Time (timeofday) in 16 interations is 1.91351s.
===== start to test GEMMEx(INT8), M=768, N=786432, K=128, test iterations:16 =====
INT8, total Time (timeofday) in 16 interations is 3.74584s.
==7890== Profiling application: ./cublastest
==7890== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   90.96%  5.17309s        32  161.66ms  110.38ms  213.00ms  [CUDA memcpy DtoH]
                    6.03%  343.08ms        16  21.443ms  21.181ms  21.899ms  volta_sgemm_int8_128x128_nt
                    2.45%  139.13ms        16  8.6957ms  7.7914ms  12.507ms  turing_h1688gemm_128x128_ldg8_nt
                    0.56%  31.810ms         5  6.3621ms  2.0160us  20.800ms  [CUDA memcpy HtoD]
      API calls:   59.41%  5.68842s        36  158.01ms  65.628us  234.50ms  cudaMemcpy2D
                   22.32%  2.13684s         8  267.10ms  33.173us  1.18106s  cudaHostAlloc
                   10.39%  995.07ms         9  110.56ms  1.0420us  652.45ms  cudaFree
                    7.79%  746.06ms         6  124.34ms  59.643us  447.81ms  cudaFreeHost
                    0.05%  4.8069ms         6  801.16us  61.777us  2.4387ms  cudaMallocPitch
                    0.02%  1.7026ms        32  53.204us  29.020us  69.090us  cudaLaunchKernel
                    0.01%  852.75us         3  284.25us  277.59us  295.24us  cuDeviceTotalMem
                    0.01%  610.51us       285  2.1420us     158ns  97.318us  cuDeviceGetAttribute
                    0.00%  414.18us         3  138.06us  7.6690us  384.40us  cudaMalloc
                    0.00%  276.94us        80  3.4610us     933ns  13.357us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  123.31us       169     729ns     428ns  7.0680us  cudaFuncSetAttribute
                    0.00%  119.43us         3  39.809us  33.549us  46.820us  cuDeviceGetName
                    0.00%  40.201us         1  40.201us  40.201us  40.201us  cudaMemcpy
                    0.00%  17.637us        16  1.1020us     517ns  7.7770us  cudaEventCreateWithFlags
                    0.00%  12.096us        32     378ns     225ns     577ns  cudaGetLastError
                    0.00%  8.4200us         1  8.4200us  8.4200us  8.4200us  cuDeviceGetPCIBusId
                    0.00%  6.4310us        11     584ns     345ns  1.7760us  cudaDeviceGetAttribute
                    0.00%  5.9760us         2  2.9880us  2.9040us  3.0720us  cuInit
                    0.00%  5.0820us         1  5.0820us  5.0820us  5.0820us  cudaGetDevice
                    0.00%  3.7190us         5     743ns     250ns  2.1850us  cuDeviceGetCount
                    0.00%  2.0360us         4     509ns     189ns     983ns  cuDeviceGet
                    0.00%  1.3220us         2     661ns     526ns     796ns  cuDriverGetVersion
                    0.00%     884ns         3     294ns     290ns     304ns  cuDeviceGetUuid
```
## Questions:

1. In our typical settings, M=768, N=786432, K=128, GEMM with INT8 (volta_sgemm_int8_128x128_nt) is much slower than FP16 (turing_h1688gemm_128x128_ldg8_nt), 21.443ms vs. 8.6957ms. I changed to CUDA version from 10.1 to 11.2, the performane results are same.

2. We would like to use UINT8 instead of INT8, How to configure the cublasGemmEx? It is not clear in the cuBLAS manual. I try to use CUDA_R_8U instead of CUDA_R_8I, but the results seems wrong.
