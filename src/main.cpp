#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <ctime>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>

#include "core.hpp"
#include "utils.hpp"

int main()
{
  int M = 768;
  int N = 768*1024;
  int K = 128;

  unsigned int iterations = 16;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  float* A;
  float* B;
  cudaMallocHost(&A,  M*sizeof(float)* K);
  cudaMallocHost(&B,  N*sizeof(float)* K);
  fill_data_host(A, M*K);
  fill_data_host(B, N*K);

  printf("===== start to test HGEMM, M=%d, N=%d, K=%d, test iterations:%d =====\n", M, N, K, iterations);
  fp16_test(handle, A, B, M, N, K, iterations);
  printf("===== start to test GEMMEx(INT8), M=%d, N=%d, K=%d, test iterations:%d =====\n", M, N, K, iterations);
  int8_test(handle, A, B, M, N, K, iterations);
  //fp16_test(handle, M, N, K, iterations);

  return 0;
}
