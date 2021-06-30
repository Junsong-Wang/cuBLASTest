#include "core.hpp"

void int8_test(cublasHandle_t& handle, float* A, float* B, int M, int N, int K, unsigned int iterations)
{
  const float alf_float = -2.0;
  const float bet_float = 0.0;
  const float *alpha_float = &alf_float;
  const float *beta_float = &bet_float;

  //convert from float to fp16
  unsigned char* A_host;
  unsigned char* B_host;
  float* C_host;
  cudaMallocHost(&A_host, M*sizeof(unsigned char) * K);
  cudaMallocHost(&B_host, N*sizeof(unsigned char) * K);
  cudaMallocHost(&C_host, M*sizeof(float) * N);
  array_float2int8(A, A_host, M*K, 64.0);
  array_float2int8(B, B_host, N*K, 64.0);


  //INT8 GEMM TEST
  unsigned char* A_ptr;
  unsigned char* B_ptr;
  float* C_ptr;
  size_t A_pitch;
  size_t B_pitch;
  size_t C_pitch;
  cudaMallocPitch(&A_ptr, &A_pitch, M*sizeof(unsigned char), K);
  cudaMallocPitch(&B_ptr, &B_pitch, N*sizeof(unsigned char), K);
  cudaMallocPitch(&C_ptr, &C_pitch, M*sizeof(float), N);
  int lda = A_pitch / sizeof(unsigned char);
  int ldb = B_pitch / sizeof(unsigned char);
  int ldc = C_pitch / sizeof(float);

  cudaMemcpy2D(A_ptr, A_pitch, A_host, M*sizeof(unsigned char), M*sizeof(unsigned char), K, cudaMemcpyHostToDevice);
  cudaMemcpy2D(B_ptr, B_pitch, B_host, N*sizeof(unsigned char), N*sizeof(unsigned char), K, cudaMemcpyHostToDevice);

  struct timeval start, end;
  gettimeofday(&start, NULL);
  for(unsigned int i = 0; i < iterations; i++){
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, alpha_float,
                 A_ptr, CUDA_R_8I, lda, B_ptr, CUDA_R_8I, ldb, beta_float,
                 C_ptr, CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaMemcpy2D(C_host,  M  * sizeof(float), C_ptr,  C_pitch,  M * sizeof(float), N, cudaMemcpyDeviceToHost);
  }
  gettimeofday(&end, NULL );
  double time_cost =  ( end.tv_sec - start.tv_sec ) + (end.tv_usec - start.tv_usec)/1000000.0;
  std::cout << "INT8, total Time (timeofday) in "<< iterations <<" interations is " << time_cost << "s." << std::endl;

  cudaFreeHost(A_host);
  cudaFreeHost(B_host);
  cudaFreeHost(C_host);

  cudaFree(A_ptr);
  cudaFree(B_ptr);
  cudaFree(C_ptr);
}

void fp16_test(cublasHandle_t& handle, float* A, float* B, int M, int N, int K, unsigned int iterations)
{
  const __half alf_half = __float2half(-2.0);
  const __half bet_half = __float2half(0.0);
  const __half *alpha_half = &alf_half;
  const __half *beta_half = &bet_half;

  //convert from float to fp16
  __half* A_host;
  __half* B_host;
  __half* C_host;
  cudaMallocHost(&A_host, M*sizeof(__half) * K);
  cudaMallocHost(&B_host, N*sizeof(__half) * K);
  cudaMallocHost(&C_host, M*sizeof(__half) * N);
  array_float2half(A, A_host, M*K);
  array_float2half(B, B_host, N*K);

  //copy to GPU device memory
  __half* A_ptr;
  __half* B_ptr;
  __half* C_ptr;
  size_t A_pitch;
  size_t B_pitch;
  size_t C_pitch;
  cudaMallocPitch(&A_ptr, &A_pitch, M*sizeof(__half), K);
  cudaMallocPitch(&B_ptr, &B_pitch, N*sizeof(__half), K);
  cudaMallocPitch(&C_ptr, &C_pitch, M*sizeof(__half), N);
  int lda = A_pitch / sizeof(__half);
  int ldb = B_pitch / sizeof(__half);
  int ldc = C_pitch / sizeof(__half);

  cudaMemcpy2D(A_ptr, A_pitch, A_host, M*sizeof(__half), M*sizeof(__half), K, cudaMemcpyHostToDevice);
  cudaMemcpy2D(B_ptr, B_pitch, B_host, N*sizeof(__half), N*sizeof(__half), K, cudaMemcpyHostToDevice);

  struct timeval start, end;
  gettimeofday(&start, NULL);
  for(unsigned int i = 0; i < iterations; i++){
    //printf("%d, %d, %d, %d, %d, %d\n", M, N, K, lda, ldb, ldc);
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, alpha_half,
                A_ptr, lda, B_ptr, ldb, beta_half, C_ptr, ldc);
    cudaMemcpy2D(C_host,  M  * sizeof(__half), C_ptr,  C_pitch,  M * sizeof(__half), N, cudaMemcpyDeviceToHost);
    //print_gpu_half_data(C_ptr, C_pitch, N,  M);
  }

  gettimeofday(&end, NULL);
  double time_cost =  ( end.tv_sec - start.tv_sec ) + (end.tv_usec - start.tv_usec)/1000000.0;
  std::cout << "FP16, total Time (timeofday) in "<< iterations <<" interations is " << time_cost << "s." << std::endl;

  cudaFreeHost(A_host);
  cudaFreeHost(B_host);
  cudaFreeHost(C_host);

  cudaFree(A_ptr);
  cudaFree(B_ptr);
  cudaFree(C_ptr);
}
