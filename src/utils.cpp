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

void fill_data_host(float* a, int size)
{
  for(int i = 0; i < size; i++){
    a[i] = 0 + 1.0*rand() / RAND_MAX *(1.0 - 0);
  }
}

void array_float2half(float* array_float, __half* array_half, int size){
  for (int i = 0; i < size; i++){
    array_half[i] = __float2half(array_float[i]);
  }
}

void array_half2float(__half* array_half, float* array_float, int size){
  for (int i = 0; i < size; i++){
    array_float[i] = __half2float(array_half[i]);
  }
}

void array_float2int8(float* array_float, unsigned char* array_int8, int size, float scale){
  for (int i = 0; i < size; i++){
    array_int8[i] = (unsigned int)(array_float[i] * scale);
  }
}

void print_gpu_half_data(__half* data, size_t pitch, int rows, int cols)
{
	__half* host_fp16  = (__half*) malloc(rows * cols * sizeof(__half));
	float* host  = (float*) malloc(rows * cols   * sizeof(float));
	cudaMemcpy2D(host_fp16,  cols * sizeof(__half), data,  pitch,  cols * sizeof(__half), rows, cudaMemcpyDeviceToHost);
	array_half2float(host_fp16, host, rows * cols);

	for(int idx = 0; idx < rows * cols; idx ++){
			std::cout << host[idx] << ", ";
	}
	std::cout << std::endl;
	free(host_fp16);
	free(host);
}
