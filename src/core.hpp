#ifndef __CORE_HPP__
#define __CORE_HPP__

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

#include "utils.hpp"

void int8_test(cublasHandle_t& handle, float* A, float* B, int M, int N, int K, unsigned int iterations);
void fp16_test(cublasHandle_t& handle, float* A, float* B, int M, int N, int K, unsigned int iterations);

#define USING_CUDA_R_32I

#endif //__CORE_HPP__
