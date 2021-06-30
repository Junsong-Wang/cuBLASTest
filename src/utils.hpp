#ifndef __UTILS_HPP__
#define __UTILS_HPP__

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

void fill_data_host(float* a, int size);

void array_float2half(float* array_float, __half* array_half, int size);

void array_half2float(__half* array_half, float* array_float, int size);

void print_gpu_half_data(__half* data, size_t pitch, int rows, int cols);

void array_float2int8(float* array_float, unsigned char* array_int8, int size, float scale);

#endif //__UTILS_HPP__
