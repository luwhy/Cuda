#pragma once
#include<stdlib.h>
#include<stdio.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>

//向上取整
//__host__ __device__表示可以在哪里启动
template<typename T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

void cuda_check(cudaError_t error, const char* file, int line){
    if(error != cudaSuccess){
        printf("CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(error) cuda_check(error, __FILE__, __LINE__)

void cublas_check(cublasStatus_t status, const char* file, int line){
    if(status != CUBLAS_STATUS_SUCCESS){
        printf("cuBLAS error %s:%d: %s\n", file, line, cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}
#define CUBLAS_CHECK(status) cublas_check(status, __FILE__, __LINE__)