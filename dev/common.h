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
#define CUDACHECK(error) cuda_check(error, __FILE__, __LINE__)

void cublas_check(cublasStatus_t status, const char* file, int line){
    if(status != CUBLAS_STATUS_SUCCESS){
        printf("cuBLAS error %s:%d: %s\n", file, line, cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}
#define CUBLASCHECK(status) cublas_check(status, __FILE__, __LINE__)


float* make_random_float_01(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0;i < N;i++) {
        arr[i] = (float)rand() / RAND_MAX; //range 0-1设置随机数
    }
    return arr;
}


float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

int* make_random_int(size_t N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

float* make_zeros_float(size_t N){
    float* arr = (float*)malloc(N*sizeof(float));
    memset(arr, 0,N*sizeof(float));
    return arr;
}

float* make_ones_float(size_t N){
    float* arr =(float*)malloc(N*sizeof(float));
    for(size_t i=0;i<N;i++){
        arr[i] = 1.0f;
    }
    return arr;
}

//把内容从GPU内存复制到主机内存，并比较
template<typename T>
void validate_result(T* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance = 1e-4) {
    T* out_gpu = (T*)malloc(num_elements * sizeof(T));
    //将device_result从GPU内存复制到主机内存
    CUDACHECK(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    int nfaults = 0;
    //比较device_result和cpu_result
    for (int i = 0;i < num_elements;++i) {
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], out_gpu[i]);
        }
        if (fabs(cpu_reference[i] - out_gpu[i]) > tolerance) {
            printf("mismatch at index %d: %f %f\n", i, cpu_reference[i], out_gpu[i]);
            nfaults++;
        }
        if (nfaults > 0) {
            free(out_gpu);
            exit(EXIT_FAILURE);
        }
    }
    free(out_gpu);
}

template<typename Kernel, class... KernelArgs>
float benchmark(int repeats, Kernel kernel, KernelArgs&&... args) {
    cudaEvent_t start, stop;
    //创建计时操作
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    //插入标记
    cudaCheck(cudaEventRecord(start,nullptr));
    for(int i =0;i<repeats;++i){
        kernel(std::forward<KernelArgs>(args)...);
    }
    cudaCheck(cudaEventRecord(stop,nullptr));
    //阻塞 CPU，直到该事件被触发（即流执行到该事件的位置）
    cudaCheck(cudaEventSynchronize(start));
    cudaCheck(cudaEventSynchronize(stop));
    float elapsed;
    //计算两个事件之间经过的时间，单位是毫秒。
    cudaCheck(cudaEventElapsedTime(&elapsed,start,stop));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    return elapsed/repeats;
}