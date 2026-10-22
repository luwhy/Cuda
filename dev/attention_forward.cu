#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<float.h>
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include<cooperative_groups/reduce.h>
#include"common.h"

void attention_forward_cpu(float *output, float *preatt, float *att, float *inp, int B, int T,
                           int C, int NH) {
    int C3 = C * 3;
    int hs = C / NH;
    float scale = 1.0 / sqrtf(hs);
    //告诉编译器，把下面紧跟着的 3 层嵌套 for 循环，直接“拍扁”并分配给 CPU 的多个核心去同时执行（多线程并行加速）
    //加上了 collapse(3)，OpenMP 会把外层的 b、中层的 t、内层的 h 总共 3 层循环合并成一个一维的大循环,然后再把这些任务平均分给几十个CPU线程
    #pragma omp parallel for collapse(3)
}

int main(int argc, char** argv){

    return 0;
}
