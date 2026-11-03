#include "common.h"
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>


static cublasHandle_t cublas_handle;

// NH代表多头数量
// C代表每个token的特征维度
// 3C代表query、key、value三个向量
// hs就是每个head分到的特征维度
void attention_forward_cpu(float *out, float *preatt, float *att, float *inp, int B, int T, int C,
                           int NH) {
    int C3 = C * 3;
    int hs = C / NH;
    float scale = 1.0 / sqrtf(hs);
// 告诉编译器，把下面紧跟着的 3 层嵌套 for 循环，直接“拍扁”并分配给 CPU
// 的多个核心去同时执行（多线程并行加速） 加上了 collapse(3)，OpenMP 会把外层的 b、中层的 t、内层的
// h 总共 3 层循环合并成一个一维的大循环,然后再把这些任务平均分给几十个CPU线程
#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                // 当前BATCH中，第t个tocken的第h个head的query向量的起始地址
                float *query_t = inp + b * T * C3 + t * C3 + h * hs;
                // 输出Attention分数，NH个head的，T个token的，注意力分数矩阵
                float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                // 经过softmax归一化后的注意力分数矩阵
                float *att_bth = att + b * NH * T * T + h * T * T + t * T;
                float maxval = -10000.0f;
                for (int t2 = 0; t2 < T; t2++) {
                    // 这里是在找当前query前面所有的key
                    float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    float val = 0.0f;
                    for (int i = 0; i < hs; ++i) {
                        val += query_t[i] * key_t2[i];
                    }
                    if (val > maxval) {
                        maxval = val;
                    }
                    preatt_bth[t2] = val;
                }
                float expsum = 0.0f;
                // 不能穿越未来的词
                for (int t2 = 0; t2 <= t; ++t2) {
                    float expv = expf(scale * (preatt_bth[t2] - maxval));
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                // 准备归一化
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        att_bth[t2] = 0.0f;
                    }
                }
                float *out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; ++i) {
                    out_bth[i] = 0.0f;
                }
                for (int t2 = 0; t2 <= t; ++t2) {
                    float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}


//其实就是算每个out和目标的误差
//d其实表示梯度，不带d表示前向传播特征
//dinp：输入张量 inp 的梯度。形状是 (B, T, 3C)。这个函数算出来的 Q、K、V 的误差，最终都要累加到这块内存里
//dpreatt：Softmax 之前（原始打分）的梯度。形状 (B, NH, T, T)。
//datt：Softmax 之后（概率权重）的梯度。形状 (B, NH, T, T)。
//dout整个 Attention 模块输出的梯度。形状 (B, T, C)。这是上一层传回来的误差。
// inp：前向传播时的原始输入（包含 Q, K, V）。反向传播需要用到前向时的值来算导数。
// att：前向传播时算出的注意力概率矩阵（Softmax 后的结果）。算 Softmax 导数必须用它。
// B, T, C, NH：批次大小、序列长度、总特征维度、头数。
void attention_backward_cpu(float* dinp,float* dpreatt,float*datt,
    float*dout,float* inp,float*att,int B,int T,int C, int NH){
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C/NH;
    float scale = 1.0 / sqrtf(hs);
    for(int b=0;b<B;b++){
        for(int t=0;t<T;t++){
            for(int h=0;h<NH;h++){
                float* att_bth = att + b * NH * T * T + h * T * T + t * T;
                float* datt_bth = datt + b * NH * T * T + h * T * T + t * T;
                float* dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
                float* dquery_t = dinp + b * T * 3 * C + t * 3 * C + h * hs;
                float* query_t = inp + b * T * 3 * C + t * 3 * C + h * hs;

                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0;t2 < T;t2++) {
                    float* value_t2 = inp + b * T * 3 * C + t2 * 3 * C + h * hs + C * 2;
                    float* dvalue_t2 = dinp + b * T * 3 * C + t2 * 3 * C + h * hs + C * 2;
                    for (int i = 0; i < hs;++i) {
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }
                for (int t2 = 0;t2 <= t;++t2) {
                    for (int t3 = 0;t3 < t;t3++) {
                        float  indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += scale * local_derivative * datt_bth[t2];
                    }
                }
                for(int t2=0;t2<=t;t2++){
                    float* key_t2 = inp + b * T * 3 * C + t2 * 3 * C + h * hs + C;
                    float* dkey_t2 = dinp + b * T * 3 * C + t2 * 3 * C + h * hs + C;
                    for(int i=0;i<hs;++i){
                        dquery_t[i]+=key_t2[i]*dpreatt_bth[t2];
                        dkey_t2[i]+=query_t[i]*dpreatt_bth[t2];
                    }
                }
            }
        }
    }
}

__global__ void permute_kernel(float* q, float* k, float* v, const float* inp, int B, int T, int C, int NH, int d) {

}



int main(int argc, char **argv) { 
    srand(0);
    int B = 4;
    int T = 1024;
    int C = 768;
    int NH = 12;
    int deviceID=0;
    CUDACHECK(cudaSetDevice(deviceID));
    cublasCreate(&cublas_handle);
    int kernel_num =1;
    if(argc>1){
        kernel_num=atoi(argv[1]);
    }
    printf("using kernel %d",kernel_num);
    //输入
    float *inp = make_random_float(B * T * 3 * C);
    float *qkvr = (float *)malloc(B * T * 3 * C * sizeof(float));
    float *preatt = (float *)malloc(B * T * NH * T * sizeof(float));
    float *att = (float *)malloc(B * NH * T * T * sizeof(float));
    float *vaccum = (float *)malloc(B * T * C * sizeof(float));
    float *out = (float *)malloc(B * T * C * sizeof(float));
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);
    float *d_inp,*d_qkvr, *d_preatt, *d_att, *d_vaccum, *d_out;
    CUDACHECK(cudaMalloc(&d_inp, B*T*3*C*sizeof(float)));
    CUDACHECK(cudaMalloc(&d_qkvr, B*T*3*C*sizeof(float)));
    CUDACHECK(cudaMalloc(&d_preatt, B*T*NH*T*sizeof(float)));
    CUDACHECK(cudaMalloc(&d_att, B*NH*T*T*sizeof(float)));
    CUDACHECK(cudaMalloc(&d_vaccum, B*T*C*sizeof(float)));
    CUDACHECK(cudaMalloc(&d_out, B*T*C*sizeof(float)));
    CUDACHECK(cudaMemcpy(d_inp,inp,B*T*3*C*sizeof(float),cudaMemcpyHostToDevice));
    printf("Checking the forward pass...");
    printf("[preatt]\n");
    
    float * dout =make_random_float(B*T*C);
    float * dinp =make_zeros_float(B*T*3*C);
    float * dpreatt =make_zeros_float(B*NH*T*T);
    float * datt =make_zeros_float(B*NH*T*T);
    attention_backward_cpu(dinp,dpreatt,datt,dout,inp,att,B,T,C,NH);
    
    
    free(inp);
    free(qkvr);
    free(preatt);
    free(att);
    free(vaccum);
    free(out);
    CUDACHECK(cudaFree(d_inp));
    CUDACHECK(cudaFree(d_qkvr));
    CUDACHECK(cudaFree(d_preatt));
    CUDACHECK(cudaFree(d_att));
    CUDACHECK(cudaFree(d_vaccum));
    CUDACHECK(cudaFree(d_out));
    cublasDestroy(cublas_handle);
    return 0;
}
