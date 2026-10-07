#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>
#include"common.h"

void adamw_cpu(float *params_memory, const float *grads_memory, float *m_memory, float *v_memory,
               int t, long num_parameters, float learning_rate = 1e-3, float beta1 = 0.9,
               float beta2 = 0.999, float eps = 1e-8, float weight_decay = 0.0) {
    for (int i = 0; i < num_parameters; ++i) {
        float param = params_memory[i];
        float grad = grads_memory[i];
        float m = beta1 * m_memory[i] + (1.0f - beta1) * grad;
        float v = beta2 * v_memory[i] + (1.0f - beta2) * grad * grad;
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        m_memory[i] = m;
        v_memory[i] = v;
        params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}
//实际上就是说a+b*(c-a) = a + bc - ab = a(1-b) + bc
//fma 是 GPU 上的单条硬件指令，比“先乘后加”两条指令更快。
//NVIDIA GPU 对 fma 有专门的硬件支持，吞吐量很高。
//fma 把乘法和加法合并成一个指令，中间结果不丢失精度
__device__ inline float lerp(float start,float end, float weight){
    return fma(weight ,end,fma(-weight,start,start));
}
// 将 (1.0f - powf(beta1, t))在cpu中计算
// 将 (1.0f - powf(beta2, t))在cpu中计算
__global__ void adamw_kernel_1(float *params_memory, const float *grads_memory, float *m_memory,
                               float *v_memory, long num_parameters, float learning_rate,
                               float beta1, float beta2, float beta1_correction,
                               float beta2_correction, float eps, float weight_decay) {
    // blockIdx.x：当前 block 的编号
    // blockDim.x：每个 block 中线程的数量
    // threadIdx.x：当前线程在 block 中的编号
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_parameters)
        return;
    m_memory[i] = beta1 * m_memory[i] + (1.0f - beta1) * grads_memory[i];
    v_memory[i] = beta2 * v_memory[i] + (1.0f - beta2) * grads_memory[i] * grads_memory[i];

    float m_hat = m_memory[i] / beta1_correction;
    float v_hat = v_memory[i] / beta2_correction;
    params_memory[i] -=
        learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params_memory[i]);
}

int main(int argc, char** argv) {
    return 0;
}
