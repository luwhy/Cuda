#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<float.h>
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include<cooperative_groups/reduce.h>
#include"common.h"
//NH代表多头数量
//C代表每个token的特征维度
//3C代表query、key、value三个向量
//hs就是每个head分到的特征维度
void attention_forward_cpu(float *output, float *preatt, float *att, float *inp, int B, int T,
                           int C, int NH) {
    int C3 = C * 3;
    int hs = C / NH;
    float scale = 1.0 / sqrtf(hs);
    //告诉编译器，把下面紧跟着的 3 层嵌套 for 循环，直接“拍扁”并分配给 CPU 的多个核心去同时执行（多线程并行加速）
    //加上了 collapse(3)，OpenMP 会把外层的 b、中层的 t、内层的 h 总共 3 层循环合并成一个一维的大循环,然后再把这些任务平均分给几十个CPU线程
    #pragma omp parallel for collapse(3)
    for(int b =0; b<B;b++){
        for(int t=0;t<T;t++){
            for(int h=0;h<NH;h++){
                //当前BATCH中，第t个tocken的第h个head的query向量的起始地址
                float *query_t = inp + b * T * C3 + t * C3 + h * hs;
                //输出Attention分数，NH个head的，T个token的，注意力分数矩阵
                float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                //经过softmax归一化后的注意力分数矩阵
                float *att_bth = att + b * NH * T * T + h * T * T + t * T;
                float maxval= -10000.0f;
                for(int t2=0;t2<T;t2++){
                    //这里是在找当前query前面所有的key
                    float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    float val = 0.0f;
                    for(int i=0; i<hs; ++i){
                        val+= query_t[i]*key_t2[i];
                    }
                    if(val>maxval){
                        maxval = val;
                    }
                    preatt_bth[t2]=val;
                }
                float expsum = 0.0f;
                //不能穿越未来的词
                for (int t2 = 0; t2 <= t; ++t2) {
                    float expv = expf(scale * (preatt_bth[t2] - maxval));
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                //准备归一化
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        att_bth[t2] = 0.0f;
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv){

    return 0;
}
