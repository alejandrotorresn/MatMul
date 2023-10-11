#ifndef CUDA_HEADER_H_
#define CUDA_HEADER_H_
#pragma once

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_DIM 32

namespace cuda {
    void cuda_sgemm_naive(const float* matA, const float* matB, float* matC, 
                        const int& M, const int& N, const int& K, 
                        const dim3 blockDim, const dim3 gridDim);

    void cuda_sgemm_tiled(const float* matA, const float* matB, float* matC, 
                        const int& M, const int& N, const int& K, 
                        const dim3 blockDim, const dim3 gridDim);

    int cublas_sgemm(const float* matA, const float* matB, float* matC, 
                        const int& M, const int& N, const int& K);
    
    int cublas_sgemmAsync(const float* matA, const float* matB, float* matC, 
                        const int& M, const int& N, const int& K);
    
    int cublas_sgemmTensor(const float* matA, const float* matB, float* matC, 
                        const int& M, const int& N, const int& K, 
                        const dim3 blockDim, const dim3 gridDim);

}
#endif 
