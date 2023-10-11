#include <iostream>
#include "cuda_mm.hpp"


__global__ void sgemm_naive(const float *matA, const float *matB, float *matC, 
                            size_t M, size_t N, size_t K) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if ( col < M && row < N ) {
        float element_c = 0.f;
        for ( int i = 0; i < K; i++ )
            element_c += matA[col * K + i] * matB[i * N + row];
    
        matC[col * N + row] = element_c;
    }
    
}

__global__ void sgemm_tiled(const float *matA, const float *matB, float *matC, 
                            size_t M, size_t N, size_t K) {
    // BLOCK_DIM
    __shared__ float sA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float sB[BLOCK_DIM][BLOCK_DIM];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_DIM + ty;
    int col = bx * BLOCK_DIM + tx;

    float element_c =0.f;

    for ( int i = 0; i < ( (K-1)/BLOCK_DIM + 1 ); i++ ) {
        if ( row < N && i * BLOCK_DIM + tx < K )
            sA[ty][tx] = matA[row * K + i * BLOCK_DIM + tx];
        else
            sA[ty][tx] = 0.f;

        if ( i * BLOCK_DIM + ty < K && col < M )
            sB[ty][tx] = matB[(i * BLOCK_DIM + ty) * M + col];
        else
            sB[ty][tx] = 0.f;

        __syncthreads();

        for ( int j = 0; j < BLOCK_DIM; j++ )
            element_c += sA[ty][j] * sB[j][tx];

        __syncthreads();
    }

    if ( row < N && col < M)
        matC[row * M + col] = element_c;

}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half( in[idx] );
        //printf("%f : %f \n", in[idx], __half2float(out[idx]) );
        }
}

void cuda::cuda_sgemm_naive(const float* matA, const float* matB, float* matC, 
                    const int& M, const int& N, const int& K, 
                    const dim3 blockDim, const dim3 gridDim) {
    float *d_A, *d_B, *d_C;
    // allocation memory space
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));
    
    // copy initial value for gpu  memory
    cudaMemcpy(d_A, matA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matB, K * N * sizeof(float), cudaMemcpyHostToDevice);

    sgemm_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // copy data from the gpu
    cudaMemcpy(matC, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void cuda::cuda_sgemm_tiled(const float* matA, const float* matB, float* matC, 
                    const int& M, const int& N, const int& K, 
                    const dim3 blockDim, const dim3 gridDim) {
    float *d_A, *d_B, *d_C;
    // allocation memory space
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));
    
    // copy initial value for gpu  memory
    cudaMemcpy(d_A, matA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matB, K * N * sizeof(float), cudaMemcpyHostToDevice);

    sgemm_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // copy data from the gpu
    cudaMemcpy(matC, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int cuda::cublas_sgemm(const float* matA, const float* matB, float* matC, 
                        const int& M, const int& N, const int& K) {
    
    cublasStatus_t stat;
    cublasHandle_t handle;

    float *d_A, *d_B, *d_C;
    float alpha, beta;

    alpha = 1.f;
    beta = 0.f;

    // allocation memory space
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    stat = cublasCreate(&handle); // initialize CUBLAS context

    cudaMemcpy(d_A, matA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matB, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, matC, M * N * sizeof(float), cudaMemcpyHostToDevice);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS initialization failed" << "\n";
        return EXIT_FAILURE;
    }

    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_B, N,
                        d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(matC, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(handle);
    
    return EXIT_SUCCESS;

}


int cuda::cublas_sgemmAsync(const float* matA, const float* matB, float* matC, 
                        const int& M, const int& N, const int& K) {
    
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cudaStream_t stream;

    float *d_A, *d_B, *d_C;
    float alpha, beta;

    alpha = 1.f;
    beta = 0.f;

    stat = cublasCreate(&handle);
    if ( stat != CUBLAS_STATUS_SUCCESS ) {
        std::cout << "CUBLAS initialization failed!\n";
        return EXIT_FAILURE;
    }
    
    // allocation memory space
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    cudaStat = cudaStreamCreate(&stream);

    cublasSetMatrixAsync(M, K, sizeof(*d_A), matA, M, d_A, M, stream);
    cublasSetMatrixAsync(K, N, sizeof(*d_B), matB, K, d_B, K, stream);
    cublasSetMatrixAsync(M, N, sizeof(*d_C), matC, M, d_C, M, stream);

    cublasSetStream(handle, stream);

    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_B, N,
                        d_A, K, &beta, d_C, N);

    cublasGetMatrixAsync(M, N, sizeof(*d_C), d_C, M, matC, M, stream);
    cudaStreamSynchronize(stream);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return EXIT_SUCCESS;
}


int cuda::cublas_sgemmTensor(const float* matA, const float* matB, float* matC, 
                        const int& M,  const int& N,  const int& K, 
                        const dim3 blockDim, const dim3 gridDim) {

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cudaStream_t stream;

    float *d_A, *d_B, *d_C;
    float alpha, beta;

    alpha = 1.f;
    beta = 0.f;

    stat = cublasCreate(&handle);
    if ( stat != CUBLAS_STATUS_SUCCESS ) {
        std::cout << "CUBLAS initialization failed!\n";
        return EXIT_FAILURE;
    }
    
    // allocation memory space
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    cudaStat = cudaStreamCreate(&stream);
    stat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    cublasSetMatrixAsync(M, K, sizeof(*d_A), matA, M, d_A, M, stream);
    cublasSetMatrixAsync(K, N, sizeof(*d_B), matB, K, d_B, K, stream);
    cublasSetMatrixAsync(M, N, sizeof(*d_C), matC, M, d_C, M, stream);

    cublasSetStream(handle, stream);

    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_B, N,
                        d_A, K, &beta, d_C, N);

    cublasGetMatrixAsync(M, N, sizeof(*d_C), d_C, M, matC, M, stream);
    cudaStreamSynchronize(stream);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return EXIT_SUCCESS;
    
}
