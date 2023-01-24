/*
    argv[] ->   N: rows of A
    argv[] ->   M: columns of B
    argv[] ->   K: columns of A and rows of B
    argv[] ->   newMatrix: if 1 create new values of matrix A and B. If 0 read from file
    argv[] ->   iter: number if iterations of matrix multiplications
    argv[] ->   prsize_tOp: if 1 prsize_t.
*/
#include <iostream>
#include <bits/stdc++.h>
#include <sys/time.h>
#include "MatMul.h"

#define BLOCK_DIM 16

using namespace std;

bool verification(const float *C_cpu, const float *C_gpu, size_t length);
void mulMat(const float *A, const float *B, float *C, size_t N, size_t M, size_t K);

__global__ void matMul_cuda(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);

int main(int argc, char **argv) {

    float *matA, *matB, *matC;
    //float *matC_cpu;
    float *d_A, *d_B, *d_C;
    size_t N, M, K;
    size_t newMatrix;
    size_t iter;
    size_t prsize_tOp;

    N = atoi(argv[1]);
    M = atoi(argv[2]);
    K = atoi(argv[3]);
    newMatrix = atoi(argv[4]);
    iter = atoi(argv[5]);
    prsize_tOp = atoi(argv[6]);

    // Timer
    struct timeval start, end;

    // Allocation memory space
    matA = (float *)malloc(M * K * sizeof(float));
    matB = (float *)malloc(K * N * sizeof(float));
    matC = (float *)malloc(M * N * sizeof(float));
    //matC_cpu = (float *)malloc(M * N * sizeof(float));

    if ( newMatrix == 1 ) {
        // Create randomized values and save them to file
        random_init(matA, M * K, true);
        saveMat("matrixA.dat", matA, M * K);
        random_init(matB, K * N, true);
        saveMat("matrixB.dat", matB, K * N);
    } else if (newMatrix == 0) {
        // Read matrices from file
        readMat("matrixA.dat", matA, N*K);
        readMat("matrixB.dat", matB, K*M); 
    }

    random_init(matC, M * N, false);

    // unsync the I/O of C and C++
    ios_base::sync_with_stdio(false);

    const dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    const dim3 gridDim((N + BLOCK_DIM - 1)/BLOCK_DIM, (M + BLOCK_DIM - 1)/BLOCK_DIM);
    
    /*size_t block_D;

        if (K < BLOCK_DIM)
            block_D = K;
        else
            block_D = BLOCK_DIM;*/

    // allocation memory space
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    // copy initial value for gpu  memory
    cudaMemcpy(d_A, matA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matB, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // start timer
    gettimeofday(&start, NULL);

    for (size_t i=0; i<iter; i++) {
        matMul_cuda<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }

    // end timer
    gettimeofday(&end, NULL);

    // copy data from the gpu
    cudaMemcpy(matC, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    

    // Calculating total time taken
    double time_taken;
    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;

    cout << endl;
    cout << "       Total time: " << fixed << time_taken/iter << setprecision(6);
    cout << " sec." << endl;

    // verification
    /*mulMat(matA, matB, matC_cpu, M, N, K);
    cout << endl << "CPU" << endl;
    prsize_tMat(matC_cpu, M, N);

    cout << endl << "GPU" << endl;
    if (prsize_tOp == 1)
        prsize_tMat(matC, N, M);

    if (verification(matC, matC_cpu, M*N))
        prsize_tf("SUCCESS!!\n");
    else
        prsize_tf("Error\n");*/

    if (prsize_tOp == 1)
        printMat(matC, N, M);

    free(matA);
    free(matB);
    free(matC);
    //free(matC_cpu);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

bool verification(const float *C_cpu, const float *C_gpu, size_t length) {
    float epsilon = 0.001;
    float error = 0.f;
    for (size_t i=0; i<length; i++) {
        error += (abs(C_gpu[i] - C_cpu[i])/abs(C_cpu[i]))*100;
        if (abs(C_cpu[i] - C_gpu[i]) >= epsilon) {
            cout << "Error: " << error/length << endl;
            return false;
        }
    }
    cout << "Error: " << error/length << endl;
    return true;
}

__global__ void matMul_cuda(const float *A, const float *B, float *C, size_t M, size_t N, size_t K) {
    size_t bid_x = blockIdx.x; //* blockDim.x;
    size_t bid_y = blockIdx.y; //* blockDim.y;
    size_t tid_x = threadIdx.x;
    size_t tid_y = threadIdx.y;

    float element_c = 0.f;

    __shared__ float s_tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float s_tile_B[BLOCK_DIM][BLOCK_DIM];

    size_t aBegin = K * BLOCK_DIM * bid_y;
    size_t aEnd = aBegin + K - 1;
    size_t aStep = BLOCK_DIM;

    size_t bBegin = BLOCK_DIM * bid_x;
    size_t bStep = BLOCK_DIM * N;

    for (size_t i=aBegin, j=bBegin; i<=aEnd; i+=aStep, j+=bStep) {
        s_tile_A[tid_y][tid_x] = A[i + K * tid_y + tid_x];
        s_tile_B[tid_y][tid_x] = B[j + N * tid_y + tid_x];

        __syncthreads();

        for (size_t k=0; k<BLOCK_DIM; ++k) {
            element_c += s_tile_A[tid_y][k] * s_tile_B[k][tid_x];
        }

        __syncthreads();
    }
    size_t cIdx = N * BLOCK_DIM * bid_y + BLOCK_DIM * bid_x;

    C[cIdx + N * tid_y + tid_x] = element_c;
}

void mulMat(const float *A, const float *B, float *C, size_t N, size_t M, size_t K) {
    for (size_t row=0; row<M; row++) {
        for (size_t col=0; col<N; col++) {
            float element_c = 0.f;
            for (size_t e=0; e<K; e++) {
                element_c += A[row * K + e] * B[e * N + col];
            }
            C[row * N + col] = element_c;
        }
    }
}
