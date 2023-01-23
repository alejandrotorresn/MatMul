/*
    argv[] ->   N: rows of A
    argv[] ->   M: columns of B
    argv[] ->   K: columns of A and rows of B
    argv[] ->   newMatrix: if 1 create new values of matrix A and B. If 0 read from file
    argv[] ->   iter: number if iterations of matrix multiplications
    argv[] ->   printOp: if 1 print.
*/
#include <iostream>
#include <bits/stdc++.h>
#include <sys/time.h>
#include "MatMul.h"

#define BLOCK_DIM 64

using namespace std;

bool verification(const float *C_cpu, const float *C_gpu, int length);
void mulMat(const float *A, const float *B, float *C, int N, int M, int K);

__global__ void matMul_cuda(const float *A, const float *B, float *C, int M, int N, int K, const int block_D);

int main(int argc, char **argv) {

    float *matA, *matB, *matC;
    //float *matC_cpu;
    float *d_A, *d_B, *d_C;
    int N, M, K;
    int newMatrix;
    int iter;
    int printOp;

    N = atoi(argv[1]);
    M = atoi(argv[2]);
    K = atoi(argv[3]);
    newMatrix = atoi(argv[4]);
    iter = atoi(argv[5]);
    printOp = atoi(argv[6]);

    // Timer
    struct timeval start, end;

    // Allocation memory space
    matA = (float *)malloc(M * K * sizeof(float));
    matB = (float *)malloc(K * N * sizeof(float));
    matC = (float *)malloc(M * N * sizeof(float));
    //matC_cpu = (float *)malloc(M * N * sizeof(float));

    // allocation memory space
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

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

    // start timer
    gettimeofday(&start, NULL);

    // copy initial value for gpu  memory
    cudaMemcpy(d_A, matA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matB, K * N * sizeof(float), cudaMemcpyHostToDevice);

    int block_D;

    if (K < BLOCK_DIM)
        block_D = K;
    else
        block_D = BLOCK_DIM;

    dim3 blockDim(block_D, block_D);
    dim3 gridDim((N + block_D - 1)/block_D, (M + block_D - 1)/block_D);
    
    for (int i=0; i<iter; i++) {
        matMul_cuda<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, block_D);
        cudaDeviceSynchronize();
    }
    
    // copy data from the gpu
    cudaMemcpy(matC, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // end timer
    gettimeofday(&end, NULL);

    // Calculating total time taken
    double time_taken;
    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;

    cout << endl;
    cout << "       Total time: " << fixed << time_taken << setprecision(6);
    cout << " sec." << endl;

    // verification
    /*mulMat(matA, matB, matC_cpu, M, N, K);
    cout << endl << "CPU" << endl;
    printMat(matC_cpu, M, N);

    cout << endl << "GPU" << endl;
    if (printOp == 1)
        printMat(matC, N, M);

    if (verification(matC, matC_cpu, M*N))
        printf("SUCCESS!!\n");
    else
        printf("Error\n");*/

    if (printOp == 1)
        printMat(matC, N, M);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(matA);
    free(matB);
    free(matC);
    //free(matC_cpu);

    return 0;
}

bool verification(const float *C_cpu, const float *C_gpu, int length) {
    float epsilon = 0.001;
    float error = 0.f;
    for (int i=0; i<length; i++) {
        error += (abs(C_gpu[i] - C_cpu[i])/abs(C_cpu[i]))*100;
        if (abs(C_cpu[i] - C_gpu[i]) >= epsilon) {
            cout << "Error: " << error/length << endl;
            return false;
        }
    }
    cout << "Error: " << error/length << endl;
    return true;
}

__global__ void matMul_cuda(const float *A, const float *B, float *C, int M, int N, int K, const int block_D) {
    int bid_x = blockIdx.x * blockDim.x;
    int bid_y = blockIdx.y * blockDim.y;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    float element_c = 0.f;

    __shared__ float s_tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float s_tile_B[BLOCK_DIM][BLOCK_DIM];

    int aBegin = K * block_D * bid_y;
    int aEnd = aBegin + K - 1;
    int aStep = block_D;

    int bBegin = block_D * bid_x;
    int bStep = block_D * N;

    for (int i=aBegin, j=bBegin; i<=aEnd; i+=aStep, j+=bStep) {
        s_tile_A[tid_y][tid_x] = A[i + K * tid_y + tid_x];
        s_tile_B[tid_x][tid_y] = B[j + N * tid_x + tid_y];

        __syncthreads();

        for (int k=0; k<block_D; ++k) {
            element_c += s_tile_A[tid_y][k] * s_tile_B[k][tid_x];
        }

        __syncthreads();
    }
    int cIdx = N * block_D * bid_y + block_D * bid_x;

    C[cIdx + N * tid_y + tid_x] = element_c;
}

void mulMat(const float *A, const float *B, float *C, int N, int M, int K) {
    for (int row=0; row<M; row++) {
        for (int col=0; col<N; col++) {
            float element_c = 0.f;
            for (int e=0; e<K; e++) {
                element_c += A[row * K + e] * B[e * N + col];
            }
            C[row * N + col] = element_c;
        }
    }
}
