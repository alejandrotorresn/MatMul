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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "MatMul.h"

using namespace std;

int main(int argc, char **argv) {
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cudaStream_t stream;

    float *matA, *matB, *matC;
    float *d_A, *d_B, *d_C;
    int N, M, K;
    float alpha, beta;
    int newMatrix;
    int iter;
    int printOp;

    N = atoi(argv[1]);
    M = atoi(argv[2]);
    K = atoi(argv[3]);
    newMatrix = atoi(argv[4]);
    iter = atoi(argv[5]);
    printOp = atoi(argv[6]);

    alpha = 1.f;
    beta = 1.f;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cout << "CUBLAS initialization failed" << endl;
        return EXIT_FAILURE;
    }

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

    cudaStat = cudaStreamCreate(&stream);

    cublasSetMatrixAsync(M, K, sizeof(*d_A), matA, M, d_A, M, stream);
    cublasSetMatrixAsync(K, N, sizeof(*d_B), matB, K, d_B, K, stream);
    cublasSetMatrixAsync(M, N, sizeof(*d_C), matC, M, d_C, M, stream);

    cublasSetStream(handle, stream);

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);

    cublasGetMatrixAsync(M, N, sizeof(*d_C), d_C, M, matC, M, stream);
    cudaStreamSynchronize(stream);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    // end timer
    gettimeofday(&end, NULL);

    // Calculating total time taken
    double time_taken;
    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;

    cout << endl;
    cout << "       Total time: " << fixed << time_taken << setprecision(6);
    cout << " sec." << endl;

    if (printOp == 1)
        printMat(matC, N, M);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(matA);
    free(matB);
    free(matC);

    return 0;
}