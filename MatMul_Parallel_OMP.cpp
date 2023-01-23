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

using namespace std;

static void matMul(const float *A, const float *B, float *C, int N, int M, int K);

int main(int argc, char **argv) {

    float *matA, *matB, *matC;
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

    // Allocation memory space
    matA = (float *)malloc(M * K * sizeof(float));
    matB = (float *)malloc(K * N * sizeof(float));
    matC = (float *)malloc(M * N * sizeof(float));

    // Timer
    struct timeval start, end;

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
    
    // unsync the I/O of C and C++
    ios_base::sync_with_stdio(false);

    // start timer
    gettimeofday(&start, NULL);

    for (int i=0; i<iter; i++)
        matMul(matA, matB, matC, N, M, K);

    // end timer
    gettimeofday(&end, NULL);

    // Calculating total time
    double time_taken;
    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;

    cout << "\tTotal time: " << fixed << time_taken << setprecision(6);
    cout << " sec." << endl;

    if (printOp == 1)
        printMat(matC, N, M);

    free(matA);
    free(matB);
    free(matC);
    
    return 0;
}


static void matMul(const float *A, const float *B, float *C, int N, int M, int K) {
    int row, col, e;
    #pragma omp parallel for private(row, col, e) shared(C)
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