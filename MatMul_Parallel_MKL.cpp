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
#include <mkl.h>
#include "MatMul.h"

using namespace std;

int main(int argc, char **argv) {

    float *matA, *matB, *matC;
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
    beta = 0.f;

    // Allocation memory space
    matA = (float *)mkl_malloc(M * K * sizeof(float), 64);
    matB = (float *)mkl_malloc(K * N * sizeof(float), 64);
    matC = (float *)mkl_malloc(M * N * sizeof(float), 64);

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
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, matA, K, matB, N, beta, matC, N);

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

    mkl_free(matA);
    mkl_free(matB);
    mkl_free(matC);

    return 0;
}

