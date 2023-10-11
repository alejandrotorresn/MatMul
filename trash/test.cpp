#include <iostream>
#include "../include/matrix.hpp"

int main() {

    float* matA;
    float* matB;
    int N = 3;
    int rowsA, rowsB, colsA, colsB;

    matA = (float *)malloc(N * N * sizeof(float));
    matB = (float *)malloc(N * N * sizeof(float));

    std::string pathA = "../data/matC_serial_3.dat";
    std::string pathB = "../data/matC_cuda_3.dat";

    matrix::readMat(pathA.data(), matA, rowsA, colsA);
    matrix::readMat(pathB.data(), matB, rowsB, colsB);
    
    std::cout << "Error: " << matrix::compare(matA, matB, rowsA, colsB) << std::endl;

    return 0;
}