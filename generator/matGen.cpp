/*
    g++ test.cpp -o test -fopenmp -lpthread 
    ulimit -s unlimited
*/
#include <iostream>
#include <array>
#include <bits/stdc++.h> 
#include "../include/matrix.hpp"

const int N = 304;


int main() {
    int rows, cols;
    std::array<float, N*N> mat;
    mat.fill(0);

    std::string path1 = std::string("../data/matA_") + std::to_string(N) + std::string(".dat");
    std::string path2 = std::string("../data/matB_") + std::to_string(N) + std::string(".dat");

    matrix::randInit(mat);
    matrix::writeMat(path1.data(), mat, N, N);
    matrix::randInit(mat);
    matrix::writeMat(path2.data(), mat, N, N);
    
    //matrix::readMat(path.data(), matB, rows, cols);
    //matrix::writeMat(path1.data(), matB, N, N);
    //matrix::printM(matA, rows, cols);
    //matrix::printM(matB, rows, cols);
    //if (matrix::compare(matA, matB))
    //    std::cout << "Ok\n";
    return 0;
}