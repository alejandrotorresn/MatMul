/*
    Matrix Multiplication Tests
    nvcc -lcublas -ljsoncpp matMul.cu ../include/cuda_mm.cu -o matMul
*/
#include <iostream>
#include <fstream>
#include <array>
#include <bits/stdc++.h>
#include <string>
#include <json/json.h>
#include "../../include/matrix.hpp"
#include "../../include/cuda_mm.hpp"

int main() {
    std::ios_base::sync_with_stdio(false);

    float *matA, *matB, *matC;
    int rowsA, colsA;
    int rowsB, colsB;
    int N;
    
    std::ifstream ifs("../../conf/settings.json");
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);

    std::string path = obj["path"].asString();
    int iter = obj["iterations"].asInt();
    Json::Value& sizes = obj["sizes"];

    for ( int i = 0; i < sizes.size(); i++ ) {
        N = sizes[i]["N"].asInt();

        const dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
        const dim3 gridDim((N + BLOCK_DIM - 1)/BLOCK_DIM, (N + BLOCK_DIM - 1)/BLOCK_DIM);

        // Allocation memory space
        matA = (float *)malloc(N * N * sizeof(float));
        matB = (float *)malloc(N * N * sizeof(float));
        matC = (float *)malloc(N * N * sizeof(float));

        std::string pathA = std::string("../" + path + "matA_") + std::to_string(N) + std::string(".dat");
        std::string pathB = std::string("../" + path + "matB_") + std::to_string(N) + std::string(".dat");

        matrix::readMat(pathA.data(), matA, rowsA, colsA);
        matrix::readMat(pathB.data(), matB, rowsB, colsB);

        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matC, N, N);
            cuda::cuda_sgemm_tiled(matA, matB, matC, N, N, N, blockDim, gridDim);
        }

        free(matA);
        free(matB);
        free(matC);
    }

    ifs.close();

    return 0;
}