/*
    Matrix Multiplication Tests
    nvcc -lcublas -ljsoncpp matMul.cu ../include/cuda_mm.cu -o matMul
    nvcc -lcublas -ljsoncpp matMul.cu ../../include/cuda_mm.cu -I /opt/papi/include  /opt/papi/lib/libpapi.a ../handle_error.cu -o matMul
*/
#include <iostream>
#include <fstream>
#include <array>
#include <bits/stdc++.h>
#include <chrono>
#include <string>
#include <papi.h>
#include <json/json.h>
#include "../../include/matrix.hpp"
#include "../../include/cuda_mm.hpp"

void handle_error(int retval);

int main() {
    std::ios_base::sync_with_stdio(false);

    float *matA, *matB, *matC;
    int rowsA, colsA;
    int rowsB, colsB;
    int retval;
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

        // Allocation memory space
        matA = (float *)malloc(N * N * sizeof(float));
        matB = (float *)malloc(N * N * sizeof(float));
        matC = (float *)malloc(N * N * sizeof(float));

        std::string pathA = std::string("../" + path + "matA_") + std::to_string(N) + std::string(".dat");
        std::string pathB = std::string("../" + path + "matB_") + std::to_string(N) + std::string(".dat");
        
        matrix::readMat(pathA.data(), matA, rowsA, colsA);
        matrix::readMat(pathB.data(), matB, rowsB, colsB);

        retval = PAPI_hl_region_begin("computation");
        if (retval != PAPI_OK)
            handle_error(1);

        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matC, N, N);
            cuda::cublas_sgemmAsync(matA, matB, matC, N, N, N);
        }

        retval = PAPI_hl_region_end("computation");
        if (retval != PAPI_OK)
            handle_error(1);
        
        free(matA);
        free(matB);
        free(matC);
    }

    ifs.close();

    return 0;
}
