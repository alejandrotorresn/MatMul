/*
    Matrix Multiplication Tests
    nvcc -lcublas -ljsoncpp matMul.cu ../include/cuda_mm.cu -o matMul
*/
#include <iostream>
#include <fstream>
#include <array>
#include <bits/stdc++.h>
#include <chrono>
#include <string>
#include <json/json.h>
#include "../include/matrix.hpp"
#include "../include/cuda_mm.hpp"


int main() {
    std::ios_base::sync_with_stdio(false);

    float *matA, *matB, *matC, *matD, *matE, *matF, *matG,  *matH;
    int rowsA, colsA;
    int rowsB, colsB;
    int rowsC, colsC;

    auto t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int N;

    double time;

    std::ofstream results;
    results.open("../results/results_gpu.json");
    
    std::ifstream ifs("../conf/settings.json");
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);

    std::string path = obj["path"].asString();
    int iter = obj["iterations"].asInt();
    Json::Value& sizes = obj["sizes"];

    Json::Value out_results;

    for ( int i = 0; i < sizes.size(); i++ ) {
        N = sizes[i]["N"].asInt();

        const dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
        const dim3 gridDim((N + BLOCK_DIM - 1)/BLOCK_DIM, (N + BLOCK_DIM - 1)/BLOCK_DIM);

        // Allocation memory space
        matA = (float *)malloc(N * N * sizeof(float));
        matB = (float *)malloc(N * N * sizeof(float));
        matC = (float *)malloc(N * N * sizeof(float));
        matD = (float *)malloc(N * N * sizeof(float));
        matE = (float *)malloc(N * N * sizeof(float));
        matF = (float *)malloc(N * N * sizeof(float));
        matG = (float *)malloc(N * N * sizeof(float));
        matH = (float *)malloc(N * N * sizeof(float));

        std::string pathA = std::string(path + "matA_") + std::to_string(N) + std::string(".dat");
        std::string pathB = std::string(path + "matB_") + std::to_string(N) + std::string(".dat");
        std::string pathC = std::string(path + "matC_serial_") + std::to_string(N) + std::string(".dat");
        
        matrix::readMat(pathA.data(), matA, rowsA, colsA);
        matrix::readMat(pathB.data(), matB, rowsB, colsB);
        matrix::readMat(pathC.data(), matC, rowsC, colsC);

        // Cuda Naive
        time = 0.0;        
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matD, N, N);
            t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            cuda::cuda_sgemm_naive(matA, matB, matD, N, N, N, blockDim, gridDim);
            t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            time += (double)(t2 - t1)/1e+6;
        }   
        
        out_results["cudaNaive"][std::to_string(N)]["time"] = (time)/iter;
        out_results["cudaNaive"][std::to_string(N)]["error"] = matrix::compare(matD, matC, N, N);

        // Cuda Tiled
        time = 0.0;
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matE, N, N);
            t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            cuda::cuda_sgemm_tiled(matA, matB, matE, N, N, N, blockDim, gridDim);
            t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            time += (double)(t2 - t1)/1e+6;
        }   
        
        out_results["cudaTiled"][std::to_string(N)]["time"] = (time)/iter;
        out_results["cudaTiled"][std::to_string(N)]["error"] = matrix::compare(matE, matC, N, N);

        // cuBLAS
        time = 0.0;
        
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matF, N, N);
            t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            cuda::cublas_sgemm(matA, matB, matF, N, N, N);
            t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            time += (double)(t2 - t1)/1e+6;
        }   
        
        out_results["cuBLAS"][std::to_string(N)]["time"] = (time)/iter;
        out_results["cuBLAS"][std::to_string(N)]["error"] = matrix::compare(matF, matC, N, N);

        // cuBLAS Async
        time = 0.0;
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matG, N, N);
            t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            cuda::cublas_sgemmAsync(matA, matB, matG, N, N, N);
            t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            time += (double)(t2 - t1)/1e+6;
        }   
        
        out_results["cuBLAS_async"][std::to_string(N)]["time"] = (time)/iter;
        out_results["cuBLAS_async"][std::to_string(N)]["error"] = matrix::compare(matG, matC, N, N);

        // cuBLAS Tensor Core
        time = 0.0;
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matH, N, N);
            t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            cuda::cublas_sgemmTensor(matA, matB, matH, N, N, N, blockDim, gridDim);
            t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            time += (double)(t2 - t1)/1e+6;
        }   
        
        out_results["cuBLAS_Tensor"][std::to_string(N)]["time"] = (time)/iter;
        out_results["cuBLAS_Tensor"][std::to_string(N)]["error"] = matrix::compare(matH, matC, N, N);      

        free(matA);
        free(matB);
        free(matC);
        free(matD);
        free(matE);
        free(matF);
        free(matG);
        free(matH);
    }

    Json::StyledWriter styledWriter;
    results << styledWriter.write(out_results);

    ifs.close();
    results.close();

    return 0;
}