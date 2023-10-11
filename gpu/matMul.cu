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
#include <jsoncpp/json/json.h>
#include "../include/matrix.hpp"
#include "../include/cuda_mm.hpp"


int main() {
    std::ios_base::sync_with_stdio(false);

    float *matA, *matB, *matC, *matD, *matE, *matF, *matG,  *matH;
    int rowsA, colsA;
    int rowsB, colsB;
    int rowsC, colsC;

    auto t1 = std::chrono::steady_clock::now();
    auto t2 = std::chrono::steady_clock::now();
    int N;

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
        t1 = std::chrono::steady_clock::now();
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matD, N, N);
            cuda::cuda_sgemm_naive(matA, matB, matD, N, N, N, blockDim, gridDim);
        }   
        t2 = std::chrono::steady_clock::now();
        out_results["Times"]["cudaNaive"][std::to_string(N)]["time"] = ( (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0f ) / iter;
        std::string path_save_cuda_naive = std::string("../data/matC_cudaNaive_") + std::to_string(N) + std::string(".dat");
        matrix::writeMat(path_save_cuda_naive.data(), matD, N, N);

        // Cuda Tiled
        t1 = std::chrono::steady_clock::now();
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matE, N, N);
            cuda::cuda_sgemm_naive(matA, matB, matE, N, N, N, blockDim, gridDim);
        }   
        t2 = std::chrono::steady_clock::now();
        out_results["Times"]["cudaTiled"][std::to_string(N)]["time"] = ( (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0f ) / iter;
        std::string path_save_cuda_tiled = std::string("../data/matC_cudaTiled_") + std::to_string(N) + std::string(".dat");
        matrix::writeMat(path_save_cuda_tiled.data(), matE, N, N);



        // cuBLAS
        t1 = std::chrono::steady_clock::now();
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matF, N, N);
            cuda::cublas_sgemm(matA, matB, matF, N, N, N);
        }   
        t2 = std::chrono::steady_clock::now();
        out_results["Times"]["cuBLAS"][std::to_string(N)]["time"] = ( (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0f ) / iter;
        std::string path_save_cuBLAS = std::string("../data/matC_cuBLAS_") + std::to_string(N) + std::string(".dat");
        matrix::writeMat(path_save_cuBLAS.data(), matF, N, N);

        // cuBLAS Async
        t1 = std::chrono::steady_clock::now();
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matG, N, N);
            cuda::cublas_sgemmAsync(matA, matB, matG, N, N, N);
        }   
        t2 = std::chrono::steady_clock::now();
        out_results["Times"]["cuBLAS_async"][std::to_string(N)]["time"] = ( (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0f ) / iter;
        std::string path_save_cuBLAS_async = std::string("../data/matC_cuBLAS_async_") + std::to_string(N) + std::string(".dat");
        matrix::writeMat(path_save_cuBLAS_async.data(), matG, N, N);

        // cuBLAS Tensor Core
        t1 = std::chrono::steady_clock::now();
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matH, N, N);
            cuda::cublas_sgemmTensor(matA, matB, matH, N, N, N, blockDim, gridDim);
        }   
        t2 = std::chrono::steady_clock::now();
        out_results["Times"]["cuBLAS_Tensor"][std::to_string(N)]["time"] = ( (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0f ) / iter;
        std::string path_save_cuBLAS_tensor = std::string("../data/matC_cuBLAS_Tensor_") + std::to_string(N) + std::string(".dat");
        matrix::writeMat(path_save_cuBLAS_tensor.data(), matH, N, N);

        out_results["Error"][std::to_string(N)]["cudaNaive"] = matrix::compare(matD, matC, N, N);
        out_results["Error"][std::to_string(N)]["cudaTiled"] = matrix::compare(matE, matC, N, N);
        out_results["Error"][std::to_string(N)]["cuBLAS"] = matrix::compare(matF, matC, N, N);
        out_results["Error"][std::to_string(N)]["cuBLAS_async"] = matrix::compare(matG, matC, N, N);
        out_results["Error"][std::to_string(N)]["cuBLAS_tensor"] = matrix::compare(matH, matC, N, N);

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