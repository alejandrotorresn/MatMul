#include <iostream>
#include <fstream>
#include <array>
#include <bits/stdc++.h>
#include <chrono>
#include <string>
#include <jsoncpp/json/json.h>
#include "../include/matrix.hpp"
#include "../include/matmul.hpp"

int main() {
    std::ios_base::sync_with_stdio(false);

    float *matA, *matB, *matC, *matD;
    int rowsA, colsA;
    int rowsB, colsB;
    int rowsC, colsC;
    auto t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int N;

    std::ofstream results;
    results.open("../results/results_sycl_gpu.json");
    
    std::ifstream ifs("../conf/settings.json");
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);

    std::string path = obj["path"].asString();
    int iter = obj["iterations"].asInt();
    Json::Value& sizes = obj["sizes"];

    Json::Value out_results;

    for ( int i = 0; i < sizes.size(); i++ ) {
        size_t N = sizes[i]["N"].asInt();
        
        matA = (float *)mkl_malloc(N*N*sizeof(float), 32);
        matB = (float *)mkl_malloc(N*N*sizeof(float), 32);
        matC = (float *)mkl_malloc(N*N*sizeof(float), 32);
        matD = (float *)mkl_malloc(N*N*sizeof(float), 32);

        
        std::string pathA = std::string(path + "matA_") + std::to_string(N) + std::string(".dat");
        std::string pathB = std::string(path + "matB_") + std::to_string(N) + std::string(".dat");
        std::string pathC = std::string(path + "matC_serial_") + std::to_string(N) + std::string(".dat");
        
        matrix::readMat(pathA.data(), matA, rowsA, colsA);
        matrix::readMat(pathB.data(), matB, rowsB, colsB);
        matrix::readMat(pathC.data(), matC, rowsC, colsC);

        // syCL
        double time = 0.0;
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matD, N, N);
            t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            matmul::sycl_mm_gpu(matA, matB, matD, N, N, N);
            t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            time += (double)(t2 - t1)/1e+6;
        }   
        
        //std::cout << std::setw(14) << std::fixed << std::setprecision(4) << "MKL (ms): " << (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0f/iter << "\n";
        out_results["Times"]["sycl"][std::to_string(N)]["time"] = (time)/iter;

        // Save results
        std::string path_save_sycl = std::string("../data/matC_sycl_") + std::to_string(N) + std::string(".dat");
        matrix::writeMat(path_save_sycl.data(), matD, N, N);

        // MKL Vs Serial
        out_results["Error"][std::to_string(N)]["sycl_gpu"] = matrix::compare(matD, matC, N, N);

        mkl_free(matA);
        mkl_free(matB);
        mkl_free(matC);
        mkl_free(matD);

    }

    Json::StyledWriter styledWriter;
    results << styledWriter.write(out_results);

    ifs.close();
    results.close();
    return 0;
}    

