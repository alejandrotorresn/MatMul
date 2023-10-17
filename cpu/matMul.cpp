/*
    Matrix Multiplication Tests
    module load mkl/latest icc/latest compiler/latest 
    icpx -fsycl -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl  -DMKL_ILP64  -m64 -qopenmp -ljsoncpp -I"${MKLROOT}/include" -O3 -mavx2 -mfma matMul.cpp -o matMul
    icpx -fsycl -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl  -DMKL_ILP64  -m64 -qopenmp -ljsoncpp -I"${MKLROOT}/include" -O3 -mavx2 -mfma -mavx512f -mavx512vl -mavx512bw -mavx512dq  matMul.cpp -o matMul
*/
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

    float *matA, *matB, *matC, *matD, *matE, *matF, *matG;
    int rowsA, colsA;
    int rowsB, colsB;
    auto t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int N;
    double time;

    std::ofstream results;
    results.open("../results/results_cpu.json");
    
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
        matE = (float *)mkl_malloc(N*N*sizeof(float), 32);
        matF = (float *)mkl_malloc(N*N*sizeof(float), 32);
        matG = (float *)mkl_malloc(N*N*sizeof(float), 32);

        
        std::string pathA = std::string(path + "matA_") + std::to_string(N) + std::string(".dat");
        std::string pathB = std::string(path + "matB_") + std::to_string(N) + std::string(".dat");
        
        matrix::readMat(pathA.data(), matA, rowsA, colsA);
        matrix::readMat(pathB.data(), matB, rowsB, colsB);

        // Serial
        time = 0.0;
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matC, N, N);
            t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            matmul::serialNaive(matA, matB, matC, N, N, N);
            t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            time += (double)(t2 - t1)/1e+6;
        }   
        
        //std::cout << std::setw(14) << std::fixed << std::setprecision(4) << "Serial (ms): " << (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0f/iter << "\n";
        out_results["serial"][std::to_string(N)]["time"] = (time)/iter;
        out_results["serial"][std::to_string(N)]["error"] = matrix::compare(matC, matC, N, N);

        // OpenMP
        time = 0.0;
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matD, N, N);
            t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            matmul::ompNaive(matA, matB, matD, N, N, N);
            t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            time += (double)(t2 - t1)/1e+6;
        }
        
        //std::cout << std::setw(14) << std::fixed << std::setprecision(4) << "OpenMP (ms): " << (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0f/iter << "\n";
        out_results["openMP"][std::to_string(N)]["time"] = (time)/iter;
        out_results["openMP"][std::to_string(N)]["error"] = matrix::compare(matD, matC, N, N);

        // AVX2
        time = 0.0;
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matE, N, N);
            t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            matmul::avx2_omp(matA, matB, matE, N, N, N);
            t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            time += (double)(t2 - t1)/1e+6;
        }   
        
        //std::cout << std::setw(14) << std::fixed << std::setprecision(4) << "avx2 (ms): " << (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0f/iter << "\n";
        out_results["avx2"][std::to_string(N)]["time"] = (time)/iter;
        out_results["avx2"][std::to_string(N)]["error"] = matrix::compare(matE, matC, N, N);

        // AVX512
        /*time = 0.0;
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matF, N, N);
            t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            matmul::avx512_omp(matA, matB, matF, N, N, N);
            t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            time += (double)(t2 - t1)/1e+6;
        }   
        
        //std::cout << std::setw(14) << std::fixed << std::setprecision(4) << "avx512 (ms): " << (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0f/iter << "\n";
        out_results["avx512"][std::to_string(N)]["time"] = (time)/iter;
        out_results["avx512"][std::to_string(N)]["error"] = matrix::compare(matF, matC, N, N);
        */

        // MKL
        time = 0.0;
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matG, N, N);
            t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            matmul::parallelMKL(matA, matB, matG, N, N, N);
            t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            time += (double)(t2 - t1)/1e+6;
        }   
        
        //std::cout << std::setw(14) << std::fixed << std::setprecision(4) << "MKL (ms): " << (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0f/iter << "\n";
        out_results["mkl"][std::to_string(N)]["time"] = (time)/iter;
        out_results["mkl"][std::to_string(N)]["error"] = matrix::compare(matG, matC, N, N);

        // Save results
        std::string path_save_serial = std::string("../data/matC_serial_") + std::to_string(N) + std::string(".dat");
        matrix::writeMat(path_save_serial.data(), matC, N, N);
        
        std::string path_save_omp = std::string("../data/matC_omp_") + std::to_string(N) + std::string(".dat");
        matrix::writeMat(path_save_omp.data(), matD, N, N);

        std::string path_save_avx2 = std::string("../data/matC_avx2_") + std::to_string(N) + std::string(".dat");
        matrix::writeMat(path_save_avx2.data(), matE, N, N);
        
        //std::string path_save_avx512 = std::string("../data/matC_avx51_") + std::to_string(N) + std::string(".dat");
        //matrix::writeMat(path_save_avx512.data(), matF, N, N);
        
        std::string path_save_mkl = std::string("../data/matC_mkl_") + std::to_string(N) + std::string(".dat");
        matrix::writeMat(path_save_mkl.data(), matG, N, N);

        mkl_free(matA);
        mkl_free(matB);
        mkl_free(matC);
        mkl_free(matD);
        mkl_free(matE);
        mkl_free(matF);
        mkl_free(matG);

    }

    Json::StyledWriter styledWriter;
    results << styledWriter.write(out_results);

    ifs.close();
    results.close();
    return 0;
}