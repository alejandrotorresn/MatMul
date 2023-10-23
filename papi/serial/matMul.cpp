/*
    Matrix Multiplication Tests
    module load mkl/latest icc/latest compiler/latest 
    icpx -fsycl -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl  -DMKL_ILP64  -m64 -qopenmp -ljsoncpp -I"${MKLROOT}/include" -O3 -mavx2 -mfma matMul.cpp -o matMul
    icpx -fsycl -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl  -DMKL_ILP64  -m64 -qopenmp -ljsoncpp -I"${MKLROOT}/include" -O3 -mavx2 -mfma -mavx512f -mavx512vl -mavx512bw -mavx512dq  matMul.cpp -o matMul
    // PAPI
    icpx -fsycl -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -m64 -ljsoncpp -I"${MKLROOT}/include" -I /opt/papi/include ../handle_error.c /opt/papi/lib/libpapi.a -O3 matMul.cpp -o matMul
*/
#include <iostream>
#include <fstream>
#include <array>
#include <bits/stdc++.h>
#include <chrono>
#include <string>
#include <papi.h>
#include <jsoncpp/json/json.h>
#include "../../include/matrix.hpp"
#include "../../include/matmul.hpp"

void handle_error(int retval);

int main() {
    std::ios_base::sync_with_stdio(false);

    float *matA, *matB, *matC;
    int rowsA, colsA;
    int rowsB, colsB;
    int retval;
    auto t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int N;
    double time;
    
    std::ifstream ifs("../../conf/settings.json");
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);

    std::string path = obj["path"].asString();
    int iter = obj["iterations"].asInt();
    Json::Value& sizes = obj["sizes"];

    for ( int i = 0; i < sizes.size(); i++ ) {
        size_t N = sizes[i]["N"].asInt();
        
        matA = (float *)mkl_malloc(N*N*sizeof(float), 32);
        matB = (float *)mkl_malloc(N*N*sizeof(float), 32);
        matC = (float *)mkl_malloc(N*N*sizeof(float), 32);

        
        std::string pathA = std::string("../" + path + "matA_") + std::to_string(N) + std::string(".dat");
        std::string pathB = std::string("../" + path + "matB_") + std::to_string(N) + std::string(".dat");
        
        matrix::readMat(pathA.data(), matA, rowsA, colsA);
        matrix::readMat(pathB.data(), matB, rowsB, colsB);
        
        time = 0.0;

        retval = PAPI_hl_region_begin("computation");
        if (retval != PAPI_OK)
            handle_error(1);
        
        for ( size_t i = 0; i < iter; i++ ) {
            matrix::init_zero(matC, N, N);
            t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            matmul::serialNaive(matA, matB, matC, N, N, N);
            t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            time += (double)(t2 - t1)/1e+6;
        }

        retval = PAPI_hl_region_end("computation");
        if (retval != PAPI_OK)
            handle_error(1);

        mkl_free(matA);
        mkl_free(matB);
        mkl_free(matC);
    }

    ifs.close();
    return 0;
}