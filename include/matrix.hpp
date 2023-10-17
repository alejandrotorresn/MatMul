#pragma once
#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <cmath>
#include <bits/stdc++.h>


namespace matrix {

    template <typename T, size_t size>
    void randInit(std::array<T, size>& data) {
        if (std::is_same<T, int>::value) {
            #pragma omp parallel for
            for ( size_t i = 0; i < data.size(); i++ )
                data[i] = rand()%100;
        } else if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
            #pragma omp parallel for
            for ( size_t i = 0; i < data.size(); i++ ) {
                data[i] = (T)std::rand()/(T)RAND_MAX;
            }
        } else {
            std::cout << "The generator only accepts int, float and double data types.\n";
            exit(0);
        }
    }


    template <typename T>
    void readMat(const char* path, T* vector, int& rows, int& cols) {
        std::vector<int> dim_vec;
        std::string dim;
        int token;
        int i = 0;
        
        std::ifstream matIn(path, std::ios_base::in);

        if ( matIn.fail() ) {
            std::cout << "Error opening file(s):" << path << "\n";
            exit(0);
        }

        if ( matIn.is_open() ) {
            std::getline(matIn, dim);
            std::stringstream ss(dim);
            while ( ss >> token) {
                dim_vec.push_back(token);
            }
            rows = dim_vec[0];
            cols = dim_vec[1];
            
            while ( !matIn.eof() ) {
                matIn >> vector[i];
                i++;
            }
            matIn.close();
        } else {
            matIn.close();
            exit(0);
        }
    }

    template <typename T, size_t size>
    void writeMat(const char* path, std::array<T, size>& vector, const int& rows, const int& cols) {
        std::ofstream matOut;
        matOut.open(path);

        if ( matOut.fail() ) {
            std::cout << "Error! wrong path! \n";
            exit(0);
        }

        matOut << rows << " " << cols << "\n";
        for ( size_t i = 0; i < vector.size(); i++ )
            matOut << std::fixed << std::setprecision(std::numeric_limits<T>::digits10 + 3) << vector[i] << " ";
        matOut.close();

    }

    template <typename T>
    void writeMat(const char* path, T* vector, const int& rows, const int& cols) {
        size_t len = rows * cols;
        std::ofstream matOut;
        matOut.open(path);

        if ( matOut.fail() ) {
            std::cout << "Error! wrong path! \n";
            exit(0);
        }

        matOut << rows << " " << cols << "\n";
        for ( size_t i = 0; i < len; i++ )
            matOut << std::fixed << std::setprecision(std::numeric_limits<T>::digits10 + 3) << vector[i] << " ";
        matOut.close();

    }

    template <typename T>
    float compare(const T* vec1, const T* vec2, const int& rows, const int& cols) {
        float error_mse = 0.f;
        int len = rows * cols;

        for ( size_t i = 0; i < len; i++ )
            error_mse += pow((vec2[i] - vec1[i]), 2);
            
        //std::cout << std::setw(12) << std::fixed << std::setprecision(9) << "Error: " << error/len << "\n";
        //std::cout << std::setw(12) << std::fixed << "Total fails: " << fails << "\n";
        return error_mse/len;
    }
    

    template <typename T>
    void printM(const T& vector, const int& rows, const int& cols) {
        for ( size_t i = 0; i < rows*cols; i ++ ) {
            if (i == 0 | i%cols != 0)
                std::cout << std::setw(12) << std::fixed << std::setprecision(9) << vector[i] << " ";
            else {
                std::cout << "\n" << std::setw(12) << std::fixed << std::setprecision(9) << vector[i] <<" ";
            }
        }
        std::cout << std::endl;
        //std::cout << "length: " << vector.size() << "\n";
        std::cout << "rows: " << rows << "\n";
        std::cout << "cols: " << cols << "\n";
    }

    inline void init_zero(float* mat, const int M, const int N) {
    #pragma omp parallel for
        for ( size_t i = 0; i < M*N; i++ ) {
            mat[i] = 0.0;
        }
    }

}