#pragma once
#include <mkl.h>
#include <immintrin.h>

// use these arrays to perform masked load and store
// operations for any residual columns.
const uint32_t ZR = 0;
const uint32_t MV = 0x80000000;

alignas(32) const uint32_t c_Mask0[8] {ZR, ZR, ZR, ZR, ZR, ZR, ZR, ZR};
alignas(32) const uint32_t c_Mask1[8] {MV, ZR, ZR, ZR, ZR, ZR, ZR, ZR};
alignas(32) const uint32_t c_Mask2[8] {MV, MV, ZR, ZR, ZR, ZR, ZR, ZR};
alignas(32) const uint32_t c_Mask3[8] {MV, MV, MV, ZR, ZR, ZR, ZR, ZR};
alignas(32) const uint32_t c_Mask4[8] {MV, MV, MV, MV, ZR, ZR, ZR, ZR};
alignas(32) const uint32_t c_Mask5[8] {MV, MV, MV, MV, MV, ZR, ZR, ZR};
alignas(32) const uint32_t c_Mask6[8] {MV, MV, MV, MV, MV, MV, ZR, ZR};
alignas(32) const uint32_t c_Mask7[8] {MV, MV, MV, MV, MV, MV, MV, ZR};

const uint32_t *c_MaskMovLUT[8] {
    c_Mask0, c_Mask1, c_Mask2, c_Mask3, c_Mask4, c_Mask5, c_Mask7, c_Mask7
};

namespace matmul {

    template <typename T>
    inline void serialNaive(const T& matA, const T& matB, T& matC, const int N, const int M, const int K) {
        for (size_t m = 0; m < M; m++ ) {
            for (size_t k = 0; k < K; k++ ) {
                for (size_t n = 0; n < N; n++ ) {
                    matC[m * M + n] += matA[m * M + k] * matB[k * K + n]; 
                }
            }
        }
    }

    template <typename T>
    inline void ompNaive(const T& matA, const T& matB, T& matC, const int N, const int M, const int K) {
        size_t n, m, k;
        #pragma omp parallel for private(n, m, k) shared(matC)
        for (m = 0; m < M; m++ ) {
            for (k = 0; k < K; k++ ) {
                for (n = 0; n < N; n++ ) {
                    matC[m * M + n] += matA[m * M + k] * matB[k * K + n]; 
                }
            }
        }       
    }

    template <typename T>
    inline void parallelMKL(const T& matA, const T& matB, T& matC, const int N, const int M, const int K) {
        float alpha = 1.0;
        float beta = 0.0;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, matA, K, matB, N, beta, matC, N);
    }

    inline void avx2_omp(const float *A, const float *B, float *C, int N, int M, int K) {
        /*
            A -> N * K
            B -> K * M
            C -> N * M
            N: rows in A -> C
            M: cols in B -> C
            K: cols in A and rows in B
        */
        const size_t num_simd_elements = 8;
        size_t num_residuals_cols = N % num_simd_elements;
        __m256i res_mask = _mm256_load_si256( (__m256i *)c_MaskMovLUT[num_residuals_cols]);

        // Repeat for each row in C
        size_t i, j, k;
        //#pragma omp parallel for private(i, k)
        #pragma omp parallel for shared(C, A, B, N, M, K, res_mask, num_residuals_cols) private(i,k) default(none)
        for (size_t i=0; i<N; i++) {
            size_t j = 0;
            // Repeat while 8 or more columns remain
            while (j + num_simd_elements <= M) {
                __m256 c_vals = _mm256_setzero_ps();
                // Calculate prodcuts for C[i][j:j+7]
                for (size_t k=0; k<K; k++) {
                    __m256 a_vals = _mm256_broadcast_ss(&A[i * K + k]);
                    __m256 b_vals = _mm256_loadu_ps(&B[k * M + j]);
                    c_vals = _mm256_fmadd_ps(a_vals, b_vals, c_vals);
                }
                _mm256_storeu_ps(&C[i * M + j], c_vals);
                j += num_simd_elements;
            }

            if (num_residuals_cols) {
                __m256 c_vals = _mm256_setzero_ps();
                for (size_t k=0; k<K; k++) {
                    __m256 a_vals = _mm256_broadcast_ss(&A[i * K + k]);
                    __m256 b_vals = _mm256_maskload_ps(&B[k * M + j], res_mask);
                    c_vals = _mm256_fmadd_ps(a_vals, b_vals, c_vals);
                }
                _mm256_maskstore_ps(&C[i * M + j], res_mask, c_vals);
            }
        }
    }

    inline void avx512_omp(const float *A, const float *B, float *C, int N, int M, int K) {
        /*
            A -> N * K
            B -> K * M
            C -> N * M
            N: rows in A -> C
            M: cols in B -> C
            K: cols in A and rows in B
        */

    const size_t num_simd_elements = 16;
    size_t num_residual_cols = M % num_simd_elements;

    __mmask16 res_mask = (__mmask16)((1 << num_residual_cols) - 1);

        size_t i, j, k;
        #pragma omp parallel for shared(C, A, B, N, M, K,) private(i, k)
        for (size_t i=0; i<N; i++) {
            size_t j = 0;
            while (j + num_simd_elements <= M) {
                __m512 c_vals = _mm512_setzero_ps();
                for (size_t k=0; k<K; k++) {
                    __m512 a_vals = _mm512_set1_ps(A[i * K + k]);
                    __m512 b_vals = _mm512_loadu_ps(&B[k * M + j]);
                    c_vals = _mm512_fmadd_ps(a_vals, b_vals, c_vals);
                }
                _mm512_storeu_ps(&C[i * M + j], c_vals);
                j += num_simd_elements;
            }

            if (num_residual_cols != 0) {
                __m512 c_vals = _mm512_setzero_ps();
                for (size_t k = 0; k<K; k++) {
                    __m512 a_vals = _mm512_set1_ps(A[i * K + k]);
                    __m512 b_vals = _mm512_maskz_loadu_ps(res_mask, &B[k * K + j]);
                    c_vals = _mm512_fmadd_ps(a_vals, b_vals, c_vals);
                }
                _mm512_mask_storeu_ps(&C[i * M + j], res_mask, c_vals);
            }
        }
    }


}