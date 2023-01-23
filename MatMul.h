#pragma once

extern void random_init(float *data, int length, bool flag);
extern void printMat(const float *C, int N, int M);
extern void readMat(const char path[100], float *vector, int length);
extern void saveMat(const char path[100], float *vector, int length);
extern void matInv(float *mat, int rows, int cols);