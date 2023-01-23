#include <iostream>
#include <fstream>
#include "MatMul.h"

using namespace std;

void random_init(float *data, int length, bool flag) {
    if (flag) {
        for (int i=0; i<length; i++)
            data[i] = (float)rand()/(float)RAND_MAX;
    } else {
        for (int i=0; i<length; i++)
            data[i] = 0.0;
    }
}

void printMat(const float *C, int N, int M) {
    cout << endl;
    for (int row=0; row<M; row++) {
        for (int col=0; col<N; col++) {
            cout << C[row * N + col] << " ";
        }
        cout << endl;
    }
}

void readMat(const char path[100], float *vector, int length) {
    ifstream mat_in(path, ios_base::in);
    if (mat_in.is_open() == true) {
        for (int i=0; i<length; i++)
            mat_in >> vector[i];
        mat_in.close();
    } else {
        cout<< "File could not be open!" <<endl;
        mat_in.close();
    }
}

void saveMat(const char path[100], float *vector, int length) {
    ofstream mat_out;
    mat_out.open(path);
    for (int i=0; i<length; i++)
        mat_out << vector[i] << " ";
    mat_out.close();
}

void matInv(float *mat, int rows, int cols) {
    float *matId = (float *)malloc(rows * cols * sizeof(float));
    random_init(matId, rows * cols, false);

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            if ( (i*rows + j) !=  (j*rows + i) ) {
                if ( matId[i * rows + j] == 0.0 ) {
                    float aux = mat[i * rows + j];
                    mat[i * rows + j] = mat[j * rows + i];
                    mat[j * rows + i] = aux;
                    matId[i * rows + j] = 1;
                    matId[j * rows + i] = 1;
                }
            }
        }
    }
    free(matId);
}