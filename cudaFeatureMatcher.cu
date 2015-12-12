#include <iostream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "Matrix.h"

/*
 *Source: http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool
        abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/*
 *Pre-conditions:
 *    `descriptor1` and `descriptor2` are each arrays of length `length`
 *Post-conditions:
 *    Returns the L2 distance of the two descriptors interpreted as vectors.
 */
float computeL2Distance(float* descriptor1, float* descriptor2, int length) {
    float distance = 0;
    for (int i = 0; i < length; ++i) {
        distance += pow(descriptor1[i] - descriptor2[i], 2);
    }
    return pow(distance, 0.5);
}

/*
 *Pre-conditions:
 *    `descriptors` is a n x k matrix
 *    `distanceMat` is a n x n matrix
 *Post-conditions:
 *    distanceMat[j][i] is the Euclidean distance from descriptors[j][:]
 *    to descriptors[i][:]. Note that Euclidean distance is symmetric.
 */
void cpuComputeDistanceMat(const Matrix descriptors, Matrix distanceMat) {
    assert(descriptors.height == distanceMat.height);
    assert(distanceMat.height == distanceMat.width);

    int n = descriptors.height;
    int k = descriptors.width;

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            float* descriptor1 = descriptors.elements + j * k;
            float* descriptor2 = descriptors.elements + i * k;
            float* dst = distanceMat.elements + j * n + i;
            *dst = computeL2Distance(descriptor1, descriptor2, k);
        }
    }
}

/*
 *Pre-conditions:
 *    start < stop
 *    0 <= i < mat.height for i = start, stop
 *Post-conditions:
 *    Returns submat, which is a view of mat[start:stop][:]
 *    submat contains rows start, start+1, ..., stop-1 of mat
 */
Matrix getSubmatrix(const Matrix mat, const int start, const int stop) {
    assert(start < stop);
    assert(0 <= start && start < mat.height);
    assert(0 <= stop && stop < mat.height);

    Matrix submat;
    submat.width = mat.width;
    submat.height = stop - start;
    submat.elements = mat.elements + start * submat.width;

    return submat;
}

/*
 *Pre-conditions:
 *    mat.height >= 2
 *    0 <= col < mat.width
 *Post-conditions:
 *    mat[idx1][col] and mat[idx2][col] are the smallest and second smallest
 *    elements in mat[:][col]
 */
void getIndexOfTwoSmallestInColumn(const Matrix mat, int col, int& idx1, int& idx2) {
    assert(mat.height >= 2);
    assert(0 <= col && col < mat.width);

    float val1 = FLT_MAX, val2 = FLT_MAX;
    idx1 = -1, idx2 = -1;

    for (int j = 0; j < mat.height; ++j) {
        float currVal = mat.elements[j * mat.width + col];

        // Push current value to first place
        if (currVal < val1) {
            val2 = val1;
            idx2 = idx1;
            val1 = currVal;
            idx1 = j;
        }

        // Push current value to second place
        else if (currVal < val2) {
            val2 = currVal;
            idx2 = j;
        }
    }
}

void test() {
    Matrix descriptors = AllocateMatrix(n, k, 1);
    printMatrix(descriptors);
    for (int i = 0; i < descriptors.width; ++i) {
        int idx1, idx2;
        getIndexOfTwoSmallestInColumn(descriptors, i, idx1, idx2);
        printf("for col %i, indices are %i, %i, vals are %f %f\n",
                i, idx1, idx2, descriptors.elements[idx1 * descriptors.width + i],
                descriptors.elements[idx2 * descriptors.width + i]
              );
    }
    FreeMatrix(&descriptors);
}


int main(void) {

    int n = 4;  // sum of all feature counts
    int k = 4;  // size of one descriptor

    Matrix descriptors = AllocateMatrix(n, k, 1);
    Matrix distanceMat = AllocateMatrix(n, n, 0);

    cpuComputeDistanceMat(descriptors, distanceMat);
    printMatrix(distanceMat);

    FreeMatrix(&descriptors);
    FreeMatrix(&distanceMat);

    /*cv::Mat image = cv::imread( "outputImages/result.jpg", 1 );*/
    /*printf("size = (%i, %i)\n", image.rows, image.cols);*/


    return 0;
}
