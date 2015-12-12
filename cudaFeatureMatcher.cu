#include <iostream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "Matrix.h"

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

int main(void) {

    int n = 10;  // sum of all feature counts
    int k = 2;  // size of one descriptor

    Matrix descriptors = AllocateMatrix(n, k, 1);
    Matrix distanceMat = AllocateMatrix(n, n, 0);
    cpuComputeDistanceMat(descriptors, distanceMat);
    printMatrix(descriptors);
    printMatrix(distanceMat);

    FreeMatrix(&descriptors);
    FreeMatrix(&distanceMat);

    /*cv::Mat image = cv::imread( "outputImages/result.jpg", 1 );*/
    /*printf("size = (%i, %i)\n", image.rows, image.cols);*/


    return 0;
}
