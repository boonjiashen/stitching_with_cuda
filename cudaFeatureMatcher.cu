#include <iostream>
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

int main(void) {

    int n = 10;  // sum of all feature counts
    int k = 4;  // size of one descriptor

    float* ptr = NULL;
    gpuErrchk( cudaMallocManaged(&ptr, n*k*sizeof(float)) );

    /*Matrix mat = AllocateMatrix(n, k, 1);*/
    /*printMatrix(mat);*/
    /*FreeMatrix(&mat);*/

    /*cv::Mat image = cv::imread( "outputImages/result.jpg", 1 );*/
    /*printf("size = (%i, %i)\n", image.rows, image.cols);*/


    return 0;
}
