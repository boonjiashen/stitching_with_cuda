/*
 *See if we can get a GPU to throw cudaErrorMemoryAllocation
 */
#include <iostream>
#include <cstdio>

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
    float* ptr = NULL;
    size_t size = pow(10, 9) * sizeof(float);
    gpuErrchk( cudaMalloc((void**)&ptr, size) );
    printf("Successfully allocated %zu bytes.\n", size);
    cudaFree(ptr);

    /*Matrix mat = AllocateMatrix(n, k, 1);*/
    /*printMatrix(mat);*/
    /*FreeMatrix(&mat);*/

    /*cv::Mat image = cv::imread( "outputImages/result.jpg", 1 );*/
    /*printf("size = (%i, %i)\n", image.rows, image.cols);*/


    return 0;
}
