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

void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;
    std::wcout << "NBody.GPU" << std::endl << "=========" << std::endl << std::endl;

    std::wcout << "CUDA version:   v" << CUDART_VERSION << std::endl;    

    int devCount;
    cudaGetDeviceCount(&devCount);
    std::wcout << "CUDA Devices: " << std::endl << std::endl;

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        std::wcout << "Device " << i << ": " << props.name << ": compute capability " << props.major << "." << props.minor << std::endl;
        std::wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << std::endl;
        std::wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << std::endl;
        std::wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << std::endl;
        std::wcout << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;

        std::wcout << "  Warp size:         " << props.warpSize << std::endl;
        std::wcout << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
        std::wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
        std::wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << std::endl;
        std::wcout << std::endl;
    }
}

int main(void) {
    DisplayHeader();

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
