#include <iostream>
#include <fstream>
#include <cassert>
using namespace std;

class CudaTimer {

    private:
        cudaEvent_t inclusiveStart, inclusiveStop;
        bool ticked;  // If tic has been run

    public:

    CudaTimer() {
        cudaEventCreate(&inclusiveStart);
        cudaEventCreate(&inclusiveStop);
        ticked = false;
    }
    ~CudaTimer() {
        cudaEventDestroy(inclusiveStart);
        cudaEventDestroy(inclusiveStop);
    }

    void tic() {
        cudaEventRecord(inclusiveStart, 0);
        ticked = true;
    }

    // Returns in milliseconds elapsed time since the last tic()
    float toc() {
        assert(ticked);
        float inclusive_timing; 
        cudaEventRecord(inclusiveStop, 0);
        cudaEventSynchronize(inclusiveStop);
        cudaEventElapsedTime(&inclusive_timing, inclusiveStart, inclusiveStop);
        return inclusive_timing;
    }
};
