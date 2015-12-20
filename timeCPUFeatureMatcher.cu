#include <iostream>
#include <cassert>
#include "gpuFeatureMatcher.cuh"
#include "CudaTimer.cuh"

/*
 *Gets exclusive and inclusive timing of one run of computing the correspondence
 *matrix from randomly generated descriptors. Computation is done on GPU.
 *Pre-conditions:
 *    descriptorDim, numImages > 0
 *    numDescriptorsPerImage > 1, so that we can do best of two neighbors
 *    matching without corner cases.
 *Post-conditions:
 *    Assigns timings in milliseconds to `inclusiveTimingMS` and `exclusiveTimingMS`
 */
void timeOneRunOfCorrespondenceFromDescriptors(
        const int numImages, const int numDescriptorsPerImage,
        const int descriptorDim, const float matchConfidence,
        float& timingMS) {
    assert(numImages > 0);
    assert(descriptorDim > 0);
    assert(numDescriptorsPerImage > 1);

    const int n_i = numDescriptorsPerImage;  
    const int k = descriptorDim;
    const int n = numDescriptorsPerImage * numImages;  // Total number of descriptors

    const int epsilon = 0.001;  // amount of tolerable error

    // Compute cumulative sum as input downstream
    int cumNumDescriptors[numImages];
    for (int i=0;i<numImages;++i) { cumNumDescriptors[i]=(i+1)*n_i; }

    /*printf("n=%i, k=%i, numImages=%i\n", n, k, numImages);*/

    CudaTimer exclusiveTimer, inclusiveTimer;

    // Allocate memory in host and device
    Matrix<float> descriptorsH = AllocateMatrix<float>(n, k, 1);
    Matrix<int> correspondenceMatH = AllocateMatrix<int>(numImages, n, 0);

    // Compute correspondence matrix in device
    exclusiveTimer.tic();
    computeCorrespondenceMatFromDescriptors(descriptorsH, cumNumDescriptors,
            numImages, correspondenceMatH, matchConfidence);
    timingMS = exclusiveTimer.toc();

    FreeMatrix(&descriptorsH);
    FreeMatrix(&correspondenceMatH);

}

int main(void) {
    float timingMS;
    int k = 32;
    int n_i = 100;
    float matchConfidence = 0.1;
    int numIters = 1;  // number of iterations per parameter set
    int runID = 0;
    for (int exponent = 1; exponent <= 7; ++exponent) {
        int base = 2;
        int numImages = pow(base, exponent);

        for (int iterNum = 0; iterNum < numIters; ++iterNum) {
            timeOneRunOfCorrespondenceFromDescriptors(numImages, n_i, k,
                    matchConfidence, timingMS);
            printf("<run isGPU='0' ID='%i' numImages='%i' numDescriptorsPerImage='%i' "
                    "descriptorDim='%i' matchConfidence='%f' "
                    "timingMS='%f' />\n",
                    runID, numImages, n_i, k, matchConfidence,
                    timingMS
                  );
            ++runID;
        }
    }
    return 0;
}
