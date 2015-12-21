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
        float& inclusiveTimingMS,
        float& exclusiveTimingMS) {
    assert(numImages > 0);
    assert(descriptorDim > 0);
    assert(numDescriptorsPerImage > 1);

    const int n_i = numDescriptorsPerImage;  
    const int k = descriptorDim;
    const int n = numDescriptorsPerImage * numImages;  // Total number of descriptors

    // Compute cumulative sum as input downstream
    int cumNumDescriptors[numImages];
    for (int i=0;i<numImages;++i) { cumNumDescriptors[i]=(i+1)*n_i; }

    /*printf("n=%i, k=%i, numImages=%i\n", n, k, numImages);*/

    CudaTimer exclusiveTimer, inclusiveTimer;

    // Allocate memory in host and device
    Matrix<float> descriptorsH = AllocateMatrix<float>(n, k, 1);
    Matrix<float> descriptorsD = AllocateDeviceMatrix<float>(descriptorsH);
    Matrix<int> correspondenceMatH = AllocateMatrix<int>(numImages, n, 0);
    Matrix<int> correspondenceMatD =
        AllocateDeviceMatrix<int>(correspondenceMatH);

    inclusiveTimer.tic();

    // Copy descriptor elements to device
    CopyToDeviceMatrix(descriptorsD, descriptorsH);

    // Compute correspondence matrix in device
    exclusiveTimer.tic();
    gpuComputeCorrespondenceMatFromDescriptors(
            descriptorsD, cumNumDescriptors,
            numImages, correspondenceMatD, matchConfidence);
    exclusiveTimingMS = exclusiveTimer.toc();

    // Copy correspondence elements back to host
    CopyFromDeviceMatrix(correspondenceMatH, correspondenceMatD);

    inclusiveTimingMS = inclusiveTimer.toc();

    FreeMatrix(&descriptorsH);
    FreeMatrix(&correspondenceMatH);
    FreeDeviceMatrix(&descriptorsD);
    FreeDeviceMatrix(&correspondenceMatD);

    /*cv::Mat image = cv::imread( "outputImages/result.jpg", 1 );*/
    /*printf("size = (%i, %i)\n", image.rows, image.cols);*/

}

void timeOneRunOfCorrespondenceFromColumnwiseDescriptors(
        const int numImages, const int numDescriptorsPerImage,
        const int descriptorDim, const float matchConfidence,
        float& inclusiveTimingMS,
        float& exclusiveTimingMS) {
    assert(numImages > 0);
    assert(descriptorDim > 0);
    assert(numDescriptorsPerImage > 1);

    const int n_i = numDescriptorsPerImage;  
    const int k = descriptorDim;
    const int n = numDescriptorsPerImage * numImages;  // Total number of descriptors

    // Compute cumulative sum as input downstream
    int cumNumDescriptors[numImages];
    for (int i=0;i<numImages;++i) { cumNumDescriptors[i]=(i+1)*n_i; }

    /*printf("n=%i, k=%i, numImages=%i\n", n, k, numImages);*/

    CudaTimer exclusiveTimer, inclusiveTimer;

    // Allocate memory in host and device
    Matrix<float> descriptorsH = AllocateMatrix<float>(k, n, 1);
    Matrix<float> descriptorsD = AllocateDeviceMatrix<float>(descriptorsH);
    Matrix<int> correspondenceMatH = AllocateMatrix<int>(numImages, n, 0);
    Matrix<int> correspondenceMatD =
        AllocateDeviceMatrix<int>(correspondenceMatH);

    inclusiveTimer.tic();

    // Copy descriptor elements to device
    CopyToDeviceMatrix(descriptorsD, descriptorsH);

    // Compute correspondence matrix in device
    exclusiveTimer.tic();
    gpuComputeCorrespondenceMatFromColumnwiseDescriptors(
            descriptorsD, cumNumDescriptors,
            numImages, correspondenceMatD, matchConfidence);
    exclusiveTimingMS = exclusiveTimer.toc();

    // Copy correspondence elements back to host
    CopyFromDeviceMatrix(correspondenceMatH, correspondenceMatD);

    inclusiveTimingMS = inclusiveTimer.toc();
    
    /*// Double-check corrrespondence values match the CPU version*/
    /*// Overwrites device results that were transferred to host*/
    /*transpose(descriptorsH);*/
    /*computeCorrespondenceMatFromDescriptors(descriptorsH, cumNumDescriptors,*/
            /*numImages, correspondenceMatH, matchConfidence);*/
    /*printf("RMSE between CPU space-optim and GPU coalesced = %f\n",*/
            /*getRMSEHostAndDevice(correspondenceMatH, correspondenceMatD));*/

    FreeMatrix(&descriptorsH);
    FreeMatrix(&correspondenceMatH);
    FreeDeviceMatrix(&descriptorsD);
    FreeDeviceMatrix(&correspondenceMatD);

    /*cv::Mat image = cv::imread( "outputImages/result.jpg", 1 );*/
    /*printf("size = (%i, %i)\n", image.rows, image.cols);*/

}

int main(void) {
    float inclusiveTimingMS, exclusiveTimingMS;
    int k = 32;
    int n_i = 100;
    float matchConfidence = 0.1;
    int numIters = 1;  // number of iterations per parameter set
    int runID = 0;
    for (int exponent = 1; exponent <= 11; exponent+=1) {
        int base = 2;
        int numImages = pow(base, exponent);

        for (int iterNum = 0; iterNum < numIters; ++iterNum) {
            timeOneRunOfCorrespondenceFromColumnwiseDescriptors(
                    numImages, n_i, k,
                    matchConfidence, inclusiveTimingMS, exclusiveTimingMS);
            printf("<run isGPU='1' ID='%i' numImages='%i' numDescriptorsPerImage='%i' "
                    "descriptorDim='%i' matchConfidence='%f' "
                    "inclusiveTimingMS='%f' exclusiveTimingMS='%f' />\n",
                    runID, numImages, n_i, k, matchConfidence, inclusiveTimingMS,
                    exclusiveTimingMS
                  );
            ++runID;
        }
    }
    return 0;
}
