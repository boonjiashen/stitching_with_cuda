#include <iostream>
#include <cassert>
#include "cpuFeatureMatcher.h"
#include <vector>
#include <set>

#define BLOCK_SIZE 1024

/*
 *Each thread is in charge of computing one element in distanceMat
 */
__global__ void kernelDistanceMat(const Matrix<float> descriptors, Matrix<float> distanceMat) {
    int matX = blockDim.x * blockIdx.x + threadIdx.x;
    int matY = blockIdx.y;
    int n = descriptors.height;
    int k = descriptors.width;

    if (matX >= n) { return; }
    float ssd = 0;  // sum of squared distances
    for (int i = 0; i < k; ++i) {
        float dist = descriptors.elements[matY * k + i] -
            descriptors.elements[matX * k + i];
        ssd += dist * dist;
    }
    distanceMat.elements[matY * n + matX] = sqrtf(ssd);
}

/*
 *See CPU-equivalent.
 *Pre-conditions:
 *    `descriptors` and `distanceMat` are in GPU memory
 */
void gpuComputeDistanceMat(const Matrix<float> descriptors, Matrix<float> distanceMat) {
    assert(descriptors.height == distanceMat.height);
    assert(distanceMat.height == distanceMat.width);

    int n = descriptors.height;

    dim3 dimGrid((n + BLOCK_SIZE - 1)/BLOCK_SIZE, n);
    dim3 dimBlock(BLOCK_SIZE);
    kernelDistanceMat<<<dimGrid,dimBlock>>>(descriptors, distanceMat);

}

/*
 *See getRMSE()
 *Pre-conditions:
 *    A is in host memory and BDevice is in device memory.
 */
template <typename T>
float getRMSEHostAndDevice(const Matrix<T> A, const Matrix<T> BDevice) {
    assert(A.height == BDevice.height);
    assert(A.width == BDevice.width);

    Matrix<T> BHost = AllocateMatrix<T>(BDevice.height, BDevice.width, 0);
    CopyFromDeviceMatrix<T>(BHost, BDevice);
    float rmse = getRMSE(A, BHost);
    FreeMatrix(&BHost);

    return rmse;
}

/*
 *See gpuComputeCorrespondenceVec()
 *Pre-conditions:
 *    Every thread is responsible for computing one element in
 *    `correspondenceMat`
 */
__global__ void gpuComputeCorrespondenceVec(
        const Matrix<float> distanceSubmat,
        Matrix<int> correspondenceMat, float matchConfidence) {
    int n = distanceSubmat.width;  // no. of descriptors matched from
    int n_i = distanceSubmat.height;  // no. of descriptors matched to
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) { return; }

    float dist1 = FLT_MAX, dist2 = FLT_MAX; // smallest & 2nd smallest distance
    int idx1 = -1;  // index of match with smallest distance

    for (int j = 0; j < n_i; ++j) {
        float currDist = distanceSubmat.elements[j * n + tid];

        if (currDist < dist1) {
            dist2 = dist1;
            dist1 = currDist;
            idx1 = j;
        }
        else if (currDist < dist2) {
            dist2 = currDist;
        }
    }

    if (dist1 / dist2 < 1 - matchConfidence) {
        correspondenceMat.elements[tid] = idx1;
    }
    else {
        correspondenceMat.elements[tid] = -1;
    }
}

/*
 *See computeCorrespondenceMat()
 *Pre-conditions:
 *    `distanceMat` and `correspondenceMat` are in device memory
 *    `cumNumDescriptors` is in host memory
 */
void gpuComputeCorrespondenceMat(const Matrix<float> distanceMat,
        int* cumNumDescriptors,
        int numImages, Matrix<int> correspondenceMat, float matchConfidence) {
    assert(distanceMat.height == distanceMat.width);
    assert(correspondenceMat.height == numImages);
    assert(correspondenceMat.width == distanceMat.width);
    assert(cumNumDescriptors[numImages-1] == distanceMat.height);

    int n = distanceMat.height;

    dim3 dimGrid((n + BLOCK_SIZE - 1)/BLOCK_SIZE, n);
    dim3 dimBlock(BLOCK_SIZE);
    for (int imgIdx = 0; imgIdx < numImages; ++imgIdx) {
        int start = imgIdx > 0 ? cumNumDescriptors[imgIdx-1] : 0;
        int stop = cumNumDescriptors[imgIdx];
        Matrix<float> src = getSubmatrix<float>(distanceMat, start, stop);
        Matrix<int> dst = getSubmatrix<int>(correspondenceMat, imgIdx, imgIdx + 1);

        gpuComputeCorrespondenceVec<<<dimGrid,dimBlock>>>(src, dst, matchConfidence);
    }
}
