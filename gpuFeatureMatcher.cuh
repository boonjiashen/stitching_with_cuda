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
 *    element member in `distanceMat` and `correspondenceMat` point to device memory
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

// See CPU-equivalent.
__device__ float gpuComputeL2Distance(float* descriptor1, float* descriptor2, int length) {
    float ssd = 0;  // sum of squared distances
    for (int i = 0; i < length; ++i) {
        float dist = descriptor1[i] - descriptor2[i];
        ssd += dist * dist;
    }
    return sqrtf(ssd);
}


/*
 *Kernel to compute correspondence directly from descriptors.
 *A thread with thread ID tid computes the correspondence of descriptor[tid][:]
 *to the range of descriptor[idxToStart:idxToStop][:] (excludes idxToStop).
 *It assigns this correspondence to correspondenceVec[tid]
 *Pre-conditions:
 *    element member of `descriptors` and `correspondenceVec` point to device memory
 *    `descriptors` is a n x k matrix
 *    `correspondenceVec` is a 1 x n matrix
 *    idxToStart < idxToStop
 *    0 <= idxToStart, idxToStop <= n
 *Post-conditions:
 *    Assigns values to correspondenceVec
 */
__global__ void kernelComputeCorrespondence(
        const Matrix<float> descriptors,
        Matrix<int> correspondenceVec,
        int idxToStart, int idxToStop, float matchConfidence) {

    int n = descriptors.height;
    int k = descriptors.width;
    assert(correspondenceVec.height == 1);
    assert(correspondenceVec.width == n);
    assert(idxToStart < idxToStop);
    assert(0 <= idxToStart && idxToStart < n);
    assert(0 < idxToStop && idxToStop <= n);

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) { return; }

    float dist1 = FLT_MAX, dist2 = FLT_MAX; // smallest & 2nd smallest distance
    int idx1 = -1;  // index of match with smallest distance
    float* descriptorFrom = descriptors.elements + tid * k;
    for (int j = idxToStart; j < idxToStop; ++j) {
        float* descriptorTo = descriptors.elements + j * k;
        float currDist = gpuComputeL2Distance(descriptorFrom, descriptorTo, k);

        if (currDist < dist1) {
            dist2 = dist1;
            dist1 = currDist;

            // Decrement by idxToStart because we want correspondence to start from 0
            // i.e. treat descriptors[idxToStart:idxToStop] as the descriptors
            // 0 to idxToStop-idxToStart-1 of the 'from' image
            idx1 = j - idxToStart;
        }
        else if (currDist < dist2) {
            dist2 = currDist;
        }
    }

    int correspondence = (dist1 / dist2 < 1 - matchConfidence) ? idx1 : -1;
    correspondenceVec.elements[tid] = correspondence;
}

/*
 *Space-optimal feature matcher. Computes correspondence matrix directly from
 *descriptors.
 *A kernel is launched for every row in correspondenceMat. Every thread
 *computes one element in correspondenceMat.
 *See CPU-equivalent.
 *Pre-conditions:
 *    element member of `descriptors` and `correspondenceMat` point to device memory
 *    `cumNumDescriptors` points to host memory
 */
void gpuComputeCorrespondenceMatFromDescriptors(
        const Matrix<float> descriptors,
        int* cumNumDescriptors,
        int numImages, Matrix<int> correspondenceMat, float matchConfidence) {
    assert(correspondenceMat.height == numImages);
    assert(correspondenceMat.width == descriptors.height);
    assert(cumNumDescriptors[numImages-1] == descriptors.height);

    int n = descriptors.height;

    dim3 dimGrid((n + BLOCK_SIZE - 1)/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE);
    for (int imgIdx = 0; imgIdx < numImages; ++imgIdx) {
        int idxToStart = imgIdx > 0 ? cumNumDescriptors[imgIdx-1] : 0;
        int idxToStop = cumNumDescriptors[imgIdx];

        Matrix<int> correspondenceVec = getSubmatrix(correspondenceMat,
                imgIdx, imgIdx+1);
        kernelComputeCorrespondence<<<dimGrid,dimBlock>>>(
                descriptors, correspondenceVec,
                idxToStart, idxToStop, matchConfidence);
    }

}
