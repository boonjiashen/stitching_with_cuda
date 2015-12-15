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

int main(void) {

    int numDescriptors[] = {10, 20, 30, 40, 50};
    const int numImages = sizeof(numDescriptors) / sizeof(numDescriptors[0]);
    int k = 32;  // size of one descriptor
    float matchConf = 0.1;

    // Compute cumulative sum as input downstream
    int cumNumDescriptors[numImages];
    cumsum(numDescriptors, cumNumDescriptors, numImages);
    int n = cumNumDescriptors[numImages-1];  // sum of all feature counts

    cout << "Num descriptors:\n\t";
    for (int i=0;i<numImages;++i) { cout << numDescriptors[i] << " "; }
    cout << endl;

    printf("n=%i, k=%i, numImages=%i\n", n, k, numImages);

    Matrix<float> descriptorsH = AllocateMatrix<float>(n, k, 1);

    // Initialize descriptors in device
    Matrix<float> descriptorsD = AllocateDeviceMatrix<float>(descriptorsH);
    CopyToDeviceMatrix(descriptorsD, descriptorsH);

    // Compute distance matrix in device
    Matrix<float> distanceMatD = AllocateDeviceMatrix<float>(AllocateMatrix<float>(n, n, 2));
    gpuComputeDistanceMat(descriptorsD, distanceMatD);
    cout << "Computed distance mat with GPU\n";

    // Compute distance matrix in host
    Matrix<float> distanceMatH = AllocateMatrix<float>(n, n, 0);
    cout << "Allocated distance matrix in host\n";
    cout << "address of descriptorsH = " << descriptorsH.elements << endl;
    cout << "address of distanceMatH = " << distanceMatH.elements << endl;

    cpuComputeDistanceMat(descriptorsH, distanceMatH);
    cout << "Computed distance mat with CPU\n";

    // Compute correspondence matrix in host
    Matrix<int> correspondenceMatH = AllocateMatrix<int>(numImages, n, 0);
    computeCorrespondenceMat(distanceMatH, cumNumDescriptors, numImages,
            correspondenceMatH, matchConf);

    // Compute correspondence matrix in device
    Matrix<int> correspondenceMatD =
        AllocateDeviceMatrix<int>(correspondenceMatH);
    gpuComputeCorrespondenceMat(distanceMatD, cumNumDescriptors, numImages,
            correspondenceMatD, matchConf);

    /*printf("Host correspondence mat:\n");*/
    /*printMatrix(correspondenceMatH);*/
    /*printf("Device correspondence mat:\n");*/
    /*printMatrixD(correspondenceMatD);*/
    printf("Error between correspondence mat in D and H = %f\n",
            getRMSEHostAndDevice(correspondenceMatH, correspondenceMatD));

    /*printf("Host descriptor mat:\n");*/
    /*printMatrix(descriptorsH);*/
    /*printf("Device descriptor mat:\n");*/
    /*printMatrixD(descriptorsD);*/
    printf("Error between descriptor in D and H = %f\n",
            getRMSEHostAndDevice(descriptorsH, descriptorsD));

    /*printf("Host distance mat:\n");*/
    /*printMatrix(distanceMatH);*/
    /*printf("Device distance mat:\n");*/
    /*printMatrixD(distanceMatD);*/
    printf("Error between distance mat in D and H = %f\n",
            getRMSEHostAndDevice(distanceMatH, distanceMatD));

    /*printCorrespondence(descriptorsH, numDescriptors, numImages, matchConf);*/

    /*cpuComputeDistanceMat(descriptors, distanceMat);*/
    /*computeCorrespondenceMat(distanceMat, cumNumDescriptors, numImages,*/
            /*corrMat, matchConf);*/
    /*std::cout << "corrMat:\n";*/
    /*printMatrix(corrMat);*/

    FreeMatrix(&descriptorsH);
    FreeMatrix(&distanceMatH);
    FreeMatrix(&correspondenceMatH);
    FreeDeviceMatrix(&descriptorsD);
    FreeDeviceMatrix(&distanceMatD);
    FreeDeviceMatrix(&correspondenceMatD);

    /*cv::Mat image = cv::imread( "outputImages/result.jpg", 1 );*/
    /*printf("size = (%i, %i)\n", image.rows, image.cols);*/


    return 0;
}
