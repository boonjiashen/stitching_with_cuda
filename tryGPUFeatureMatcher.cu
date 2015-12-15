#include <iostream>
#include <cassert>
#include "gpuFeatureMatcher.cuh"

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
