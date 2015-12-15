#include <iostream>
#include <cassert>
#include "cpuFeatureMatcher.h"
#include <vector>
#include <set>

int testCorrMatFromDistanceMat(void) {

    int numDescriptors[] = {2, 3, 4, 5, 6};
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

    // Compute correspondence matrix from descriptors
    Matrix<int> corrMatFromDescriptors = AllocateMatrix<int>(numImages, n, 0);
    computeCorrespondenceMatFromDescriptors(descriptorsH, cumNumDescriptors,
            numImages, corrMatFromDescriptors, matchConf);

    /*printf("Host correspondence mat from distance matrix:\n");*/
    /*printMatrix(correspondenceMatH);*/
    /*printf("Correspondence mat from descriptors:\n");*/
    /*printMatrix(corrMatFromDescriptors);*/
    printf("Error between correspondence mat from distance matrix and descriptors = %f\n",
            getRMSE(correspondenceMatH, corrMatFromDescriptors));

    /*printf("Host descriptor mat:\n");*/
    /*printMatrix(descriptorsH);*/
    /*printf("Device descriptor mat:\n");*/
    /*printMatrixD(descriptorsD);*/
    //printf("Error between descriptor in D and H = %f\n",
            //getRMSEHostAndDevice(descriptorsH, descriptorsD));

    /*printf("Host distance mat:\n");*/
    /*printMatrix(distanceMatH);*/
    /*printf("Device distance mat:\n");*/
    /*printMatrixD(distanceMatD);*/
    //printf("Error between distance mat in D and H = %f\n",
            //getRMSEHostAndDevice(distanceMatH, distanceMatD));

    /*printCorrespondence(descriptorsH, numDescriptors, numImages, matchConf);*/

    /*cpuComputeDistanceMat(descriptors, distanceMat);*/
    /*computeCorrespondenceMat(distanceMat, cumNumDescriptors, numImages,*/
            /*corrMat, matchConf);*/
    /*std::cout << "corrMat:\n";*/
    /*printMatrix(corrMat);*/

    FreeMatrix(&descriptorsH);
    FreeMatrix(&distanceMatH);
    FreeMatrix(&correspondenceMatH);

    /*cv::Mat image = cv::imread( "outputImages/result.jpg", 1 );*/
    /*printf("size = (%i, %i)\n", image.rows, image.cols);*/


    return 0;
}

int main(void) {

    int numDescriptors[] = {1500, 1500, 1500, 1500, 1500};
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

    // Compute correspondence matrix from descriptors
    Matrix<int> corrMatFromDescriptors = AllocateMatrix<int>(numImages, n, 0);
    computeCorrespondenceMatFromDescriptors(descriptorsH, cumNumDescriptors,
            numImages, corrMatFromDescriptors, matchConf);

    cout << "Done computing from descriptors\n";

    /*printf("Host correspondence mat from distance matrix:\n");*/
    /*printMatrix(correspondenceMatH);*/
    /*printf("Correspondence mat from descriptors:\n");*/
    /*printMatrix(corrMatFromDescriptors);*/

    FreeMatrix(&descriptorsH);
    FreeMatrix(&corrMatFromDescriptors);

    /*cv::Mat image = cv::imread( "outputImages/result.jpg", 1 );*/
    /*printf("size = (%i, %i)\n", image.rows, image.cols);*/


    return 0;
}
