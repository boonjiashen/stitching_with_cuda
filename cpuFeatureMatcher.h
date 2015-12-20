#include <iostream>
#include <cassert>
#include "Matrix.h"
#include <vector>
#include <set>
#include <cfloat>

typedef std::pair<int, int> match;

/*
 *Pre-conditions:
 *    `descriptor1` and `descriptor2` are each arrays of length `length`
 *Post-conditions:
 *    Returns the L2 distance of the two descriptors interpreted as vectors.
 */
float computeL2Distance(float* descriptor1, float* descriptor2, int length) {
    float distance = 0;
    for (int i = 0; i < length; ++i) {
        distance += pow(descriptor1[i] - descriptor2[i], 2);
    }
    return pow(distance, 0.5);
}

/*
 *Pre-conditions:
 *    `descriptors` is a n x k matrix
 *    `distanceMat` is a n x n matrix
 *Post-conditions:
 *    distanceMat[j][i] is the Euclidean distance from descriptors[j][:]
 *    to descriptors[i][:]. Note that Euclidean distance is symmetric.
 *Known bugs:
 *    binary produces no stdout with n = 10000 and k = 32.
 */
void cpuComputeDistanceMat(const Matrix<float> descriptors, Matrix<float> distanceMat) {
    assert(descriptors.height == distanceMat.height);
    assert(distanceMat.height == distanceMat.width);

    int n = descriptors.height;
    int k = descriptors.width;

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            float* descriptor1 = descriptors.elements + j * k;
            float* descriptor2 = descriptors.elements + i * k;
            float* dst = distanceMat.elements + j * n + i;
            *dst = computeL2Distance(descriptor1, descriptor2, k);
        }
    }
}

/*
 *Pre-conditions:
 *    `A` and `B` are of the same size and are on host memory.
 *Post-conditions:
 *    Returns the root mean squared error between A and B
 *    if number of elements is more than 0, else returns 0
 */
template <typename T>
float getRMSE(const Matrix<T> A, const Matrix<T> B) {
    assert(A.height == B.height);
    assert(A.width == B.width);

    int numel = A.height * A.width;  // number of elements
    if (numel == 0) { return 0; }

    float sse = 0;  // sum of squared error
    for (int i = 0; i < numel; ++i) {
        sse = pow(A.elements[i] - B.elements[i], 2);
    }

    return sqrt(sse / numel);
}

/*
 *Pre-conditions:
 *    start < stop
 *    0 <= start < mat.height
 *    0 < start <= mat.height
 *Post-conditions:
 *    Returns submat, which is a view of mat[start:stop][:]
 *    submat contains rows start, start+1, ..., stop-1 of mat
 */
template<typename T>
Matrix<T> getSubmatrix(const Matrix<T> mat, const int start, const int stop) {
    assert(start < stop);
    assert(0 <= start && start < mat.height);
    assert(0 < stop && stop <= mat.height);

    Matrix<T> submat;
    submat.width = mat.width;
    submat.height = stop - start;
    submat.elements = mat.elements + start * submat.width;

    return submat;
}

/*
 *Pre-conditions:
 *    mat.height >= 2
 *    0 <= col < mat.width
 *Post-conditions:
 *    mat[idx1][col] and mat[idx2][col] are the smallest and second smallest
 *    elements in mat[:][col]
 */
template<typename T>
void getIndexOfTwoSmallestInColumn(const Matrix<T> mat, int col, int& idx1, int& idx2) {
    assert(mat.height >= 2);
    assert(0 <= col && col < mat.width);

    float val1 = FLT_MAX, val2 = FLT_MAX;
    idx1 = -1, idx2 = -1;

    for (int j = 0; j < mat.height; ++j) {
        float currVal = mat.elements[j * mat.width + col];

        // Push current value to first place
        if (currVal < val1) {
            val2 = val1;
            idx2 = idx1;
            val1 = currVal;
            idx1 = j;
        }

        // Push current value to second place
        else if (currVal < val2) {
            val2 = currVal;
            idx2 = j;
        }
    }
}

/*
 *Computes the correspondence of one feature given its distance from other
 *features
 *Pre-conditions:
 *    `distances` is an array of length `len`
 *    distances[i] is the distance of a fixed feature to feature i
 *Post-conditions:
 *    Returns index i, such that feature i corresponds to the fixed feature, if
 *    there is correspondence, else returns -1
 */
int computeCorrespondenceScalar(float* distances, int len, float matchConfidence) {

    Matrix<float> M;
    M.height = len;
    M.width = 1;
    M.elements = distances;

    int idx1, idx2;
    getIndexOfTwoSmallestInColumn(M, 0, idx1, idx2);
    float dist1 = distances[idx1];
    float dist2 = distances[idx2];

    if (dist1 / dist2 < 1 - matchConfidence) {
        return idx1;
    }
    else {
        return -1;
    }
}

/*
 *Computes the correspondence of descriptor[idxFrom] against descriptors 
 *descriptor[idxToStart:idxToStop] (excludes idxToStop)
 *Pre-conditions:
 *    descriptors is a n x k matrx
 *    0 <= idxFrom, idxToStart < n
 *    0 < idxToStop <= n
 *    idxToStart < idxToStop
 *Post-conditions:
 *    Returns i s.t. descriptor[idxFrom] corresponds to descriptor idxToStart+i,
 *    if there is correspondence, else returns -1
 *    0 <= i < idxToStop-idxToStart
 */
int computeCorrespondenceScalar(const Matrix<float> descriptors,
        int idxFrom, int idxToStart, int idxToStop, float matchConfidence) {
    assert(0 <= idxFrom && idxFrom < descriptors.height);
    assert(0 <= idxToStart && idxToStart < descriptors.height);
    assert(0 < idxToStop && idxToStop <= descriptors.height);

    int n_i = idxToStop - idxToStart;
    int k = descriptors.width;
    float distances[n_i];
    float* descriptorFrom = descriptors.elements + k * idxFrom;
    for (int j = 0; j < n_i; ++j) {
        float* descriptorTo = descriptors.elements + k * (j + idxToStart);
        distances[j] = computeL2Distance(descriptorFrom, descriptorTo, k);
    }
    return computeCorrespondenceScalar(distances, n_i, matchConfidence);
}


/*
 *Compute correspondence matrix given descriptors. This has better space
 *complexity than first computing the full distance matrix
 *Pre-conditions:
 *    `descriptors` is a n x k matrix
 *    cumNumDescriptors is an array of length `numImages`
 *    cumNumDescriptors[numImages-1] = n
 *    correspondenceMat is a numImages x n matrix
 *Post-conditions:
 *    correspondenceMat[j][i] is the i-th feature's corresponding feature in
 *    image j. It is -1 if there's no correspondence.
 *    -1 <= correspondenceMat[j][i] < n_j, where n_j is the number of features
 *    in image j.
 */
void computeCorrespondenceMatFromDescriptors(
        const Matrix<float> descriptors,
        int* cumNumDescriptors,
        int numImages, Matrix<int> correspondenceMat, float matchConfidence) {
    assert(correspondenceMat.height == numImages);
    assert(correspondenceMat.width == descriptors.height);
    assert(cumNumDescriptors[numImages-1] == descriptors.height);

    int n = descriptors.height;

    for (int imgIdx = 0; imgIdx < numImages; ++imgIdx) {
        int idxToStart = imgIdx > 0 ? cumNumDescriptors[imgIdx-1] : 0;
        int idxToStop = cumNumDescriptors[imgIdx];

        for (int idxFrom = 0; idxFrom < n; ++idxFrom) {
            int* dst = correspondenceMat.elements + imgIdx * n + idxFrom;
            *dst = computeCorrespondenceScalar(descriptors, idxFrom,
                    idxToStart, idxToStop, matchConfidence);
        }
    }
}
 

/*
 *Pre-conditions:
 *    `distanceSubmat` is a n_i x n matrix
 *    `correspondenceMat` is a 1 x n matrix
 *Post-conditions:
 *    See `computeCorrespondenceMat`
 */
void computeCorrespondenceVec(const Matrix<float> distanceSubmat,
        Matrix<int> correspondenceMat, float matchConfidence) {
    assert(correspondenceMat.height == 1);
    assert(correspondenceMat.width == distanceSubmat.width);

    int n = distanceSubmat.width;
    for (int i = 0; i < n; ++i) {
        int idx1, idx2;
        getIndexOfTwoSmallestInColumn(distanceSubmat, i, idx1, idx2);
        float dist1 = distanceSubmat.elements[idx1 * distanceSubmat.width + i];
        float dist2 = distanceSubmat.elements[idx2 * distanceSubmat.width + i];

        if (dist1 / dist2 < 1 - matchConfidence) {
            correspondenceMat.elements[i] = idx1;
        }
        else {
            correspondenceMat.elements[i] = -1;
        }
    }
}

/*
 *Pre-conditions:
 *    `distanceMat` is a n x n matrix
 *    `distanceMat` is the result of computeDistanceMat()
 *    cumNumDescriptors is an array of length `numImages`
 *    cumNumDescriptors[numImages-1] = distanceMat.height
 *    correspondenceMat is a numImages * n matrix
 *
 *    Let start = cumNumDescriptors[i-1], stop = cumNumDescriptors[i].
 *    Then distanceMat[start:stop][:] refers to descriptors of image i.
 *Post-conditions:
 *    correspondenceMat[j][i] is the i-th feature's corresponding feature in
 *    image j. It is -1 if there's no correspondence.
 *    -1 <= correspondenceMat[j][i] < n_j, where n_j is the number of features
 *    in image j.
 */
void computeCorrespondenceMat(const Matrix<float> distanceMat,
        int* cumNumDescriptors,
        int numImages, Matrix<int> correspondenceMat, float matchConfidence) {
    assert(distanceMat.height == distanceMat.width);
    assert(correspondenceMat.height == numImages);
    assert(correspondenceMat.width == distanceMat.width);
    assert(cumNumDescriptors[numImages-1] == distanceMat.height);

    for (int imgIdx = 0; imgIdx < numImages; ++imgIdx) {
        int start = imgIdx > 0 ? cumNumDescriptors[imgIdx-1] : 0;
        int stop = cumNumDescriptors[imgIdx];
        Matrix<float> src = getSubmatrix<float>(distanceMat, start, stop);
        Matrix<int> dst = getSubmatrix<int>(correspondenceMat, imgIdx, imgIdx + 1);

        computeCorrespondenceVec(src, dst, matchConfidence);
    }
}

/*
 *Pre-conditions:
 *    src and dst are arrays of length `length`.
 *    length > 0
 *Post-conditions:
 *    dst is the cumulative sum of src.
 */
template<typename T>
void cumsum(T* src, T* dst, int length) {
    assert(length > 0);
    dst[0] = src[0];
    for (int i = 1; i < length; ++i) {
        dst[i] = dst[i-1] + src[i];
    }
}

/*
 *Pre-conditions:
 *    `allDescriptors` is a n x k matrix
 *    numDescriptors = [n1, n2, ..., n_m] where n_i is the number of descriptors
 *    of image i and m is numImages
 *    n = sum(n1, n2, ..., n_m)
 *Post-conditions:
 *    Prints the feature correspondence between each image pair, based on
 *    OpenCV's Best2NearestMatcher
 */
void printCorrespondence(const Matrix<float>& allDescriptors,
        int* numDescriptors, int numImages, float matchConf) {
    std::vector<std::set<match> > correspondenceSets;

    // cumsum of n_i = [n_0, n_0 + n_1, ..., n]
    int* cumNumDesc = (int*)malloc(sizeof(int) * (numImages));
    cumsum<int>(numDescriptors, cumNumDesc, numImages);

    // Compute distance matrix
    int n = allDescriptors.height;
    Matrix<float> distanceMat = AllocateMatrix<float>(n, n, 0);
    cpuComputeDistanceMat(allDescriptors, distanceMat);

    // Compute correspondence matrix
    Matrix<int> corrMat = AllocateMatrix<int>(numImages, n, 0);
    computeCorrespondenceMat(distanceMat, cumNumDesc, numImages,
            corrMat, matchConf);

    std::cout << "distanceMat\n";
    printMatrix(distanceMat);
    std::cout << "corrMat\n";
    printMatrix(corrMat);

    for (int imgIdx1 = 0; imgIdx1 < numImages - 1; ++imgIdx1) {
        for (int imgIdx2 = imgIdx1 + 1; imgIdx2 < numImages; ++imgIdx2) {
            int numDescriptors1 = numDescriptors[imgIdx1];
            int numDescriptors2 = numDescriptors[imgIdx2];

            int* src = NULL;

            // Correspondence from imgIdx2 to imgIdx1
            src = corrMat.elements + imgIdx1 * n + cumNumDesc[imgIdx2-1];
            for (int i = 0; i < numDescriptors2; ++i) {
                if (src[i] > -0.5) {
                    printf("im%i-%i corr %i-%i\n", imgIdx1, imgIdx2, src[i], i);
                }
            }

            // Correspondence from imgIdx1 to imgIdx2
            src = corrMat.elements + imgIdx2 * n +
                    (imgIdx1 > 0 ? cumNumDesc[imgIdx1-1] : 0);
            for (int i = 0; i < numDescriptors1; ++i) {
                if (src[i] > -0.5) {
                    printf("im%i-%i corr %i-%i\n", imgIdx1, imgIdx2, i, src[i]);
                }
            }

        }
    }

    free(cumNumDesc);
}
