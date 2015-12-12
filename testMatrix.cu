#include <iostream>
#include <opencv2/opencv.hpp>
#include "Matrix.h"

int main(void) {
    std::cout << "Hello!\n";

    Matrix mat = AllocateMatrix(10, 10, 1);
    printMatrix(mat);

    cv::Mat image = cv::imread( "outputImages/result.jpg", 1 );
    printf("size = (%i, %i)\n", image.rows, image.cols);

    return 0;
}
