#include <iostream>
#include <opencv2/opencv.hpp>
#include "../Matrix.h"

int main(void) {
    std::cout << "Hello!\n";

    AllocateMatrix<float>(10, 10, 1);

    return 0;
}
