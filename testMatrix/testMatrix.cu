#include <iostream>
#include "../Matrix.h"

int main(void) {
    std::cout << "Hello!\n";

    Matrix<float> M = AllocateMatrix<float>(10, 10, 1);
    printMatrix(M);
    FreeMatrix<float>(&M);
    /*func<int>();*/

    return 0;
}
