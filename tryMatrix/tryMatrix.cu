#include <iostream>
#include "../Matrix.h"

void tryTranspose() {
    Matrix<float> A = AllocateMatrix<float>(5, 4, 1);
    cout << "Before transpose:\n";
    printMatrix(A);
    transpose(A);
    cout << "After transpose:\n";
    printMatrix(A);
    FreeMatrix<float>(&A);
}

int main(void) {
    tryTranspose();
    std::cout << "Hello!\n";

    Matrix<float> M = AllocateMatrix<float>(10, 10, 1);
    printMatrix(M);
    FreeMatrix<float>(&M);
    /*func<int>();*/

    return 0;
}
