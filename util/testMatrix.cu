#include <iostream>
#include "Matrix.h"
using namespace std;

int main(void) {
    cout << "Hello!\n";

    Matrix mat = AllocateMatrix(10, 10, 1);
    printMatrix(mat);

    return 0;
}
