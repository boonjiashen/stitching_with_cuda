/*
* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:   
*
* This source code is subject to NVIDIA ownership rights under U.S. and 
* international Copyright laws.  
*
* This software and the information contained herein is PROPRIETARY and 
* CONFIDENTIAL to NVIDIA and is being provided under the terms and 
* conditions of a Non-Disclosure Agreement.  Any reproduction or 
* disclosure to any third party without the express written consent of 
* NVIDIA is prohibited.     
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
* OR PERFORMANCE OF THIS SOURCE CODE.  
*
* U.S. Government End Users.  This source code is a "commercial item" as 
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
* "commercial computer software" and "commercial computer software 
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
* and is provided to the U.S. Government only as a commercial end item.  
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
* source code with only those rights set forth herein.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
// includes, project
#include <iostream>

using namespace std;

// Matrix Structure declaration
template <typename T>
struct Matrix {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    T* elements;
};

template<typename T> Matrix<T> AllocateDeviceMatrix(const Matrix<T> M);

template<typename T> Matrix<T> AllocateMatrix(int height, int width, int init);
template<typename T> void CopyToDeviceMatrix(Matrix<T> Mdevice, const Matrix<T> Mhost);
template<typename T> void CopyFromDeviceMatrix(Matrix<T> Mhost, const Matrix<T> Mdevice);
template<typename T> void FreeDeviceMatrix(Matrix<T>* M);
template<typename T> void FreeMatrix(Matrix<T>* M);

template<typename T> void printMatrix(const Matrix<T> M, char* fmt);
void printMatrix(const Matrix<float>);
void printMatrix(const Matrix<int>);


template<typename T>
void printMatrix(const Matrix<T> M, char* fmt) {
    int numel = M.height * M.width;
    for (int i = 0; i < numel; ++i) {
        printf(fmt, M.elements[i]);
        if ((i+1)%M.width == 0) {
            printf("\n");
        }
    }
}


void printMatrix(const Matrix<float> M) {
    char fmt[] = "%5.2f";
    printMatrix<float>(M, fmt);
}

void printMatrixD(const Matrix<float> matrixD) {
    Matrix<float> matrixH = AllocateMatrix<float>
        (matrixD.height, matrixD.width, 0);
    CopyFromDeviceMatrix(matrixH, matrixD);
    printMatrix(matrixH);
    FreeMatrix(&matrixH);
}

void printMatrix(const Matrix<int> M) {
    char fmt[] = "%2i";
    printMatrix<int>(M, fmt);
}


// Allocate a device matrix of same size as M.
template<typename T>
Matrix<T> AllocateDeviceMatrix(const Matrix<T> M)
{
    Matrix<T> Mdevice = M;
    int size = M.width * M.height * sizeof(T);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
template<typename T>
Matrix<T> AllocateMatrix(int height, int width, int init)
{
    Matrix<T> M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (T*) malloc(size*sizeof(T));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
        //TODO Is this correct? typecasting RANDMAX
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (T)RAND_MAX);
		if(rand() % 2)
			M.elements[i] = - M.elements[i];
	}
    return M;
}	


// Copy a host matrix to a device matrix.
template<typename T>
void CopyToDeviceMatrix(Matrix<T> Mdevice, const Matrix<T> Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(T);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
template<typename T>
void CopyFromDeviceMatrix(Matrix<T> Mhost, const Matrix<T> Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(T);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
template<typename T>
void FreeDeviceMatrix(Matrix<T>* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
template<typename T>
void FreeMatrix(Matrix<T>* M)
{
    free(M->elements);
    M->elements = NULL;
}
