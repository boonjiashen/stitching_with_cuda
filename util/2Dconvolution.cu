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

/* Matrix multiplication: C = A * B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
// includes, project
#include "2Dconvolution.h"
#include "CudaTimer.cuh"
#include <iostream>

using namespace std;
////////////////////////////////////////////////////////////////////////////////
// declarations, forward

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
bool CompareResults(float* A, float* B, int elements, float eps);
bool ReadParams(int* params, int size, char* file_name);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

Matrix PadMatrix(const Matrix& M, const int deltaHeight, const int deltaWidth);
void ExtractFromPadded(Matrix M, const Matrix& Mpadded);

void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);

////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
////////////////////////////////////////////////////////////////////////////////
__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P)
{

    extern volatile __shared__ float shmem[];

    // (x, y) coordinate on the grid
    int g_y = threadIdx.y + blockDim.y * blockIdx.y;
    int g_x = threadIdx.x + blockDim.x * blockIdx.x;
    int gridWidth = gridDim.x * blockDim.x;
    int gridHeight = gridDim.y * blockDim.y;
    /*printf("(gy,gx)=(%i,%i) ", g_y, g_x);*/

    int dst_x, dst_y, src_x, src_y;
    int dx, dy;  // direction that data is moving on the grid [dx, dy]

#define MOVE_MEMORY \
    do { \
        dst_x = threadIdx.x + (dx + 1) * RADIUS; \
        dst_y = threadIdx.y + (dy + 1) * RADIUS; \
        src_x = (g_x + dx * RADIUS + gridWidth) % gridWidth; \
        src_y = (g_y + dy * RADIUS + gridHeight) % gridHeight; \
        shmem[dst_y * (BLOCK_SIZE + 4) + dst_x] = N.elements[src_y * gridWidth + src_x]; \
    } while (0)

    // Populate bottom-right of shared memory
    dx = 1; dy = 1;
    MOVE_MEMORY;

    // Populate top-left of shared memory
    if (threadIdx.x < 2 * RADIUS || threadIdx.y < 2 * RADIUS) {
        dx = -1; dy = -1;
        MOVE_MEMORY;
    }

    // Populate top-right and bottom-left
    int populatesBL = threadIdx.x < 2 * RADIUS &&  \
                      blockDim.y - threadIdx.y <= 2 * RADIUS;
    int populatesTR = threadIdx.y < 2 * RADIUS &&  \
                      blockDim.x - threadIdx.x <= 2 * RADIUS;
    if (populatesBL || populatesTR) {
        dx = populatesTR ? 1 : -1;
        dy = populatesTR ? -1 : 1;
        MOVE_MEMORY;
    }

    /*printf("%i,%3.1f,%3.1f ",dst_y * (BLOCK_SIZE + 4) + dst_x,shmem[dst_y * (BLOCK_SIZE + 4) + dst_x], N.elements[src_y * gridWidth + src_x]);*/


    /*if (threadIdx.x == 0 && threadIdx.y == 0) {*/
        /*for (int j = 0; j < 20; ++j) {*/
            /*for (int i = 0; i < 20; ++i) {*/
                /*printf("%3.1f ", shmem[j * 20 + i]);*/
            /*}*/
            /*printf("\n");*/
        /*}*/
    /*}*/

    // Get result
    float result = 0.0f;
    for (int j = 0; j < 5; ++j) {
        for (int i = 0; i < 5; ++i) {
            result += M.elements[j * 5 + i] *  \
                      shmem[(threadIdx.y + j) * (BLOCK_SIZE + 4) + threadIdx.x + i];
        }
    }
    P.elements[g_y * gridWidth + g_x] = result;

}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	Matrix  M;
	Matrix  N;
	Matrix  P;
	
	srand(2013);
	
	if(argc != 5 && argc != 4) 
	{
        // ./a.out  - size is of default values
        // ./a.out <H>  - input size is HxH
        // ./a.out <H> <W> - input size is HxW
		// Allocate and initialize the matrices
		M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 1);
        /*unsigned int N_height = rand() % 1024) + 1;*/
        unsigned int N_height = 32;
        unsigned int N_width = 32;
        if (argc == 2) {
            N_height = atoi(argv[1]);
            N_width = N_height;
        }
        if (argc == 3) {
            N_height = atoi(argv[1]);
            N_width = atoi(argv[2]);
        }
        /*unsigned int N_height = 32;*/
        /*unsigned int N_width = 32;*/
		N  = AllocateMatrix(N_height, N_width, 1);
		P  = AllocateMatrix(N.height, N.width, 0);
        /*printf("N h,w = %i,%i\n", N.height, N.width);*/
	}
	else
	{
		// Allocate and read in matrices from disk
		int* params = (int*)malloc(2 * sizeof(int));
		unsigned int data_read = 2;
      	if(ReadParams(params, data_read, argv[1])){
         	printf("Error reading parameter file\n");
         	return 1;
      	}

		M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 0);
		N  = AllocateMatrix(params[0], params[1], 0);		
		P  = AllocateMatrix(params[0], params[1], 0);
		(void)ReadFile(&M, argv[2]);
		(void)ReadFile(&N, argv[3]);
	}

    CudaTimer GPUTimer, CPUTimer;
    float GPU_inclusive_ms, CPU_ms;

	// M * N on the device
    GPUTimer.tic();
    ConvolutionOnDevice(M, N, P);
    GPU_inclusive_ms = GPUTimer.toc();
    
    /*// compute the matrix multiplication on the CPU for comparison*/
    /*Matrix reference = AllocateMatrix(P.height, P.width, 0);*/
    /*CPUTimer.tic();*/
    /*computeGold(reference.elements, M.elements, N.elements, N.height, N.width);*/
    /*CPU_ms = CPUTimer.toc();*/

    printf("<P_height>%i</P_height>\n"
            "<P_width>%i</P_width>\n"
            "<GPU_inclusive_ms>%f</GPU_inclusive_ms>\n"
            "<CPU_ms>%f</CPU_ms>\n"
            , P.height, P.width, GPU_inclusive_ms, CPU_ms);
        
    // in this case check if the result is equivalent to the expected soluion

    /*bool res = CompareResults(reference.elements, P.elements, P.width * P.height, 0.01f);;*/
    /*printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");*/
    
    if(argc == 5)
    {
		WriteFile(P, argv[4]);
	}
	else if(argc == 2)
	{
	    WriteFile(P, argv[1]);
	}   

	// Free matrices
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{

    // Pad N to have at least 2 more rows and 2 more columns (for convolution
    // to ignore borders), and multiple of BLOCK_SIZE because that's the
    // requirement of the kernel
    unsigned int NpaddedHeight = (N.height + 2 + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    unsigned int NpaddedWidth = (N.width + 2 + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    Matrix Npadded = PadMatrix(N, NpaddedHeight - N.height,
            NpaddedWidth - N.width);
    /*printf("Npadded h,w = %i,%i\n", Npadded.height, Npadded.width);*/

    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(Npadded);
    CopyToDeviceMatrix(Nd, Npadded);

    // Allocate P on the device
    Matrix Ppadded = AllocateMatrix(Npadded.height, Npadded.width, 0);
    Matrix Pd = AllocateDeviceMatrix(Ppadded);
    /*CopyToDeviceMatrix(Pd, P); // Clear memory*/
    /*printf("Ppadded h,w = %i,%i\n", Ppadded.height, Ppadded.width);*/

    /*cout << "N:\n";*/
    /*for (int j = 0; j < N.height; ++j) {*/
        /*for (int i = 0; i < N.width; ++i) {*/
            /*printf("%3.1f ", N.elements[j * N.width + i]);*/
        /*}*/
        /*printf("\n");*/
    /*}*/

    // Setup the execution configuration
    /*for (int i = 0; i < N.height * N.width; ++i) {*/
        /*cout << i << "=" << N.elements[i] << " "; } cout << endl;*/
 
    // Launch the device computation threads!
    dim3 dimGrid(Nd.width / BLOCK_SIZE, Nd.height / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    unsigned int shmem_size = pow(BLOCK_SIZE + 2 * RADIUS, 2) * sizeof(float);
    CudaTimer kernelTimer;
    kernelTimer.tic();
    ConvolutionKernel<<<dimGrid,dimBlock,shmem_size>>>(Md, Nd, Pd);
    float kernel_timing_ms = kernelTimer.toc();

    printf("<kernel_timing_ms>%f</kernel_timing_ms>\n", kernel_timing_ms);

    // Read P from the device
    CopyFromDeviceMatrix(Ppadded, Pd); 
    ExtractFromPadded(P, Ppadded);

    // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);

    FreeMatrix(&Npadded);
    FreeMatrix(&Ppadded);

}


// The submatrix of dimensions M.width by M.height of Mpadded is copied over 
// from Mpadded into M.  Note that the assumption is that M.pitch <= M.width;
void ExtractFromPadded(Matrix M, const Matrix& Mpadded)
{
    if( Mpadded.width<M.width ) {
        printf("Error extracting data from padded matrix: ");
        printf("Trying to extract %i (padded) rows from %i rows.\n", M.width, Mpadded.width);
        exit(1);
    }

    if( Mpadded.height<M.height ) {
        printf("Error extracting data from padded matrix: Height too small%d, %d\n", Mpadded.height, M.height);
        exit(1);
    }

    for( int i=0; i<M.height; i++) {
        memcpy(&M.elements[i*M.width],
                &Mpadded.elements[i*Mpadded.width],
                M.width*sizeof(float));
    }

    return;
}

Matrix PadMatrix(const Matrix& M, const int deltaHeight, const int deltaWidth)
{
    Matrix Mpadded;
    Mpadded.height = M.height + deltaHeight;
    Mpadded.width = M.width + deltaWidth;
    Mpadded.pitch = M.width;
    // Use calloc because it initializes memory elements
    Mpadded.elements = (float*) calloc(Mpadded.width*Mpadded.height, sizeof(float));

    // copy entries of original matrix only if asked to
    for( int i=0; i<M.height; i++) {
        memcpy(&Mpadded.elements[i*Mpadded.width],
                &M.elements[i*M.width],
                M.width*sizeof(float));
    }

    return Mpadded;
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
		if(rand() % 2)
			M.elements[i] = - M.elements[i];
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

//compare the data stored in two arrays on the host
bool CompareResults(float* A, float* B, int elements, float eps)
{
    int count = 0;
    for(unsigned int i = 0; i < elements; i++){
        float error = A[i]-B[i];
        if (error > eps || error < -eps) {
            count += 1;
            /*printf("%i ", i);*/
        }
        /*if(error>eps){*/
        /*return false;*/
        /*} */
    }
    printf("No. elements wrong = %i\n", count);
    return count == 0;
}

bool ReadParams(int* params, int size, char* file_name){
   ifstream ifile(file_name);
   int i=0;
   for(int i=0; i<size; i++){
      if(ifile.fail()==false){
         ifile>>params[i];
      }
   }
   return (i==size)? 1:0;

}

// Read a 16x16 floating point matrix in from file
int ReadFile(Matrix* M, char* file_name)
{
   unsigned int data_read = M->height * M->width;
   std::ifstream ifile(file_name);

   for(unsigned int i = 0; i < data_read; i++){
      ifile>>M->elements[i];
   }
   ifile.close();
   return data_read;

}



// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
   std::ofstream ofile(file_name);
   for(unsigned int i = 0; i < M.width*M.height; i++){
      ofile<<M.elements[i];
   }
   ofile.close();
}

