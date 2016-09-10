/*
 * 
 * NICK Local Image thresholding Algorithm
 * M. Hassan Najafi
 * Najaf011@umn.edu
 */
 
// includes, system
#include <stdlib.h>
#include <stdio.h> //FILE*, fopen, fclose, fread, fwrite
#include <string.h>
#include <math.h>
#include <time.h> //clock_t, clock

// includes, project
//#include <cutil.h>

// includes, kernels
#include "Nick_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void NickOnDevice(const Matrix N, Matrix P_global, Matrix P_shared1, Matrix P_shared2);
bool comparefe(float* A, float* B, int len, float tol);

float cpu_run_time;
float gpu_run_time;
float gpu_run_time_shared1;
float gpu_run_time_shared2;
float gpu_total_time;
float gpu_total_time_shared1;
float gpu_total_time_shared2;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {


	
	//Matrix  M;
	Matrix  N;
	Matrix  P_global;
	Matrix  P_shared1;
	Matrix  P_shared2;
	
	srand(2012);
       int sizex, sizey;
	
	if(argc != 4 && argc != 3) 
	{
		// Allocate and initialize the matrices
		//M  = AllocateMatrix(WINDOW_SIZE, WINDOW_SIZE, 1);
		N  = AllocateMatrix((rand() % 1024) + 1, (rand() % 1024) + 1, 1);
		P_global  = AllocateMatrix(N.height, N.width, 0);
		P_shared1  = AllocateMatrix(N.height, N.width, 0);
		P_shared2  = AllocateMatrix(N.height, N.width, 0);
	}
	else
	{
		// Allocate and read in matrices from disk
                sizex = atoi(argv[1]);
                sizey = atoi(argv[2]);

		//M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 0);
		N  = AllocateMatrix(sizex, sizey, 0);		
		P_global  = AllocateMatrix(sizex, sizey, 0);
		P_shared1  = AllocateMatrix(sizex, sizey, 0);
		P_shared2  = AllocateMatrix(sizex, sizey, 0);
		//(void)ReadFile(&M, argv[2]);
		(void)ReadFile(&N, argv[2]);
	}

	printf("=====================================================\n" );
	printf("Input Matrix Dimension = %d x %d\n", N.height,N.width  );

    NickOnDevice(N, P_global, P_shared1, P_shared2);
    
    // compute the matrix convolution on the CPU for comparison
    Matrix reference = AllocateMatrix(P_global.height, P_global.width, 0);
	
	clock_t start = clock();
        float cpu_run_time;
    computeGold(reference.elements, N.elements, N.height, N.width);
        cpu_run_time = (clock()-start)*1000./CLOCKS_PER_SEC;
	
	printf("Elapsed time for running CPU kernel = %f ms\n", cpu_run_time );
	
	printf("\nSpeedup CPU/GPU Kernel (global)= %f \n", cpu_run_time/gpu_run_time );
	printf("Speedup CPU/GPU Kernel(shared1)= %f \n", cpu_run_time/gpu_run_time_shared1 );
	printf("Speedup CPU/GPU Kernel(shared2)= %f \n\n", cpu_run_time/gpu_run_time_shared2 );
	
	printf("Speedup CPU/GPU Total (global)= %f \n", cpu_run_time/gpu_total_time );
	printf("Speedup CPU/GPU Total (shared1)= %f \n", cpu_run_time/gpu_total_time_shared1 );
	printf("Speedup CPU/GPU Total (shared2)= %f \n\n", cpu_run_time/gpu_total_time_shared2 );
	
    // in this case check if the result is equivalent to the expected soluion
    bool res_global = comparefe(reference.elements, P_global.elements, P_global.width * P_global.height, 0.001f);
	bool res_shared1 = comparefe(reference.elements, P_shared1.elements, P_shared1.width * P_shared1.height, 0.001f);
	bool res_shared2 = comparefe(reference.elements, P_shared2.elements, P_shared2.width * P_shared2.height, 0.001f);
    printf("Test CUDA kernel global %s \n", (1 == res_global) ? "PASSED" : "FAILED");
	printf("Test CUDA kernel shared1 %s \n", (1 == res_shared1) ? "PASSED" : "FAILED");
	printf("Test CUDA kernel shared2 %s \n", (1 == res_shared2) ? "PASSED" : "FAILED");
	
	
//	float p_out;
//	float r_out;
//	int Miss_Match_count=0;
//	  for(int row = 0; row < P.height; ++row) {
//	      for(int col = 0; col < P.width; ++col) {	  
//			  p_out = P.elements[row*P.width + col];
//			  r_out = reference.elements[row*P.width + col];
//			  if (p_out-r_out>0.001f | r_out-p_out>0.001f)
//			  //	printf("Miss match : P[%d]  Produced : %.5f, Expected : %.5f\n ",row*P.width + col, p_out, r_out );
//				Miss_Match_count++;
//		  }
//	  }
//		printf("Total Number of Miss matches: %d\n", Miss_Match_count);
//		printf("=====================================================\n" );
	
    
    if(argc == 4)
    {
		WriteFile(P_global, argv[3]);
	}
 

    FreeMatrix(&N);
    FreeMatrix(&P_global);
	FreeMatrix(&P_shared1);
	FreeMatrix(&P_shared2);
	return 0;
}



////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void NickOnDevice(const Matrix N, Matrix P_global, Matrix P_shared1, Matrix P_shared2)
{
//	float Overhead_allocation;
	float Overhead_Copy;
	//Define a timer from cutil.h
	//MyTimer_overhead is for measuring overhead time
	
	//MyTimer_kernel is for measuring  time elapsed for running kernels
 
	//Allocating memory and measuring their overhead time
        clock_t start = clock();
	Matrix Nd = AllocateDeviceMatrix(N);
	Matrix Pd_global = AllocateDeviceMatrix(P_global);	
	Matrix Pd_shared1 = AllocateDeviceMatrix(P_shared1);
	Matrix Pd_shared2 = AllocateDeviceMatrix(P_shared2);
        Overhead_Copy = (clock() - start) * 1000./CLOCKS_PER_SEC;	
//	Overhead_allocation = cutGetTimerValue(MyTimer_overhead);
//	printf("Elapsed time - Overhead - Allocating = %f ms\n", Overhead_allocation );
	
        start = clock();
	//Copy N to constant Memory
	//cudaMemcpyToSymbol(Nc, N.elements , IMAGE_WIDTH*IMAGE_HEIGTH*sizeof(float));
	//Copy N to Device Memory 
    CopyToDeviceMatrix(Nd, N);	    
    //Clearing memory
    CopyToDeviceMatrix(Pd_global, P_global);	
        Overhead_Copy += (clock() - start) * 1000./CLOCKS_PER_SEC;	
	
	
	CopyToDeviceMatrix(Pd_shared1, P_shared1);
	CopyToDeviceMatrix(Pd_shared2, P_shared2);	
	
	//Sets 16 KB for shared memory and 48 KB for SM's L1 cache	
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	
    // Setup the execution configuration	
	dim3 dimGrid, dimBlock, dimBlock_Shared1, dimBlock_Shared2;

 	dimBlock.x = BLOCK_SIZE ;
 	dimBlock.y = BLOCK_SIZE ;
 	dimBlock.z = 1;	
	
	dimBlock_Shared1.x = BLOCK_SIZE ;
 	dimBlock_Shared1.y = BLOCK_SIZE ;
 	dimBlock_Shared1.z = 1;	
	
	dimBlock_Shared2.x = BLOCK_SIZE+WINDOW_SIZE-1 ;
	dimBlock_Shared2.y = BLOCK_SIZE+WINDOW_SIZE-1 ;
	dimBlock_Shared2.z = 1 ;
	
 	dimGrid.x = (Nd.width+BLOCK_SIZE-1)/BLOCK_SIZE;
	dimGrid.y = (Nd.height+BLOCK_SIZE-1)/BLOCK_SIZE;
	dimGrid.z = 1;
	
	
	//1. All accesses from global memory
        start = clock();
			NickKernel<<<dimGrid, dimBlock>>>( Nd, Pd_global ,Nd.width, Nd.height );
 	cudaDeviceSynchronize();	
	gpu_run_time = (clock() - start)*1000./CLOCKS_PER_SEC;
	
	//2. Some accesses from shared memory and some from global memory
        start = clock();
			NickKernel_shared1<<<dimGrid, dimBlock_Shared1>>>( Nd, Pd_shared1, Nd.width, Nd.height );
 	cudaDeviceSynchronize();	
	gpu_run_time_shared1 = (clock() - start)*1000./CLOCKS_PER_SEC;
	
	//3. All accesses from shared memory
        start = clock();
		NickKernel_shared2<<<dimGrid, dimBlock_Shared2>>>( Nd, Pd_shared2, Nd.width, Nd.height );
 	cudaDeviceSynchronize();	
	gpu_run_time_shared2 = (clock() - start)*1000./CLOCKS_PER_SEC;
	
	
	printf("Elapsed time for running Kernels on GPUs (global)= %f ms\n", gpu_run_time );
	printf("Elapsed time for running Kernels on GPUs (shared1) = %f ms\n", gpu_run_time_shared1 );
	printf("Elapsed time for running Kernels on GPUs (shared2) = %f ms\n", gpu_run_time_shared2 );	
	
	start = clock();
    // Read P from the device
    CopyFromDeviceMatrix(P_global, Pd_global); 
        Overhead_Copy += (clock() - start) * 1000./CLOCKS_PER_SEC;	
	CopyFromDeviceMatrix(P_shared1, Pd_shared1);
	CopyFromDeviceMatrix(P_shared2, Pd_shared2);
	
	printf("Elapsed time - Overhead - Copy = %f ms\n", Overhead_Copy );
	


	gpu_total_time = gpu_run_time + Overhead_Copy;
	gpu_total_time_shared1 = gpu_run_time_shared1 + Overhead_Copy;
	gpu_total_time_shared2 = gpu_run_time_shared2 + Overhead_Copy;
	printf("Total GPU time (global)= %f ms\n", gpu_total_time );
	printf("Total GPU time (shared1)= %f ms\n", gpu_total_time_shared1 );
	printf("Total GPU time (shared2)= %f ms\n", gpu_total_time_shared2 );
	
    // Free device matrices
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd_global);
	FreeDeviceMatrix(&Pd_shared1);
	FreeDeviceMatrix(&Pd_shared2);


}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}


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

// Read a 16x16 floating point matrix in from file
int ReadFile(Matrix* M, char* file_name)
{
	unsigned int data_read = M->height * M->width;
        FILE* infile=fopen(file_name, "r");
	fread(M->elements, sizeof(float), data_read, infile);
        fclose(infile);
	return data_read;
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
    FILE* outfile=fopen(file_name, "w");
    fread(M.elements, sizeof(float), M.width*M.height, outfile);
    //cutWriteFilef(file_name, M.elements, M.width*M.height,
    //                   0.0001f);
}
bool comparefe(float* A, float* B, int len, float tol)
{
    for (int z=0;z<len;z++)
       if (fabs(A[z]-B[z]) > tol) return false;
    return true;
}
