#include "../../common/common.h"
#include "hello.h"

/*
 * A simple introduction to programming in CUDA. 
 * This program prints "Hello World from GPU! from 10 CUDA threads running on the GPU.
 */

__global__ void helloFromGPU()
{
    printf("Hello World from GPU!\n");
}

void helloFromCPU()
{
	printf("Hello World from CPU!\n\n");

	helloFromGPU<<<1, 10>>>();
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaDeviceReset());
}