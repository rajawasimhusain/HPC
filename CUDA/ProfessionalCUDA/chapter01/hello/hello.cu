#include "../../common/common.h"

/*
 * A simple introduction to programming in CUDA. 
 * This program prints "Hello World from GPU! from 10 CUDA threads running on the GPU.
 */

__global__ void helloFromGPU()
{
		//printf("### Hello World from GPU ID:%d ###\n", threadIdx.x);
	for (int i = 0; i < 1024*1024; i++)
	{
		int a = 10;
		int b = 20;
		int c = a + b;
	}
}

int main(int argc, char **argv)
{
		// print message on CPU side
    printf("+++ Hello World from CPU +++\n");

		// call function to execute on GPU and halt CPU thread until GPU is done
    helloFromGPU<<<1024, 1024>>>();    
		CHECK(cudaDeviceSynchronize());
		
		// print message on CPU side
		printf("Press any key to exit ... ");
		getchar();

		// release all the GPU related resources
		CHECK(cudaDeviceReset());
		return 0;
}


