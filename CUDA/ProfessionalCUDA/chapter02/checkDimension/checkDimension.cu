#include "../../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Display the dimensionality of a thread block and grid from the host and
 * device.
 */

__global__ void checkIndex(void)
{		
    printf("(%2d,%2d,%2d )\t(%2d,%2d,%2d )\t(%2d,%2d,%2d )\t(%2d,%2d,%2d )\n", 
			     threadIdx.x, threadIdx.y, threadIdx.z, 
			     blockIdx.x, blockIdx.y, blockIdx.z, 
			     blockDim.x, blockDim.y, blockDim.z, 
			     gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv)
{
    // define total data element
    int nElem = 25;

    // define grid and block structure
    dim3 block(5);
    dim3 grid((nElem + block.x - 1) / block.x);

    // check grid and block dimension from host side
		printf("--- CPU side --- \n");
		printf("  gridDim\t  blockDim\n");
		printf("(%2d,%2d,%2d )\t(%2d,%2d,%2d )", grid.x, grid.y, grid.z, block.x, block.y, block.z);    

    // check grid and block dimension from device side
		printf("\n\n--- GPU side --- \n");
		printf(" threadIdx\t  blockIdx\t  blockDim\t  gridDim\n");
    checkIndex<<<grid, block>>>();

    // reset device before you leave
    CHECK(cudaDeviceReset());

		getchar();
    return(0);
}
