#include "../../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Demonstrate defining the dimensions of a block of threads and a grid of blocks from the host.
 */

int main(int argc, char **argv)
{
    // define total data element
    int nElem = 1024;

		printf("grid\tblock\n");

    // define grid and block structure
    dim3 block (1024);
    dim3 grid  ((nElem + block.x - 1) / block.x);
    printf("%4d\t %4d\n", grid.x, block.x);

    // reset block
    block.x = 512;
    grid.x  = (nElem + block.x - 1) / block.x;
		printf("%4d\t %4d\n", grid.x, block.x);

    // reset block
    block.x = 256;
    grid.x  = (nElem + block.x - 1) / block.x;
		printf("%4d\t %4d\n", grid.x, block.x);

    // reset block
    block.x = 128;
    grid.x  = (nElem + block.x - 1) / block.x;
		printf("%4d\t %4d\n", grid.x, block.x);

    // reset device before you leave
    CHECK(cudaDeviceReset());

		getchar();
    return(0);
}

