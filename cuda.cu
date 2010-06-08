#include "cuda.h"


extern "C" void test() {
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(1, 1, 1);

	kernel<<< 1, 1, 0 >>>();
}

__global__ void kernel()
{
   unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
   unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
}
