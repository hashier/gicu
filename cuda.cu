#include "cuda.h"
#include "gicu.h"

extern "C" void filter( guchar *d_image, gint width, gint height) {

	dim3 blockDim( 16, 16, 1);
	dim3 gridDim( width / blockDim.x + 1, height / blockDim.y + 1, 1);
	kernel<<< gridDim, blockDim, 0 >>>( d_image, width, height);

}

__global__ void kernel( guchar *d_image, gint width, gint height) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	d_image[y*width+x] = 128;
}
