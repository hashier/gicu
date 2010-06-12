#include "cuda.h"

__global__ void grey( guchar *d_image, gint width, gint height, guint channels, guint step) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	x *= channels;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	d_image[y*step+x]   = 128;
	d_image[y*step+x+1] = 128;
	d_image[y*step+x+2] = 128;
}