#include "cuda.h"

__global__ void box( guchar *d_image, gint width, gint height, guint channels, guint step) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	x *= channels;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	d_image[y*step+x]   = 10;
	d_image[y*step+x+1] = 10;
	d_image[y*step+x+2] = 10;
}
