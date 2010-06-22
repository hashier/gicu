#include "cuda.h"

__global__ void box( guchar *d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	d_image[y*step+x] = 255 - filterParm.radius;

}
