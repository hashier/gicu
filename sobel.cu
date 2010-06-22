#include "cuda.h"

__global__ void sobel( guchar *d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm) {

	unsigned char pix00 = tex2D( tex, 1,1 );

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	d_image[y*step+x] = pix00;

}
