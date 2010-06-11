#include "cuda.h"
//#include "gicu.h"

extern "C" void filter( guchar *d_image, gint width, gint height, guint channels) {

	dim3 blockDim( 16, 16, 1);
	dim3 gridDim( width / blockDim.x + 1, height / blockDim.y + 1, 1);
	guint step = channels * width;
	kernel<<< gridDim, blockDim, 0 >>>( d_image, width, height, channels, step);

}

__global__ void kernel( guchar *d_image, gint width, gint height, guint channels, guint step) {
	for ( int i = 0; i < channels; i++) {
		unsigned int x = blockIdx.x * blockDim.x * channels + threadIdx.x * channels-i;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		d_image[y*step+x] = 128;
	}
}
