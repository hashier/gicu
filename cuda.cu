#include "cuda.h"

extern "C" void filter( guchar* d_image, gint width, gint height, guint channels, cuda_filter mode) {

	dim3 blockDim( 16, 16, 1);
	dim3 gridDim( width / blockDim.x + 1, height / blockDim.y + 1, 1);
	guint step = channels * width;

	switch ( mode) {
		case GREY:
			grey<<< gridDim, blockDim, 0 >>>( d_image, width, height, channels, step);
			break;
		case BOX:
			box<<< gridDim, blockDim, 0 >>>( d_image, width, height, channels, step);
			break;
	}

}
