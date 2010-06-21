#include "cuda.h"

// Texture reference for reading image
static texture<guchar, 2> tex;

// arrays are optimized for 2D access so we'll use arrays
// insted of single row pointer memory addresses
static cudaArray *array = NULL;

// Shared Mem on the dev is declared with __shared__
extern __shared__ unsigned char LocalBlock[];


void filter(
		guchar* d_image, gint width, gint height,
		guint channels) {

	dim3 blockDim( 16, 16, 1);
	int x = 0, y = 0;
	if ( width % blockDim.x)
		x = 1;
	if ( height % blockDim.y)
		y = 1;

	dim3 gridDim( width / blockDim.x + x, height / blockDim.y + y, 1);
	guint step = channels * width;


	switch ( filterParm.cuda_filter) {
		case GREY:
			greyGRAY<<< gridDim, blockDim, 0 >>>( d_image, width, height, channels, step);
			break;
			
		case BOX:
			box<<< gridDim, blockDim, 0 >>>( d_image, width, height, channels, step, filterParm);
			break;
			
		case SOBEL:
			break;
			
		case AVERAGE:
			break;
			
		default:
			g_printerr("Filter not found");
			break;
	}

}

extern "C" void cuda_init( ) {
}

extern "C" void setupTexture(gint width, gint height) {
	cudaChannelFormatDesc desc;

// 	desc = cudaCreateChannelDesc<unsigned char>();
	int e = (int)sizeof(guchar) * 8;
	desc = cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);

	cutilSafeCall(cudaMallocArray(&array, &desc, width, height));
}

extern "C" void updateTexture(gint width, gint height, guchar *data, gint channel) {
	cutilSafeCall(cudaMemcpyToArray(
			array,
			0, 0,
			data, /* 0 0 <- h und w offset */
			channel * sizeof(guchar) * width * height, cudaMemcpyHostToDevice));
}

extern "C" void deleteTexture( ) {
	cutilSafeCall(cudaFreeArray(array));
}

extern "C" void bindTexture( ) {
	/* clamp x and y axis to the boarder */
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false;

	cutilSafeCall(cudaBindTextureToArray(tex, array));
}

extern "C" void unbindTexture( ) {
	cutilSafeCall(cudaUnbindTexture(tex));
}
