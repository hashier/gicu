#ifndef __cuda_H_
#define __cuda_H_

/* CUDA */
#include <cuda_runtime.h>
#include <cutil_inline.h>

#include <glib/gtypes.h>
#include <glib/gmessages.h>

#include "gicu.h"


/**
 * @brief Calls the correct CUDA-kernel
 * @param[in] d_image Pointer to device memory (the rendered image will be saved there)
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] channels How many channels does the image have
 * @param[in] d_image_temp This is only needed for the Boxing algo. The boxing algo needes a temp image (pointer to dev mem as well)
 *
 */
extern "C" void filter(guchar *d_image, gint width, gint height, guint channels, guchar *d_image_temp);

__global__ void greyRGB  ( guchar *d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm);
__global__ void grey  ( guchar *d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm);
__global__ void box   ( guchar* d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm);
__global__ void test  ( guchar* d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm);
__global__ void sobelTex( guchar* d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm);
__global__ void d_boxfilter_x_tex( guchar* od, int w, int h, int r);
__global__ void d_boxfilter_y_global(guchar* id, guchar *od, int w, int h, int r, int offset, gboolean do_bin);
__device__ void d_boxfilter_y(guchar* id, guchar* od, int w, int h, int r, uint x, int offset);
__device__ void d_boxfilter_y_bin(guchar *id, guchar *od, int w, int h, int r, uint x, int offset);
__global__ void AVGShared(
		uchar4 *pc, unsigned short step,
		short BlockWidth, short SharedPitch,
		short w, short h, float fScale,
		int radiusAVG, int offset, gboolean do_bin );

/**
 * @brief nothing
 */
extern "C" void cuda_init( );

/**
 * @brief sets the texture up for CUDA
 * @param[in] width the width of the image
 * @param[in] height the height of the image
 *
 * width and height is needed to calculate the needed size for the texture
 */
extern "C" void setupTexture(gint width, gint height);
/**
 * @brief binds the array to the texture
 */
extern "C" void bindTexture( );

/**
 * @brief Updates the texture with a new image
 * @param[in] width of the image
 * @param[in] height of the new image
 * @param[in] data Pointer to the image
 * @param[in] channels of the picture
 */
extern "C" void updateTexture(gint width, gint height, guchar *data, gint channels);

/**
 * @brief unbindes the texture from the array
 */
extern "C" void unbindTexture( );
/**
 * @brief frees the array
 */
extern "C" void deleteTexture( );


#endif
