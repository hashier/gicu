#ifndef __cuda_H_
#define __cuda_H_

/* CUDA */
#include <cuda_runtime.h>
#include <cutil_inline.h>

#include <glib/gtypes.h>
#include <glib/gmessages.h>

#include "gicu.h"


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

extern "C" void cuda_init( );

extern "C" void setupTexture(gint width, gint height);
extern "C" void bindTexture( );

extern "C" void updateTexture(gint width, gint height, guchar *data, gint channel);

extern "C" void unbindTexture( );
extern "C" void deleteTexture( );


#endif