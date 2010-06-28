#ifndef __cuda_H_
#define __cuda_H_

/* CUDA */
#include <cuda_runtime.h>
#include <cutil_inline.h>

#include <glib/gtypes.h>
#include <glib/gmessages.h>

#include "gicu.h"


extern "C" void filter(guchar *d_image, gint width, gint height, guint channels);

__global__ void greyRGB  ( guchar *d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm);
__global__ void grey ( guchar *d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm);
__global__ void box  ( guchar* d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm);
__global__ void sobelTex( guchar* d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm);

extern "C" void cuda_init( );

extern "C" void setupTexture(gint width, gint height);
extern "C" void bindTexture( );

extern "C" void updateTexture(gint width, gint height, guchar *data, gint channel);

extern "C" void unbindTexture( );
extern "C" void deleteTexture( );


#endif