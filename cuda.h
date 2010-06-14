#ifndef __cuda_H_
#define __cuda_H_

/* CUDA */
#include <cuda_runtime.h>
#include <cutil_inline.h>

#include <glib/gtypes.h>
#include <glib/gmessages.h>

typedef enum _cuda_filter {
	GREY,
	BOX,
	SOBEL,
	AVERAGE
} cuda_filter;


extern "C" void filter(guchar *d_image, gint width, gint height, guint channels,
				cuda_filter mode, gint radius = 3, gint offset = 5);

__global__ void grey ( guchar *d_image, gint width, gint height, guint channels, guint step);
__global__ void box  ( guchar *d_image, gint width, gint height, guint channels, guint step);
__global__ void sobel( guchar *d_image, gint width, gint height, guint channels, guint step);

extern "C" void cuda_init( );

extern "C" void setupTexture(gint width, gint height);
extern "C" void updateTexture(gint width, gint height, guchar *data, gint channel);

extern "C" void deleteTexture( );

extern "C" void bindTexture( );
extern "C" void unbindTexture( );


#endif