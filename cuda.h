#ifndef __cuda_H_
#define __cuda_H_

/* CUDA */
#include <cuda_runtime.h>
#include <cutil_inline.h>

/* Gimp plug-in header */
#include <libgimp/gimp.h>

static enum _cuda_filter {
	GREY,
	BOX
} cuda_filter = {
	BOX
};

extern "C" void filter( guchar* d_image, gint width, gint height, guint channels, enum _cuda_filter mode);
__global__ void grey( guchar* d_image, gint width, gint height, guint channels, guint step);
__global__ void box( guchar *d_image, gint width, gint height, guint channels, guint step);


#endif
