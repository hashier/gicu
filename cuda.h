#ifndef __cuda_H_
#define __cuda_H_

/* CUDA */
#include <cuda_runtime.h>
#include <cutil_inline.h>

/* Gimp plug-in header */
#include <libgimp/gimp.h>


extern "C" void filter( guchar* d_image, gint width, gint height);
__global__ void kernel( guchar* d_image, gint width, gint height);

#endif
