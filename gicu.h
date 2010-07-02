#ifndef __gicu_H_
#define __gicu_H_


/* Gimp plug-in header */
#include <libgimp/gimp.h>
#include <libgimpwidgets/gimpwidgetstypes.h>


typedef enum _cuda_filter {
	GREY,
	BOXBIN,
	SOBEL,
	BOX,
	TEST
} cuda_filter;


typedef struct _FilterParameter {
	gint radius;
	gint offset;
	gboolean preview;
	enum _cuda_filter cuda_filter;
} FilterParameter;

extern FilterParameter filterParm;


void query( void);
void run(
		const gchar     *name,
		gint            nparams,
		const GimpParam *param,
		gint            *nreturn_vals,
		GimpParam       **return_vals);
void cuda( GimpDrawable* drawable, GimpPreview* preview);


#endif
