#ifndef __gicu_H_
#define __gicu_H_

/* Gimp plug-in header */
#include <libgimp/gimp.h>

typedef struct _param {
	gint radius;
	gint offset;
} Para;

/* Set up default values for options */
static Para pvals = {
	3,  /* radius */
	5   /* offset */
};


static void query( void);
static void run(
		const gchar     *name,
		gint            nparams,
		const GimpParam *param,
		gint            *nreturn_vals,
		GimpParam       **return_vals);
static void cuda( GimpDrawable *drawable);


#endif
