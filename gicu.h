#ifndef __gicu_H_
#define __gicu_H_

static void query (void);
static void run   (
		const gchar     *name,
		gint            nparams,
		const GimpParam *param,
		gint            *nreturn_vals,
		GimpParam       **return_vals);
static void blur (GimpDrawable *drawable);

#endif
