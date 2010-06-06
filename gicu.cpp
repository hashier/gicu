/*
 * gicu
 *
 * gimp CUDA
 *
 * Gimp Plug-In welches die Grafikkarte benutzt um die Berechnungen
 * zu beschleunigen
 *
 */

/* System headers */


/* gimp */
#include <libgimp/gimp.h>


/* CUDA */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>


//
// HEADER
//
static void query (void);
static void run   (
		const gchar     *name,
		gint            nparams,
		const GimpParam *param,
		gint            *nreturn_vals,
		GimpParam       **return_vals);


//
// Basics
//
GimpPlugInInfo PLUG_IN_INFO = {
	NULL,
	NULL,
	query,
	run
};


//
// Main
//
MAIN()

static void query (void) {

	char gicu_help[] =
		"In the beginning, there was Wilber, Wilber the gimp. \
The graphic was without form and void, and darkness was \
upon the face of the desktop, and the Spirit of Wilber was \
moving over the face of the bitstream.\n\nAnd Wilber said, \
\"<Toolbox>/File/New,\" and there was an image. And Wilber \
saw that the image was good, and Wilber separated the \
image into drawables. And Wilber looked down at what he \
had wrought, and Wilber said, \"Oh golly.\" For Wilber \
had made the drawables of the layer according to their \
kinds, and the drawables of the channel according to their \
kinds, and the drawables of the mask according to their kinds...";

	/* required parameters for image
	 * see for more information:
	 * http://www.gimp.org/docs/plug-in/sect-essentials.html#sect-main
	 */
	static GimpParamDef args[] = {
		{
			GIMP_PDB_INT32,
			(gchar*) "run-mode",
			(gchar*) "Interactive, non-interactive"
		},
		{
			GIMP_PDB_IMAGE,
			(gchar*) "image",
			(gchar*) "Input image"
		},
		{
			GIMP_PDB_DRAWABLE,
			(gchar*) "drawable",
			(gchar*) "Input drawable"
		}
	};

	/* install procedure
	 * this registers the plugin into the PDB
	 */
	gimp_install_procedure (
			"plug-in-gicu",
			"Use CUDA to speed up things",
			gicu_help,
			"Christopher Loessl",
			"Copyright: as - is by Christopher Loessl",
			"SummerSemster 2010",
			"_CUDA-Filter",
			"RGB*, GRAY*",
			GIMP_PLUGIN,
			G_N_ELEMENTS (args), // this parameters are passed to the function
			0,                   // num of parameters
			args,                // same but return thingy stuff
			NULL);

	/* Where to put the plugin */
	gimp_plugin_menu_register (
			"plug-in-gicu",
			"<Image>/Filters/CUDA");
}


static void run (
		const gchar     *name,
		gint            nparams,
		const GimpParam *param,
		gint            *nreturn_vals,
		GimpParam       **return_vals) {

	/* return thingy */
	static GimpParam values[1];

	GimpPDBStatusType status = GIMP_PDB_SUCCESS;
	GimpRunMode run_mode;

	/* CUDA */

	/* Setting mandatory output values */
	*nreturn_vals = 1;
	*return_vals = values;

	values[0].type = GIMP_PDB_STATUS;
	values[0].data.d_status = status;

	/* Getting run_mode */
	run_mode = (GimpRunMode)param[0].data.d_int32;

	if ( run_mode != GIMP_RUN_NONINTERACTIVE)
		g_message("values = return werte!\n\
param = uebergebene parameter und der 0te ist run-mode");


	int deviceCount = 5;
	cudaGetDeviceCount(&deviceCount);

	g_message("blub: %d\n", deviceCount);

	unsigned char *speicher;
	size_t size;
	size = 5 * 5 * sizeof (unsigned char);
	cutilSafeCall(cudaMallocHost((void**)&speicher, size));

}

