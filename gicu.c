#include <libgimp/gimp.h>

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

	// required parameters for image
	// see for more information:
	// http://www.gimp.org/docs/plug-in/sect-essentials.html#sect-main
	static GimpParamDef args[] = {
		{
			GIMP_PDB_INT32,
			"run-mode",
			"Interactive, non-interactive"
		},
		{
			GIMP_PDB_IMAGE,
			"image",
			"Input image"
		},
		{
			GIMP_PDB_DRAWABLE,
			"drawable",
			"Input drawable"
		}
	};

	// install procedure
	// this registers the plugin into the PDB
	gimp_install_procedure (
			"plug-in-gicu",
			"Use CUDA to speed up things",
			"In the beginning, there was Wilber, Wilber the gimp. The graphic was without form and void, and darkness was upon the face of the desktop, and the Spirit of Wilber was moving over the face of the bitstream.\n\nAnd Wilber said, \"<Toolbox>/File/New,\" and there was an image. And Wilber saw that the image was good, and Wilber separated the image into drawables. And Wilber looked down at what he had wrought, and Wilber said, \"Oh golly.\" For Wilber had made the drawables of the layer according to their kinds, and the drawables of the channel according to their kinds, and the drawables of the mask according to their kinds...",
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

	// Where to put the plugin
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

	static GimpParam values[1];
	GimpPDBStatusType status = GIMP_PDB_SUCCESS;
	GimpRunMode run_mode;

	/* Setting mandatory output values */
	*nreturn_vals = 1;
	*return_vals = values;

	values[0].type = GIMP_PDB_STATUS;
	values[0].data.d_status = status;

	/* Getting run_mode - we won't display a dialog if
	 * we are in NONINTERACTIVE mode */
	run_mode = param[0].data.d_int32;

	if ( run_mode != GIMP_RUN_NONINTERACTIVE)
		g_message("Hello, world!\n");
}

