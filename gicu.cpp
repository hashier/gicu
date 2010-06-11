/*
 * gicu
 *
 * gimp CUDA
 *
 * Gimp Plug-In welches die Grafikkarte benutzt um die Berechnungen
 * zu beschleunigen
 *
 * Copyright
 * Christopher Loessl (4044723)
 *
 * License
 * as-is
 *
 * Coded for
 * PADI (Praktische Aspekte der Informatiker)
 * Institut CG
 *
 */


/* eigene */
#include "cuda.h"
#include "gicu.h"


// Basics
// Function pnts to the functions
GimpPlugInInfo PLUG_IN_INFO = {
	NULL,
	NULL,
	query,
	run
};


// Main
MAIN()

static void query( void) {

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
	gimp_plugin_menu_register(
			"plug-in-gicu",
			"<Image>/Filters/CUDA");
}


static void run(
		const gchar     *name,
		gint            nparams,
		const GimpParam *param,
		gint            *nreturn_vals,
		GimpParam       **return_vals) {

	/* return thingy */
	static GimpParam values[1];
	GimpPDBStatusType status = GIMP_PDB_SUCCESS;
	GimpRunMode run_mode;
	GimpDrawable *drawable;

	/* Setting mandatory output values */
	*nreturn_vals = 1;
	*return_vals = values;
	values[0].type = GIMP_PDB_STATUS;
	values[0].data.d_status = status;

	/* Getting run_mode */
	run_mode = (GimpRunMode)param[0].data.d_int32;

	/* Where to paint */
	drawable = gimp_drawable_get( param[2].data.d_drawable);

	gimp_progress_init ("CUDA-Filter...");
	cuda (drawable);

	gimp_displays_flush();
	gimp_drawable_detach( drawable);

}


static void cuda( GimpDrawable *drawable) {
	gint         channels;
	gint         x1, y1, x2, y2;
	GimpPixelRgn rgn_in, rgn_out;
	gint         width, height;

	size_t size;
	guchar *h_image;
	guchar *d_image;

	gimp_drawable_mask_bounds (
			drawable->drawable_id,
			&x1, &y1,
			&x2, &y2);

	channels = gimp_drawable_bpp( drawable->drawable_id);
	width = x2 - x1;
	height = y2 - y1;
	size = width * height * channels;

	/* allocate mem on the gpu */
	cutilSafeCall( cudaMalloc( (void**)&d_image, size));

	gimp_pixel_rgn_init(
			&rgn_in,
			drawable,
			x1, y1,
			width, height,
			FALSE, FALSE);
	gimp_pixel_rgn_init(
			&rgn_out,
			drawable,
			x1, y1,
			width, height,
			TRUE, TRUE);

	/* where the final image will be saved in the host memory */
	/* allocate mem for output image */
// 	cutilSafeCall( cudaMallocHost( (void**)&h_image, size));
	h_image = g_new( guchar, size);
	gimp_pixel_rgn_get_rect( &rgn_in, h_image, x1, y1, width, height);

	gimp_progress_init ( "CUDA-Gimp-Plug-In");
	gimp_progress_pulse();

	/* CUDA cp image */
	gimp_progress_set_text( "Copying the image to graphics card");
	cutilSafeCall( (cudaMemcpy( d_image, h_image, size, cudaMemcpyHostToDevice)));
	
	/* CUDA running kernel */
	gimp_progress_set_text( "Running CUDA-Kernel");
// 	g_message("test: %d\n", channels);
// 	g_message("w: %d\nh: %d\n", width, height);
	filter( d_image, width, height, channels);
	
	/* CUDA cp image back to host */
	gimp_progress_set_text( "Copying the image back to host");
	cutilSafeCall( (cudaMemcpy( h_image, d_image, size, cudaMemcpyDeviceToHost)));

	gimp_progress_end();

	/* save image back to gimp core */
	gimp_pixel_rgn_set_rect (
			&rgn_out,
			h_image,
			x1, y1,
			width, height);

	gimp_drawable_flush( drawable);
	gimp_drawable_merge_shadow( drawable->drawable_id, TRUE);
	gimp_drawable_update( drawable->drawable_id, x1, y1, width, height);

}

