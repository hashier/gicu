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
#include "gui.h"


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
		},
		{
			GIMP_PDB_INT32,
			(gchar*) "radius",
			(gchar*) "set the radius of the kernel"
		},
			GIMP_PDB_INT32,
			(gchar*) "offset",
			(gchar*) "to reduce the noice"
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
			/* "RGB, GRAY", */
			"GRAY",
			GIMP_PLUGIN,
			G_N_ELEMENTS (args), // this number of parameters are passed to the function
			0,                   // num of parameters returned by the function
			args,                // where is all the stuff saved (where to get the info)
			NULL);               // where to save the information

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
	GimpPDBStatusType status;
	GimpRunMode run_mode;
	GimpDrawable *drawable;

	/* Setting mandatory output values */
	status = GIMP_PDB_SUCCESS;
	*nreturn_vals = 1;
	*return_vals = values;
	values[0].type = GIMP_PDB_STATUS;
	values[0].data.d_status = status;

	/* Getting run_mode */
	run_mode = (GimpRunMode)param[0].data.d_int32;

// 	g_message("nparams: %d\n", nparams);
// 	filterParm.radius = 4;
// 	g_message("nparams: %d\n", nparams);

	/* Where to paint */
	drawable = gimp_drawable_get( param[2].data.d_drawable);


// 	switch ( run_mode) {
// 		case GIMP_RUN_INTERACTIVE:
// 			/* Get parameters from last run */
// 			gimp_get_data( "plug-in-gicu", &filterParm);
// 
// 			if (! gicu_dialog (drawable)) {
// 				return;
// 			}
// 			break;
// 
// 		case GIMP_RUN_NONINTERACTIVE:
// 			/* 0 run mode
// 			 * 1 image
// 			 * 2 drawable
// 			 * 3 radius
// 			 * 4 offset
// 			 */
// 			if ( nparams != 5) {
// 				status = GIMP_PDB_CALLING_ERROR;
// 			}
// 			if ( status == GIMP_PDB_SUCCESS) {
// 				filterParm.radius = param[3].data.d_int32;
// 				filterParm.offset = param[4].data.d_int32;
// 			}
// 			break;
// 
// 		case GIMP_RUN_WITH_LAST_VALS:
// 			/*  Get options last values if needed  */
// 			gimp_get_data( "plug-in-gicu", &filterParm);
// 			break;
// 
// 		default:
// 			break;
// 	}

	cuda( drawable);

	gimp_displays_flush();
	gimp_drawable_detach( drawable);

	/*  Finally, set options in the core  */
	if ( run_mode == GIMP_RUN_INTERACTIVE) {
		gimp_set_data( "plug-in-gicu", &filterParm, sizeof( FilterParameter));
	}

}


static void cuda( GimpDrawable *drawable) {
	gint         channels;
	gint         x1, y1, x2, y2;
	GimpPixelRgn rgn_in, rgn_out;
	gint         width, height;
	gint         radius = 3;

	size_t size;
	guchar *h_image;
	guchar *d_image;

	cuda_filter cuda_filter;

	cuda_filter = GREY;
// 	cuda_filter = BOX;


	gimp_progress_init( "CUDA-Filter...");

	gimp_drawable_mask_bounds (
			drawable->drawable_id,
			&x1, &y1,
			&x2, &y2);

	channels = gimp_drawable_bpp( drawable->drawable_id);
	width = x2 - x1;
	height = y2 - y1;
	size = width * height * channels * ( 2 * radius);

	/* read data (image) from here */
	gimp_pixel_rgn_init(
			&rgn_in,
			drawable,
			x1, y1,
			width, height,
			FALSE, FALSE);

	/* write new image to here
	 * and use shadow tiles if ! preview
	 */
	gimp_pixel_rgn_init(
			&rgn_out,
			drawable,
			x1, y1,
			width, height,
			TRUE, TRUE);

	/* allocate mem on the gpu */
	cutilSafeCall( cudaMalloc( (void**)&d_image, size));

	/* where the final image will be saved in the host memory */
	/* allocate mem for output image */
// 	cutilSafeCall( cudaMallocHost( (void**)&h_image, size));
	h_image = g_new( guchar, size);

	/* Version with takes radius into account */
	gimp_pixel_rgn_get_rect(
			&rgn_in,
			h_image,
			x1 - radius, y1 - radius,
			width + 2 * radius, height + 2 * radius);

	/* TEST
	 * can you get pixel that aren't in the region
	 * yes you can!
	guchar *pixel = g_new( guchar, 1);
	gimp_pixel_rgn_get_pixel(&rgn_in,pixel, x1-1,y1);
	g_message("bla: %d\n", *pixel);
	*/

	gimp_progress_init( "CUDA-Gimp-Plug-In");
	gimp_progress_pulse( );
	
	/* CUDA cp image */
	gimp_progress_set_text( "Copying the image to graphics card");
	cutilSafeCall( (cudaMemcpy( d_image, h_image, size, cudaMemcpyHostToDevice)));

	/* timer */
// 	GTimer timer = g_timer_new time( );

	/* CUDA running kernel */
	gimp_progress_set_text( "Running CUDA-Kernel");
	filter( d_image, width, height, channels, cuda_filter);

// 	g_print ("blur() took %g seconds.\n", g_timer_elapsed (timer));
// 	g_timer_degstroy (timer);

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

