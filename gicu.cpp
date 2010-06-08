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
#include "cuda.h"
#include "gicu.h"

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

	GimpDrawable     *drawable;

	/* Setting mandatory output values */
	*nreturn_vals = 1;
	*return_vals = values;

	values[0].type = GIMP_PDB_STATUS;
	values[0].data.d_status = status;

	/* Getting run_mode */
	run_mode = (GimpRunMode)param[0].data.d_int32;

	/*
	if ( run_mode != GIMP_RUN_NONINTERACTIVE)
		g_message("values = return werte!\n\
param = uebergebene parameter und der 0te ist run-mode");
	*/


	int deviceCount = 5;
	cudaGetDeviceCount(&deviceCount);

	/* Prints the num of all avail. CUDA cards */
	/*
	g_message("blub: %d\n", deviceCount);
	*/

	unsigned char *speicher;
	size_t size;
	size = 5 * 5 * sizeof (unsigned char);
	cutilSafeCall(cudaMallocHost((void**)&speicher, size));

	drawable = gimp_drawable_get( param[2].data.d_drawable);

	gimp_progress_init ("My Blur...");

	blur (drawable);

	gimp_displays_flush ();
	gimp_drawable_detach (drawable);

	test();

}


static void blur (GimpDrawable *drawable) {
    gint         i, j, k, channels;
    gint         x1, y1, x2, y2;
    GimpPixelRgn rgn_in, rgn_out;
    guchar      *row1, *row2, *row3;
    guchar      *outrow;

    gimp_drawable_mask_bounds (drawable->drawable_id,
            &x1, &y1,
            &x2, &y2);
    channels = gimp_drawable_bpp (drawable->drawable_id);

    gimp_pixel_rgn_init (&rgn_in,
            drawable,
            x1, y1,
            x2 - x1, y2 - y1,
            FALSE, FALSE);
    gimp_pixel_rgn_init (&rgn_out,
            drawable,
            x1, y1,
            x2 - x1, y2 - y1,
            TRUE, TRUE);

    /* Initialise enough memory for row1, row2, row3, outrow */
    row1 = g_new (guchar, channels * (x2 - x1));
    row2 = g_new (guchar, channels * (x2 - x1));
    row3 = g_new (guchar, channels * (x2 - x1));
    outrow = g_new (guchar, channels * (x2 - x1));

    for (i = y1; i < y2; i++)
    {
        /* Get row i-1, i, i+1 */
        gimp_pixel_rgn_get_row (&rgn_in,
                row1,
                x1, MAX (y1, i - 1),
                x2 - x1);
        gimp_pixel_rgn_get_row (&rgn_in,
                row2,
                x1, i,
                x2 - x1);
        gimp_pixel_rgn_get_row (&rgn_in,
                row3,
                x1, MIN (y2 - 1, i + 1),
                x2 - x1);

        for (j = x1; j < x2; j++)
        {
            /* For each layer, compute the average of the nine
             * pixels */
            for (k = 0; k < channels; k++)
            {
                int sum = 0;
                sum = row1[channels * MAX ((j - 1 - x1), 0) + k]           +
                    row1[channels * (j - x1) + k]                        +
                    row1[channels * MIN ((j + 1 - x1), x2 - x1 - 1) + k] +
                    row2[channels * MAX ((j - 1 - x1), 0) + k]           +
                    row2[channels * (j - x1) + k]                        +
                    row2[channels * MIN ((j + 1 - x1), x2 - x1 - 1) + k] +
                    row3[channels * MAX ((j - 1 - x1), 0) + k]           +
                    row3[channels * (j - x1) + k]                        +
                    row3[channels * MIN ((j + 1 - x1), x2 - x1 - 1) + k];
                outrow[channels * (j - x1) + k] = sum / 9;
            }

        }

        gimp_pixel_rgn_set_row (&rgn_out,
                outrow,
                x1, i,
                x2 - x1);

        if (i % 10 == 0)
            gimp_progress_update ((gdouble) (i - y1) / (gdouble) (y2 - y1));
    }

    g_free (row1);
    g_free (row2);
    g_free (row3);
    g_free (outrow);

    gimp_drawable_flush (drawable);
    gimp_drawable_merge_shadow (drawable->drawable_id, TRUE);
    gimp_drawable_update (drawable->drawable_id,
            x1, y1,
            x2 - x1, y2 - y1);
}

