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
static void blur (GimpDrawable *drawable);


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

}

static void blur (GimpDrawable *drawable) {
    gint         i, j, k, channels;
    gint         x1, y1, x2, y2;
    GimpPixelRgn rgn_in, rgn_out;
    guchar       output[4];

    /* Gets upper left and lower right coordinates,
     * and layers number in the image */
    gimp_drawable_mask_bounds (drawable->drawable_id,
            &x1, &y1,
            &x2, &y2);
    channels = gimp_drawable_bpp (drawable->drawable_id);

g_print("x1: %d\ny1: %d\nx2: %d\ny2: %d\n", x1, y1, x2, y2);

x2 += 20;
y2 += 20;

g_print("x1: %d\ny1: %d\nx2: %d\ny2: %d\n\n", x1, y1, x2, y2);


    /* Initialises two PixelRgns, one to read original data,
     * and the other to write output data. That second one will
     * be merged at the end by the call to
     * gimp_drawable_merge_shadow() */
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

    for (i = x1; i < x2; i++)
    {
        for (j = y1; j < y2; j++)
        {
            guchar pixel[9][4];

            /* Get nine pixels */
            gimp_pixel_rgn_get_pixel (&rgn_in,
                    pixel[0],
                    MAX (i - 1, x1),
                    MAX (j - 1, y1));
            gimp_pixel_rgn_get_pixel (&rgn_in,
                    pixel[1],
                    MAX (i - 1, x1),
                    j);
            gimp_pixel_rgn_get_pixel (&rgn_in,
                    pixel[2],
                    MAX (i - 1, x1),
                    MIN (j + 1, y2 - 1));

            gimp_pixel_rgn_get_pixel (&rgn_in,
                    pixel[3],
                    i,
                    MAX (j - 1, y1));
            gimp_pixel_rgn_get_pixel (&rgn_in,
                    pixel[4],
                    i,
                    j);
            gimp_pixel_rgn_get_pixel (&rgn_in,
                    pixel[5],
                    i,
                    MIN (j + 1, y2 - 1));

            gimp_pixel_rgn_get_pixel (&rgn_in,
                    pixel[6],
                    MIN (i + 1, x2 - 1),
                    MAX (j - 1, y1));
            gimp_pixel_rgn_get_pixel (&rgn_in,
                    pixel[7],
                    MIN (i + 1, x2 - 1),
                    j);
            gimp_pixel_rgn_get_pixel (&rgn_in,
                    pixel[8],
                    MIN (i + 1, x2 - 1),
                    MIN (j + 1, y2 - 1));

            /* For each layer, compute the average of the
             * nine */
            for (k = 0; k < channels; k++)
            {
                int tmp, sum = 0;
                for (tmp = 0; tmp < 9; tmp++)
                    sum += pixel[tmp][k];
                output[k] = sum / 9;
            }

            gimp_pixel_rgn_set_pixel (&rgn_out,
                    output,
                    i, j);
        }

        if (i % 10 == 0)
            gimp_progress_update ((gdouble) (i - x1) / (gdouble) (x2 - x1));
    }

    /*  Update the modified region */
    gimp_drawable_flush (drawable);
    gimp_drawable_merge_shadow (drawable->drawable_id, TRUE);
    gimp_drawable_update (drawable->drawable_id,
            x1, y1,
            x2 - x1, y2 - y1);
}

