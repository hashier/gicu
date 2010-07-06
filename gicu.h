#ifndef __gicu_H_
#define __gicu_H_


/* Gimp plug-in header */
#include <libgimp/gimp.h>
#include <libgimpwidgets/gimpwidgetstypes.h>

/**
 * @mainpage Doxygen documentation for the gicu project
 *
 * This project was coded for<br>
 * <b>Praktische Aspekte der Informatik</b>
 */

/**
 * @brief All available filters
 *
 * In this enum all available filters are saved
 */
typedef enum _cuda_filter {
	GREY,
	SOBEL,
	BOXBIN,
	BOX,
	AVERAGE,
	AVERAGEBIN,
	TEST
} cuda_filter;


/**
 * @brief struct to save the parameters
 *
 * In this struct all information is stored which is needed to run the filters
 */
typedef struct _FilterParameter {
	/// The selected radius
	gint radius;
	/// The selected offset
	gint offset;
	/// Should be a preview shown
	gboolean preview;
	/// The CUDA-Filter
	enum _cuda_filter cuda_filter;
} FilterParameter;

extern FilterParameter filterParm;

/**
 * @brief this is run while gimps initialises the plug-ins
 *
 * This sets up the name of the plugin, help, input parameters and stuff like that
 */
void query( void);

/**
 * @brief this is the function which will be run, if the plugin is selected
 * @param[in] name The Name of the plug-in
 * @param[in] nparams Number of parameters
 * @param[in] param Pointer to a GimpParam
 * @param[in] nreturn_vals Number of return Values
 * @param[in] return_vals GimpParam Pointer to a GimpParam for the return values
 *
 * This is called everytime the plug-in is selected.
 * Whether through clicking on it in the menu or through a script.
 */
void run(
		const gchar     *name,
		gint            nparams,
		const GimpParam *param,
		gint            *nreturn_vals,
		GimpParam       **return_vals);

/**
 * @brief This handels all the CUDA stuff
 * @param[in] drawable Which GimpDrawable to use to read in the data
 * @param[in] preview  Where should the preview displayed
 *
 * Call this function to set up the CUDA stuff and run it
 */
void cuda( GimpDrawable* drawable, GimpPreview* preview);


#endif
