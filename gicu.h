#ifndef __gicu_H_
#define __gicu_H_


/* Gimp plug-in header */
#include <libgimp/gimp.h>
#include <libgimpwidgets/gimpwidgetstypes.h>
#include "gui.h"

/**
 * @mainpage Doxygen documentation for the gicu project
 *
 * This project was coded for<br>
 * <H2><b>Praktische Aspekte der Informatik - SoSe 2010</b></H2>
 * <p>
 * Prof. Dr.-Ing. Marcus Magnor<br>
 * Christian Lipski
 *
 * <H2>How to implement another CUDA-Filter?</H2>
 * <p><p>
 * One has to alter 4 things in the Code
 * <p>
 * <b>1st:</b> Add an name to the enum _cuda_filter in gicu.h
 * <p>
 * <b>2nd:</b> Write the CUDA code in cuda.cu
 * <p>
 * <b>3rd:</b> Alter the update_gui function in gui.cpp and
 * <p>
 * <b>4th:</b> Add the new Filter name to the combobox so you can select it
 * <p>
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
 * @brief Typedef struct to save all the information about the filter
 *
 * In this struct all the information are stored that is needed to run the filters.
 * This includes radius, offset, preview on/off and the CUDA-Filter
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

/**
 * @brief This var holds the all the config information 
 *
 * For more information see: typedef struct _FilterParameter
 */
extern FilterParameter filterParm;

/**
 * @brief Stores pointers to the important functions
 *
 * Which function to run
 * - during initialization
 * - quiting gimp
 * - query
 * - run
 */
extern GimpPlugInInfo PLUG_IN_INFO;

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
