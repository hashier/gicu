#ifndef __gui_H_
#define __gui_H_


/* Gimp plug-in header */
#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>

#include "gicu.h"

/**
 * @brief Opens the pref. dialog
 *
 * Here you can set radius/offset/value
 * depending on your choosen filter
 */
gboolean gicu_dialog (GimpDrawable *drawable);

/**
 * @brief Updates the spinners and labels
 *
 * This function has to be called every time the CUDA-Filter changes.
 * It disables unneeded options
 */
static void add( GtkWidget *w);


#endif
