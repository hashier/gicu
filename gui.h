#ifndef __gui_H_
#define __gui_H_


/* Gimp plug-in header */
#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>

#include "gicu.h"

/**
 * @brief Opens the pref. dialog
 *
 * param[in] drawable We'll get our image from this drawable
 *
 * In this dialog you can set radius/offset/value
 * depending on your choosen filter
 */
gboolean gicu_dialog (GimpDrawable *drawable);

#endif
