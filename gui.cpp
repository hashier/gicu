#include "gui.h"


static GtkWidget *label_offset;
static GtkWidget *label_radius;
static GtkWidget *spinbutton_radius;
static GtkWidget *spinbutton_offset;
static GtkWidget *combo_box;

/**
 * @brief Updates the gui (spinners, labels etc.)
 *
 * param[in] w the GtkWidget which caused the "trouble" (;
 *
 * This function has to be called every time the CUDA-Filter changes.
 * It disables unneeded options
 */
static void update_gui( GtkWidget *w) {

	int value = -1;

	gimp_int_combo_box_get_active( GIMP_INT_COMBO_BOX( combo_box), &value);
	switch ( value) {
		case GREY:
			gtk_label_set_text_with_mnemonic( GTK_LABEL( label_radius), "_Value");
			gtk_widget_set_visible( label_offset, FALSE);
			gtk_widget_set_visible( spinbutton_offset, FALSE);
			gtk_widget_set_visible( label_radius, TRUE);
			gtk_widget_set_visible( spinbutton_radius, TRUE);
			break;

		case SOBEL:
			gtk_widget_set_visible( label_offset, FALSE);
			gtk_widget_set_visible( spinbutton_offset, FALSE);
			gtk_widget_set_visible( label_radius, FALSE);
			gtk_widget_set_visible( spinbutton_radius, FALSE);
			break;

		case BOXBIN:
			gtk_label_set_text_with_mnemonic( GTK_LABEL( label_radius), "_Radius");
			gtk_label_set_text_with_mnemonic( GTK_LABEL( label_offset), "_Offset");
			gtk_widget_set_visible( label_offset, TRUE);
			gtk_widget_set_visible( spinbutton_offset, TRUE);
			gtk_widget_set_visible( label_radius, TRUE);
			gtk_widget_set_visible( spinbutton_radius, TRUE);
			break;

		case BOX:
			gtk_label_set_text_with_mnemonic( GTK_LABEL( label_radius), "_Radius");
			gtk_label_set_text_with_mnemonic( GTK_LABEL( label_offset), "_Offset");
			gtk_widget_set_visible( label_offset, FALSE);
			gtk_widget_set_visible( spinbutton_offset, FALSE);
			gtk_widget_set_visible( label_radius, TRUE);
			gtk_widget_set_visible( spinbutton_radius, TRUE);
			break;

		case TEST:
			gtk_label_set_text_with_mnemonic( GTK_LABEL( label_radius), "_Radius");
			gtk_label_set_text_with_mnemonic( GTK_LABEL( label_offset), "_Offset");
			gtk_widget_set_visible( label_offset, FALSE);
			gtk_widget_set_visible( spinbutton_offset, FALSE);
			gtk_widget_set_visible( label_radius, FALSE);
			gtk_widget_set_visible( spinbutton_radius, FALSE);
			break;

		case AVERAGE:
			filterParm.radius = 7;
			gtk_label_set_text_with_mnemonic( GTK_LABEL( label_radius), "_Radius = FIX 7!");
			gtk_label_set_text_with_mnemonic( GTK_LABEL( label_offset), "_Offset");
			gtk_widget_set_visible( label_offset, FALSE);
			gtk_widget_set_visible( spinbutton_offset, FALSE);
			gtk_widget_set_visible( label_radius, TRUE);
			gtk_widget_set_visible( spinbutton_radius, FALSE);
			break;

		case AVERAGEBIN:
			filterParm.radius = 7;
			gtk_label_set_text_with_mnemonic( GTK_LABEL( label_radius), "_Radius = FIX 7!");
			gtk_label_set_text_with_mnemonic( GTK_LABEL( label_offset), "_Offset");
			gtk_widget_set_visible( label_offset, TRUE);
			gtk_widget_set_visible( spinbutton_offset, TRUE);
			gtk_widget_set_visible( label_radius, TRUE);
			gtk_widget_set_visible( spinbutton_radius, FALSE);
			break;

		default:
			gtk_label_set_text_with_mnemonic( GTK_LABEL( label_radius), "_Radius");
			gtk_label_set_text_with_mnemonic( GTK_LABEL( label_offset), "_Offset");
			gtk_widget_set_visible( label_offset, FALSE);
			gtk_widget_set_visible( spinbutton_offset, FALSE);
			gtk_widget_set_visible( label_radius, TRUE);
			gtk_widget_set_visible( spinbutton_radius, TRUE);
			break;
	}
}

gboolean gicu_dialog (GimpDrawable *drawable) {

	GtkWidget *dialog;
	GtkWidget *main_vbox;
	GtkWidget *main_hbox;
	GtkWidget *preview;
	GtkWidget *frame;
	GtkWidget *alignment;
	GtkObject *spinbutton_adj_radius;
	GtkObject *spinbutton_adj_offset;
	gboolean   run;

	int x1, y1, x2, y2, width;


	gimp_ui_init ("gicu", FALSE);

	dialog = gimp_dialog_new (
			"CUDA-Filter parameter", "gicu",
			NULL, GTK_DIALOG_MODAL,
			gimp_standard_help_func, "plug-in-gicu",
			GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
			GTK_STOCK_OK,     GTK_RESPONSE_OK,
			NULL);

	main_vbox = gtk_vbox_new( FALSE, 5);
	gtk_container_add( GTK_CONTAINER( GTK_DIALOG( dialog)->vbox), main_vbox);
	gtk_widget_show (main_vbox);

	preview = gimp_drawable_preview_new( drawable, &filterParm.preview);
	gtk_box_pack_start( GTK_BOX( main_vbox), preview, FALSE, FALSE, 0);
	gtk_widget_show( preview);

	frame = gimp_frame_new( "CUDA-Filter-Options");
	gtk_box_pack_start( GTK_BOX( main_vbox), frame, FALSE, FALSE, 0);
	gtk_widget_show( frame);

	alignment = gtk_alignment_new( 0.5, 0.5, 1, 1);
	gtk_container_add( GTK_CONTAINER ( frame), alignment);
	gtk_alignment_set_padding( GTK_ALIGNMENT( alignment), 5, 5, 5, 5);
	gtk_widget_show( alignment);

	main_hbox = gtk_hbox_new( FALSE, 5);
	gtk_container_set_border_width( GTK_CONTAINER( main_hbox), 5);
	gtk_container_add( GTK_CONTAINER( alignment), main_hbox);
	gtk_widget_show( main_hbox);

	label_radius = gtk_label_new_with_mnemonic( "_Radius:");
	gtk_box_pack_start( GTK_BOX( main_hbox), label_radius, FALSE, FALSE, 5);
	gtk_label_set_justify( GTK_LABEL( label_radius), GTK_JUSTIFY_RIGHT);
// 	gtk_widget_show( label_radius);

	spinbutton_radius = gimp_spin_button_new (
			&spinbutton_adj_radius, filterParm.radius,
			1, 255, 1, 0, 0, 5, 0);
	gtk_box_pack_start( GTK_BOX( main_hbox), spinbutton_radius, FALSE, FALSE, 0);
// 	gtk_widget_show( spinbutton_radius);

	label_offset = gtk_label_new_with_mnemonic( "_Offset:");
	gtk_box_pack_start( GTK_BOX( main_hbox), label_offset, FALSE, FALSE, 5);
	gtk_label_set_justify( GTK_LABEL( label_offset), GTK_JUSTIFY_RIGHT);
// 	gtk_widget_show( label_offset);

	spinbutton_offset = gimp_spin_button_new (
			&spinbutton_adj_offset, filterParm.offset,
			0, 255, 1, 0, 0, 5, 0);
	gtk_box_pack_start( GTK_BOX( main_hbox), spinbutton_offset, FALSE, FALSE, 0);
// 	gtk_widget_show( spinbutton_offset);

	combo_box = gimp_int_combo_box_new (
			("Grey"), GREY,
			("Sobel"), SOBEL,
			("Boxing"), BOX,
			("Boxing (Bin)"), BOXBIN,
			("Fast Average"), AVERAGE,
			("Fast Average (Bin)"), AVERAGEBIN,
			("Test"), TEST,
			NULL);
	gtk_box_pack_start( GTK_BOX( main_vbox), combo_box, FALSE, FALSE, 0);
	gtk_widget_show( combo_box);


	/* Wenn das Bild invalid ist, cuda ausfuehren */
	g_signal_connect_swapped (
			preview, "invalidated",
			G_CALLBACK( cuda),
			drawable);

	/* wenn sich radius/offset/... aendert, Bild invalid markieren */
	g_signal_connect_swapped (
			spinbutton_adj_radius, "value_changed",
			G_CALLBACK( gimp_preview_invalidate),
			preview);
	g_signal_connect_swapped (
			spinbutton_adj_offset, "value_changed",
			G_CALLBACK( gimp_preview_invalidate),
			preview);
	g_signal_connect_swapped (
			combo_box, "changed",
			G_CALLBACK( gimp_preview_invalidate),
			preview);

	/* Wenn sich .... aendert, es auch in filterParm speichern */
	g_signal_connect (
			spinbutton_adj_radius, "value_changed",
			G_CALLBACK( gimp_int_adjustment_update),
			&filterParm.radius);
	g_signal_connect (
			spinbutton_adj_offset, "value_changed",
			G_CALLBACK( gimp_int_adjustment_update),
			&filterParm.offset);
	/* Set signals to the combobox */
	/* Wenn sich wert aendert, filterparm speicher */
	gimp_int_combo_box_connect(
			GIMP_INT_COMBO_BOX( combo_box), 1, /* Default active value */
			G_CALLBACK( gimp_int_combo_box_get_active), &filterParm.cuda_filter);
	g_signal_connect (
			GIMP_INT_COMBO_BOX( combo_box), "changed",
			G_CALLBACK( update_gui),
			NULL);


	cuda( drawable, GIMP_PREVIEW( preview));

	gtk_widget_show( dialog);

	run = ( gimp_dialog_run( GIMP_DIALOG( dialog)) == GTK_RESPONSE_OK);

	gimp_drawable_mask_bounds(
			drawable->drawable_id,
			&x1, &y1,
			&x2, &y2);
	width  = x2 - x1;

	if ( width % 4 ) {
		if ( filterParm.cuda_filter == AVERAGE || filterParm.cuda_filter == AVERAGEBIN) {
			g_message("Width is not a multiply of 4\n");
			run = FALSE;
		}
	}

	gtk_widget_destroy( dialog);

	return run;

}
