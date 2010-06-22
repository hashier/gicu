#include "gui.h"

static void add( GtkWidget *w) {

}

gboolean gicu_dialog (GimpDrawable *drawable) {

	GtkWidget *dialog;
	GtkWidget *main_vbox;
	GtkWidget *main_hbox;
	GtkWidget *preview;
	GtkWidget *frame;
	GtkWidget *label_radius;
	GtkWidget *label_offset;
	GtkWidget *alignment;
	GtkWidget *spinbutton_radius;
	GtkWidget *spinbutton_offset;
	GtkObject *spinbutton_adj_radius;
	GtkObject *spinbutton_adj_offset;
	gboolean   run;
	GtkWidget *combo_box;


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
	gtk_widget_show( label_radius);

	spinbutton_radius = gimp_spin_button_new (
			&spinbutton_adj_radius, filterParm.radius,
			1, 255, 1, 0, 0, 5, 0);
	gtk_box_pack_start( GTK_BOX( main_hbox), spinbutton_radius, FALSE, FALSE, 0);
	gtk_widget_show( spinbutton_radius);

	label_offset = gtk_label_new_with_mnemonic( "_Offset:");
	gtk_box_pack_start( GTK_BOX( main_hbox), label_offset, FALSE, FALSE, 5);
	gtk_label_set_justify( GTK_LABEL( label_offset), GTK_JUSTIFY_RIGHT);
	gtk_widget_show( label_offset);

	spinbutton_offset = gimp_spin_button_new (
			&spinbutton_adj_offset, filterParm.offset,
			1, 255, 1, 0, 0, 5, 0);
	gtk_box_pack_start( GTK_BOX( main_hbox), spinbutton_offset, FALSE, FALSE, 0);
	gtk_widget_show( spinbutton_offset);

	/*
	combo_box = gtk_combo_box_new_text( );
	gtk_combo_box_append_text( GTK_COMBO_BOX( combo_box), "Grey");
	gtk_combo_box_append_text( GTK_COMBO_BOX( combo_box), "Box");
	gtk_combo_box_append_text( GTK_COMBO_BOX( combo_box), "Sobel");
	gtk_combo_box_append_text( GTK_COMBO_BOX( combo_box), "Average");
	*/
	combo_box = gimp_int_combo_box_new (
			("Grey"), GREY,
			("Box"), BOX,
			("Sobel"), SOBEL,
			("Average"), AVERAGE,
			NULL);


	gtk_box_pack_start( GTK_BOX( main_vbox), combo_box, FALSE, FALSE, 0);
	gtk_widget_show( combo_box);


	/* Wenn das Bild invalid ist, cuda ausfuehren */
	g_signal_connect_swapped (
			preview, "invalidated",
			G_CALLBACK( cuda),
			drawable);


	/* wenn radius sich aendert, Bild invalid markieren */
	g_signal_connect_swapped (
			spinbutton_adj_radius, "value_changed",
			G_CALLBACK( gimp_preview_invalidate),
			preview);
	/* wenn offset sich aendert, Bild invalid markieren */
	g_signal_connect_swapped (
			spinbutton_adj_offset, "value_changed",
			G_CALLBACK( gimp_preview_invalidate),
			preview);
	g_signal_connect_swapped (
			combo_box, "changed",
			G_CALLBACK( gimp_preview_invalidate),
			preview);


	/* Wenn sich radius aendert, es in filterParm speichern */
	g_signal_connect (
			spinbutton_adj_radius, "value_changed",
			G_CALLBACK( gimp_int_adjustment_update),
			&filterParm.radius);
	/* Wenn sich offset aendert, es in filterParm speichern */
	g_signal_connect (
			spinbutton_adj_offset, "value_changed",
			G_CALLBACK( gimp_int_adjustment_update),
			&filterParm.offset);

	/* Set signals to the combobox */
	/* Wenn sich wert aendert, filterparm speicher */
	gimp_int_combo_box_connect(
			GIMP_INT_COMBO_BOX( combo_box), 1, /* Default active value */
			G_CALLBACK( gimp_int_combo_box_get_active), &filterParm.cuda_filter);

	cuda( drawable, GIMP_PREVIEW( preview));

	gtk_widget_show( dialog);

	run = ( gimp_dialog_run( GIMP_DIALOG( dialog)) == GTK_RESPONSE_OK);

	gtk_widget_destroy( dialog);

	return run;

}
