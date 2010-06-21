#include "gui.h"


gboolean gicu_dialog (GimpDrawable *drawable) {

	GtkWidget *dialog;
	GtkWidget *main_vbox;
	GtkWidget *main_hbox;
	GtkWidget *preview;
	GtkWidget *frame;
	GtkWidget *radius_label;
	GtkWidget *alignment;
	GtkWidget *spinbutton;
	GtkObject *spinbutton_adj;
	GtkWidget *frame_label;
	gboolean   run;

	gimp_ui_init ("gicu", FALSE);

	dialog = gimp_dialog_new (
			"CUDA-Filter parameter", "gicu",
			NULL, GTK_DIALOG_MODAL,
			gimp_standard_help_func, "plug-in-gicuu",
			GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
			GTK_STOCK_OK,     GTK_RESPONSE_OK,
			NULL);

	main_vbox = gtk_vbox_new (FALSE, 6);
	gtk_container_add (GTK_CONTAINER (GTK_DIALOG (dialog)->vbox), main_vbox);
	gtk_widget_show (main_vbox);

	preview = gimp_drawable_preview_new (drawable, &filterParm.preview);
	gtk_box_pack_start (GTK_BOX (main_vbox), preview, TRUE, TRUE, 0);
	gtk_widget_show (preview);

	frame = gimp_frame_new ("Blur radius");
	gtk_box_pack_start (GTK_BOX (main_vbox), frame, FALSE, FALSE, 0);
	gtk_widget_show (frame);

	alignment = gtk_alignment_new (0.5, 0.5, 1, 1);
	gtk_widget_show (alignment);
	gtk_container_add (GTK_CONTAINER (frame), alignment);
	gtk_alignment_set_padding (GTK_ALIGNMENT (alignment), 6, 6, 6, 6);

	main_hbox = gtk_hbox_new (FALSE, 12);
	gtk_container_set_border_width (GTK_CONTAINER (main_hbox), 12);
	gtk_widget_show (main_hbox);
	gtk_container_add (GTK_CONTAINER (alignment), main_hbox);

	radius_label = gtk_label_new_with_mnemonic ("_Radius:");
	gtk_widget_show (radius_label);
	gtk_box_pack_start (GTK_BOX (main_hbox), radius_label, FALSE, FALSE, 6);
	gtk_label_set_justify (GTK_LABEL (radius_label), GTK_JUSTIFY_RIGHT);

	spinbutton = gimp_spin_button_new (&spinbutton_adj, filterParm.radius,
										1, 255, 1, 0, 0, 5, 0);
	gtk_box_pack_start (GTK_BOX (main_hbox), spinbutton, FALSE, FALSE, 0);
	gtk_widget_show (spinbutton);

	g_signal_connect_swapped (
			preview, "invalidated",
			G_CALLBACK (cuda),
			drawable);
	g_signal_connect_swapped (
			spinbutton_adj, "value_changed",
			G_CALLBACK (gimp_preview_invalidate),
			preview);

	cuda (drawable, GIMP_PREVIEW (preview));

	g_signal_connect (
			spinbutton_adj, "value_changed",
			G_CALLBACK (gimp_int_adjustment_update),
			&filterParm.radius);

	gtk_widget_show (dialog);

	run = (gimp_dialog_run (GIMP_DIALOG (dialog)) == GTK_RESPONSE_OK);

	gtk_widget_destroy (dialog);

	return run;


}
