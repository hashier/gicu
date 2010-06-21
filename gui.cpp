#include "gui.h"

gint add (GtkObject a, gint *t) {
// 	g_print("test1: %d\n",*t);
// 	g_print("test2: %d\n",filterParm.radius);
	//*t = 3;
	//return *t + 1;
}
gboolean gicu_dialog (GimpDrawable *drawable) {

        GtkWidget *dialog;
        GtkWidget *main_vbox;
        GtkWidget *main_hbox;
        GtkWidget *frame;
        GtkWidget *radius_label;
        GtkWidget *alignment;
        GtkWidget *spinbutton;
        GtkObject *spinbutton_adj;
        GtkWidget *frame_label;
        gboolean   run;

        gimp_ui_init ("myblur", FALSE);

        dialog = gimp_dialog_new ("My blur", "myblur",
                                  NULL, GTK_DIALOG_MODAL,
                                  gimp_standard_help_func, "plug-in-myblur",

                                  GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
                                  GTK_STOCK_OK,     GTK_RESPONSE_OK,

                                  NULL);

        main_vbox = gtk_vbox_new (FALSE, 6);
        gtk_container_add (GTK_CONTAINER (GTK_DIALOG (dialog)->vbox), main_vbox);
        gtk_widget_show (main_vbox);

        frame = gtk_frame_new (NULL);
        gtk_widget_show (frame);
        gtk_box_pack_start (GTK_BOX (main_vbox), frame, TRUE, TRUE, 0);
        gtk_container_set_border_width (GTK_CONTAINER (frame), 6);

        alignment = gtk_alignment_new (0.5, 0.5, 1, 1);
        gtk_widget_show (alignment);
        gtk_container_add (GTK_CONTAINER (frame), alignment);
        gtk_alignment_set_padding (GTK_ALIGNMENT (alignment), 6, 6, 6, 6);

        main_hbox = gtk_hbox_new (FALSE, 0);
        gtk_widget_show (main_hbox);
        gtk_container_add (GTK_CONTAINER (alignment), main_hbox);

        radius_label = gtk_label_new_with_mnemonic ("_Radius:");
        gtk_widget_show (radius_label);
        gtk_box_pack_start (GTK_BOX (main_hbox), radius_label, FALSE, FALSE, 6);
        gtk_label_set_justify (GTK_LABEL (radius_label), GTK_JUSTIFY_RIGHT);

        spinbutton_adj = gtk_adjustment_new (filterParm.radius, 1, 255, 1, 0, 0);
        spinbutton = gtk_spin_button_new (GTK_ADJUSTMENT (spinbutton_adj), 1, 0);
        gtk_widget_show (spinbutton);
        gtk_box_pack_start (GTK_BOX (main_hbox), spinbutton, FALSE, FALSE, 6);
        gtk_spin_button_set_numeric (GTK_SPIN_BUTTON (spinbutton), TRUE);

        frame_label = gtk_label_new ("Modify radius");
        gtk_widget_show (frame_label);
        gtk_frame_set_label_widget (GTK_FRAME (frame), frame_label);
        gtk_label_set_use_markup (GTK_LABEL (frame_label), TRUE);

        g_signal_connect (spinbutton_adj, "value_changed",
                          G_CALLBACK (gimp_int_adjustment_update),
                          &filterParm.radius);
        g_signal_connect (spinbutton_adj, "value_changed",
                          G_CALLBACK (add),
                          &filterParm.radius);
        gtk_widget_show (dialog);

        run = (gimp_dialog_run (GIMP_DIALOG (dialog)) == GTK_RESPONSE_OK);

        gtk_widget_destroy (dialog);

        return run;

}
