#nvcc  -I/usr/include/gimp-2.0 -I/usr/include/gtk-2.0 -I/usr/include/glib-2.0 -I/usr/lib64/glib-2.0/include -I/usr/lib64/gtk-2.0/include -I/usr/include/atk-1.0 -I/usr/include/cairo -I/usr/include/pango-1.0 -I/usr/include/pixman-1 -I/usr/include/freetype2 -I/usr/include/libpng12   -o /home/hashier/.gimp-2.6/plug-ins/gicu gicu.c  -lgimpui-2.0 -lgimpwidgets-2.0 -lgimpmodule-2.0 -lgimp-2.0 -lgimpmath-2.0 -lgimpconfig-2.0 -lgimpcolor-2.0 -lgimpbase-2.0 -lgtk-x11-2.0 -lgdk-x11-2.0 -latk-1.0 -lgio-2.0 -lpangoft2-1.0 -lgdk_pixbuf-2.0 -lpangocairo-1.0 -lcairo -lpango-1.0 -lfreetype -lfontconfig -lgobject-2.0 -lgmodule-2.0 -lglib-2.0
nvcc -I/opt/cuda/include/ -I/opt/cuda/sdk/C/common/inc -I/usr/include/gimp-2.0 -I/usr/include/gtk-2.0 -I/usr/include/glib-2.0 -I/usr/lib64/glib-2.0/include -I/usr/lib64/gtk-2.0/include -I/usr/include/atk-1.0 -I/usr/include/cairo -I/usr/include/pango-1.0 -I/usr/include/pixman-1 -I/usr/include/freetype2 -I/usr/include/libpng12   -o /home/hashier/.gimp-2.6/plug-ins/gicu cuda.cu gicu.cpp  -lgimpui-2.0 -lgimpwidgets-2.0 -lgimpmodule-2.0 -lgimp-2.0 -lgimpmath-2.0 -lgimpconfig-2.0 -lgimpcolor-2.0 -lgimpbase-2.0 -lgtk-x11-2.0 -lgdk-x11-2.0 -latk-1.0 -lgio-2.0 -lpangoft2-1.0 -lgdk_pixbuf-2.0 -lpangocairo-1.0 -lcairo -lpango-1.0 -lfreetype -lfontconfig -lgobject-2.0 -lgmodule-2.0 -lglib-2.0 -L/opt/cuda/lib64 -L/opt/cuda/sdk/C/lib -lcutil -lcudart

