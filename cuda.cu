#include "cuda.h"

// Texture reference for reading image
texture<guchar, 2> tex;

// arrays are optimized for 2D access so we'll use arrays
// insted of single row pointer memory addresses
cudaArray *array = NULL;

// Shared Mem on the dev is declared with __shared__
extern __shared__ unsigned char LocalBlock[];


double fScale = 1.0;


void filter(
		guchar *d_image, gint width, gint height,
		guint channels, guchar *d_image_temp) {

	int add = 0;
	guint step = channels * width;

	if ( width % numThreads || height % numThreads) {
		add = 1;
	}

	switch ( filterParm.cuda_filter) {
		case GREY:
			grey<<< height, 384, 0 >>>( d_image, width, height, channels, step, filterParm);
			break;

		case BOXBIN: {
			d_boxfilter_x_tex<<< height / numThreads +add, numThreads >>>( d_image_temp, width, height, filterParm.radius);
			d_boxfilter_y_global<<< width / numThreads +add, numThreads >>>( d_image_temp, d_image, width, height, filterParm.radius, filterParm.offset, TRUE);
		}
			break;

		case SOBEL:
			sobelTex<<< height, 384, 0 >>>( d_image, width, height, channels, step, filterParm);
			break;

		case BOX: {
			d_boxfilter_x_tex<<< height / numThreads +add, numThreads >>>( d_image_temp, width, height, filterParm.radius);
			d_boxfilter_y_global<<< width / numThreads +add, numThreads >>>( d_image_temp, d_image, width, height, filterParm.radius, filterParm.offset, FALSE);
		}
			break;

		case TEST:
			test<<< height, 384, 0 >>>( d_image, width, height, channels, step, filterParm);
			break;

		case AVERAGE:
		case AVERAGEBIN: {
			if ( width % 4) {
				g_message("Width is not a multiply of 4\nThe resulting image will probably be f*ed up\n");
			}

			dim3 threads(16,16);
			int BlockWidth = 32;

			// 0==0 -> 1      0!=0 -> 0
			// Image 1280 x 1024 -> 16,4 ThreadDim and 4,256 BlockDim
			// SharedPitch -> 384
			// sharedMem -> 2304
			// SharedPitch ist durch 64Teilbar, Rechnung siehe unten bei iw&=~3
			// 16 Threads * 4 Pixel = 64
			dim3 blocks = dim3(width/(4*BlockWidth)+(0!=width%(4*BlockWidth)),
					height/threads.y+(0!=height%threads.y));
			int SharedPitch = ~0x3f&(4*(BlockWidth+2*filterParm.radius)+0x3f);
			int sharedMem = SharedPitch*(threads.y+2*filterParm.radius);

			// for the shared kernel, width must be divisible by 4
			// 3 dec = 11bin
			// ~3 dec = 00bin
			// letze beiden stellen 0 -> durch 4 teilbar
			width &= ~3;

			if ( filterParm.radius != 7 )
				printf("Wegen Optimierung ist nur Radius 7 Erlaubt in diesem Modus: Das zu sehende Bidl wird DEFINITIV Fehler enthalten\n");

			printf("AVERAGE_FAST: radius: %d  offset: %d  threads: %d,%d  blocks: %d,%d  step: %d  sharedMem: %d  width: %d\n",
					filterParm.radius, filterParm.offset, threads.x, threads.y, blocks.x, blocks.y, SharedPitch, sharedMem, width);

			AVGShared<<<blocks, threads, sharedMem>>>((uchar4 *) d_image,
					width,
					BlockWidth, SharedPitch,
					width, height, fScale,
					filterParm.radius, filterParm.offset, ( filterParm.cuda_filter == AVERAGEBIN) );
		}

			break;

		default:
			g_printerr("Filter not found");
			break;
	}

}

extern "C" void cuda_init( ) {
}

extern "C" void setupTexture( gint width, gint height) {
	cudaChannelFormatDesc desc;

// 	desc = cudaCreateChannelDesc<unsigned char>();
	int e = (int)sizeof( guchar) * 8;
	desc = cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);

	cutilSafeCall(cudaMallocArray( &array, &desc, width, height));
}

extern "C" void bindTexture( ) {
	/* clamp x and y axis to the boarder */
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false;

	cutilSafeCall( cudaBindTextureToArray( tex, array));
}

extern "C" void updateTexture( gint width, gint height, guchar *data, gint channels) {
	cutilSafeCall(cudaMemcpyToArray(
			array,
			0, 0, /* 0 0 <- h und w offset */
			data,
			channels * sizeof( guchar) * width * height, cudaMemcpyHostToDevice));
}

extern "C" void unbindTexture( ) {
	cutilSafeCall( cudaUnbindTexture( tex));
}

extern "C" void deleteTexture( ) {
	cutilSafeCall( cudaFreeArray( array));
}


/*
 * ALL the CUDA Functions
 */


__device__ unsigned char ComputeSobel(
		unsigned char ul, // upper left
		unsigned char um, // upper middle
		unsigned char ur, // upper right
		unsigned char ml, // middle left
		unsigned char mm, // middle (unused)
		unsigned char mr, // middle right
		unsigned char ll, // lower left
		unsigned char lm, // lower middle
		unsigned char lr, // lower right
		float fScale )
{
    short Horz = ul + 2*ml + ll - ur - 2*mr - lr;
    short Vert = ul + 2*um + ur - ll - 2*lm - lr;

    short Sum = (short) (fScale*(abs(Horz)+abs(Vert)));
    if ( Sum < 0 ) return 0; else if ( Sum > 0xff ) return 0xff;
    return (unsigned char) Sum;
}

__global__ void
sobelTex( guchar *d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm) {

	double fScale = 1.0;

	unsigned char *pSobel =
			(unsigned char *) (((char *) d_image)+blockIdx.x*step);
	for ( int i = threadIdx.x; i < width; i += blockDim.x ) {
		unsigned char pix00 = tex2D( tex, (float) i-1, (float) blockIdx.x-1 );
		unsigned char pix01 = tex2D( tex, (float) i+0, (float) blockIdx.x-1 );
		unsigned char pix02 = tex2D( tex, (float) i+1, (float) blockIdx.x-1 );
		unsigned char pix10 = tex2D( tex, (float) i-1, (float) blockIdx.x+0 );
		unsigned char pix11 = tex2D( tex, (float) i+0, (float) blockIdx.x+0 );
		unsigned char pix12 = tex2D( tex, (float) i+1, (float) blockIdx.x+0 );
		unsigned char pix20 = tex2D( tex, (float) i-1, (float) blockIdx.x+1 );
		unsigned char pix21 = tex2D( tex, (float) i+0, (float) blockIdx.x+1 );
		unsigned char pix22 = tex2D( tex, (float) i+1, (float) blockIdx.x+1 );
		pSobel[i] = ComputeSobel(
				pix00, pix01, pix02,
				pix10, pix11, pix12,
				pix20, pix21, pix22,
				fScale );
	}

}

__global__ void
box( guchar *d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm) {

	// blockIdx.x * Pitch (image.width) = Startpointer auf die Idx.x te Zeile
	unsigned char *p =
			(unsigned char *) (((char *) d_image)+blockIdx.x*step);
	int b = 0;

	for ( int i = threadIdx.x; i < width; i += blockDim.x ) {
		if(b==1) {
			if(blockIdx.x % 2 ) {
				p[i] = 255;
			} else {
				p[i] = 0;
			}
		} else {
			if(blockIdx.x % 2 ) {
				p[i] = 0;
			} else {
				p[i] = 255;
			}
		}
		b=1;
	}

}

__global__ void
test( guchar *d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm) {

	// blockIdx.x * Pitch (image.width) = Startpointer auf die Idx.x te Zeile
	unsigned char *p =
			(unsigned char *) (((char *) d_image)+blockIdx.x*step);
	int b = 0;

	for ( int i = threadIdx.x; i < width; i += blockDim.x ) {
		if(b==1) {
			if(blockIdx.x % 2 ) {
				p[i] = 255;
			} else {
				p[i] = 0;
			}
		} else {
			if(blockIdx.x % 2 ) {
				p[i] = 0;
			} else {
				p[i] = 255;
			}
		}
		b=1;
	}

}

/* OLD OUTDATED CODE */
// __global__ void greyRGB( guchar* d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm) {
// 
// 	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
// 	x *= channels;
// 	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
// 	d_image[y*step+x]   = 128;
// 	d_image[y*step+x+1] = 128;
// 	d_image[y*step+x+2] = 128;
// 
// }

__global__ void
grey( guchar* d_image, gint width, gint height, guint channels, guint step, FilterParameter filterParm) {

	for ( int i = threadIdx.x; i < width; i += blockDim.x ) {
		d_image[blockIdx.x*step+i] = filterParm.radius;
	}

}

__global__ void
sobelSharedTex(
		uchar4 *pSobelOriginal, unsigned short SobelPitch,
		short BlockWidth, short SharedPitch,
		short w, short h, float fScale, FilterParameter filterParm ) {


	int Radius = filterParm.radius;
	// pSobelOriginal > Pointer auf den Speicher in der Graka
	// SobelPitch > 1280 (BilderBreite)
	// BlockWidth > 80
	// SharedPitch > 384 // Zeilenlaenge
	// w > 1280
	// h > 1024
	// sharedMem > 2304 >> 48 * 48 = 2304
	// threads > 16,4
	// block   > 4,256
	// Radius 1 = Ich brauche links/rechts/oben/unten 1 extra pixel
	// radius darf max die haelfte des Blockes sein in x und y

	// u und v sind die KOs des Pixels, das ich kopieren will
	// u ist 4*80 = 320 -> 4*320 = 1280  -->  Der 320er Anfang jedes Blockes
	// auf u (anfang des 320er Blockes) muss dann noch der Zu nehmende Pixel addiert werden
	short u = 4*blockIdx.x*BlockWidth;
	short v = blockIdx.y*blockDim.y + threadIdx.y;
	short ib;

	// SharedIdx > Zeilenanfang vom SharedMem
	// 384 > Zeilenbreite vom SharedMem
	int SharedIdx = threadIdx.y * SharedPitch;

	// ib geht komplett durch von 0-81
	// ib geht 16er schritte
	// 4*ib = 64
	// damit hat man einheitliches lesen
	// t0 liest 4byte
	// t1 liest 4byte
	// -> 16Threads a 4byte = 64byte
	for ( ib = threadIdx.x; ib < BlockWidth+2*Radius; ib += blockDim.x ) {
		LocalBlock[SharedIdx+4*ib+0] = tex2D( tex,
				(float) (u+4*ib-Radius+0), (float) (v-Radius) );
		LocalBlock[SharedIdx+4*ib+1] = tex2D( tex,
				(float) (u+4*ib-Radius+1), (float) (v-Radius) );
		LocalBlock[SharedIdx+4*ib+2] = tex2D( tex,
				(float) (u+4*ib-Radius+2), (float) (v-Radius) );
		LocalBlock[SharedIdx+4*ib+3] = tex2D( tex,
				(float) (u+4*ib-Radius+3), (float) (v-Radius) );
	}
	if ( threadIdx.y < Radius*2 ) {
		//
		// copy trailing Radius*2 rows of pixels into shared
		//
		// bis v-Radius wurde alles kopiert, fehlt also noch 2*Radius
		// also alle Werte nochmal gleich nur SharedIdx anpassen auf die richte Stelle
		// und bei der y KO einfach blockDim.y uebrspringen weil die ja schon kopiert sind
		// v-Radius .. v lief ja bis blockDim.y-1 (-Radius)
		SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;
		for ( ib = threadIdx.x; ib < BlockWidth+2*Radius; ib += blockDim.x ) {
			LocalBlock[SharedIdx+4*ib+0] = tex2D( tex,
					(float) (u+4*ib-Radius+0), (float) (v+blockDim.y-Radius) );
			LocalBlock[SharedIdx+4*ib+1] = tex2D( tex,
					(float) (u+4*ib-Radius+1), (float) (v+blockDim.y-Radius) );
			LocalBlock[SharedIdx+4*ib+2] = tex2D( tex,
					(float) (u+4*ib-Radius+2), (float) (v+blockDim.y-Radius) );
			LocalBlock[SharedIdx+4*ib+3] = tex2D( tex,
					(float) (u+4*ib-Radius+3), (float) (v+blockDim.y-Radius) );
		}
	}

	__syncthreads();

	u >>= 2;    // index as uchar4 from here
	uchar4 *pSobel = (uchar4 *) (((char *) pSobelOriginal)+v*SobelPitch);
	SharedIdx = threadIdx.y * SharedPitch;

	for ( ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x ) {

		unsigned char pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+0];
		unsigned char pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+1];
		unsigned char pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+2];
		unsigned char pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+0];
		unsigned char pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+1];
		unsigned char pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+2];
		unsigned char pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+0];
		unsigned char pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+1];
		unsigned char pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+2];

		uchar4 out;

		out.x = ComputeSobel(pix00, pix01, pix02,
				pix10, pix11, pix12,
				pix20, pix21, pix22, fScale );

		pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+3];
		pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+3];
		pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+3];
		out.y = ComputeSobel(pix01, pix02, pix00,
				pix11, pix12, pix10,
				pix21, pix22, pix20, fScale );

		pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+4];
		pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+4];
		pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+4];
		out.z = ComputeSobel( pix02, pix00, pix01,
				pix12, pix10, pix11,
				pix22, pix20, pix21, fScale );

		pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+5];
		pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+5];
		pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+5];
		out.w = ComputeSobel( pix00, pix01, pix02,
				pix10, pix11, pix12,
				pix20, pix21, pix22, fScale );
		if ( u+ib < w/4 && v < h ) {
			pSobel[u+ib] = out;
		}
	}

	__syncthreads();
}

// texture version
// texture fetches automatically clamp to edge of image
__global__ void
d_boxfilter_x_tex( guchar *od, int w, int h, int r) {
	float scale = 1.0f / (2*r+1);
	unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;

	// In der Reihe den Vordersten Pixel berechnen
	float t = 0.0f;
	for(int x=-r; x<=r; x++) {
		t += tex2D(tex, x, y);
	}
	od[y*w] = t * scale;

	// Optimiert
	// und nun immer vorne ein Pixel abziehen
	// und hinten einen Pixel adden
	for(int x=1; x<w; x++) {
		t += tex2D(tex, x + r, y);
		t -= tex2D(tex, x - r - 1, y);
		od[y*w+x] = t * scale;
	}
}

__global__ void
d_boxfilter_y_global(guchar *id, guchar *od, int w, int h, int r, int offset, gboolean do_bin) {
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if ( do_bin)
		d_boxfilter_y_bin(&id[x], &od[x], w, h, r, x, offset);
	else
		d_boxfilter_y(&id[x], &od[x], w, h, r, x, offset);
}

// process column
__device__ void
d_boxfilter_y(guchar *id, guchar *od, int w, int h, int r, uint x, int offset) {
	float scale = 1.0f / (2*r+1);

	float t;
	// do left edge
	t = id[0] * (r+1);
	for (int y = 1; y <= r; y++) {
		t += id[y*w];
	}
	// Average Filter
	od[0] = t * scale;

	for(int y = 1; y <= r; y++) {
		t += id[(y+r)*w];
		t -= id[0];
		// Average Filter
		od[y*w] = t * scale;
	}

	// main loop
	for(int y = r+1; y < h-r; y++) {
		t += id[(y+r)*w];
		t -= id[((y-r)*w)-w];
		od[y*w] = t * scale;
	}

	// do right edge
	for (int y = h-r; y < h; y++) {
		t += id[(h-1)*w];
		t -= id[((y-r)*w)-w];
		od[y*w] = t * scale;
	}
}

// process column
__device__ void
d_boxfilter_y_bin(guchar *id, guchar *od, int w, int h, int r, uint x, int offset) {
	float scale = 1.0f / (2*r+1);

	float t;
	// do left edge
	t = id[0] * (r+1);
	for (int y = 1; y <= r; y++) {
		t += id[y*w];
	}
	// Average Filter
	//od[0] = t * scale;
	// Binaerisierung
	if ( tex2D(tex, x, 0) < ((t * scale) + offset ))
		od[0] = 0;
	else
		od[0] = 255;

	for(int y = 1; y <= r; y++) {
		t += id[(y+r)*w];
		t -= id[0];
		// Average Filter
		//od[y*w] = t * scale;
		// Binaerisierung
		if ( tex2D(tex, x, y) < ((t * scale) + offset ))
			od[y*w] = 0;
		else
			od[y*w] = 255;
	}

	// main loop
	for(int y = r+1; y < h-r; y++) {
		t += id[(y+r)*w];
		t -= id[((y-r)*w)-w];
		//od[y*w] = t * scale;
		if ( tex2D(tex, x, y) < ((t * scale) + offset ))
			od[y*w] = 0;
		else
			od[y*w] = 255;
	}

	// do right edge
	for (int y = h-r; y < h; y++) {
		t += id[(h-1)*w];
		t -= id[((y-r)*w)-w];
		//od[y*w] = t * scale;
		if ( tex2D(tex, x, y) < ((t * scale) + offset ))
			od[y*w] = 0;
		else
			od[y*w] = 255;
	}
}


/* Fast Average Filter for a fixed radius of 7!
 * Reused Code
 */
__global__ void
AVGShared( uchar4 *pc, unsigned short step,
		short BlockWidth, short SharedPitch,
		short w, short h, float fScale,
		int radius, int offset, gboolean do_bin) {
	// pc > Pointer auf den Speicher in der Graka
	// step > 1280 (BilderBreite)
	// BlockWidth > 80
	// SharedPitch > 384
	// w > 1280
	// h > 1024
	// sharedMem > 2304 >> 48 * 48 = 2304
	// threads > 16,4
	// block   > 4,256
	// Radius 1 = Ich brauche links/rechts/oben/unten 1 extra pixel
	// radius darf max die haelfte des Blockes sein in x und y

	// u und v sind die KOs des Pixels, das ich kopieren will
	// u ist 4*80 = 320 -> 4*320 = 1280  -->  Der 320er Anfang jedes Blockes
	// auf u (anfang des 320er Blockes) muss dann noch der Zu nehmende Pixel addiert werden
	short u = 4*blockIdx.x*BlockWidth;
	short v = blockIdx.y*blockDim.y + threadIdx.y;
	short ib;

	// SharedIdx > Zeilenanfang vom SharedMem
	// 384 > Zeilenbreite vom SharedMem
	int SharedIdx = threadIdx.y * SharedPitch;


	// ib geht komplett durch von 0-81
	// ib geht 16er schritte
	// 4*ib = 64
	// damit hat man einheitliches lesen
	// t0 liest 4byte
	// t1 liest 4byte
	// -> 16Threads a 4byte = 64byte
	for ( ib = threadIdx.x; ib < BlockWidth+2*radius; ib += blockDim.x ) {
		LocalBlock[SharedIdx+4*ib+0] = tex2D( tex,
				(float) (u+4*ib-radius+0), (float) (v-radius) );
		LocalBlock[SharedIdx+4*ib+1] = tex2D( tex,
				(float) (u+4*ib-radius+1), (float) (v-radius) );
		LocalBlock[SharedIdx+4*ib+2] = tex2D( tex,
				(float) (u+4*ib-radius+2), (float) (v-radius) );
		LocalBlock[SharedIdx+4*ib+3] = tex2D( tex,
				(float) (u+4*ib-radius+3), (float) (v-radius) );
	}
	if ( threadIdx.y < radius*2 ) {
		//
		// copy trailing Radius*2 rows of pixels into shared
		//
		// bis v-Radius wurde alles kopiert, fehlt also noch 2*Radius
		// also alle Werte nochmal gleich nur SharedIdx anpassen auf die richte Stelle
		// und bei der y KO einfach blockDim.y uebrspringen weil die ja schon kopiert sind
		// v-Radius .. v lief ja bis blockDim.y-1 (-Radius)
		SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;
		for ( ib = threadIdx.x; ib < BlockWidth+2*radius; ib += blockDim.x ) {
			LocalBlock[SharedIdx+4*ib+0] = tex2D( tex,
					(float) (u+4*ib-radius+0), (float) (v+blockDim.y-radius) );
			LocalBlock[SharedIdx+4*ib+1] = tex2D( tex,
					(float) (u+4*ib-radius+1), (float) (v+blockDim.y-radius) );
			LocalBlock[SharedIdx+4*ib+2] = tex2D( tex,
					(float) (u+4*ib-radius+2), (float) (v+blockDim.y-radius) );
			LocalBlock[SharedIdx+4*ib+3] = tex2D( tex,
					(float) (u+4*ib-radius+3), (float) (v+blockDim.y-radius) );
		}
	}


	__syncthreads();


	u >>= 2;    // index as uchar4 from here
	uchar4 *pSobel = (uchar4 *) (((char *) pc)+v*step);
	SharedIdx = threadIdx.y * SharedPitch;


	unsigned char p;
	int j,k;




	for ( ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x ) {

		uchar4 out;
		out.x = 10;
		out.y = 80;
		out.z = 160;
		out.w = 254;

		volatile int avg = 0;
		volatile int avg1 = 0;
		volatile int l0 = 0;
		volatile int l1 = 0;
		volatile int l2 = 0;
		volatile int ll0 = 0;
		volatile int ll1 = 0;
		volatile int ll2 = 0;

		/* prob. the fasted, but not working ):
		volatile int pixrm1 = 0;
		volatile int pixrm2 = 0;
		volatile int pixrm3 = 0;
		volatile int avg = 0;
		volatile int avg15 = 0;
		volatile int avg16 = 0;
		volatile int avg17 = 0;

		for ( j = 0; j < 15; j++) {
			for ( k = 0; k < 15; k++) {
				/*
				if ( k == 15 ) {
					avg15 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+k];
					continue;
				} else if ( k == 16 ) {
					avg16 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+k];
					continue;
				} else if ( k == 17 ) {
					avg17 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+k];
					continue;
				}
				*/ /*
				avg1 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+k];
				if ( k == 0 ) {
					pixrm1 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+k];
				} else if ( k == 1 ) {
					pixrm2 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+k];
				} else if ( k == 2 ) {
					pixrm3 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+k];
				}
			}
		}
		p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+7];
		if ( p  < ((avg / 225) + 5)) {
			out.x = 0;
		} else {
			out.x = 255;
		}

		avg -= pixrm1;

		for ( j = 0; j < 15; j++) {
			avg += LocalBlock[SharedIdx+4*ib+j*SharedPitch+15];
		}
		p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+8];
		if ( p  < ((avg / 225) + 5)) {
			out.y = 0;
		} else {
			out.y = 255;
		}

		avg -= pixrm2;

		for ( j = 0; j < 15; j++) {
			avg += LocalBlock[SharedIdx+4*ib+j*SharedPitch+15];
		}
		p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+9];
		if ( p  < ((avg / 225) + 5)) {
			out.z = 0;
		} else {
			out.z = 255;
		}

		avg -= pixrm3;

		for ( j = 0; j < 15; j++) {
			avg += LocalBlock[SharedIdx+4*ib+j*SharedPitch+15];
		}
		p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+10];
		if ( p  < ((avg / 225) + 5)) {
			out.w = 0;
		} else {
			out.w = 255;
		}
*/ /* prob. the fasted, but not working ): */

		/* 3 bis 15 fuellen */
		avg = 0;
		avg1 = 0;
		for ( j = 0; j < 15; j++) {
			for ( k = 3; k < 15; k++) {
				avg += LocalBlock[SharedIdx+4*ib+j*SharedPitch+k];
			}
		}


		/* 0 1 2 und 15 16 17 fuellen */
		for ( j = 0; j < 15; j++) {
			l0 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+0];
			l1 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+1];
			l2 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+2];

			ll0 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+15];
			ll1 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+16];
			ll2 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+17];
		}

		if ( do_bin) {
			p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+7];
			if ( p < (( (avg + l0 + l1 + l2) / 225 ) + offset ))
				out.x = 0;
			else
				out.x = 255;
		} else
			out.x = (avg + l0 + l1 + l2) / 225;

		if ( do_bin) {
			p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+8];
			if ( p < (( (avg + l1 + l2 + ll0) / 225 ) + offset ))
				out.y = 0;
			else
				out.y = 255;
		} else
			out.y = (avg + l1 + l2 + ll0) / 225;

		if ( do_bin) {
			p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+9];
			if ( p < (( (avg + l2 + ll0 + ll1) / 225 ) + offset ))
				out.z = 0;
			else
				out.z = 255;
		} else
			out.z = (avg + l2 + ll0 + ll1) / 225;

		if ( do_bin) {
			p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+10];
			if ( p < (( (avg + ll0 + ll1 + ll2) / 225 ) + offset ))
				out.w = 0;
			else
				out.w = 255;
		} else
			out.w = (avg + ll0 + ll1 + ll2) / 225;


/*
		p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+7];
		if ( p  < ((avg / 225) + 5)) {
			out.x = 0;
		} else {
			out.x = 255;
		}

		p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+8];
		if ( p  < ((avg1 / 225) + 5)) {
			out.y = 0;
		} else {
			out.y = 255;
		}

		p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+9];
		if ( p  < ((avg2 / 225) + 5)) {
			out.z = 0;
		} else {
			out.z = 255;
		}

		p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+10];
		if ( p  < ((avg3 / 225) + 5)) {
			out.w = 0;
		} else {
			out.w = 255;
		}
*/

/*
		for ( j = 0; j < 15; j++) {
			for ( k = 15; k < 18; k++) {
				if ( k == 15 ) {
					avg15 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+k];
					continue;
				} else if ( k == 16 ) {
					avg16 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+k];
					continue;
				} else if ( k == 17 ) {
					avg17 += LocalBlock[SharedIdx+4*ib+j*SharedPitch+k];
					continue;
				}
			}
		}

		avg15 += ((avg - pixrm1) / 225);
		avg16 += ((avg15 - pixrm2) / 225);
		avg17 += ((avg16 - pixrm3) / 225);

		p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+8];
		if ( p  < (avg15 + 5) ) {
			out.y = 0;
		} else {
			out.y = 255;
		}

		p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+9];
		if ( p  < (avg16 + 5)) {
			out.z = 0;
		} else {
			out.z = 255;
		}

		p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+10];
		if ( p  < (avg17 + 5)) {
			out.w = 0;
		} else {
			out.w = 255;
		}
*/

/*
		avg -= pixrm2;

		for ( j = 0; j < 15; j++) {
			avg += LocalBlock[SharedIdx+4*ib+j*SharedPitch+16];
		}
		p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+9];
		if ( p  < ((avg / 225) + 5)) {
			out.z = 0;
		} else {
			out.z = 255;
		}

		avg -= pixrm3;

		for ( j = 0; j < 15; j++) {
			avg += LocalBlock[SharedIdx+4*ib+j*SharedPitch+17];
		}
		p = LocalBlock[SharedIdx+4*ib+7*SharedPitch+10];
		if ( p  < ((avg / 225) + 5)) {
			out.w = 0;
		} else {
			out.w = 255;
		}
*/


		if ( u+ib < w/4 && v < h ) {
#ifdef __DEVICE_EMULATION__
			printf("1: Wert x: %d  Wert y: %d  Wert z: %d  Wert w: %d\n", out.x, out.y, out.z, out.w);
#endif
			pSobel[u+ib] = out;
		}

	}

	__syncthreads();
}
