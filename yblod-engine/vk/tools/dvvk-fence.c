/* Does iHD put implicit (write) fences on decoded surfaces? For each decoded
 * frame, straight off the decoder: export the surface dmabuf, export its
 * readers' sync_file (DMA_BUF_IOCTL_EXPORT_SYNC_FILE, READ) and check whether
 * it is still pending; then time vaSyncSurface. If decodes are ever caught
 * pending and the fence signals no earlier than vaSyncSurface would return,
 * the engine can wait on the GPU instead of calling vaSyncSurface.
 *
 *   dvvk-fence FILE [FRAMES] [render node]      (headless) */
#include "decode.h"
#include <va/va_drmcommon.h>
#include <linux/dma-buf.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int surface_fence(VADisplay dpy, VASurfaceID s, int *pending, double *signal_ms, double *sync_ms)
{
    VADRMPRIMESurfaceDescriptor desc;
    if (vaExportSurfaceHandle(dpy, s, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                              VA_EXPORT_SURFACE_READ_ONLY | VA_EXPORT_SURFACE_COMPOSED_LAYERS, &desc))
        return -1;
    struct dma_buf_export_sync_file arg = {.flags = DMA_BUF_SYNC_READ, .fd = -1};
    double t0 = now_ms();
    int r = ioctl(desc.objects[0].fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &arg);
    *pending = 0;
    *signal_ms = 0;
    if (r == 0) {
        struct pollfd p = {.fd = arg.fd, .events = POLLIN};
        *pending = poll(&p, 1, 0) == 0;
        if (*pending) {
            poll(&p, 1, 1000);
            *signal_ms = now_ms() - t0;
        }
        close(arg.fd);
    }
    double t1 = now_ms();
    vaSyncSurface(dpy, s);
    *sync_ms = now_ms() - t1;
    for (uint32_t i = 0; i < desc.num_objects; i++)
        close(desc.objects[i].fd);
    return r;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: dvvk-fence FILE [FRAMES] [NODE]\n");
        return 2;
    }
    int frames = argc > 2 ? atoi(argv[2]) : 96;
    struct dv_decoder d;
    if (dv_decoder_open(&d, argv[1], argc > 3 ? argv[3] : "/dev/dri/renderD128", 0) < 0) {
        fprintf(stderr, "open failed\n");
        return 1;
    }
    int n = 0, pending[2] = {0}, failed = 0;
    double sync_after_signal_max = 0, signal_max = 0, sync_max = 0;
    struct dv_pair p;
    while (n < frames && dv_decoder_next(&d, &p) == 1) {
        VASurfaceID s[2] = {p.bl_surface, p.el_surface};
        for (int l = 0; l < 2; l++) {
            if (s[l] == VA_INVALID_SURFACE)
                continue;
            int pend;
            double sig, syn;
            if (surface_fence(d.display, s[l], &pend, &sig, &syn) < 0) {
                failed++;
                continue;
            }
            pending[l] += pend;
            if (pend) {
                if (sig > signal_max)
                    signal_max = sig;
                if (syn > sync_after_signal_max)
                    sync_after_signal_max = syn;
            } else if (syn > sync_max)
                sync_max = syn;
        }
        if (!p.dovi && n == 0)
            fprintf(stderr, "no DOVI metadata on the first frame\n");
        dv_pair_free(&p);
        n++;
    }
    printf("{\"file\":\"%s\",\"profile\":%d,\"el\":%d,\"frames\":%d,\"export_failures\":%d,"
           "\"pending_at_receive\":[%d,%d],\"pending_signal_ms_max\":%.3f,"
           "\"vasync_after_fence_ms_max\":%.3f,\"vasync_when_signalled_ms_max\":%.3f}\n",
           argv[1], d.profile, d.el_present, n, failed, pending[0], pending[1], signal_max,
           sync_after_signal_max, sync_max);
    dv_decoder_close(&d);
    return 0;
}
