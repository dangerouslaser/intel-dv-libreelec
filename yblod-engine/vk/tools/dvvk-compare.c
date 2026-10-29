/* The phase 1 gate: yblod's OpenCL engine and the Vulkan engine on the same
 * decoded surfaces, every output byte compared. Headless (render node, a
 * surfaceless EGL context for the OpenCL engine's GL texture, as mpv-yblod
 * gives it one; no display, no audio).
 *
 *   dvvk-compare [options] FILE
 *     --frames N        frames to compare (default 48)
 *     --el-scaler S     0 nearest (mpv-yblod's default), 1 Annex B
 *     --fel 0|1         FEL off/on
 *     --dump DIR        on a mismatch, both frames as DIR/{cl,vk}-N.bgra
 *     --verify          also run yblod's dv_source_renderer_check on the OpenCL side
 *     --overlay N       an OSD-like synthetic overlay (1: bars, gray panel,
 *                       colour ramps with alpha ramps, antialiased stripes at
 *                       odd and even x, opaque blocks); covered pixels'
 *                       differences are reported separately
 *     --async           submit each Vulkan frame straight off the decoder, its
 *                       decode waited for on the GPU (the write fence), before
 *                       the OpenCL engine's vaSyncSurface: checks fence-only
 *                       reads byte for byte
 *     --sources         give the Vulkan engine the overlay the way mpv will:
 *                       pattern N as a dmabuf (KWin's forwarded shell OSD)
 *                       plus synthetic libass subtitle parts under it,
 *                       composited on the GPU; the OpenCL engine gets the
 *                       same picture composited on the CPU
 *     --subs-move       with --sources: the subtitles move every frame (the
 *                       engine prepares only their old and new area)
 *     --mutate KIND     rewrite each frame's RPU to reach routes the library
 *                       doesn't: mmr2 / mmr3 (MMR chroma, single layer: the
 *                       piecewise route, order-2 or general), mixed (FEL: Cr
 *                       polynomial, Cb per-channel MMR), wide (FEL: a Cb
 *                       coefficient past 32 bits: mmr96), denom14 (FEL: 2^-14
 *                       coefficients: the 128-bit MMR), equal (FEL with a
 *                       full-size synthetic EL), nv12 (8-bit synthetic BL
 *                       with MMR chroma); combine with '+' (mmr2+denom14
 *                       on a MEL: the 128-bit MMR, reachable only without
 *                       a residual)
 */
#define _GNU_SOURCE
#include "decode.h"
#include <libavutil/dovi_meta.h>
#include "../include/dv_engine_vk.h"
#include <native-dv/dv_source_renderer.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <gbm.h>
#include <drm_fourcc.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define W 3840
#define H 2160

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/* mpv-yblod's engine_yblod.c load_env_file: KEY=VALUE, no overwrite. */
static void load_env(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char *s = line;
        while (*s == ' ' || *s == '\t')
            s++;
        if (*s == '#' || *s == '\n' || !*s)
            continue;
        char *eq = strchr(s, '=');
        if (!eq)
            continue;
        *eq = 0;
        char *v = eq + 1;
        v[strcspn(v, "\r\n")] = 0;
        setenv(s, v, 0);
    }
    fclose(f);
}

struct bo {
    struct gbm_bo *bo;
    int fd;
    uint32_t stride;
    EGLImageKHR image;
    GLuint tex;
};

static int make_bo(struct gbm_device *gbm, struct bo *b)
{
    b->bo = gbm_bo_create(gbm, W, H, GBM_FORMAT_ARGB8888, GBM_BO_USE_LINEAR | GBM_BO_USE_RENDERING | GBM_BO_USE_SCANOUT);
    if (!b->bo)
        return -1;
    b->fd = gbm_bo_get_fd(b->bo);
    b->stride = gbm_bo_get_stride(b->bo);
    return b->fd < 0 ? -1 : 0;
}

static int read_bo(struct bo *b, uint8_t *dst)
{
    uint32_t stride = 0;
    void *map = NULL;
    uint8_t *p = gbm_bo_map(b->bo, 0, 0, W, H, GBM_BO_TRANSFER_READ, &stride, &map);
    if (!p)
        return -1;
    for (int y = 0; y < H; y++)
        memcpy(dst + (size_t)y * W * 4, p + (size_t)y * stride, W * 4);
    gbm_bo_unmap(b->bo, map);
    return 0;
}

/* RGBA8 premultiplied sRGB, memory order r,g,b,a. */
static void make_overlay(uint8_t *o, int kind)
{
    memset(o, 0, (size_t)W * H * 4);
    if (!kind)
        return;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            uint8_t *p = o + ((size_t)y * W + x) * 4;
            unsigned r = 0, g = 0, b = 0, a = 0;
            if (y >= 1700 && y < 1960 && x >= 160 && x < 3680) {          /* glass bar: gray, alpha 200 */
                a = 200; r = g = b = 30 + (unsigned)(x % 40);
                if (y >= 1820 && y < 1832 && x < 160 + (3680 - 160) * 6 / 10) { a = 255; r = g = b = 235; }
            } else if (y >= 300 && y < 520 && x >= 200 && x < 3640) {     /* colour ramp, alpha ramp */
                a = (unsigned)((x - 200) * 255 / 3440);
                unsigned h = (unsigned)(x * 6 / 3840), f = (unsigned)(x % 640) * 255 / 640;
                unsigned rgb[3] = {255, f, 0};
                if (h == 1) { rgb[0] = 255 - f; rgb[1] = 255; rgb[2] = 0; }
                if (h == 2) { rgb[0] = 0; rgb[1] = 255; rgb[2] = f; }
                if (h == 3) { rgb[0] = 0; rgb[1] = 255 - f; rgb[2] = 255; }
                if (h == 4) { rgb[0] = f; rgb[1] = 0; rgb[2] = 255; }
                if (h >= 5) { rgb[0] = 255; rgb[1] = 0; rgb[2] = 255 - f; }
                r = (rgb[0] * a + 127) / 255; g = (rgb[1] * a + 127) / 255; b = (rgb[2] * a + 127) / 255;
            } else if (y >= 700 && y < 900 && x >= 300 && x < 2300) {     /* antialiased stripes */
                int m = x % 7;
                a = m == 2 || m == 4 ? 128 : m == 3 ? 255 : 0;
                r = g = b = a == 255 ? 250 : a == 128 ? 120 : 0;
                if ((y / 20) % 2) { r = a; g = a * 3 / 4; b = a / 3; }    /* warm text */
            } else if (y >= 1000 && y < 1200 && x >= 2600 && x < 3600) {  /* opaque blocks */
                a = 255;
                unsigned k = (unsigned)((x - 2600) / 125);
                r = k & 1 ? 255 : 40; g = k & 2 ? 255 : 40; b = k & 4 ? 255 : 40;
            }
            p[0] = (uint8_t)r; p[1] = (uint8_t)g; p[2] = (uint8_t)b; p[3] = (uint8_t)a;
        }
}

static void set_mmr(AVDOVIReshapingCurve *c, int own, int order, int64_t one)
{
    for (int p = 0; p + 1 < c->num_pivots; p++) {
        c->mapping_idc[p] = AV_DOVI_MAPPING_MMR;
        c->mmr_order[p] = (uint8_t)order;
        c->mmr_constant[p] = one / 64;
        memset(c->mmr_coef[p], 0, sizeof(c->mmr_coef[p]));
        c->mmr_coef[p][0][own] = one - one / 16;
        c->mmr_coef[p][0][0] = one / 50;
        c->mmr_coef[p][0][3] = -one / 100;
        c->mmr_coef[p][0][5] = one / 40;
        if (order > 1) { c->mmr_coef[p][1][own] = one / 100; c->mmr_coef[p][1][4] = -one / 200; }
        if (order > 2) { c->mmr_coef[p][2][5] = one / 250; c->mmr_coef[p][2][6] = one / 300; }
    }
}

/* Rewrites a copy of the frame's AVDOVIMetadata (ffmpeg-yblod layout). */
static void mutate_one(uint8_t *buf, const char *kind);
/* KIND may combine mutations with '+', applied in order. */
static void mutate(uint8_t *buf, const char *kinds)
{
    char copy[128];
    snprintf(copy, sizeof(copy), "%s", kinds);
    for (char *save = NULL, *k = strtok_r(copy, "+", &save); k; k = strtok_r(NULL, "+", &save))
        mutate_one(buf, k);
}

static void mutate_one(uint8_t *buf, const char *kind)
{
    AVDOVIMetadata *m = (AVDOVIMetadata *)buf;
    AVDOVIRpuDataHeader *h = (AVDOVIRpuDataHeader *)(buf + m->header_offset);
    AVDOVIDataMapping *map = (AVDOVIDataMapping *)(buf + m->mapping_offset);
    int64_t one = (int64_t)1 << h->coef_log2_denom;
    if (!strcmp(kind, "mmr2") || !strcmp(kind, "mmr3") || !strcmp(kind, "nv12")) {
        int order = !strcmp(kind, "mmr3") ? 3 : 2;
        set_mmr(&map->curves[1], 1, order, one);
        set_mmr(&map->curves[2], 2, order, one);
        if (!strcmp(kind, "nv12")) {
            h->bl_bit_depth = 8;
            for (int c = 0; c < 3; c++) {
                AVDOVIReshapingCurve *cv = &map->curves[c];
                for (int p = 0; p < cv->num_pivots; p++)
                    cv->pivots[p] = (uint16_t)(cv->pivots[p] >> 2);
                if (cv->pivots[cv->num_pivots - 1] <= cv->pivots[0])
                    cv->pivots[cv->num_pivots - 1] = 255;
            }
        }
    } else if (!strcmp(kind, "mixed")) {
        AVDOVIReshapingCurve *c = &map->curves[2];
        for (int p = 0; p + 1 < c->num_pivots; p++) {
            c->mapping_idc[p] = AV_DOVI_MAPPING_POLYNOMIAL;
            c->poly_order[p] = 1;
            c->poly_coef[p][0] = one / 32;
            c->poly_coef[p][1] = one - one / 16;
            c->poly_coef[p][2] = 0;
        }
    } else if (!strcmp(kind, "wide")) {
        AVDOVIReshapingCurve *c = &map->curves[1];
        for (int p = 0; p + 1 < c->num_pivots; p++)
            c->mmr_coef[p][0][6] = 300 * one;
    } else if (!strcmp(kind, "denom14")) {
        /* The same values in Q14 (all coefficients are in units of 2^-denom). */
        int shift = h->coef_log2_denom - 14;
        h->coef_log2_denom = 14;
        for (int c = 0; c < 3; c++) {
            AVDOVIReshapingCurve *cv = &map->curves[c];
            for (int p = 0; p < 8; p++) {
                for (int k = 0; k < 3; k++) cv->poly_coef[p][k] >>= shift;
                cv->mmr_constant[p] >>= shift;
                for (int o = 0; o < 3; o++) for (int k = 0; k < 7; k++) cv->mmr_coef[p][o][k] >>= shift;
            }
            map->nlq[c].vdr_in_max >>= shift;
            map->nlq[c].linear_deadzone_slope >>= shift;
            map->nlq[c].linear_deadzone_threshold >>= shift;
        }
    } else if (!strcmp(kind, "noresidual")) {
        h->disable_residual_flag = 1;
    } else if (!strcmp(kind, "equal")) {
        h->el_spatial_resampling_filter_flag = 0;
    }
}

/* A synthetic decoded surface (VA, filled through vaDeriveImage): smooth
 * gradients plus noise; for an EL, codes near the NLQ offset. */
static VASurfaceID synthetic_surface(VADisplay dpy, int nv12, unsigned w, unsigned h, int el, unsigned seed)
{
    VASurfaceAttrib attr = {.type = VASurfaceAttribPixelFormat, .flags = VA_SURFACE_ATTRIB_SETTABLE,
                            .value = {.type = VAGenericValueTypeInteger, .value.i = nv12 ? VA_FOURCC_NV12 : VA_FOURCC_P010}};
    VASurfaceID s = VA_INVALID_SURFACE;
    if (vaCreateSurfaces(dpy, nv12 ? VA_RT_FORMAT_YUV420 : VA_RT_FORMAT_YUV420_10, w, h, &s, 1, &attr, 1))
        return VA_INVALID_SURFACE;
    VAImage img;
    void *ptr = NULL;
    if (vaDeriveImage(dpy, s, &img) || vaMapBuffer(dpy, img.buf, &ptr))
        return VA_INVALID_SURFACE;
    uint32_t r = seed * 2654435761u + 1;
    for (unsigned plane = 0; plane < 2; plane++) {
        unsigned pw = plane ? w : w, ph = plane ? h / 2 : h;   /* chroma: w/2 pairs = w samples */
        for (unsigned y = 0; y < ph; y++) {
            uint8_t *row = (uint8_t *)ptr + img.offsets[plane] + (size_t)y * img.pitches[plane];
            for (unsigned x = 0; x < pw; x++) {
                r ^= r << 13; r ^= r >> 17; r ^= r << 5;
                unsigned v;
                if (el)
                    v = 512 + (r % 9) - 4 + ((x / 64 + y / 64) % 3);
                else
                    v = (plane ? 300 + (x * 400 / pw) : 64 + (x * 800 / pw) + (y * 100 / ph)) + (r % 5);
                if (v > 1023) v = 1023;
                if (nv12)
                    row[x] = (uint8_t)(v >> 2);
                else
                    ((uint16_t *)row)[x] = (uint16_t)(v << 6);
            }
        }
    }
    vaUnmapBuffer(dpy, img.buf);
    vaDestroyImage(dpy, img.image_id);
    vaSyncSurface(dpy, s);
    return s;
}

/* What the OpenCL engine gets for --sources: the parts in order (libass
 * "over", premultiplied), then the shell's picture over them. */
static void compose_cpu(uint8_t *out, const uint8_t *shell_rgba, const dvvk_sub_part *parts, int n)
{
    uint8_t *subs = calloc((size_t)W * H, 4);
    for (int o = 0; o < n; o++) {
        const dvvk_sub_part *q = &parts[o];
        double cr = ((q->color >> 24) & 255) / 255.0, cg = ((q->color >> 16) & 255) / 255.0,
               cb = ((q->color >> 8) & 255) / 255.0, ca = (255 - (q->color & 255)) / 255.0;
        for (uint32_t y = 0; y < q->dh; y++)
            for (uint32_t x = 0; x < q->dw; x++) {
                int px = q->x + (int)x, py = q->y + (int)y;
                if (px < 0 || py < 0 || px >= W || py >= H)
                    continue;
                double a = ((const uint8_t *)q->data)[y * q->stride + x] / 255.0 * ca;
                uint8_t *d = subs + ((size_t)py * W + (size_t)px) * 4;
                double dst[4] = {d[0] / 255.0, d[1] / 255.0, d[2] / 255.0, d[3] / 255.0};
                double src[4] = {cr * a, cg * a, cb * a, a};
                for (int c = 0; c < 4; c++)
                    d[c] = (uint8_t)lround((src[c] + dst[c] * (1 - a)) * 255.0);
            }
    }
    for (size_t i = 0; i < (size_t)W * H; i++) {
        const uint8_t *g = shell_rgba + i * 4, *b = subs + i * 4;
        uint8_t *o = out + i * 4;
        double a = g[3] / 255.0;
        for (int c = 0; c < 4; c++) {
            long v = lround((g[c] / 255.0 + b[c] / 255.0 * (1 - a)) * 255.0);
            o[c] = (uint8_t)(v > 255 ? 255 : v);
        }
        for (int c = 0; c < 3; c++)
            if (o[c] > o[3]) o[c] = o[3];
    }
    free(subs);
}

static unsigned luma12(const uint8_t *p) { return (unsigned)p[1] * 16u + (p[0] & 15u); }
static unsigned chroma12(const uint8_t *p) { return (unsigned)p[2] * 16u + (p[0] >> 4); }

int main(int argc, char **argv)
{
    int frames = 48, scaler = 0, fel = 1, verify = 0, overlay = 0;
    const char *mutation = NULL;
    int async = 0, sources = 0, subs_move = 0;
    const char *file = NULL, *dump = NULL, *node = "/dev/dri/renderD128";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc)
            frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--el-scaler") && i + 1 < argc)
            scaler = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fel") && i + 1 < argc)
            fel = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc)
            dump = argv[++i];
        else if (!strcmp(argv[i], "--sources"))
            sources = 1;
        else if (!strcmp(argv[i], "--subs-move"))
            subs_move = 1;
        else if (!strcmp(argv[i], "--async"))
            async = 1;
        else if (!strcmp(argv[i], "--mutate") && i + 1 < argc)
            mutation = argv[++i];
        else if (!strcmp(argv[i], "--overlay") && i + 1 < argc)
            overlay = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify"))
            verify = 1;
        else
            file = argv[i];
    }
    if (!file) {
        fprintf(stderr, "usage: dvvk-compare [--frames N] [--el-scaler 0|1] [--fel 0|1] [--dump DIR] [--verify] FILE\n");
        return 2;
    }
    load_env("/usr/lib/kodi/native-dv.env");
    setenv("NEO_CACHE_DIR", "/var/tmp/dvvk/neo-cache", 0);

    struct dv_decoder dec;
    if (dv_decoder_open(&dec, file, node, 0) < 0) {
        fprintf(stderr, "decode: can't open %s\n", file);
        return 1;
    }
    int drm = open(node, O_RDWR | O_CLOEXEC);
    struct gbm_device *gbm = gbm_create_device(drm);
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_display = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay dpy = get_display(EGL_PLATFORM_GBM_KHR, gbm, NULL);
    if (!eglInitialize(dpy, NULL, NULL) || !eglBindAPI(EGL_OPENGL_ES_API)) {
        fprintf(stderr, "EGL init failed\n");
        return 1;
    }
    const EGLint ctx_attr[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attr);
    if (!ctx || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        fprintf(stderr, "EGL context failed\n");
        return 1;
    }
    PFNEGLCREATEIMAGEKHRPROC create_image = (void *)eglGetProcAddress("eglCreateImageKHR");
    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target = (void *)eglGetProcAddress("glEGLImageTargetTexture2DOES");

    struct bo cl = {0}, vk = {0};
    if (make_bo(gbm, &cl) || make_bo(gbm, &vk)) {
        fprintf(stderr, "gbm_bo_create failed\n");
        return 1;
    }
    const EGLint img_attr[] = {EGL_WIDTH, W, EGL_HEIGHT, H, EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_ARGB8888,
                               EGL_DMA_BUF_PLANE0_FD_EXT, cl.fd, EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
                               EGL_DMA_BUF_PLANE0_PITCH_EXT, (EGLint)cl.stride, EGL_NONE};
    cl.image = create_image(dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, img_attr);
    glGenTextures(1, &cl.tex);
    glBindTexture(GL_TEXTURE_2D, cl.tex);
    image_target(GL_TEXTURE_2D, cl.image);
    glFinish();

    void *ocl = dv_source_renderer_create_storage((uintptr_t)dec.display, (uintptr_t)dpy, (uintptr_t)ctx, cl.tex,
                                                  "/usr/share/native-dv", DV_SOURCE_OUTPUT_BGRA);
    if (!ocl) {
        fprintf(stderr, "OpenCL engine didn't start\n");
        return 1;
    }
    dv_source_renderer_set_el_scaler(ocl, (unsigned)scaler);
    dv_source_renderer_set_fel(ocl, fel);
    /* mpv-yblod's GUI colour: BT.709 -> BT.2020 -> the common LMS, 203 nits */
    static const double rgb[9] = {.6274, .3293, .0433, .0691, .9195, .0114, .0164, .0880, .8956};
    static const double lms[9] = {7222, 8771, 390, 2654, 12430, 1300, 0, 422, 15962};
    double rgb_to_lms[9] = {0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                rgb_to_lms[3 * i + j] += lms[3 * i + k] * rgb[3 * k + j] / 16384.0;
    uint8_t *gui = malloc((size_t)W * H * 4);
    make_overlay(gui, overlay);
    /* --sources: the shell's picture as a linear ARGB8888 dmabuf, and libass
     * parts (A8 glyph-like bitmaps, a colour each) composited under it. */
    struct bo shell = {0};
    uint8_t *shell_rgba = malloc((size_t)W * H * 4);
    dvvk_sub_part parts[3];
    uint8_t *masks[3] = {0};
    if (sources && overlay) {
        if (make_bo(gbm, &shell))
            return 1;
        uint32_t stride = 0;
        void *map = NULL;
        uint8_t *p = gbm_bo_map(shell.bo, 0, 0, W, H, GBM_BO_TRANSFER_WRITE, &stride, &map);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                const uint8_t *s = gui + ((size_t)y * W + x) * 4;
                uint8_t *d = p + (size_t)y * stride + (size_t)x * 4;
                d[0] = s[2]; d[1] = s[1]; d[2] = s[0]; d[3] = s[3];   /* ARGB8888: B,G,R,A */
            }
        gbm_bo_unmap(shell.bo, map);
        const uint32_t colours[3] = {0xffffff00u, 0x10101000u, 0xffd02080u};   /* white, black outline, translucent */
        const int px[3] = {1200, 1196, 1600}, py[3] = {1500, 1496, 600};
        for (int k = 0; k < 3; k++) {
            int pw = 900 + 8 * (k == 1), ph = 120 + 8 * (k == 1);
            masks[k] = malloc((size_t)pw * ph);
            for (int y = 0; y < ph; y++)
                for (int x = 0; x < pw; x++) {
                    int glyph = ((x / 30) % 3 != 2) && ((y / 20) % 4 != 3);
                    int edge = (x % 30 == 0 || x % 30 == 19) ? 128 : 255;
                    masks[k][(size_t)y * pw + x] = (uint8_t)(glyph ? edge : (k == 1 && x % 7 == 0 ? 60 : 0));
                }
            parts[k] = (dvvk_sub_part){.x = px[k], .y = py[k], .w = (uint32_t)pw, .h = (uint32_t)ph, .dw = (uint32_t)pw,
                                       .dh = (uint32_t)ph, .stride = (uint32_t)pw, .format = 1, .color = colours[k],
                                       .data = masks[k]};
        }
        dvvk_sub_part ordered[3] = {parts[1], parts[0], parts[2]};
        memcpy(parts, ordered, sizeof(parts));
        memcpy(shell_rgba, gui, (size_t)W * H * 4);
        compose_cpu(gui, shell_rgba, parts, 3);
    }
    GLuint gui_tex = 0;
    if (overlay) {
        if (dv_source_renderer_prepare_overlays(ocl)) {
            fprintf(stderr, "OpenCL overlay programs failed\n");
            return 1;
        }
        glGenTextures(1, &gui_tex);
        glBindTexture(GL_TEXTURE_2D, gui_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, gui);
        glFinish();
    }
    dvvk_engine *vke = NULL;
    if (dvvk_create(drm, dec.display, &vke) || dvvk_set_output(vke, 0, vk.fd, vk.stride)) {
        fprintf(stderr, "Vulkan engine didn't start\n");
        return 1;
    }
    if (overlay && sources) {
        dvvk_dmabuf db = {.width = W, .height = H, .format = DRM_FORMAT_ARGB8888, .modifier = DRM_FORMAT_MOD_LINEAR,
                          .num_planes = 1, .fd = {shell.fd, -1, -1, -1}, .stride = {shell.stride}};
        if (dvvk_set_overlay(vke, &db, 1) || dvvk_set_subtitles(vke, parts, 3)) {
            fprintf(stderr, "Vulkan overlay sources refused\n");
            return 1;
        }
    } else if (overlay)
        dvvk_set_overlay_rgba(vke, gui, (size_t)W * 4);
    size_t ov_differ_total = 0, ov_compared = 0;
    unsigned ov_max_dy = 0, ov_max_dc = 0;
    int overlay_before = 0;

    uint8_t *a = malloc((size_t)W * H * 4), *b = malloc((size_t)W * H * 4);
    unsigned packet_id = 0;
    int refresh = 1, n = 0, identical = 0, mismatched = 0, refused_both = 0, refused_one = 0, range_flagged = 0;
    double cl_ms = 0, vk_ms = 0;
    struct dv_pair p;
    while (n < frames && dv_decoder_next(&dec, &p) == 1) {
        n++;
        if (!p.dovi) {
            dv_pair_free(&p);
            continue;
        }
        unsigned w = (unsigned)p.bl->width, h = (unsigned)p.bl->height;
        const void *meta = p.dovi->data;
        size_t meta_size = p.dovi->size;
        uint8_t *mutated = NULL;
        VASurfaceID bl_surface = p.bl_surface, el_surface = p.el ? p.el_surface : 0xffffffffu, made[2] = {VA_INVALID_SURFACE, VA_INVALID_SURFACE};
        int synthetic_nv12 = 0;
        if (mutation) {
            mutated = malloc(meta_size);
            memcpy(mutated, meta, meta_size);
            mutate(mutated, mutation);
            meta = mutated;
            if (!strcmp(mutation, "equal") && p.el)
                el_surface = made[1] = synthetic_surface(dec.display, 0, w, h, 1, (unsigned)n);
            if (!strcmp(mutation, "nv12")) {
                bl_surface = made[0] = synthetic_surface(dec.display, 1, w, h, 0, (unsigned)n);
                el_surface = 0xffffffffu;
                synthetic_nv12 = 1;
            }
        }
        if (sources && overlay && subs_move && n > 1) {
            for (int k = 0; k < 3; k++)
                parts[k].x += 37;
            dvvk_set_subtitles(vke, parts, 3);
            compose_cpu(gui, shell_rgba, parts, 3);
            glBindTexture(GL_TEXTURE_2D, gui_tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, gui);
            glFinish();
        }
        /* mpv: aspect-fit destination, x/y even, w/h multiples of 4 */
        double scale = fmin(3840.0 / w, 2160.0 / h);
        int x0 = (int)lround((3840 - w * scale) / 2), y0 = (int)lround((2160 - h * scale) / 2);
        int x1 = (int)lround((3840 + w * scale) / 2), y1 = (int)lround((2160 + h * scale) / 2);
        unsigned x = (unsigned)x0 & ~1u, y = (unsigned)y0 & ~1u;
        unsigned dw = (unsigned)(x1 - (int)x) & ~3u, dh = (unsigned)(y1 - (int)y) & ~3u;
        int nv12 = synthetic_nv12 || ((AVHWFramesContext *)p.bl->hw_frames_ctx->data)->sw_format == AV_PIX_FMT_NV12;
        int top_left = !synthetic_nv12 && p.bl->chroma_location == AVCHROMA_LOC_TOPLEFT;
        int64_t pts = p.bl->pts;

        dvvk_frame f = {.plan = {.width = w, .height = h, .x = x, .y = y, .dw = dw, .dh = dh,
                                 .chroma_top_left = (uint32_t)top_left, .has_el = el_surface != 0xffffffffu, .nv12 = (uint32_t)nv12,
                                 .el_scaler = (uint32_t)scaler, .fel = (uint32_t)fel, .packet_id = packet_id,
                                 .refresh = (uint32_t)refresh, .overlay = (uint32_t)(overlay != 0),
                                 .overlay_visible_before = (uint32_t)overlay_before, .pts = pts, .white_nits = 203.0},
                        .bl_surface = bl_surface, .el_surface = el_surface,
                        .metadata = meta, .metadata_size = meta_size, .output = 0,
                        .flags = sources && overlay ? DVVK_FRAME_AUTO_OVERLAY : 0};
        memcpy(f.plan.rgb_to_lms, rgb_to_lms, sizeof(rgb_to_lms));
        uint32_t flags = 0;
        uint64_t ticket = 0;
        int rv = 0;
        double vk_t0 = now_ms();
        if (async)
            rv = dvvk_submit(vke, &f, &ticket);
        double vk_submit_ms = now_ms() - vk_t0;

        /* OpenCL, as mpv-yblod's engine_yblod.c render() */
        dv_source_geometry g = {.width = w, .height = h, .canvas_width = W, .canvas_height = H, .x = x, .y = y,
                                .chroma_siting = el_surface == 0xffffffffu && top_left ? DV_SOURCE_CHROMA_TOP_LEFT : DV_SOURCE_CHROMA_CENTER_LEFT,
                                .destination_width = dw, .destination_height = dh};
        dv_source_input in = {sizeof(in), nv12 ? DV_SOURCE_INPUT_NV12 : DV_SOURCE_INPUT_P010, nv12 ? 8u : 10u};
        double t0 = now_ms();
        int rc = dv_source_renderer_submit_input(ocl, bl_surface, el_surface, w, h, meta, meta_size, (uint64_t)n, pts, &g, &in);
        if (!rc)
            rc = overlay ? dv_source_renderer_present_overlay(ocl, (uint64_t)n, pts, packet_id, refresh, gui_tex, 0,
                                                              rgb_to_lms, 203.0)
                         : dv_source_renderer_present(ocl, (uint64_t)n, pts, packet_id, refresh);
        if (dv_source_renderer_drain(ocl)) {
            fprintf(stderr, "OpenCL drain failed\n");
            return 1;
        }
        cl_ms += now_ms() - t0;
        if (!rc && verify) {
            read_bo(&cl, a);
            int c = dv_source_renderer_check(ocl, a, (size_t)W * H * 4);
            if (c)
                printf("frame %d: OpenCL self-check %d\n", n, c);
        }

        /* Vulkan */
        t0 = now_ms();
        if (async) {
            if (!rv && ticket) {
                dvvk_wait(vke, ticket, UINT64_MAX);
                int done = 0;
                dvvk_poll(vke, ticket, &done, &flags);
            }
            vk_ms += vk_submit_ms;
        } else {
            rv = dvvk_render_sync(vke, &f, &flags);
            vk_ms += now_ms() - t0;
        }

        /* The OpenCL engine refuses a frame with a range error; the Vulkan
         * engine presents it clamped and flags it (the agreed behaviour). */
        if (rc == -4 && !rv && (flags & 3u)) {
            range_flagged++;
        } else if (rc || rv) {
            if (rc && rv)
                refused_both++;
            else
                refused_one++;
            if ((rc == 0) != (rv == 0) && refused_one < 4)
                printf("frame %d: OpenCL %d, Vulkan %d (flags %u)\n", n, rc, rv, flags);
            if (rc && rv && refused_both == 1)
                printf("frame %d: both refuse (OpenCL %d, Vulkan %d, flags %u)\n", n, rc, rv, flags);
        } else {
            read_bo(&cl, a);
            read_bo(&vk, b);
            size_t diff = 0, first = (size_t)-1, packet_px = 0, outside = 0, covered = 0;
            for (size_t i = 0; i < (size_t)W * H; i++) {
                unsigned px = (unsigned)(i % W), py = (unsigned)(i / W);
                int under = 0;
                /* the pixel pair's chroma blend reads center-1..center+1 */
                if (overlay)
                    for (int dx = -1; dx <= 1; dx++) {
                        int qx = (int)(px & ~1u) + dx;
                        qx = qx < 0 ? 0 : qx >= W ? W - 1 : qx;
                        under |= gui[((size_t)py * W + (size_t)qx) * 4 + 3] != 0;
                    }
                ov_compared += (size_t)under;
                if (!memcmp(a + 4 * i, b + 4 * i, 4))
                    continue;
                if (under) {
                    covered++;
                    unsigned dy = (unsigned)abs((int)luma12(a + 4 * i) - (int)luma12(b + 4 * i));
                    unsigned dc = (unsigned)abs((int)(chroma12(a + 4 * i) >> 1) - (int)(chroma12(b + 4 * i) >> 1)) * 2;
                    if (dy > ov_max_dy) {
                        ov_max_dy = dy;
                        if (dy > 16)
                            printf("frame %d: covered pixel %u,%u luma %u vs %u (gui %02x%02x%02x%02x)\n", n, px, py,
                                   luma12(a + 4 * i), luma12(b + 4 * i), gui[i * 4], gui[i * 4 + 1], gui[i * 4 + 2], gui[i * 4 + 3]);
                    }
                    if (dc > ov_max_dc) ov_max_dc = dc;
                    continue;
                }
                if (first == (size_t)-1)
                    first = i;
                diff++;
                if (i < 5 * 3072)
                    packet_px++;
                if (py < y || py >= y + dh || px < x || px >= x + dw)
                    outside++;
            }
            ov_differ_total += covered;
            if (!diff)
                identical++;
            else {
                mismatched++;
                if (mismatched <= 4) {
                    unsigned fx = (unsigned)(first % W), fy = (unsigned)(first / W);
                    const uint8_t *pa = a + 4 * first, *pb = b + 4 * first;
                    printf("frame %d: %zu uncovered pixels differ (%zu in packet rows, %zu outside the picture); first %u,%u "
                           "cl %02x%02x%02x vk %02x%02x%02x; flags %u\n",
                           n, diff, packet_px, outside, fx, fy, pa[2], pa[1], pa[0], pb[2], pb[1], pb[0], flags);
                }
                if (dump && mismatched <= 2) {
                    char path[4096];
                    snprintf(path, sizeof(path), "%s/cl-%d.bgra", dump, n);
                    FILE *o = fopen(path, "wb");
                    if (o) { fwrite(a, 1, (size_t)W * H * 4, o); fclose(o); }
                    snprintf(path, sizeof(path), "%s/vk-%d.bgra", dump, n);
                    o = fopen(path, "wb");
                    if (o) { fwrite(b, 1, (size_t)W * H * 4, o); fclose(o); }
                }
            }
        }
        /* mpv-yblod: the picture was shown */
        if (!rc) {
            packet_id = (packet_id + 1) & 15;
            refresh = 0;
            overlay_before = overlay != 0;
        }
        free(mutated);
        for (int k = 0; k < 2; k++)
            if (made[k] != VA_INVALID_SURFACE) {
                dvvk_reset(vke);   /* forget the import before the id can be reused */
                vaDestroySurfaces(dec.display, &made[k], 1);
            }
        dv_pair_free(&p);
    }
    uint64_t fence_waits = 0, vasync_waits = 0;
    dvvk_decode_waits(vke, &fence_waits, &vasync_waits);
    if (async)
        printf("decode waits: %llu on the GPU (write fence), %llu vaSyncSurface\n",
               (unsigned long long)fence_waits, (unsigned long long)vasync_waits);
    printf("{\"file\":\"%s\",\"el_scaler\":%d,\"fel\":%d,\"frames\":%d,\"identical\":%d,\"mismatched\":%d,"
           "\"refused_both\":%d,\"refused_one\":%d,\"range_flagged\":%d,\"opencl_ms\":%.2f,\"vulkan_ms\":%.2f,"
           "\"overlay\":%d,\"overlay_pixels_compared\":%zu,\"overlay_pixels_differ\":%zu,"
           "\"overlay_max_luma_delta\":%u,\"overlay_max_chroma_delta\":%u}\n",
           file, scaler, fel, n, identical, mismatched, refused_both, refused_one, range_flagged, n ? cl_ms / n : 0,
           n ? vk_ms / n : 0, overlay, ov_compared, ov_differ_total, ov_max_dy, ov_max_dc);
    dvvk_destroy(vke);
    dv_source_renderer_destroy(ocl);
    dv_decoder_close(&dec);
    return mismatched || refused_one ? 1 : 0;
}
