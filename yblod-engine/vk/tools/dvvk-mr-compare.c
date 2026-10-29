/* CB1's Master Reference against yblod's Vulkan engine on the same decoded
 * surfaces, every output byte compared. Headless: a render node, a
 * surfaceless EGL context for CB1 (libplacebo on GLES, as Kodi gives it one)
 * and the Vulkan engine on the same VADisplay. No display, no audio.
 *
 *   dvvk-mr-compare [options] FILE
 *     --frames N        frames to compare (default 48)
 *     --start S         seek to S seconds first
 *     --fel 0|1         give both engines the EL (default 1)
 *     --ipt             CB1's Master Reference IPT output instead of YCC
 *     --yblod           the Vulkan engine as YBLOD runs it (its own siting and
 *                       chroma filters), not CB1's spatial contract
 *     --flip            CB1's flip_y (default off: row 0 is the HDMI top row)
 *     --dump DIR        on a mismatch, both frames as DIR/{cb1,vk}-N.bgra
 *     --node PATH       render node (default /dev/dri/renderD128)
 *     --engine vk|cl    YBLOD's engine: Vulkan (default) or OpenCL (Intel NEO;
 *                       its production settings, /usr/lib/kodi/native-dv.env)
 *     --only cb1|vk     time one engine alone (no comparison; vk = YBLOD's)
 *     --warmup N        frames left out of the timing (default 12)
 *
 * Timing is per picture on the CPU clock, synchronous (each engine finishes
 * before the next starts), from decoded surfaces (vaSyncSurface first), and
 * includes each engine's per-picture imports and CPU preparation.
 *
 * Both engines get an unscaled picture (CB1 doesn't call resized Master
 * output bit-identical). The Vulkan engine runs with the Annex B EL scaler,
 * CB1's only Master EL filter. Pixels are the HDMI tunnel's 12-bit 4:2:2
 * (luma, and Cb on even / Cr on odd columns); the metadata packets ride in
 * the first rows' low bits, so those rows are counted on their own.
 */
#define _GNU_SOURCE
#include "decode.h"
#include <libavutil/dovi_meta.h>
#include "../include/dv_engine_vk.h"
#include <native-dv/dv_source_renderer.h>
#include "dvbridge_render.h"
#include "dvbridge_dmabuf.h"
#include "dvbridge_gl_frame.h"
#include <libplacebo/opengl.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <gbm.h>
#include <drm_fourcc.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define W 3840
#define H 2160
#define META_PIXELS (4 * W)   /* the rows the metadata packets ride in */

/* The engine is dlopened, as Kodi does: the image has libdv_engine_vk.so.0. */
static struct {
    int (*create)(int, void *, dvvk_engine **);
    int (*set_output)(dvvk_engine *, uint32_t, int, uint32_t);
    int (*render_sync)(dvvk_engine *, const dvvk_frame *, uint32_t *);
    void (*reset)(dvvk_engine *);
    void (*destroy)(dvvk_engine *);
} vk;

/* YBLOD's OpenCL engine, dlopened too. */
static struct {
    void *(*create)(uintptr_t, uintptr_t, uintptr_t, unsigned, const char *, unsigned);
    int (*set_el_scaler)(void *, unsigned);
    int (*set_fel)(void *, int);
    int (*submit)(void *, unsigned, unsigned, unsigned, unsigned, const void *, size_t, uint64_t, int64_t,
                  const dv_source_geometry *, const dv_source_input *);
    int (*present)(void *, uint64_t, int64_t, unsigned, int);
    int (*drain)(void *);
    int (*destroy)(void *);
} cl;

static int load_opencl(void)
{
    void *lib = dlopen("libdv_source_private.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return -1;
    }
    cl.create = dlsym(lib, "dv_source_renderer_create_storage");
    cl.set_el_scaler = dlsym(lib, "dv_source_renderer_set_el_scaler");
    cl.set_fel = dlsym(lib, "dv_source_renderer_set_fel");
    cl.submit = dlsym(lib, "dv_source_renderer_submit_input");
    cl.present = dlsym(lib, "dv_source_renderer_present");
    cl.drain = dlsym(lib, "dv_source_renderer_drain");
    cl.destroy = dlsym(lib, "dv_source_renderer_destroy");
    return cl.create && cl.set_el_scaler && cl.set_fel && cl.submit && cl.present && cl.drain && cl.destroy ? 0 : -1;
}

/* mpv-yblod's engine_yblod.c load_env_file: KEY=VALUE, no overwrite. */
static void load_env(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char *k = line;
        while (*k == ' ' || *k == '\t')
            k++;
        char *eq = strchr(k, '=');
        if (*k == '#' || !eq)
            continue;
        *eq = 0;
        eq[1 + strcspn(eq + 1, "\r\n")] = 0;
        setenv(k, eq + 1, 0);
    }
    fclose(f);
}

static int load_engine(void)
{
    void *lib = dlopen("libdv_engine_vk.so.0", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return -1;
    }
    vk.create = dlsym(lib, "dvvk_create");
    vk.set_output = dlsym(lib, "dvvk_set_output");
    vk.render_sync = dlsym(lib, "dvvk_render_sync");
    vk.reset = dlsym(lib, "dvvk_reset");
    vk.destroy = dlsym(lib, "dvvk_destroy");
    return vk.create && vk.set_output && vk.render_sync && vk.reset && vk.destroy ? 0 : -1;
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

struct bo {
    struct gbm_bo *bo;
    int fd;
    uint32_t stride;
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

/* Memory order b,g,r,a (ARGB8888). */
static unsigned luma12(const uint8_t *p) { return (unsigned)p[1] * 16u + (p[0] & 15u); }
static unsigned chroma12(const uint8_t *p) { return (unsigned)p[2] * 16u + (p[0] >> 4); }

/* Kodi's CB1 adapter (DVBridgeGLES.cpp SourceChroma). */
static enum pl_chroma_location cb1_chroma(enum AVChromaLocation loc, int profile)
{
    if (loc == AVCHROMA_LOC_UNSPECIFIED && profile == 7)
        return PL_CHROMA_TOP_LEFT;
    if (loc == AVCHROMA_LOC_TOPLEFT)
        return PL_CHROMA_TOP_LEFT;
    if (loc == AVCHROMA_LOC_CENTER)
        return PL_CHROMA_CENTER;
    if (loc == AVCHROMA_LOC_TOP)
        return PL_CHROMA_TOP_CENTER;
    if (loc == AVCHROMA_LOC_BOTTOMLEFT)
        return PL_CHROMA_BOTTOM_LEFT;
    if (loc == AVCHROMA_LOC_BOTTOM)
        return PL_CHROMA_BOTTOM_CENTER;
    return PL_CHROMA_LEFT;
}

static int by_value(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* "median M ms, p95 P, mean A (N pictures)" */
static const char *stats(double *t, int n)
{
    static char text[128];
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += t[i];
    qsort(t, (size_t)n, sizeof(*t), by_value);
    snprintf(text, sizeof(text), "median %.2f ms, p95 %.2f, mean %.2f (%d pictures)", t[n / 2],
             t[(int)(n * 0.95)], sum / n, n);
    return text;
}

/* CB1's GL imports of decoded surfaces, kept per VA surface as Kodi keeps
 * them (its VAAPI textures and raw EL cache): imported once, not per frame. */
struct cb1_import {
    VASurfaceID surface;
    int layout_w, layout_h;
    struct dvbridge_dmabuf dmabuf;
    struct dvbridge_gl_frame frame;
};
static struct cb1_import cb1_imports[512];
static int cb1_import_count;

static struct dvbridge_gl_frame *cb1_import(pl_gpu gpu, const AVFrame *f, VASurfaceID surface, int w, int h)
{
    for (int i = 0; i < cb1_import_count; i++)
        if (cb1_imports[i].surface == surface && cb1_imports[i].layout_w == w && cb1_imports[i].layout_h == h)
            return &cb1_imports[i].frame;
    if (cb1_import_count == (int)(sizeof(cb1_imports) / sizeof(cb1_imports[0]))) {
        fprintf(stderr, "CB1 import cache full (%d surfaces)\n", cb1_import_count);
        return NULL;
    }
    struct cb1_import *c = &cb1_imports[cb1_import_count];
    memset(c, 0, sizeof(*c));
    if (!dvbridge_dmabuf_import(&c->dmabuf, f))
        return NULL;
    if (!dvbridge_gl_frame_import(&c->frame, gpu, c->dmabuf.layout, w, h, c->dmabuf.planes)) {
        dvbridge_dmabuf_release(&c->dmabuf);
        return NULL;
    }
    c->surface = surface;
    c->layout_w = w;
    c->layout_h = h;
    cb1_import_count++;
    return &c->frame;
}

static const char *master_status(enum cb1_master_status s)
{
    switch (s) {
    case CB1_MASTER_READY: return "ready";
    case CB1_MASTER_INVALID: return "invalid-metadata";
    case CB1_MASTER_UNSUPPORTED: return "unsupported-contract";
    case CB1_MASTER_GPU_UNAVAILABLE: return "gpu-unavailable";
    default: return "failed";
    }
}

static void dump_frame(const char *dir, const char *name, int n, const uint8_t *data)
{
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s-%d.bgra", dir, name, n);
    FILE *o = fopen(path, "wb");
    if (o) {
        fwrite(data, 1, (size_t)W * H * 4, o);
        fclose(o);
    }
}

int main(int argc, char **argv)
{
    int frames = 48, fel = 1, ipt = 0, contract = 1, flip = 0, warmup = 12;
    int run_cb1 = 1, run_vk = 1, opencl = 0;
    double start = 0;
    const char *file = NULL, *dump = NULL, *node = "/dev/dri/renderD128";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc)
            frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--start") && i + 1 < argc)
            start = atof(argv[++i]);
        else if (!strcmp(argv[i], "--fel") && i + 1 < argc)
            fel = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc)
            dump = argv[++i];
        else if (!strcmp(argv[i], "--node") && i + 1 < argc)
            node = argv[++i];
        else if (!strcmp(argv[i], "--engine") && i + 1 < argc)
            opencl = !strcmp(argv[++i], "cl");
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) {
            const char *e = argv[++i];
            run_cb1 = !strcmp(e, "cb1");
            run_vk = !strcmp(e, "vk");
        } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ipt"))
            ipt = 1;
        else if (!strcmp(argv[i], "--yblod"))
            contract = 0;
        else if (!strcmp(argv[i], "--flip"))
            flip = 1;
        else
            file = argv[i];
    }
    if (!file) {
        fprintf(stderr, "usage: dvvk-mr-compare [--frames N] [--start S] [--fel 0|1] [--ipt] [--yblod] "
                        "[--flip] [--dump DIR] [--node PATH] FILE\n");
        return 2;
    }
    if (opencl) {
        setenv("DV_CB1_CONTRACT", contract ? "1" : "0", 1);
        load_env("/usr/lib/kodi/native-dv.env");
        setenv("NEO_CACHE_DIR", "/tmp/neo-cache", 0);
        if (load_opencl())
            return 1;
    } else if (load_engine())
        return 1;

    struct dv_decoder dec;
    if (dv_decoder_open(&dec, file, node, start) < 0) {
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
    const EGLint ctx_attr[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 2, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attr);
    if (!ctx || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        fprintf(stderr, "EGL context failed\n");
        return 1;
    }
    PFNEGLCREATEIMAGEKHRPROC create_image = (void *)eglGetProcAddress("eglCreateImageKHR");
    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target = (void *)eglGetProcAddress("glEGLImageTargetTexture2DOES");

    /* CB1, as CDVBridgeGLES::Context::Initialize and Attach set it up */
    pl_log log = pl_log_create(PL_API_VER, pl_log_params(.log_cb = pl_log_color, .log_level = PL_LOG_WARN));
    pl_opengl gl = pl_opengl_create(log, pl_opengl_params(
        .get_proc_addr = (pl_voidfunc_t (*)(const char *))eglGetProcAddress,
        .egl_display = dpy, .egl_context = ctx));
    if (!gl) {
        fprintf(stderr, "pl_opengl_create failed\n");
        return 1;
    }
    struct dvbridge_retirement_owner *owner = dvbridge_retirement_owner_create(gl->gpu);
    struct dvbridge_renderer *cb1 = owner ? dvbridge_renderer_create_with_owner(owner) : NULL;
    if (!cb1 || !dvbridge_renderer_supports_mode(cb1, DVBRIDGE_MODE_MASTER_REFERENCE)) {
        fprintf(stderr, "CB1 renderer unavailable or without Master Reference on this GPU\n");
        return 1;
    }
    printf("GL: %s / %s\n", glGetString(GL_VENDOR), glGetString(GL_RENDERER));

    struct bo cb = {0}, out = {0};
    if (make_bo(gbm, &cb) || make_bo(gbm, &out)) {
        fprintf(stderr, "gbm_bo_create failed\n");
        return 1;
    }
    const EGLint img_attr[] = {EGL_WIDTH, W, EGL_HEIGHT, H, EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_ARGB8888,
                               EGL_DMA_BUF_PLANE0_FD_EXT, cb.fd, EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
                               EGL_DMA_BUF_PLANE0_PITCH_EXT, (EGLint)cb.stride, EGL_NONE};
    EGLImageKHR image = create_image(dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, img_attr);
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    image_target(GL_TEXTURE_2D, image);
    pl_tex target = pl_opengl_wrap(gl->gpu, pl_opengl_wrap_params(.texture = tex, .width = W, .height = H,
                                                                   .iformat = GL_RGBA8));
    if (!image || !target) {
        fprintf(stderr, "CB1 target import failed\n");
        return 1;
    }

    dvvk_engine *vke = NULL;
    void *ocl = NULL;
    if (opencl) {
        /* The OpenCL engine writes a GL texture (CL-GL interop): the output bo's. */
        const EGLint out_attr[] = {EGL_WIDTH, W, EGL_HEIGHT, H, EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_ARGB8888,
                                   EGL_DMA_BUF_PLANE0_FD_EXT, out.fd, EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
                                   EGL_DMA_BUF_PLANE0_PITCH_EXT, (EGLint)out.stride, EGL_NONE};
        EGLImageKHR out_image = create_image(dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, out_attr);
        GLuint out_tex;
        glGenTextures(1, &out_tex);
        glBindTexture(GL_TEXTURE_2D, out_tex);
        image_target(GL_TEXTURE_2D, out_image);
        glFinish();
        ocl = out_image ? cl.create((uintptr_t)dec.display, (uintptr_t)dpy, (uintptr_t)ctx, out_tex,
                                    "/usr/share/native-dv", DV_SOURCE_OUTPUT_BGRA) : NULL;
        if (!ocl) {
            fprintf(stderr, "OpenCL engine didn't start\n");
            return 1;
        }
        cl.set_el_scaler(ocl, 1);   /* Annex B, Master's EL filter */
        cl.set_fel(ocl, fel);
    } else if (vk.create(drm, dec.display, &vke) || vk.set_output(vke, 0, out.fd, out.stride)) {
        fprintf(stderr, "Vulkan engine didn't start\n");
        return 1;
    }

    uint8_t *a = malloc((size_t)W * H * 4), *b = malloc((size_t)W * H * 4);
    unsigned packet_id = 0;
    /* CB1's dvbridge_dv_metadata_output: refresh on the first picture, after
     * a step back or a gap over 0.1 s; a repeated picture keeps its value. */
    double last_pts = 0;
    int shown = 0, refresh = 1, n = 0, compared = 0, identical = 0, mismatched = 0, refused_cb1 = 0, refused_vk = 0;
    size_t luma_px = 0, chroma_px = 0, meta_px = 0, border_px = 0, total_px = 0;
    unsigned luma_max = 0, chroma_max = 0;
    size_t luma_hist[5] = {0}, chroma_hist[5] = {0};   /* |d| = 1, 2, 3-4, 5-16, more */
    double cb1_ms = 0, vk_ms = 0;
    double *cb1_t = calloc((size_t)frames + 1, sizeof(double)), *vk_t = calloc((size_t)frames + 1, sizeof(double));
    /* CB1's GPU time alone (EXT_disjoint_timer_query): its CPU work overlaps the
     * GPU in Kodi, so the CPU-clock time above is the synchronous worst case. */
    double *cb1_gpu = calloc((size_t)frames + 1, sizeof(double));
    int timed_cb1_gpu = 0;
    PFNGLGENQUERIESEXTPROC gen_queries = (void *)eglGetProcAddress("glGenQueriesEXT");
    PFNGLBEGINQUERYEXTPROC begin_query = (void *)eglGetProcAddress("glBeginQueryEXT");
    PFNGLENDQUERYEXTPROC end_query = (void *)eglGetProcAddress("glEndQueryEXT");
    PFNGLGETQUERYOBJECTUI64VEXTPROC query_result = (void *)eglGetProcAddress("glGetQueryObjectui64vEXT");
    GLuint query = 0;
    int gpu_timing = gen_queries && begin_query && end_query && query_result;
    if (gpu_timing)
        gen_queries(1, &query);
    int timed_cb1 = 0, timed_vk = 0;
    struct dv_pair p;
    while (n < frames && dv_decoder_next(&dec, &p) == 1) {
        n++;
        if (!p.dovi) {
            dv_pair_free(&p);
            continue;
        }
        AVFrame *el = fel ? p.el : NULL;
        unsigned w = (unsigned)p.bl->width, h = (unsigned)p.bl->height;
        if (w > W || h > H || (w != W && h != H)) {
            fprintf(stderr, "frame %d: %ux%u would be resized; Master compares unscaled pictures only\n", n, w, h);
            dv_pair_free(&p);
            break;
        }
        unsigned x = ((W - w) / 2) & ~1u, y = ((H - h) / 2) & ~1u;
        const void *meta = p.dovi->data;
        size_t meta_size = p.dovi->size;
        /* seconds, as Kodi gives CB1 (its 0.1 s discontinuity test) */
        double pts = p.bl->pts * av_q2d(dec.time_base);

        /* Both engines start from finished decodes. */
        vaSyncSurface(dec.display, p.bl_surface);
        if (el)
            vaSyncSurface(dec.display, p.el_surface);

        /* CB1 Master Reference, as CDVBridgeGLES::Prepare */
        struct dvbridge_gl_frame *blp = NULL, *elp = NULL;
        enum cb1_master_status cs = CB1_MASTER_FAILED;
        double t0 = now_ms();
        if (run_cb1 && (blp = cb1_import(gl->gpu, p.bl, p.bl_surface, (int)w, (int)h))) {
            struct dvbridge_gl_frame bl = *blp;
            dvbridge_gl_frame_set_chroma(&bl, cb1_chroma(p.bl->chroma_location, dec.profile));
            bl.frame.enhancement_layer = NULL;
            int el_ok = 1;
            struct dvbridge_gl_frame elf = {0};
            if (el) {
                el_ok = (elp = cb1_import(gl->gpu, el, p.el_surface, el->width, el->height)) != NULL;
                if (el_ok) {
                    elf = *elp;
                    /* CDVBridgeGLES::PrepareHardwareFrame: an unspecified EL is top-left */
                    dvbridge_gl_frame_set_chroma(&elf, cb1_chroma(el->chroma_location, 7));
                    bl.frame.enhancement_layer = &elf.frame;
                }
            }
            if (el_ok) {
                struct dvbridge_geometry g = {.source_width = (int)w, .source_height = (int)h, .x = (int)x,
                                              .y = (int)y, .width = (int)w, .height = (int)h,
                                              .target_width = W, .target_height = H};
                /* The options Kodi sets before each picture (CDVBridgeGLES::PrepareFrame
                 * with default preferences): optimized DV on, processing optimizations
                 * off, the EL "nearest" flag for an untransformed raw EL. */
                dvbridge_renderer_set_optimized_dv(cb1, true);
                dvbridge_renderer_set_processing_optimizations(cb1, false);
                dvbridge_renderer_set_fel_nearest(cb1, el != NULL);
                dvbridge_renderer_set_overlay_visible(cb1, false);
                if (gpu_timing)
                    begin_query(GL_TIME_ELAPSED_EXT, query);
                cs = (ipt ? dvbridge_render_master_reference_ipt : dvbridge_render_master_reference)(
                    cb1, &bl.frame, meta, meta_size, pts, el ? pts : NAN, g, target, flip, NULL);
                if (gpu_timing)
                    end_query(GL_TIME_ELAPSED_EXT);
                pl_gpu_finish(gl->gpu);
                if (gpu_timing && n > warmup) {
                    GLuint64 ns = 0;
                    query_result(query, GL_QUERY_RESULT_EXT, &ns);
                    cb1_gpu[timed_cb1_gpu++] = ns / 1e6;
                }
            }
        }
        double dt = now_ms() - t0;
        cb1_ms += dt;
        if (run_cb1 && n > warmup)
            cb1_t[timed_cb1++] = dt;
        if (cs == CB1_MASTER_READY)
            read_bo(&cb, a);


        /* yblod's Vulkan engine, as Kodi's yblod adapter drives it */
        int top_left = contract ? cb1_chroma(p.bl->chroma_location, dec.profile) == PL_CHROMA_TOP_LEFT
                                : p.bl->chroma_location == AVCHROMA_LOC_TOPLEFT;
        dvvk_frame f = {.plan = {.width = w, .height = h, .x = x, .y = y, .dw = w, .dh = h,
                                 .chroma_top_left = (uint32_t)top_left, .has_el = el != NULL, .nv12 = 0,
                                 .el_scaler = 1, .fel = (uint32_t)fel, .packet_id = packet_id,
                                 .refresh = (uint32_t)refresh, .pts = p.bl->pts, .white_nits = 203.0,
                                 .cb1_contract = (uint32_t)contract},
                        .bl_surface = p.bl_surface, .el_surface = el ? p.el_surface : 0xffffffffu,
                        .metadata = meta, .metadata_size = meta_size, .output = 0};
        if (shown && pts != last_pts)
            refresh = pts < last_pts || pts - last_pts > 0.1;
        else if (!shown)
            refresh = 1;
        f.plan.refresh = (uint32_t)refresh;
        uint32_t flags = 0;
        t0 = now_ms();
        int rv = -1;
        if (run_vk && opencl) {
            dv_source_geometry g = {.width = w, .height = h, .canvas_width = W, .canvas_height = H, .x = x, .y = y,
                                    .chroma_siting = (contract || !el) && top_left ? DV_SOURCE_CHROMA_TOP_LEFT
                                                                                   : DV_SOURCE_CHROMA_CENTER_LEFT,
                                    .destination_width = w, .destination_height = h};
            dv_source_input in = {sizeof(in), DV_SOURCE_INPUT_P010, 10u};
            rv = cl.submit(ocl, p.bl_surface, el ? p.el_surface : 0xffffffffu, w, h, meta, meta_size, (uint64_t)n,
                           p.bl->pts, &g, &in);
            if (!rv)
                rv = cl.present(ocl, (uint64_t)n, p.bl->pts, packet_id, refresh);
            if (cl.drain(ocl))
                rv = -8;
        } else if (run_vk)
            rv = vk.render_sync(vke, &f, &flags);
        dt = now_ms() - t0;
        vk_ms += dt;
        if (run_vk && n > warmup)
            vk_t[timed_vk++] = dt;
        if (!run_cb1 || !run_vk) {
            if (!run_cb1 && !rv) {
                packet_id = (packet_id + 1) & 15;
                last_pts = pts;
                shown++;
            }
            if (!run_vk && cs == CB1_MASTER_READY)
                dvbridge_render_commit(cb1);
            dv_pair_free(&p);
            continue;
        }

        if (cs != CB1_MASTER_READY || rv) {
            refused_cb1 += cs != CB1_MASTER_READY;
            refused_vk += rv != 0;
            if (refused_cb1 + refused_vk <= 4)
                printf("frame %d: CB1 %s, Vulkan %d (flags %u)\n", n, master_status(cs), rv, flags);
        } else {
            read_bo(&out, b);
            compared++;
            size_t diff = 0, first = (size_t)-1;
            for (size_t i = 0; i < (size_t)W * H; i++) {
                if (!memcmp(a + 4 * i, b + 4 * i, 3))
                    continue;
                diff++;
                if (first == (size_t)-1)
                    first = i;
                unsigned px = (unsigned)(i % W), py = (unsigned)(i / W);
                if (i < META_PIXELS) {
                    meta_px++;
                    continue;
                }
                if (px < x || px >= x + w || py < y || py >= y + h) {
                    border_px++;
                    continue;
                }
                unsigned dy = (unsigned)abs((int)luma12(a + 4 * i) - (int)luma12(b + 4 * i));
                unsigned dc = (unsigned)abs((int)chroma12(a + 4 * i) - (int)chroma12(b + 4 * i));
                if (dy) {
                    luma_px++;
                    luma_hist[dy == 1 ? 0 : dy == 2 ? 1 : dy <= 4 ? 2 : dy <= 16 ? 3 : 4]++;
                    if (dy > luma_max)
                        luma_max = dy;
                }
                if (dc) {
                    chroma_px++;
                    chroma_hist[dc == 1 ? 0 : dc == 2 ? 1 : dc <= 4 ? 2 : dc <= 16 ? 3 : 4]++;
                    if (dc > chroma_max)
                        chroma_max = dc;
                }
            }
            total_px += diff;
            if (!diff)
                identical++;
            else {
                mismatched++;
                if (mismatched <= 4) {
                    unsigned fx = (unsigned)(first % W), fy = (unsigned)(first / W);
                    const uint8_t *pa = a + 4 * first, *pb = b + 4 * first;
                    printf("frame %d: %zu pixels differ; first %u,%u cb1 Y%u C%u vk Y%u C%u\n", n, diff, fx, fy,
                           luma12(pa), chroma12(pa), luma12(pb), chroma12(pb));
                }
                if (dump && mismatched <= 2) {
                    dump_frame(dump, "cb1", n, a);
                    dump_frame(dump, "vk", n, b);
                }
            }
        }
        /* The picture was shown: both engines advance their packet state. */
        if (cs == CB1_MASTER_READY)
            dvbridge_render_commit(cb1);
        else
            dvbridge_renderer_reset(cb1);
        if (!rv) {
            packet_id = (packet_id + 1) & 15;
            last_pts = pts;
            shown++;
        }
        dv_pair_free(&p);
    }
    if (timed_cb1)
        printf("timing cb1: %s\n", stats(cb1_t, timed_cb1));
    if (timed_cb1_gpu)
        printf("timing cb1 gpu: %s\n", stats(cb1_gpu, timed_cb1_gpu));
    if (timed_vk)
        printf("timing vk:  %s\n", stats(vk_t, timed_vk));
    printf("{\"file\":\"%s\",\"profile\":%d,\"fel\":%d,\"ipt\":%d,\"cb1_contract\":%d,\"frames\":%d,\"compared\":%d,"
           "\"identical\":%d,\"mismatched\":%d,\"refused_cb1\":%d,\"refused_vk\":%d,"
           "\"differ_pixels\":%zu,\"metadata_row_pixels\":%zu,\"border_pixels\":%zu,"
           "\"luma_pixels\":%zu,\"luma_max\":%u,\"luma_hist\":[%zu,%zu,%zu,%zu,%zu],"
           "\"chroma_pixels\":%zu,\"chroma_max\":%u,\"chroma_hist\":[%zu,%zu,%zu,%zu,%zu],"
           "\"cb1_ms\":%.2f,\"vulkan_ms\":%.2f}\n",
           file, dec.profile, fel, ipt, contract, n, compared, identical, mismatched, refused_cb1, refused_vk,
           total_px, meta_px, border_px, luma_px, luma_max, luma_hist[0], luma_hist[1], luma_hist[2], luma_hist[3],
           luma_hist[4], chroma_px, chroma_max, chroma_hist[0], chroma_hist[1], chroma_hist[2], chroma_hist[3],
           chroma_hist[4], n ? cb1_ms / n : 0, n ? vk_ms / n : 0);
    for (int i = 0; i < cb1_import_count; i++) {
        dvbridge_gl_frame_release(&cb1_imports[i].frame);
        dvbridge_dmabuf_release(&cb1_imports[i].dmabuf);
    }
    if (vke)
        vk.destroy(vke);
    if (ocl)
        cl.destroy(ocl);
    dvbridge_renderer_destroy(cb1);
    dvbridge_retirement_owner_destroy(&owner);
    pl_tex_destroy(gl->gpu, &target);
    pl_opengl_destroy(&gl);
    pl_log_destroy(&log);
    dv_decoder_close(&dec);
    return mismatched || refused_cb1 || refused_vk ? 1 : 0;
}
