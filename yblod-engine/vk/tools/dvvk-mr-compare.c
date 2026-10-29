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
 *     --cb1-siting      give the Vulkan engine CB1's chroma siting (Profile 7
 *                       with unspecified siting is top-left there)
 *     --flip            CB1's flip_y (default off: row 0 is the HDMI top row)
 *     --dump DIR        on a mismatch, both frames as DIR/{cb1,vk}-N.bgra
 *     --node PATH       render node (default /dev/dri/renderD128)
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
    int frames = 48, fel = 1, ipt = 0, cb1_siting = 0, flip = 0;
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
        else if (!strcmp(argv[i], "--ipt"))
            ipt = 1;
        else if (!strcmp(argv[i], "--cb1-siting"))
            cb1_siting = 1;
        else if (!strcmp(argv[i], "--flip"))
            flip = 1;
        else
            file = argv[i];
    }
    if (!file) {
        fprintf(stderr, "usage: dvvk-mr-compare [--frames N] [--start S] [--fel 0|1] [--ipt] [--cb1-siting] "
                        "[--flip] [--dump DIR] [--node PATH] FILE\n");
        return 2;
    }
    if (load_engine())
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
    if (vk.create(drm, dec.display, &vke) || vk.set_output(vke, 0, out.fd, out.stride)) {
        fprintf(stderr, "Vulkan engine didn't start\n");
        return 1;
    }

    uint8_t *a = malloc((size_t)W * H * 4), *b = malloc((size_t)W * H * 4);
    unsigned packet_id = 0;
    int refresh = 1, n = 0, compared = 0, identical = 0, mismatched = 0, refused_cb1 = 0, refused_vk = 0;
    size_t luma_px = 0, chroma_px = 0, meta_px = 0, border_px = 0, total_px = 0;
    unsigned luma_max = 0, chroma_max = 0;
    size_t luma_hist[5] = {0}, chroma_hist[5] = {0};   /* |d| = 1, 2, 3-4, 5-16, more */
    double cb1_ms = 0, vk_ms = 0;
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
        double pts = (double)p.bl->pts;

        /* CB1 Master Reference, as CDVBridgeGLES::Prepare */
        struct dvbridge_dmabuf bl_import = {0}, el_import = {0};
        struct dvbridge_gl_frame bl = {0}, elf = {0};
        enum cb1_master_status cs = CB1_MASTER_FAILED;
        double t0 = now_ms();
        if (dvbridge_dmabuf_import(&bl_import, p.bl) &&
            dvbridge_gl_frame_import(&bl, gl->gpu, bl_import.layout, (int)w, (int)h, bl_import.planes)) {
            dvbridge_gl_frame_set_chroma(&bl, cb1_chroma(p.bl->chroma_location, dec.profile));
            int el_ok = 1;
            if (el) {
                el_ok = dvbridge_dmabuf_import(&el_import, el) &&
                        dvbridge_gl_frame_import(&elf, gl->gpu, el_import.layout, el->width, el->height,
                                                 el_import.planes);
                if (el_ok) {
                    dvbridge_gl_frame_set_chroma(&elf, cb1_chroma(el->chroma_location, 0));
                    bl.frame.enhancement_layer = &elf.frame;
                }
            }
            if (el_ok) {
                struct dvbridge_geometry g = {.source_width = (int)w, .source_height = (int)h, .x = (int)x,
                                              .y = (int)y, .width = (int)w, .height = (int)h,
                                              .target_width = W, .target_height = H};
                cs = (ipt ? dvbridge_render_master_reference_ipt : dvbridge_render_master_reference)(
                    cb1, &bl.frame, meta, meta_size, pts, el ? pts : NAN, g, target, flip, NULL);
                pl_gpu_finish(gl->gpu);
            }
        }
        cb1_ms += now_ms() - t0;
        if (cs == CB1_MASTER_READY)
            read_bo(&cb, a);
        if (elf.gpu)
            dvbridge_gl_frame_release(&elf);
        if (bl.gpu)
            dvbridge_gl_frame_release(&bl);
        dvbridge_dmabuf_release(&el_import);
        dvbridge_dmabuf_release(&bl_import);

        /* yblod's Vulkan engine, as Kodi's yblod adapter drives it */
        int top_left = cb1_siting ? cb1_chroma(p.bl->chroma_location, dec.profile) == PL_CHROMA_TOP_LEFT
                                  : p.bl->chroma_location == AVCHROMA_LOC_TOPLEFT;
        dvvk_frame f = {.plan = {.width = w, .height = h, .x = x, .y = y, .dw = w, .dh = h,
                                 .chroma_top_left = (uint32_t)top_left, .has_el = el != NULL, .nv12 = 0,
                                 .el_scaler = 1, .fel = (uint32_t)fel, .packet_id = packet_id,
                                 .refresh = (uint32_t)refresh, .pts = p.bl->pts, .white_nits = 203.0},
                        .bl_surface = p.bl_surface, .el_surface = el ? p.el_surface : 0xffffffffu,
                        .metadata = meta, .metadata_size = meta_size, .output = 0};
        uint32_t flags = 0;
        t0 = now_ms();
        int rv = vk.render_sync(vke, &f, &flags);
        vk_ms += now_ms() - t0;

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
            refresh = 0;
        }
        dv_pair_free(&p);
    }
    printf("{\"file\":\"%s\",\"profile\":%d,\"fel\":%d,\"ipt\":%d,\"cb1_siting\":%d,\"frames\":%d,\"compared\":%d,"
           "\"identical\":%d,\"mismatched\":%d,\"refused_cb1\":%d,\"refused_vk\":%d,"
           "\"differ_pixels\":%zu,\"metadata_row_pixels\":%zu,\"border_pixels\":%zu,"
           "\"luma_pixels\":%zu,\"luma_max\":%u,\"luma_hist\":[%zu,%zu,%zu,%zu,%zu],"
           "\"chroma_pixels\":%zu,\"chroma_max\":%u,\"chroma_hist\":[%zu,%zu,%zu,%zu,%zu],"
           "\"cb1_ms\":%.2f,\"vulkan_ms\":%.2f}\n",
           file, dec.profile, fel, ipt, cb1_siting, n, compared, identical, mismatched, refused_cb1, refused_vk,
           total_px, meta_px, border_px, luma_px, luma_max, luma_hist[0], luma_hist[1], luma_hist[2], luma_hist[3],
           luma_hist[4], chroma_px, chroma_max, chroma_hist[0], chroma_hist[1], chroma_hist[2], chroma_hist[3],
           chroma_hist[4], n ? cb1_ms / n : 0, n ? vk_ms / n : 0);
    vk.destroy(vke);
    dvbridge_renderer_destroy(cb1);
    dvbridge_retirement_owner_destroy(&owner);
    pl_tex_destroy(gl->gpu, &target);
    pl_opengl_destroy(&gl);
    pl_log_destroy(&log);
    dv_decoder_close(&dec);
    return mismatched || refused_cb1 || refused_vk ? 1 : 0;
}
