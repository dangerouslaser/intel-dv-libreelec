/* Fit the DV engine to a reference (AM9 Pro) by rendering exact frames with candidate filters.
 * Runs natively on LibreELEC against the system libplacebo/FFmpeg (the same code Kodi uses).
 *
 *   dvfit thumbs <file.mkv> <outdir> <first> <last>
 *       BL+reshape renders, every 16th pixel, for frame matching.
 *   dvfit render <file.mkv> <el.hevc|-> <outdir> <chroma,filters> <el,filters> <idx>...
 *       For each idx: one RGB16 PQ frame per base-layer chroma filter (EL fixed to the first EL
 *       filter) and per EL filter (chroma fixed to the first chroma filter).
 *       Chroma filters are libplacebo presets. EL filters are CPU kernels with the DV layer alignment
 *       libplacebo uses (luma co-sited horizontally, chroma left-sited), or "native" for libplacebo's
 *       own EL path (which then shares the chroma filter).
 */
#define PL_LIBAV_IMPLEMENTATION 1
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/filters.h>
#include <libplacebo/opengl.h>
#include <libplacebo/utils/libav.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct dec { AVFormatContext *fmt; AVCodecContext *cc; int st; AVPacket *pkt; int eof; };
static int dec_open(struct dec *d, const char *path)
{
    memset(d, 0, sizeof(*d));
    if (avformat_open_input(&d->fmt, path, NULL, NULL) < 0 || avformat_find_stream_info(d->fmt, NULL) < 0) return -1;
    d->st = av_find_best_stream(d->fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    const AVCodec *c = avcodec_find_decoder(d->fmt->streams[d->st]->codecpar->codec_id);
    d->cc = avcodec_alloc_context3(c);
    avcodec_parameters_to_context(d->cc, d->fmt->streams[d->st]->codecpar);
    d->cc->thread_count = 4;
    d->pkt = av_packet_alloc();
    return avcodec_open2(d->cc, c, NULL);
}
static AVFrame *dec_next(struct dec *d)
{
    AVFrame *f = av_frame_alloc();
    for (;;) {
        int r = avcodec_receive_frame(d->cc, f);
        if (r == 0) return f;
        if (r == AVERROR_EOF) { av_frame_free(&f); return NULL; }
        if (d->eof) { avcodec_send_packet(d->cc, NULL); continue; }
        if (av_read_frame(d->fmt, d->pkt) < 0) { d->eof = 1; avcodec_send_packet(d->cc, NULL); continue; }
        if (d->pkt->stream_index == d->st) avcodec_send_packet(d->cc, d->pkt);
        av_packet_unref(d->pkt);
    }
}

static pl_gpu gpu;
static pl_renderer rr;
static pl_tex out, el4k;
static uint16_t *px;

static void render(const struct pl_frame *img, const struct pl_filter_config *chroma)
{
    struct pl_frame dst = {.num_planes = 1, .planes = {{.texture = out, .components = 4, .component_mapping = {0, 1, 2, 3}}},
                           .repr = pl_color_repr_rgb, .color = img->color};
    struct pl_render_params p = pl_render_default_params;
    p.min_fbo_precision = 32;
    p.peak_detect_params = NULL;
    p.dither_params = NULL;
    p.frame_mixer = NULL;
    p.skip_caching_single_frame = true;
    p.border = PL_CLEAR_SKIP;
    if (chroma) p.plane_upscaler = chroma;
    if (!pl_render_image(rr, img, &dst, &p)) { fprintf(stderr, "render failed\n"); exit(1); }
    pl_tex_download(gpu, pl_tex_transfer_params(.tex = out, .ptr = px));
}
static void save(const char *dir, const char *name, int idx, int thumb)
{
    char path[512];
    snprintf(path, sizeof(path), thumb ? "%s/thumb-%d.u16" : "%s/%d-%s.rgb16", dir, idx, name);
    FILE *f = fopen(path, "wb");
    if (thumb) {
        for (int y = 0; y < 2160; y += 16) for (int x = 0; x < 3840; x += 16) fwrite(&px[4 * ((long) y * 3840 + x)], 2, 3, f);
    } else {
        /* centre quarter (rows 540..1619, cols 960..2879) keeps the files small */
        for (int y = 540; y < 1620; y++) for (int x = 960; x < 2880; x++) fwrite(&px[4 * ((long) y * 3840 + x)], 2, 3, f);
    }
    fclose(f);
}

/* ---- CPU enhancement-layer upscale with the DV alignment ---- */
static double k_l3(double x) { x = fabs(x); if (x < 1e-9) return 1; if (x >= 3) return 0; return 3 * sin(M_PI * x) * sin(M_PI * x / 3) / (M_PI * M_PI * x * x); }
static double k_lin(double x) { x = fabs(x); return x < 1 ? 1 - x : 0; }
static double cubic(double x, double B, double C)
{
    x = fabs(x);
    if (x < 1) return ((12 - 9*B - 6*C) * x*x*x + (-18 + 12*B + 6*C) * x*x + (6 - 2*B)) / 6;
    if (x < 2) return ((-B - 6*C) * x*x*x + (6*B + 30*C) * x*x + (-12*B - 48*C) * x + (8*B + 24*C)) / 6;
    return 0;
}
static double k_cr(double x) { return cubic(x, 0, 0.5); }
static double k_mitchell(double x) { return cubic(x, 1.0/3, 1.0/3); }
static double k_bspline(double x) { return cubic(x, 1, 0); }
static double k_spline36(double x)
{
    x = fabs(x);
    if (x < 1) return ((13.0/11 * x - 453.0/209) * x - 3.0/209) * x + 1;
    if (x < 2) { x -= 1; return ((-6.0/11 * x + 270.0/209) * x - 156.0/209) * x; }
    if (x < 3) { x -= 2; return ((1.0/11 * x - 45.0/209) * x + 26.0/209) * x; }
    return 0;
}
struct kern { const char *name; double (*k)(double); int r; };
static const struct kern kerns[] = {
    {"lanczos", k_l3, 3}, {"spline36", k_spline36, 3}, {"catmull_rom", k_cr, 2}, {"mitchell", k_mitchell, 2},
    {"bspline", k_bspline, 2}, {"bilinear", k_lin, 1},
};
static const struct kern *find_kern(const char *n) { for (size_t i = 0; i < sizeof(kerns)/sizeof(kerns[0]); i++) if (!strcmp(kerns[i].name, n)) return &kerns[i]; return NULL; }

/* src: w x h, 10-bit codes in uint16 (stride in elements). Output codes/1023 scaled into 16-bit unorm. */
static void upscale_plane(const uint16_t *src, int sw, int sh, int stride, const struct kern *k,
                          double scale, double shx, int comp, uint16_t *dst4k)
{
    const int dw = 3840, dh = 2160, r = k->r;
    double *tmp = malloc(sizeof(double) * dw * sh);
    for (int x = 0; x < dw; x++) {
        double c = (x + 0.5) / scale - 0.5 + shx;
        int i0 = (int) floor(c) - r + 1;
        double w[8], ws = 0;
        for (int t = 0; t < 2 * r; t++) ws += (w[t] = k->k(c - (i0 + t)));
        for (int y = 0; y < sh; y++) {
            double s = 0;
            for (int t = 0; t < 2 * r; t++) { int ii = i0 + t; ii = ii < 0 ? 0 : ii >= sw ? sw - 1 : ii; s += w[t] * src[y * stride + ii]; }
            tmp[y * dw + x] = s / ws;
        }
    }
    for (int y = 0; y < dh; y++) {
        double c = (y + 0.5) / scale - 0.5;
        int i0 = (int) floor(c) - r + 1;
        double w[8], ws = 0;
        for (int t = 0; t < 2 * r; t++) ws += (w[t] = k->k(c - (i0 + t)));
        for (int x = 0; x < dw; x++) {
            double s = 0;
            for (int t = 0; t < 2 * r; t++) { int ii = i0 + t; ii = ii < 0 ? 0 : ii >= sh ? sh - 1 : ii; s += w[t] * tmp[(size_t) ii * dw + x]; }
            double v = s / ws / 1023.0 * 65535.0;
            dst4k[((size_t) y * dw + x) * 4 + comp] = (uint16_t) (v < 0 ? 0 : v > 65535 ? 65535 : lrint(v));
        }
    }
    free(tmp);
}
static int el_prescaled(const AVFrame *ef, const struct kern *k, struct pl_frame *el)
{
    static uint16_t *buf;
    if (!buf) buf = calloc((size_t) 3840 * 2160 * 4, 2);
    /* luma: co-sited horizontally (source x = X/2), centered vertically; chroma: left sited (x = X/4) */
    upscale_plane((const uint16_t *) ef->data[0], ef->width, ef->height, ef->linesize[0] / 2, k, 2.0, 0.25, 0, buf);
    upscale_plane((const uint16_t *) ef->data[1], ef->width / 2, ef->height / 2, ef->linesize[1] / 2, k, 4.0, 0.375, 1, buf);
    upscale_plane((const uint16_t *) ef->data[2], ef->width / 2, ef->height / 2, ef->linesize[2] / 2, k, 4.0, 0.375, 2, buf);
    pl_tex_upload(gpu, pl_tex_transfer_params(.tex = el4k, .ptr = buf));
    el->num_planes = 1;
    el->planes[0] = (struct pl_plane) {.texture = el4k, .components = 3, .component_mapping = {0, 1, 2}};
    el->repr.bits = (struct pl_bit_encoding) {.sample_depth = 16, .color_depth = 16};
    el->crop = (pl_rect2df) {0, 0, 3840, 2160};
    return 0;
}

static int split(char *s, char **v, int n) { int c = 0; for (char *t = strtok(s, ","); t && c < n; t = strtok(NULL, ",")) v[c++] = t; return c; }

int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "usage: see source\n"); return 1; }
    pl_log log = pl_log_create(PL_API_VER, pl_log_params(.log_cb = pl_log_simple, .log_level = PL_LOG_WARN));
    EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    EGLint maj, min; eglInitialize(dpy, &maj, &min); eglBindAPI(EGL_OPENGL_ES_API);
    const EGLint ca[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 2, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ca);
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);
    pl_opengl gl = pl_opengl_create(log, pl_opengl_params(.get_proc_addr = (pl_voidfunc_t (*)(const char *)) eglGetProcAddress,
                                                          .egl_display = dpy, .egl_context = ctx));
    gpu = gl->gpu; rr = pl_renderer_create(log, gpu);
    pl_fmt f16 = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 16, 16, PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    out = pl_tex_create(gpu, pl_tex_params(.w = 3840, .h = 2160, .format = f16, .renderable = true, .host_readable = true));
    el4k = pl_tex_create(gpu, pl_tex_params(.w = 3840, .h = 2160, .sampleable = true, .host_writable = true,
                                            .format = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 16, 16, PL_FMT_CAP_SAMPLEABLE)));
    px = malloc(3840L * 2160 * 8);
    int thumbs = !strcmp(argv[1], "thumbs");
    struct dec bd, ed; int have_el = 0;
    if (dec_open(&bd, argv[2])) { fprintf(stderr, "open failed\n"); return 1; }
    const char *outdir; int first = 0, last = 0, want[256], nwant = 0;
    char *cf[32], *ef[32]; int ncf = 0, nef = 0;
    if (thumbs) { outdir = argv[3]; first = atoi(argv[4]); last = atoi(argv[5]); }
    else {
        if (strcmp(argv[3], "-")) { if (dec_open(&ed, argv[3])) { fprintf(stderr, "el open failed\n"); return 1; } have_el = 1; }
        outdir = argv[4]; ncf = split(argv[5], cf, 32); nef = split(argv[6], ef, 32);
        for (int i = 7; i < argc && nwant < 256; i++) { want[nwant] = atoi(argv[i]); if (want[nwant] > last) last = want[nwant]; nwant++; }
        first = 1 << 30;
    }
    pl_tex bt[4] = {0}, et[4] = {0};
    for (int idx = 0; idx <= last; idx++) {
        AVFrame *bf = dec_next(&bd), *eff = have_el ? dec_next(&ed) : NULL;
        if (!bf) break;
        int hit = idx >= first;
        for (int i = 0; i < nwant; i++) hit |= want[i] == idx;
        if (hit) {
            struct pl_frame bl, el;
            pl_map_avframe_ex(gpu, &bl, pl_avframe_params(.frame = bf, .tex = bt, .map_dovi = true));
            if (thumbs) { render(&bl, NULL); save(outdir, NULL, idx, 1); }
            else {
                if (eff) pl_map_avframe_ex(gpu, &el, pl_avframe_params(.frame = eff, .tex = et, .map_dovi = false));
                const struct pl_filter_config *c0 = pl_find_filter_config(cf[0], PL_FILTER_UPSCALING);
                /* chroma sweep, EL fixed to ef[0] */
                for (int c = 0; c < ncf; c++) {
                    const struct pl_filter_config *fc = pl_find_filter_config(cf[c], PL_FILTER_UPSCALING);
                    if (!fc) { fprintf(stderr, "unknown chroma filter %s\n", cf[c]); continue; }
                    struct pl_frame elv = el;
                    if (eff && strcmp(ef[0], "native")) { el_prescaled(eff, find_kern(ef[0]), &elv); }
                    bl.enhancement_layer = eff ? &elv : NULL;
                    render(&bl, fc);
                    char name[128]; snprintf(name, sizeof(name), "c_%s-e_%s", cf[c], eff ? ef[0] : "none"); save(outdir, name, idx, 0);
                }
                /* EL sweep, chroma fixed to cf[0] */
                for (int e = 1; eff && e < nef; e++) {
                    struct pl_frame elv = el;
                    if (strcmp(ef[e], "native")) {
                        const struct kern *k = find_kern(ef[e]);
                        if (!k) { fprintf(stderr, "unknown EL kernel %s\n", ef[e]); continue; }
                        el_prescaled(eff, k, &elv);
                    }
                    bl.enhancement_layer = &elv;
                    render(&bl, c0);
                    char name[128]; snprintf(name, sizeof(name), "c_%s-e_%s", cf[0], ef[e]); save(outdir, name, idx, 0);
                }
                if (eff) pl_unmap_avframe(gpu, &el);
            }
            pl_unmap_avframe(gpu, &bl);
            fprintf(stderr, "frame %d\n", idx);
        }
        av_frame_free(&bf); av_frame_free(&eff);
    }
    return 0;
}
