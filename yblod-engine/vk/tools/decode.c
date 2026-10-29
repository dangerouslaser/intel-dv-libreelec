/* See decode.h. */
#include "decode.h"
#include <stdlib.h>
#include <libavutil/dovi_meta.h>
#include <libavutil/opt.h>
#include <stdio.h>
#include <string.h>

#define QUEUE 32
static AVFrame *queue[2][QUEUE];
static int queued[2];

static enum AVPixelFormat pick_vaapi(AVCodecContext *c, const enum AVPixelFormat *fmts)
{
    (void)c;
    for (; *fmts != AV_PIX_FMT_NONE; fmts++)
        if (*fmts == AV_PIX_FMT_VAAPI)
            return *fmts;
    fprintf(stderr, "decode: no VAAPI format offered\n");
    return AV_PIX_FMT_NONE;
}

static AVCodecContext *open_decoder(struct dv_decoder *d, const AVCodecParameters *par)
{
    const AVCodec *codec = avcodec_find_decoder(par->codec_id);
    AVCodecContext *c = avcodec_alloc_context3(codec);
    if (!c || avcodec_parameters_to_context(c, par) < 0)
        return NULL;
    c->hw_device_ctx = av_buffer_ref(d->device);
    c->get_format = pick_vaapi;
    /* Frames the caller may hold at once beyond the decoder's own (more for
     * dvvk-mr-compare --throughput's preloaded batch). */
    const char *extra = getenv("DV_EXTRA_HW_FRAMES");
    c->extra_hw_frames = extra ? atoi(extra) : 24;
    c->pkt_timebase = d->time_base;
    if (avcodec_open2(c, codec, NULL) < 0)
        return NULL;
    return c;
}

/* Without ffmpeg-yblod's dovi_split (CB1's FFmpeg): the single-track Profile
 * 7 layout split here. Length-prefixed HEVC; the BL keeps everything but NAL
 * 63, the EL is each NAL 63's payload (the EL's own NAL) in Annex B. */
static int nal_size_len = 4;

static int split_packet(const AVPacket *in, AVPacket *out, int el)
{
    size_t room = (size_t)in->size + 64 * 4;
    if (av_new_packet(out, (int)room) < 0)
        return AVERROR(ENOMEM);
    size_t n = 0;
    const uint8_t *q = in->data, *end = in->data + in->size;
    while (end - q > nal_size_len) {
        uint32_t len = 0;
        for (int i = 0; i < nal_size_len; i++)
            len = len << 8 | q[i];
        q += nal_size_len;
        if (len < 2 || len > (size_t)(end - q))
            break;
        int type = (q[0] >> 1) & 63;
        if (el && type == 63 && len > 2) {
            static const uint8_t start[4] = {0, 0, 0, 1};
            if (n + 4 + len - 2 > room)
                break;
            memcpy(out->data + n, start, 4);
            memcpy(out->data + n + 4, q + 2, len - 2);
            n += 4 + len - 2;
        } else if (!el && type != 63) {
            memcpy(out->data + n, q - nal_size_len, nal_size_len + len);
            n += nal_size_len + len;
        }
        q += len;
    }
    av_packet_copy_props(out, in);
    out->size = (int)n;
    return 0;
}

static AVCodecContext *open_el_decoder(struct dv_decoder *d)
{
    AVCodecParameters *par = avcodec_parameters_alloc();
    avcodec_parameters_copy(par, d->fmt->streams[d->stream]->codecpar);
    av_freep(&par->extradata);
    par->extradata_size = 0;
    AVCodecContext *c = open_decoder(d, par);
    avcodec_parameters_free(&par);
    return c;
}

static AVBSFContext *open_split(struct dv_decoder *d, const char *mode)
{
    const AVBitStreamFilter *f = av_bsf_get_by_name("dovi_split");
    AVBSFContext *b = NULL;
    if (!f)
        return NULL;
    if (av_bsf_alloc(f, &b) < 0)
        return NULL;
    avcodec_parameters_copy(b->par_in, d->fmt->streams[d->stream]->codecpar);
    b->time_base_in = d->time_base;
    av_opt_set(b->priv_data, "mode", mode, 0);
    if (av_bsf_init(b) < 0)
        return NULL;
    return b;
}

int dv_decoder_open(struct dv_decoder *d, const char *path, const char *node, double start)
{
    memset(d, 0, sizeof(*d));
    queued[0] = queued[1] = 0;
    if (avformat_open_input(&d->fmt, path, NULL, NULL) < 0 || avformat_find_stream_info(d->fmt, NULL) < 0)
        return -1;
    d->stream = av_find_best_stream(d->fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (d->stream < 0)
        return -1;
    AVStream *st = d->fmt->streams[d->stream];
    d->time_base = st->time_base;
    const AVPacketSideData *sd = av_packet_side_data_get(st->codecpar->coded_side_data,
                                                         st->codecpar->nb_coded_side_data, AV_PKT_DATA_DOVI_CONF);
    if (sd) {
        const AVDOVIDecoderConfigurationRecord *cfg = (const void *)sd->data;
        d->profile = cfg->dv_profile;
        d->el_present = cfg->el_present_flag && cfg->dv_profile == 7;
    }
    if (av_hwdevice_ctx_create(&d->device, AV_HWDEVICE_TYPE_VAAPI, node, NULL, 0) < 0)
        return -1;
    d->display = ((AVVAAPIDeviceContext *)((AVHWDeviceContext *)d->device->data)->hwctx)->display;
    if (d->el_present && av_bsf_get_by_name("dovi_split")) {
        d->bsf[0] = open_split(d, "bl_rpu");
        d->bsf[1] = open_split(d, "el");
        if (!d->bsf[0] || !d->bsf[1])
            return -1;
        d->dec[0] = open_decoder(d, d->bsf[0]->par_out);
        d->dec[1] = open_decoder(d, d->bsf[1]->par_out);
        if (!d->dec[0] || !d->dec[1])
            return -1;
    } else if (d->el_present) {
        const AVCodecParameters *par = st->codecpar;
        if (par->extradata_size < 23 || par->extradata[0] != 1) {
            fprintf(stderr, "decode: Profile 7 needs hvcC (MKV/MP4) without dovi_split\n");
            return -1;
        }
        nal_size_len = (par->extradata[21] & 3) + 1;
        d->split = 1;
        d->dec[0] = open_decoder(d, par);
        d->dec[1] = open_el_decoder(d);
        if (!d->dec[0] || !d->dec[1])
            return -1;
    } else {
        d->dec[0] = open_decoder(d, st->codecpar);
        if (!d->dec[0])
            return -1;
    }
    if (start > 0)
        av_seek_frame(d->fmt, d->stream, (int64_t)(start / av_q2d(d->time_base)), AVSEEK_FLAG_BACKWARD);
    return 0;
}

static int drain_decoder(struct dv_decoder *d, int layer)
{
    for (;;) {
        if (queued[layer] >= QUEUE)
            return 0;
        AVFrame *f = av_frame_alloc();
        int r = avcodec_receive_frame(d->dec[layer], f);
        if (r < 0) {
            av_frame_free(&f);
            return r == AVERROR(EAGAIN) || r == AVERROR_EOF ? 0 : r;
        }
        queue[layer][queued[layer]++] = f;
    }
}

static int feed(struct dv_decoder *d, int layer, AVPacket *pkt)
{
    if (d->split) {
        if (!pkt)
            return avcodec_send_packet(d->dec[layer], NULL) == AVERROR_EOF ? 0 : 0;
        AVPacket *out = av_packet_alloc();
        int r = split_packet(pkt, out, layer);
        if (!r && out->size)
            r = avcodec_send_packet(d->dec[layer], out);
        av_packet_free(&out);
        return r == AVERROR(EAGAIN) ? 0 : r;
    }
    if (!d->bsf[0]) {
        int r = avcodec_send_packet(d->dec[0], pkt);
        return r == AVERROR_EOF ? 0 : r;
    }
    int r = av_bsf_send_packet(d->bsf[layer], pkt);
    if (r < 0)
        return r;
    AVPacket *out = av_packet_alloc();
    while ((r = av_bsf_receive_packet(d->bsf[layer], out)) >= 0) {
        int s = avcodec_send_packet(d->dec[layer], out);
        av_packet_unref(out);
        if (s < 0 && s != AVERROR(EAGAIN)) {
            av_packet_free(&out);
            return s;
        }
    }
    av_packet_free(&out);
    if (r == AVERROR_EOF && pkt == NULL)
        avcodec_send_packet(d->dec[layer], NULL);
    return 0;
}

static AVFrame *pop(int layer)
{
    AVFrame *f = queue[layer][0];
    memmove(queue[layer], queue[layer] + 1, sizeof(queue[0][0]) * (size_t)--queued[layer]);
    return f;
}

int dv_decoder_next(struct dv_decoder *d, struct dv_pair *p)
{
    int layers = d->dec[1] ? 2 : 1;
    memset(p, 0, sizeof(*p));
    for (;;) {
        /* A Profile 7 track whose configuration announces an EL that the
         * stream doesn't carry (RPU-only remuxes): play the BL alone. */
        if (layers == 2 && queued[1] == 0 && (queued[0] >= 8 || d->eof == 2)) {
            fprintf(stderr, "decode: no enhancement-layer frames; base layer only\n");
            avcodec_free_context(&d->dec[1]);
            av_bsf_free(&d->bsf[1]);
            layers = 1;
        }
        int ready = queued[0] > 0 && (layers == 1 || queued[1] > 0);
        if (ready) {
            p->bl = pop(0);
            if (layers == 2)
                p->el = pop(1);
            p->bl_surface = (VASurfaceID)(uintptr_t)p->bl->data[3];
            p->el_surface = p->el ? (VASurfaceID)(uintptr_t)p->el->data[3] : VA_INVALID_SURFACE;
            p->dovi = av_frame_get_side_data(p->bl, AV_FRAME_DATA_DOVI_METADATA);
            if (!p->dovi && p->el)
                p->dovi = av_frame_get_side_data(p->el, AV_FRAME_DATA_DOVI_METADATA);
            if (p->el && p->el->pts != p->bl->pts)
                fprintf(stderr, "decode: BL pts %lld paired with EL pts %lld\n", (long long)p->bl->pts,
                        (long long)p->el->pts);
            return 1;
        }
        if (d->eof == 2)
            return 0;
        if (d->eof) {
            d->eof = 2;
            for (int l = 0; l < layers; l++) {
                feed(d, l, NULL);
                if (!d->bsf[0])
                    avcodec_send_packet(d->dec[0], NULL);
                drain_decoder(d, l);
            }
            continue;
        }
        AVPacket *pkt = av_packet_alloc();
        int r = av_read_frame(d->fmt, pkt);
        if (r < 0) {
            av_packet_free(&pkt);
            d->eof = 1;
            continue;
        }
        if (pkt->stream_index == d->stream) {
            for (int l = 0; l < layers; l++) {
                AVPacket *copy = av_packet_clone(pkt);
                r = feed(d, l, copy);
                av_packet_free(&copy);
                if (r < 0)
                    fprintf(stderr, "decode: layer %d send: %s\n", l, av_err2str(r));
                if (r < 0) {
                    av_packet_free(&pkt);
                    return r;
                }
                if ((r = drain_decoder(d, l)) < 0) {
                    fprintf(stderr, "decode: layer %d receive: %s\n", l, av_err2str(r));
                    av_packet_free(&pkt);
                    return r;
                }
            }
        }
        av_packet_free(&pkt);
    }
}

void dv_pair_free(struct dv_pair *p)
{
    av_frame_free(&p->bl);
    av_frame_free(&p->el);
}

void dv_decoder_close(struct dv_decoder *d)
{
    for (int l = 0; l < 2; l++) {
        while (queued[l])
            av_frame_free(&queue[l][--queued[l]]);
        avcodec_free_context(&d->dec[l]);
        av_bsf_free(&d->bsf[l]);
    }
    av_buffer_unref(&d->device);
    avformat_close_input(&d->fmt);
}
