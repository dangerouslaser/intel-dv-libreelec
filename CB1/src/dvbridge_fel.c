/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Split and hardware-decode the enhancement layer; pair surfaces by exact presentation time. */
#include "dvbridge_fel.h"
#include "dvbridge_fel_key_match.h"
#include <libavcodec/bsf.h>
#include <libavutil/hwcontext.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define QUEUE_MAX 32
#define PACKET_BYTES_MAX (64u * 1024u * 1024u)
#define MAP_MAX (QUEUE_MAX * 2)

/* Independently coded layers can have different decode
 * order and IDR boundaries. Use FFmpeg's HEVC parser rather than assigning
 * the BL packet timestamp blindly to the encapsulated EL access unit. */
struct picture_order {
    AVCodecParserContext *parser;
    AVCodecContext *context;
    int64_t offset, maximum;
    bool seen;
};

static bool picture_identity(struct picture_order *order, const AVPacket *packet, int64_t *identity)
{
    uint8_t *output;
    int output_size;
    order->parser->output_picture_number = INT_MIN;
    int used = av_parser_parse2(order->parser, order->context, &output, &output_size,
                               packet->data, packet->size, packet->pts, packet->dts, packet->pos);
    int poc = order->parser->output_picture_number;
    if (used != packet->size || output_size <= 0 || poc == INT_MIN)
        return false;
    if (order->seen && poc == 0 && order->parser->key_frame == 1) {
        if (order->maximum == INT64_MAX)
            return false;
        order->offset = order->maximum + 1;
    }
    if (poc > 0 && order->offset > INT64_MAX - poc)
        return false;
    *identity = order->offset + poc;
    if (!order->seen || *identity > order->maximum)
        order->maximum = *identity;
    order->seen = true;
    return true;
}

struct dvbridge_fel {
    AVBSFContext *split;
    AVCodecContext *decoder;
    AVPacket *packets[QUEUE_MAX];
    AVFrame *frames[QUEUE_MAX];
    unsigned num_packets, num_frames;
    size_t packet_bytes;
    bool opened, failed, draining, sent_eof, eof;
    struct picture_order base_order, enhancement_order;
    int64_t packet_order[QUEUE_MAX];
    struct { int64_t order, pts; } times[MAP_MAX];
    unsigned num_times;
    bool synchronizing;
    int64_t anchor_pts;
    unsigned preroll_packets;
    bool sync_key, sync_el_key;
    unsigned sync_key_age;
    int sync_key_poc;
    int64_t sync_key_identity, sync_key_pts;
    int sync_base_poc[QUEUE_MAX], sync_el_poc[QUEUE_MAX];
};

static enum AVPixelFormat hardware_format(AVCodecContext *ctx, const enum AVPixelFormat *formats)
{
    bool found = false;
    for (; *formats != AV_PIX_FMT_NONE; ++formats)
        found |= *formats == AV_PIX_FMT_VAAPI;
    if (!found || !ctx->hw_device_ctx)
        return AV_PIX_FMT_NONE;
    AVBufferRef *frames = NULL;
    if (avcodec_get_hw_frames_parameters(ctx, ctx->hw_device_ctx, AV_PIX_FMT_VAAPI, &frames) < 0)
        return AV_PIX_FMT_NONE;
    AVHWFramesContext *hw = (void *)frames->data;
    hw->initial_pool_size += QUEUE_MAX + 4;
    if (av_hwframe_ctx_init(frames) < 0) {
        av_buffer_unref(&frames);
        return AV_PIX_FMT_NONE;
    }
    av_buffer_unref(&ctx->hw_frames_ctx);
    ctx->hw_frames_ctx = frames;
    return AV_PIX_FMT_VAAPI;
}

static void clear_packets(struct dvbridge_fel *f)
{
    for (unsigned i = 0; i < f->num_packets; ++i)
        av_packet_free(&f->packets[i]);
    f->num_packets = 0;
    f->packet_bytes = 0;
}

static void clear_queues(struct dvbridge_fel *f)
{
    clear_packets(f);
    for (unsigned i = 0; i < f->num_frames; ++i)
        av_frame_free(&f->frames[i]);
    f->num_packets = f->num_frames = 0;
    f->packet_bytes = 0;
    f->failed = f->draining = f->sent_eof = f->eof = false;
    f->num_times = 0;
    f->sync_key = f->sync_el_key = false;
}

void dvbridge_fel_reset(struct dvbridge_fel *f)
{
    if (!f)
        return;
    clear_queues(f);
    f->synchronizing = true;
    f->anchor_pts = AV_NOPTS_VALUE;
    f->preroll_packets = 0;
    struct picture_order *orders[] = {&f->base_order, &f->enhancement_order};
    for (unsigned i = 0; i < 2; ++i) {
        struct picture_order *order = orders[i];
        av_parser_close(order->parser);
        order->parser = av_parser_init(AV_CODEC_ID_HEVC);
        if (order->parser)
            order->parser->flags = PARSER_FLAG_COMPLETE_FRAMES;
        else
            f->failed = true;
        order->offset = order->maximum = 0;
        order->seen = false;
    }
    if (f->split)
        av_bsf_flush(f->split);
    if (f->opened)
        avcodec_flush_buffers(f->decoder);
}

void dvbridge_fel_destroy(struct dvbridge_fel *f)
{
    if (!f)
        return;
    clear_queues(f);
    av_parser_close(f->base_order.parser);
    av_parser_close(f->enhancement_order.parser);
    avcodec_free_context(&f->base_order.context);
    avcodec_free_context(&f->enhancement_order.context);
    avcodec_free_context(&f->decoder);
    av_bsf_free(&f->split);
    free(f);
}

struct dvbridge_fel *dvbridge_fel_create(const AVCodecParameters *p, AVRational time_base)
{
    if (!p || p->codec_id != AV_CODEC_ID_HEVC || time_base.num <= 0 || time_base.den <= 0)
        return NULL;
    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
    if (!codec)
        return NULL;
    struct dvbridge_fel *f = calloc(1, sizeof(*f));
    if (!f)
        return NULL;
    f->synchronizing = true;
    f->anchor_pts = AV_NOPTS_VALUE;
    if (av_bsf_list_parse_str("dovi_split=mode=el", &f->split) < 0 ||
        avcodec_parameters_copy(f->split->par_in, p) < 0)
        goto fail;
    f->split->time_base_in = time_base;
    if (av_bsf_init(f->split) < 0)
        goto fail;
    f->decoder = avcodec_alloc_context3(codec);
    if (!f->decoder || avcodec_parameters_to_context(f->decoder, f->split->par_out) < 0)
        goto fail;
    f->decoder->pkt_timebase = time_base;
    f->decoder->get_format = hardware_format;
    f->decoder->thread_count = 1;
    struct picture_order *orders[] = {&f->base_order, &f->enhancement_order};
    const AVCodecParameters *parameters[] = {p, f->split->par_out};
    for (unsigned i = 0; i < 2; ++i) {
        orders[i]->parser = av_parser_init(AV_CODEC_ID_HEVC);
        orders[i]->context = avcodec_alloc_context3(NULL);
        if (!orders[i]->parser || !orders[i]->context ||
            avcodec_parameters_to_context(orders[i]->context, parameters[i]) < 0)
            goto fail;
        orders[i]->parser->flags = PARSER_FLAG_COMPLETE_FRAMES;
    }
    return f;
fail:
    dvbridge_fel_destroy(f);
    return NULL;
}

static bool receive_frames(struct dvbridge_fel *f)
{
    while (f->num_frames < QUEUE_MAX) {
        AVFrame *frame = av_frame_alloc();
        if (!frame)
            return false;
        int result = avcodec_receive_frame(f->decoder, frame);
        if (result < 0) {
            av_frame_free(&frame);
            if (result == AVERROR_EOF)
                f->eof = true;
            return result == AVERROR(EAGAIN) || result == AVERROR_EOF;
        }
        /* This timestamp was mapped from HEVC picture order before decoding,
         * not borrowed from the BL packet carrying this EL access unit. */
        if (frame->pts == AV_NOPTS_VALUE)
            frame->pts = frame->best_effort_timestamp;
        if (frame->format != AV_PIX_FMT_VAAPI || frame->pts == AV_NOPTS_VALUE) {
            av_frame_free(&frame);
            return false;
        }
        for (unsigned i = 0; i < f->num_frames; ++i) {
            if (f->frames[i]->pts == frame->pts) {
                av_frame_free(&frame);
                return false;
            }
        }
        f->frames[f->num_frames++] = frame;
    }
    return true;
}

static bool pump(struct dvbridge_fel *f)
{
    if (f->failed)
        return false;
    if (f->synchronizing)
        return true;
    if (!f->opened)
        return true;
    if (!receive_frames(f))
        goto fail;
    while (f->num_packets && f->num_frames < QUEUE_MAX) {
        AVPacket *packet = f->packets[0];
        unsigned match = 0;
        while (match < f->num_times && f->times[match].order != f->packet_order[0])
            ++match;
        if (match == f->num_times) {
            if (f->draining)
                goto fail;
            break; /* A reference EL picture can precede its corresponding BL. */
        }
        packet->pts = f->times[match].pts;
        packet->dts = AV_NOPTS_VALUE;
        int result = avcodec_send_packet(f->decoder, packet);
        if (result == AVERROR(EAGAIN))
            break;
        if (result < 0)
            goto fail;
        --f->num_times;
        memmove(f->times + match, f->times + match + 1,
                (f->num_times - match) * sizeof(*f->times));
        f->packet_bytes -= packet->size;
        av_packet_free(&packet);
        --f->num_packets;
        memmove(f->packets, f->packets + 1, f->num_packets * sizeof(*f->packets));
        memmove(f->packet_order, f->packet_order + 1, f->num_packets * sizeof(*f->packet_order));
        if (!receive_frames(f))
            goto fail;
    }
    if (f->draining && !f->num_packets && !f->sent_eof && f->num_frames < QUEUE_MAX) {
        int result = avcodec_send_packet(f->decoder, NULL);
        if (result != AVERROR(EAGAIN)) {
            if (result < 0 && result != AVERROR_EOF)
                goto fail;
            f->sent_eof = true;
        }
        if (!receive_frames(f))
            goto fail;
    }
    return true;
fail:
    f->failed = true;
    return false;
}

bool dvbridge_fel_device(struct dvbridge_fel *f, AVBufferRef *device)
{
    if (!f || !device || ((AVHWDeviceContext *)device->data)->type != AV_HWDEVICE_TYPE_VAAPI)
        return false;
    if (f->opened)
        return f->decoder->hw_device_ctx->data == device->data && pump(f);
    f->decoder->hw_device_ctx = av_buffer_ref(device);
    if (!f->decoder->hw_device_ctx || avcodec_open2(f->decoder, f->decoder->codec, NULL) < 0) {
        f->failed = true;
        return false;
    }
    f->opened = true;
    return pump(f);
}


bool dvbridge_fel_submit(struct dvbridge_fel *f, const AVPacket *packet)
{
    if (!f || !packet || f->failed || f->draining || packet->size <= 0 ||
        packet->pts == AV_NOPTS_VALUE)
        return false;
    int64_t base_identity;
    if (f->num_times == MAP_MAX || !picture_identity(&f->base_order, packet, &base_identity))
        goto fail;
    if (f->synchronizing) {
        if (f->sync_el_key && ++f->sync_key_age >= QUEUE_MAX) {
            clear_packets(f);
            f->sync_key = f->sync_el_key = false;
            f->num_times = 0;
        }
        /* An early EL candidate retains the timestamp map across the BL key. */
        if (!f->sync_el_key && f->base_order.parser->key_frame == 1) {
            f->sync_key = true;
            f->sync_key_age = 0;
            f->sync_key_poc = f->base_order.parser->output_picture_number;
            f->sync_key_identity = base_identity;
            f->sync_key_pts = packet->pts;
            for (unsigned i = 0; i < QUEUE_MAX; ++i)
                f->sync_base_poc[i] = f->sync_el_poc[i] = INT_MIN;
            f->num_times = 0;
        } else if (!f->sync_el_key && (!f->sync_key || ++f->sync_key_age >= QUEUE_MAX)) {
            f->sync_key = false;
            f->num_times = 0;
        }
    }
    for (unsigned i = 0; i < f->num_times; ++i)
        if (f->times[i].order == base_identity)
            goto fail;
    f->times[f->num_times].order = base_identity;
    f->times[f->num_times++].pts = packet->pts;
    AVPacket *copy = av_packet_clone(packet);
    if (!copy)
        goto fail;
    int result = av_bsf_send_packet(f->split, copy);
    av_packet_free(&copy);
    if (result < 0)
        goto fail;
    while (true) {
        copy = av_packet_alloc();
        if (!copy)
            goto fail;
        result = av_bsf_receive_packet(f->split, copy);
        if (result < 0) {
            av_packet_free(&copy);
            if (result == AVERROR(EAGAIN))
                return pump(f);
            goto fail;
        }
        int64_t enhancement_identity;
        if (f->num_packets == QUEUE_MAX || copy->size <= 0 ||
            (size_t)copy->size > PACKET_BYTES_MAX - f->packet_bytes ||
            !picture_identity(&f->enhancement_order, copy, &enhancement_identity)) {
            av_packet_free(&copy);
            goto fail;
        }
        if (f->synchronizing) {
            const AVCodecParserContext *ep = f->enhancement_order.parser;
            if (f->sync_key) {
                f->sync_base_poc[f->sync_key_age] = f->base_order.parser->output_picture_number;
                f->sync_el_poc[f->sync_key_age] = ep->output_picture_number;
            }
            bool common_key = f->sync_key && !f->sync_el_key && ep->key_frame == 1 &&
                              f->sync_key_poc == ep->output_picture_number &&
                              dvbridge_fel_key_match(f->sync_base_poc, f->sync_el_poc,
                                                    f->sync_key_age + 1);
            bool early_key = f->sync_key && f->sync_el_key &&
                             f->base_order.parser->key_frame == 1 &&
                             f->sync_key_poc == f->base_order.parser->output_picture_number &&
                             dvbridge_fel_key_match(f->sync_base_poc, f->sync_el_poc,
                                                   f->sync_key_age + 1);
            /* A non-reordered BL gives the IDR packet an unambiguous display
             * time. Never use this shortcut for a BL with reordered pictures. */
            bool el_idr = ep->key_frame == 1 && ep->output_picture_number == 0 &&
                          f->base_order.context->has_b_frames == 0;
            if (!common_key && !early_key && !el_idr) {
                if (++f->preroll_packets > 1024) {
                    av_packet_free(&copy);
                    goto fail;
                }
                if (ep->key_frame == 1) {
                    clear_packets(f);
                    f->sync_key = f->sync_el_key = true;
                    f->sync_key_age = 0;
                    f->sync_key_poc = ep->output_picture_number;
                    for (unsigned i = 0; i < QUEUE_MAX; ++i)
                        f->sync_base_poc[i] = f->sync_el_poc[i] = INT_MIN;
                    f->sync_base_poc[0] = f->base_order.parser->output_picture_number;
                    f->sync_el_poc[0] = ep->output_picture_number;
                    f->num_times = 1;
                    f->times[0].order = base_identity;
                    f->times[0].pts = packet->pts;
                }
                if (!f->sync_el_key) {
                    av_packet_free(&copy);
                    if (!f->sync_key) f->num_times = 0;
                    continue;
                }
            } else if (early_key) {
                if (f->num_packets != f->sync_key_age) {
                    av_packet_free(&copy);
                    goto fail;
                }
                /* All buffered pictures now have a proven BL identity. */
                for (unsigned i = 0; i <= f->num_packets; ++i) {
                    int64_t delta = (int64_t)f->sync_el_poc[i] - f->sync_key_poc;
                    if ((delta > 0 && base_identity > INT64_MAX - delta) ||
                        (delta < 0 && base_identity < INT64_MIN - delta)) {
                        av_packet_free(&copy);
                        goto fail;
                    }
                    f->packet_order[i] = base_identity + delta;
                }
                enhancement_identity = f->packet_order[f->num_packets];
                int64_t delta = -(int64_t)f->sync_key_poc;
                if ((delta > 0 && base_identity > INT64_MAX - delta) ||
                    (delta < 0 && base_identity < INT64_MIN - delta)) {
                    av_packet_free(&copy);
                    goto fail;
                }
                f->enhancement_order.offset = base_identity + delta;
                f->enhancement_order.maximum = base_identity;
                for (unsigned i = 0; i <= f->num_packets; ++i)
                    if (f->packet_order[i] > f->enhancement_order.maximum)
                        f->enhancement_order.maximum = f->packet_order[i];
                f->anchor_pts = packet->pts;
                f->synchronizing = false;
            } else {
                clear_packets(f);
                int64_t anchor = common_key ? f->sync_key_identity : base_identity;
                f->enhancement_order.offset = anchor - ep->output_picture_number;
                f->enhancement_order.maximum = anchor;
                enhancement_identity = anchor;
                f->anchor_pts = common_key ? f->sync_key_pts : packet->pts;
                f->synchronizing = false;
            }
        }
        f->packet_order[f->num_packets] = enhancement_identity;
        f->packet_bytes += copy->size;
        f->packets[f->num_packets++] = copy;
    }
fail:
    f->failed = true;
    return false;
}

bool dvbridge_fel_drain(struct dvbridge_fel *f)
{
    if (!f)
        return false;
    if (f->synchronizing) {
        f->failed = true;
        return false; /* EOF cannot supply the missing random-access picture. */
    }
    f->draining = true;
    return pump(f);
}

int dvbridge_fel_take(struct dvbridge_fel *f, int64_t pts, AVFrame **frame)
{
    if (!f || !frame || *frame || pts == AV_NOPTS_VALUE || !pump(f))
        return -1;
    if (f->synchronizing || (f->anchor_pts != AV_NOPTS_VALUE && pts < f->anchor_pts))
        return 2; /* Preroll before a recoverable EL picture, not a substitute. */
    for (unsigned i = 0; i < f->num_frames;) {
        AVFrame *candidate = f->frames[i];
        if (candidate->pts > pts) {
            ++i;
            continue;
        }
        --f->num_frames;
        memmove(f->frames + i, f->frames + i + 1,
                (f->num_frames - i) * sizeof(*f->frames));
        if (candidate->pts == pts) {
            *frame = candidate;
            return 1;
        }
        av_frame_free(&candidate);
    }
    /* A later decoded EL frame does not prove the requested one is missing.
     * Keep bounded lookahead and never substitute a neighbouring surface. */
    return f->eof || f->num_frames == QUEUE_MAX ? -1 : 0;
}

bool dvbridge_fel_failed(const struct dvbridge_fel *f)
{
    return !f || f->failed;
}

bool dvbridge_fel_exhausted(const struct dvbridge_fel *f)
{
    return f && !f->failed && f->eof && !f->num_frames && !f->num_packets;
}
