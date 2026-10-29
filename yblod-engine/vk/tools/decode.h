/* Test-tool decoding: a Dolby Vision file through ffmpeg-yblod VAAPI, the BL
 * and (Profile 7) the EL split with dovi_split and decoded on one VADisplay,
 * paired in order as mpv's f_enhancement_pair does. Headless, render node. */
#pragma once
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavcodec/bsf.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vaapi.h>
#include <va/va.h>

struct dv_decoder {
    AVFormatContext *fmt;
    int stream;
    AVBufferRef *device;
    VADisplay display;
    AVBSFContext *bsf[2];       /* BL (+RPU), EL; NULL if single layer */
    AVCodecContext *dec[2];     /* BL, EL (EL NULL if single layer) */
    AVFrame *pending[2];
    int profile, el_present, eof;
    int split;                  /* no dovi_split: decode.c splits the layers */
    AVRational time_base;
};

struct dv_pair {
    AVFrame *bl, *el;           /* VAAPI frames; el NULL if none */
    VASurfaceID bl_surface, el_surface;
    const AVFrameSideData *dovi; /* AV_FRAME_DATA_DOVI_METADATA (BL, else EL) */
};

int dv_decoder_open(struct dv_decoder *d, const char *path, const char *node, double start);
/* 1: a pair; 0: end; <0: error. Caller frees with dv_pair_free. */
int dv_decoder_next(struct dv_decoder *d, struct dv_pair *p);
void dv_pair_free(struct dv_pair *p);
void dv_decoder_close(struct dv_decoder *d);
