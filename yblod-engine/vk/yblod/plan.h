/* The CPU half of a frame, done by yblod's own host code exactly as
 * dv_source_renderer.c and opencl_backend.c do it with the production
 * settings (native-dv.env): the route, every table the kernels read, the
 * letterbox row window, the placed metadata packets and the pack's constants.
 * Rust (src/plan.rs) mirrors this struct; dvvk_plan_layout() guards it. */
#pragma once
#include <stddef.h>
#include <stdint.h>

enum {
    DVVK_ROUTE_DUAL = 1,        /* FEL: BL + EL, residual (opencl reconstruct_p010) */
    DVVK_ROUTE_SINGLE = 2,      /* polynomial, no residual (single_polynomial_p010) */
    DVVK_ROUTE_PIECEWISE = 3,   /* general mapping (single_reshape.cl), P010 or NV12 */
};
enum { DVVK_CHROMA_POLY = 0, DVVK_CHROMA_MMR128 = 1, DVVK_CHROMA_MMR96 = 2, DVVK_CHROMA_MMR64_LUT = 3 };
enum { DVVK_PACK_SITED = 0, DVVK_PACK_422 = 1, DVVK_PACK_RESIZED = 2 };

typedef struct {
    uint32_t width, height;             /* decoded */
    uint32_t x, y, dw, dh;              /* destination in the canvas; dw/dh 0: no resize */
    uint32_t chroma_top_left, has_el, nv12;
    uint32_t el_scaler, fel;            /* 0 nearest / 1 Annex B; 0 FEL off */
    uint32_t packet_id, refresh, overlay, overlay_visible_before;
    int64_t pts;
    double rgb_to_lms[9];               /* overlay: linear GUI RGB -> the source's LMS */
    double white_nits;                  /* overlay white (mpv-yblod: 203) */
} dvvk_plan_in;

typedef struct {
    /* 8-byte members first: no padding differences with Rust. */
    int64_t mmr_lut[2][3][1024];        /* univariate MMR LUTs (dv_mmr64_build_lut) */
    unsigned char reshape[4544];        /* dv_single_reshape, the GPU struct as is */
    int32_t residual[3][1024];          /* luma, Cb, Cr residual tables */
    uint32_t mmr_params[2][96];
    uint32_t route, width, height, el_scale, nearest, chroma_phase;
    uint32_t chroma_kind[2], dual, horizontal_fused, equal;
    uint32_t row_first, row_last;       /* letterbox window (8-aligned), 0,0 = all rows */
    uint32_t layout, sampling, depth, order2, narrow_mask[4];
    uint32_t pack, active[4], black[4], placement[4], top_left, source_w, source_h;
    uint32_t packet_count, overlay_visible;
    uint32_t reserved[7];
    float gui_coeff[32];                /* overlay colour, as prepare_overlay() builds it */
    uint16_t mapping[1024];             /* luma mapping (Q16) */
    uint16_t chroma_mapping[2][1024];   /* polynomial chroma (Q16) */
    uint16_t single_table[3][1024];     /* single-layer 12-bit tables */
    uint16_t luma_table[1024];          /* piecewise luma table (12-bit) */
    unsigned char packets[640];
} dvvk_plan;

/* 0 OK; else a dv_status (-1 invalid, -2 unsupported: ordinary picture, ...). */
int dvvk_plan_frame(const dvvk_plan_in *in, const void *metadata, size_t metadata_size, dvvk_plan *out);
/* sizeof and a few offsets, for the Rust mirror. */
void dvvk_plan_layout(size_t out[6]);
