/* big-fella's Dolby Vision source-tunnel engine on Vulkan compute.
 * Phase 1 subset (synchronous; docs/vulkan-engine.md has the full API).
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "dvvk_plan.h"   /* dvvk_plan_in */
#ifdef __cplusplus
extern "C" {
#endif
typedef struct dvvk_engine dvvk_engine;
typedef struct {
    dvvk_plan_in plan;            /* geometry, scaler, packet id, refresh, overlay */
    uint32_t bl_surface, el_surface;  /* VASurfaceID; el 0xffffffff = none */
    const void *metadata; size_t metadata_size;   /* ffmpeg-yblod AVDOVIMetadata */
    uint32_t output;              /* slot */
    uint32_t flags;               /* DVVK_FRAME_AUTO_OVERLAY: the overlay (and the
                                   * metadata's overlay edge) from the engine's
                                   * sources, set_subtitles / set_overlay */
    uint64_t release_wait;        /* explicit sync: write the output only once the
                                   * release timeline reaches this (0: no wait) */
} dvvk_frame;
#define DVVK_FRAME_AUTO_OVERLAY 1u

typedef struct {
    int32_t x, y;                 /* destination in the 3840x2160 canvas */
    uint32_t w, h, dw, dh;        /* bitmap size; destination size (BGRA may scale) */
    uint32_t stride;
    uint32_t format;              /* 1 libass A8 with color, 2 BGRA premultiplied */
    uint32_t color;               /* libass 0xRRGGBBAA, AA = transparency */
    const void *data;
} dvvk_sub_part;

typedef struct {
    uint32_t width, height, format;   /* DRM fourcc: AR24, XR24, AB24, XB24 */
    uint64_t modifier;
    uint32_t num_planes;
    int fd[4];
    uint32_t offset[4], stride[4];
} dvvk_dmabuf;

/* Overlay sources for frames with DVVK_FRAME_AUTO_OVERLAY. */
int dvvk_set_subtitles(dvvk_engine *, const dvvk_sub_part *, uint32_t num);
int dvvk_set_overlay(dvvk_engine *, const dvvk_dmabuf *, uint64_t id);   /* NULL: none */
/* With flags: DVVK_OVERLAY_FLIP_Y, rows bottom first (a GL framebuffer's
 * texture). Every frame that reads the buffer leaves its fence on it as an
 * implicit read fence: a renderer that honours implicit sync may draw the
 * next contents into it at once (no copy, no CPU wait). */
#define DVVK_OVERLAY_FLIP_Y 1u
int dvvk_set_overlay_ex(dvvk_engine *, const dvvk_dmabuf *, uint64_t id, uint32_t flags);
/* CB1 Master Reference's native overlay (cb1_composite_glsl.h), for frames
 * planned with plan.cb1_contract: the overlay set with dvvk_set_overlay_ex
 * (Kodi's GUI framebuffer, premultiplied; RGBA8, or RGBA16F BT.2020 PQ with
 * input_pq) mixed into the source signal exactly as CB1 does. The matrices are
 * CB1's overlay_decode, overlay_offset and overlay_linear for the picture, in
 * column order (libplacebo's mat3 variables); the LUTs Kodi's GUI transfer
 * tables (CGuiCompositeShaderGLES, stored as GL_R16F). NULL: none. */
typedef struct {
    float rect[4];                      /* overlay_area: x0, y0, x1, y1 (x even); empty: none */
    float decode[9], offset[3], linear[9];
    uint32_t input_pq;
    float degamma[256], pq[256];
} dvvk_cb1_overlay;
int dvvk_set_cb1_overlay(dvvk_engine *, const dvvk_cb1_overlay *);
/* 1 once a fence could not be attached as an implicit fence (older kernel):
 * wait for each ticket before committing or drawing into the overlay. */
int dvvk_implicit_failed(dvvk_engine *);
/* How many overlay pixels frame `ticket` found changed since the frame
 * before (the shell's layer read by the pack); -1 when not known. */
int64_t dvvk_overlay_changes(dvvk_engine *, uint64_t ticket);
uint64_t dvvk_overlay_ticket(dvvk_engine *, uint64_t id);
int dvvk_create(int drm_fd, void *va_display, dvvk_engine **out);
int dvvk_set_output(dvvk_engine *, uint32_t slot, int dmabuf_fd, uint32_t stride);
/* 0: the picture is in the output; > 0 never; < 0: a dv_status (-2: not this
 * engine's frame, show an ordinary picture; -8: engine failure). */
int dvvk_render_sync(dvvk_engine *, const dvvk_frame *, uint32_t *error_flags);
/* Phase 1: the overlay picture for frames with plan.overlay set (RGBA8
 * premultiplied sRGB, 3840x2160, rows top first), copied in. Error flags
 * bit 8: an overlay colour the source can't represent. */
void dvvk_set_overlay_rgba(dvvk_engine *, const void *rgba, size_t stride);
/* Asynchronous: plan and submit without waiting; 0 and *ticket, or a
 * dv_status. On return the output dmabuf carries the frame's fence as an
 * implicit write fence (commit it at once). The decoded surfaces must stay
 * alive until the ticket is done. */
int dvvk_submit(dvvk_engine *, const dvvk_frame *, uint64_t *ticket);
/* Never blocks: *done when finished; *flags (1 range, 2 overflow, 0x100
 * overlay colour) once done. */
int dvvk_poll(dvvk_engine *, uint64_t ticket, int *done, uint32_t *flags);
/* 1 done, 0 timeout, < 0 failure. */
int dvvk_wait(dvvk_engine *, uint64_t ticket, uint64_t timeout_ns);
/* linux-drm-syncobj-v1: DRM syncobj fds of the frames' timeline (acquire
 * point = the frame's ticket) and of a release timeline for the compositor's
 * release points. The caller owns the fds. */
int dvvk_timelines(dvvk_engine *, int *acquire_syncobj_fd, int *release_syncobj_fd);
void dvvk_decode_waits(dvvk_engine *, uint64_t *fence, uint64_t *vasync);
/* GPU time of the most recent frame read back (ms); < 0 unknown. */
double dvvk_last_gpu_ms(dvvk_engine *);
void dvvk_reset(dvvk_engine *);
void dvvk_destroy(dvvk_engine *);
#ifdef __cplusplus
}
#endif
