#ifndef DV_SOURCE_RENDERER_H
#define DV_SOURCE_RENDERER_H
#include <stddef.h>
#include <stdint.h>
#include "source_geometry.h"
#include "source_input.h"
#ifdef __cplusplus
extern "C" {
#endif
#define DV_SOURCE_RENDERER_ABI 4u
/* Private synchronous renderer boundary. The caller owns the EGL context,
 * RGBA8 texture and paired VA surface leases. GL use must complete before
 * render; render completes the CL release before returning a usable image.
 * A failed drain/destroy requires retaining all owners until process recovery. */
unsigned dv_source_renderer_abi(void);
enum {DV_SOURCE_OUTPUT_RGBA=0,DV_SOURCE_OUTPUT_BGRA=1};
/* Explicit CL-store byte order for a separately qualified imported BO.
 * BGRA does not imply a vertical flip or prove scanout compatibility. */
void *dv_source_renderer_create_storage(uintptr_t,uintptr_t,uintptr_t,unsigned,
                                        const char *,unsigned storage);
void *dv_source_renderer_create(uintptr_t va_display,uintptr_t egl_display,
                               uintptr_t egl_context,unsigned texture,
                               const char *resource_directory);
/* Optional second same-format texture, imported once. Slot0 is create's texture,
 * slot1 is attached. Caller keeps both textures alive until successful destroy;
 * distinct names must refer to distinct storage. Attach before submitting work.
 * select may run during pending reconstruction, but never during output access.
 * Caller proves selected BO is not scanned out and prior GL use has completed.
 * This API does not transfer KMS ownership or synchronize external consumers. */
int dv_source_renderer_attach_output(void *,unsigned texture);
int dv_source_renderer_select_output(void *,unsigned slot);
int dv_source_renderer_render(void *,unsigned bl_surface,unsigned el_surface,
                              unsigned width,unsigned height,const void *metadata,
                              size_t metadata_size,uint64_t frame_id,int64_t pts,
                              unsigned packet_id,int refresh,const dv_source_geometry *);
/* One pending reconstruction per renderer. submit snapshots metadata/geometry
 * and does not acquire the output GL image. Caller keeps decoded owners until
 * successful drain/present/cancel. A pending slot cannot be overwritten.
 * poll never publishes pixels; present waits if necessary and consumes only
 * the exact submitted identity. Complete prior GL use before present. */
int dv_source_renderer_submit(void *,unsigned bl_surface,unsigned el_surface,
                              unsigned width,unsigned height,const void *metadata,
                              size_t metadata_size,uint64_t frame_id,int64_t pts,
                              const dv_source_geometry *);
/* Additive ABI4 entry point. Explicit decoded BL storage/depth must agree
 * with each other, the RPU and the imported VA views. NV12 is single-layer,
 * opt-in and center-left only. Existing submit/render remain P010-only.
 * EL, when present, retains the existing10bit P010 contract. */
int dv_source_renderer_submit_input(void *,unsigned,unsigned,unsigned,unsigned,
    const void *,size_t,uint64_t,int64_t,const dv_source_geometry *,const dv_source_input *);
int dv_source_renderer_poll(void *,uint64_t frame_id,int64_t pts,int *ready);
int dv_source_renderer_present(void *,uint64_t frame_id,int64_t pts,unsigned packet_id,int refresh);
/* Compile all enabled overlay layouts and allocate uniforms before playback.
 * Requires an idle renderer; idempotent. Overlay presentation never compiles
 * shaders and rejects if this preparation has not completed successfully. */
int dv_source_renderer_prepare_overlays(void *);
/* GUI texture is RGBA8, full canvas size, coverage alpha and premultiplied sRGB.
 * GUI storage must not alias either output, even through a different texture.
 * Caller completes GUI GL writes before calling and keeps storage alive until
 * successful drain. No submission may overwrite this pending reconstruction.
 * A nonzero result forbids presentation of the output, including partial writes.
 * rgb_to_lms and white_nits define GUI colour explicitly, not movie processing. */
int dv_source_renderer_present_overlay(void *,uint64_t,int64_t,unsigned,int,
                                      unsigned gui_texture,unsigned flip_y,
                                      const double rgb_to_lms[9],double white_nits);
/* Successful cancellation drains ownership and invalidates pending/output state. */
int dv_source_renderer_cancel(void *);
int dv_source_renderer_drain(void *);
/* Diagnostic only: compare a GL RGBA8 readback to CPU packing of retained source planes. */
int dv_source_renderer_check(void *,const unsigned char *,size_t);
int dv_source_renderer_destroy(void *);
#ifdef __cplusplus
}
#endif
#endif
