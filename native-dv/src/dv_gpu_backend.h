#ifndef DV_GPU_BACKEND_H
#define DV_GPU_BACKEND_H
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "dv_engine.h"
#include "dv_ffmpeg_metadata.h"
/* Private interface. The caller supplies a qualified, same-device VA+EGL
 * sharing context. No renderer or decoder resources are allocated here. */
void *dv_opencl_create_shared(cl_context,cl_device_id);
/* Resource location is per backend; never changes the player's working directory. */
void *dv_opencl_create_shared_at(cl_context,cl_device_id,const char *resource_directory);
/* Native source-domain renderer owns its tunnel packer. Do not build the
 * unrelated legacy colour-to-GL variant; general backend callers retain it. */
void *dv_opencl_create_source_at(cl_context,cl_device_id,const char *resource_directory);
/* Diagnostic oracle: always performs the original planar P010 unpack. */
void *dv_opencl_create_shared_planar_reference(cl_context,cl_device_id);
/* Success proves queue drain AND all acquired VA/GL ownership returned.
 * Failure requires retaining the frame owner/backend until recovery succeeds. */
dv_status dv_opencl_drain(void *);
dv_status dv_opencl_reconstruct_enqueue(void *,const dv_frame_settings *,unsigned,unsigned,
 const uint16_t *const[3],const uint16_t *const[3]);
/* EL logical dimensions are BL/2 for resampling flag1, BL for flag0.
 * Caller validates decoded dimensions; imported allocations may include padding. */
dv_status dv_opencl_reconstruct_p010(void *,const dv_frame_settings *,unsigned,unsigned,
 const cl_mem[2],const cl_mem[2]);
/* Native-grid, residual-disabled polynomial path; no enhancement input/work.
 * Caller supplies producer-complete VA images and retains the surface owner
 * until drain. This does not perform a chroma-siting conversion. */
dv_status dv_opencl_reconstruct_single_p010(void *,const dv_frame_settings *,unsigned,unsigned,
 const cl_mem[2]);
/* Opt-in DV_SINGLE_PIECEWISE=1, explicit CCM center-left guide, polynomial
 * luma plus general chroma. Same VA ownership/drain contract as above. */
dv_status dv_opencl_reconstruct_piecewise_p010(void *,const dv_single_frame_settings *,unsigned,unsigned,const cl_mem[2]);
/* Additional DV_SINGLE_NV12=1 opt-in. Explicit 8-bit NV12 R/RG UNORM views,
 * native420 center-left guide only. Matching mapping depth is mandatory.
 * Uses the same metadata identity, output/poll and VA ownership contract. */
dv_status dv_opencl_reconstruct_piecewise_nv12(void *,const dv_single_frame_settings *,unsigned,unsigned,const cl_mem[2]);
dv_status dv_opencl_piecewise_poll(void *,const dv_single_frame_settings *,unsigned,unsigned,int *);
dv_status dv_opencl_piecewise_output(void *,const dv_single_frame_settings *,unsigned,unsigned,cl_mem[3]);
/* Separate typed experimental contract; old publication APIs reject422.
 * Layout0=420,1=422. Sampling0=center-left candidate,1=top-left candidate;
 * 420 supports only its original center-left guide.422 requires both
 * DV_SINGLE_PIECEWISE=1 and DV_SINGLE_FULL_HEIGHT=1 at backend creation.
 * Layout/policy are part of identity, not inferred from buffer allocation. */
typedef struct {
    dv_single_frame_settings frame;
    uint32_t layout,sampling;
} dv_piecewise_surface_settings;
dv_status dv_opencl_reconstruct_piecewise_surface(void *,const dv_piecewise_surface_settings *,unsigned,unsigned,const cl_mem[2]);
dv_status dv_opencl_piecewise_surface_poll(void *,const dv_piecewise_surface_settings *,unsigned,unsigned,int *);
dv_status dv_opencl_piecewise_surface_output(void *,const dv_piecewise_surface_settings *,unsigned,unsigned,cl_mem[3]);
/* Explicit experimental source representation. Retained native 12-bit420
 * buffers, with completed reconstruction and validated sample bounds. Caller
 * must retain this backend slot until GPU consumers finish, resample chroma
 * explicitly and send matching SOURCE metadata. No target colour conversion. */
dv_status dv_opencl_source_output(void *,const dv_frame_settings *,unsigned,unsigned,cl_mem[3]);
/* Nonblocking readiness of the reconstruction and range-check event. DV_OK
 * with ready=0 means pending, not a publishable frame. Output is unchanged
 * on failure. This does not release VA ownership or permit reusing the slot. */
dv_status dv_opencl_source_poll(void *,const dv_frame_settings *,unsigned,unsigned,int *ready);
dv_status dv_opencl_colour_resident(void *,const dv_frame_settings *,unsigned,unsigned,uint64_t *);
/* Image path holds GL ownership through CPU correction. Always call drain on
 * return, including failures. Returned release event is retained. */
dv_status dv_opencl_colour_gl_resident(void *,const dv_frame_settings *,unsigned,unsigned,const cl_mem[3],uint64_t *,cl_event *);
/* Returned buffers/events are retained references. Success-only publication. */
dv_status dv_opencl_output(void *,cl_mem[3],cl_event *);
dv_status dv_opencl_publish_gl(void *,const cl_mem[3],cl_event *);
#endif
