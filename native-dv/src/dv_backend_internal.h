#ifndef DV_BACKEND_INTERNAL_H
#define DV_BACKEND_INTERNAL_H
#include "dv_engine.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Private synchronous hook; validated dense inputs, engine-owned scratch
 * outputs. Context borrows opaque. No silent fallback on backend failure. */
typedef int (*dv_luma_backend)(void *,const dv_intel_composer_config *,size_t,
 const uint16_t *,const uint16_t *,uint16_t *,int32_t *,int32_t *,uint16_t *);
int dv_engine_set_luma_backend(dv_engine *,dv_luma_backend,void *);
void *dv_metal_luma_create(void);
void dv_metal_luma_destroy(void *);
int dv_metal_luma_dispatch(void *,const dv_intel_composer_config *,size_t,
 const uint16_t *,const uint16_t *,uint16_t *,int32_t *,int32_t *,uint16_t *);
double dv_metal_luma_gpu_seconds(void *);
int dv_luma_benchmark(void *,const dv_intel_composer_config *,size_t,
 const uint16_t *,const uint16_t *,uint16_t *,int32_t *,int32_t *,uint16_t *);
/* Chain owns EL enlargement + luma reconstruction + MMR guide. Diagnostic
 * arrays may be NULL when diagnostics=0. Returns a dv_status, never fallback. */
typedef dv_status (*dv_chain_backend)(void *,const dv_intel_composer_config *,unsigned,unsigned,
 const uint16_t *,const uint16_t *,uint16_t *,uint16_t *,int32_t *,int32_t *,uint16_t *,int);
int dv_engine_set_chain_backend(dv_engine *,dv_chain_backend,void *);
void *dv_metal_chain_create(void);
void dv_metal_chain_destroy(void *);
double dv_metal_chain_gpu_seconds(void *);
dv_status dv_metal_chain_dispatch(void *,const dv_intel_composer_config *,unsigned,unsigned,
 const uint16_t *,const uint16_t *,uint16_t *,uint16_t *,int32_t *,int32_t *,uint16_t *,int);
dv_status dv_cpu_chain(void *,const dv_intel_composer_config *,unsigned,unsigned,
 const uint16_t *,const uint16_t *,uint16_t *,uint16_t *,int32_t *,int32_t *,uint16_t *,int);
void *dv_cpu_chain_create(void);
void dv_cpu_chain_destroy(void *);
dv_status dv_chain_benchmark(void *,const dv_intel_composer_config *,unsigned,unsigned,
 const uint16_t *,const uint16_t *,uint16_t *,uint16_t *,int32_t *,int32_t *,uint16_t *,int);
typedef dv_status (*dv_chroma_backend)(void *,const dv_intel_composer_config *,unsigned,size_t,
 const uint16_t *,const uint16_t *,const uint16_t *,const uint16_t *,uint16_t *,int32_t *,int32_t *,uint16_t *,int);
int dv_engine_set_chroma_backend(dv_engine *,dv_chroma_backend,void *);
/* Resident hook consumes GPU guide from paired chain context; CPU y pointer
 * is NULL. Requires chain attachment first. Do not use for CPU shadow timing. */
int dv_engine_set_resident_chroma_backend(dv_engine *,dv_chroma_backend,void *);
typedef dv_status (*dv_reconstruct_backend)(void *,const dv_frame_settings *,unsigned,unsigned,
 const uint16_t *const[3],const uint16_t *const[3],uint16_t *const[3],const dv_observer *);
int dv_engine_set_reconstruct_backend(dv_engine *,dv_reconstruct_backend,void *);
int dv_engine_set_colour_workers(dv_engine *,unsigned);
typedef struct {double reconstruction_seconds,colour_seconds,total_seconds;} dv_timings;
int dv_engine_get_timings(const dv_engine *,dv_timings *);
typedef dv_status (*dv_colour_backend)(void *,const dv_frame_settings *,unsigned,unsigned,
 const uint16_t *const[3],uint16_t *const[3],uint64_t *);
int dv_engine_set_colour_backend(dv_engine *,dv_colour_backend,void *);
/* Paired resident contract: without a stage observer reconstruction receives
 * three NULL CPU planes. Colour consumes the same context's resident result.
 * With diagnostics the usual CPU planes must still be produced. */
int dv_engine_set_resident_backends(dv_engine *,dv_reconstruct_backend,dv_colour_backend,void *);
dv_status dv_opencl_colour(void *,const dv_frame_settings *,unsigned,unsigned,
 const uint16_t *const[3],uint16_t *const[3],uint64_t *);
void *dv_opencl_create(void);
void dv_opencl_destroy(void *);
dv_status dv_opencl_reconstruct(void *,const dv_frame_settings *,unsigned,unsigned,
 const uint16_t *const[3],const uint16_t *const[3],uint16_t *const[3],const dv_observer *);
dv_status dv_opencl_reconstruct_resident(void *,const dv_frame_settings *,unsigned,unsigned,
 const uint16_t *const[3],const uint16_t *const[3],uint16_t *const[3],const dv_observer *);
void *dv_metal_mmr_create(void);
void dv_metal_mmr_destroy(void *);
double dv_metal_mmr_gpu_seconds(void *);
dv_status dv_metal_mmr_dispatch(void *,const dv_intel_composer_config *,unsigned,size_t,
 const uint16_t *,const uint16_t *,const uint16_t *,const uint16_t *,uint16_t *,int32_t *,int32_t *,uint16_t *,int);
dv_status dv_metal_chain_chroma_dispatch(void *,const dv_intel_composer_config *,unsigned,size_t,
 const uint16_t *,const uint16_t *,const uint16_t *,const uint16_t *,uint16_t *,int32_t *,int32_t *,uint16_t *,int);
double dv_metal_chain_chroma_gpu_seconds(void *);
dv_status dv_chroma_benchmark(void *,const dv_intel_composer_config *,unsigned,size_t,
 const uint16_t *,const uint16_t *,const uint16_t *,const uint16_t *,uint16_t *,int32_t *,int32_t *,uint16_t *,int);
dv_status dv_metal_mmr_accumulate(void *,const dv_intel_composer_config *,unsigned,size_t,const uint16_t *,const uint16_t *,const uint16_t *,uint32_t *);
#ifdef __cplusplus
}
#endif
#endif
