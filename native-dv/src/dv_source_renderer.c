#define _POSIX_C_SOURCE 200809L
#define CL_TARGET_OPENCL_VERSION 120
#include "dv_source_renderer.h"
#include "dv_gpu_backend.h"
#include "dv_backend_internal.h"
#include "dv_ffmpeg_metadata.h"
#include "source_overlay.h"
#include <CL/cl_gl.h>
#include <CL/cl_va_api_media_sharing_intel.h>
#include <va/va.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef struct {
    VADisplay va;cl_context context;cl_device_id device;cl_command_queue queue;
    cl_program program;cl_kernel kernel;cl_mem output,packets,inputs[4],planes[3];
    cl_mem other_output;unsigned output_slot,primary_texture,secondary_texture;int bgra;
    clCreateFromVA_APIMediaSurfaceINTEL_fn import;
    void *backend;int acquired,valid,queued;cl_event released;
    dv_frame_settings settings;unsigned char packed[640];unsigned count;
    dv_piecewise_surface_settings piecewise_settings;int piecewise_enabled,piecewise,space2_enabled,full_enabled,nv12_enabled;
    cl_program full_program;cl_kernel full_kernel;
    int submitted;unsigned char dm[512];size_t dm_bytes;int overlay_visible;unsigned wait_profiles;
    dv_source_geometry geometry;
    unsigned submit_traces;
    int profile_async;
    unsigned pack_profiles;
    cl_event pack_done;
    int pack_pending,pack_overlay,pack_error,pack_cached_gui;
    cl_program resize_program;
    cl_kernel resize_kernel,resize_prepare;
    cl_mem resize_rows;
    unsigned resize_key[5];
    int resize_key_valid;
    char *directory;
    cl_program overlay_program[3];cl_kernel overlay_kernel[3];
    cl_mem gui,gui_coeff,gui_error;
    int gui_prepass;cl_kernel gui_prepare;cl_mem gui_colours;
    int gui_pixel_cache,gui_cache_valid,gui_cache_invalidate,gui_sparse;cl_mem gui_pixels;
    float gui_cache_coeff[31];
    uint64_t gui_revision;
    unsigned gui_texture,gui_flip,gui_width,gui_height;
    int gui_reused;
    int gui_cache_verify;cl_kernel gui_cache_check;
    int gui_gray_lut,gui_shared;cl_kernel gui_gray_prepare;cl_mem gui_gray;
    int gui_tiled;cl_kernel gui_tiles_prepare;cl_mem gui_tiles;
    int gui_profile;unsigned gui_profiles;cl_event gui_events[3];double gui_host_ms;
} source_renderer;
static double submit_clock(void)
{struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec/1e9;}
unsigned dv_source_renderer_abi(void){return DV_SOURCE_RENDERER_ABI;}
int dv_source_renderer_drain(void *opaque)
{
    source_renderer *r=opaque;if(!r)return DV_INVALID;
    if(r->acquired){
        /* In-order packing queue: release follows every image writer. Wait
         * for this ownership transfer, not a finish before AND after it.
         * Keep a failed/uncertain event and all leases for recovery. */
        cl_mem objects[2]={r->output,r->gui};cl_uint count=r->gui?2u:1u;
        if(!r->released&&clEnqueueReleaseGLObjects(r->queue,count,objects,0,NULL,&r->released))return DV_BACKEND;
        cl_int state=1;
        if(clFlush(r->queue)||clWaitForEvents(1,&r->released)||
           clGetEventInfo(r->released,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(state),&state,NULL)||
           state!=CL_COMPLETE)return DV_BACKEND;
        clReleaseEvent(r->released);r->released=NULL;r->acquired=0;r->queued=0;
    }else if(r->queued){
        /* Failed before acquiring the image: retain conservative error-path
         * completion for any commands that were successfully submitted. */
        if(clFinish(r->queue))return DV_BACKEND;r->queued=0;
    }
    if(r->pack_done){cl_ulong start=0,end=0;
        int valid=!clGetEventProfilingInfo(r->pack_done,CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL)&&
                  !clGetEventProfilingInfo(r->pack_done,CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL)&&end>=start;
        fprintf(stderr,"{\"source_pack_profile_pts\":%lld,\"valid\":%d,\"pack_ms\":%.6f}\n",(long long)r->settings.identity.pts,valid,valid?(double)(end-start)/1e6:0);
        clReleaseEvent(r->pack_done);r->pack_done=NULL;++r->pack_profiles;}
    if(r->gui_events[0]){
        double ms[3]={0};int valid=1;
        for(unsigned i=0;i<3;++i)if(r->gui_events[i]){
            cl_ulong start=0,end=0;
            if(clGetEventProfilingInfo(r->gui_events[i],CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL)||
               clGetEventProfilingInfo(r->gui_events[i],CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL)||end<start)valid=0;
            else ms[i]=(double)(end-start)/1e6;
            clReleaseEvent(r->gui_events[i]);r->gui_events[i]=NULL;
        }
        fprintf(stderr,"{\"overlay_profile_pts\":%lld,\"valid\":%d,\"host_prepare_ms\":%.6f,\"acquire_ms\":%.6f,\"colour_ms\":%.6f,\"pack_ms\":%.6f,\"revision_reused\":%d}\n",
                (long long)r->settings.identity.pts,valid,r->gui_host_ms,ms[0],ms[1],ms[2],r->gui_reused);
        ++r->gui_profiles;
    }
    if(r->backend&&dv_opencl_drain(r->backend)!=DV_OK)return DV_BACKEND;
    for(unsigned i=0;i<3;++i)if(r->planes[i]){clReleaseMemObject(r->planes[i]);r->planes[i]=NULL;}
    for(unsigned i=0;i<4;++i)if(r->inputs[i]){clReleaseMemObject(r->inputs[i]);r->inputs[i]=NULL;}
    if(r->gui){clReleaseMemObject(r->gui);r->gui=NULL;}
    if(r->pack_pending){
        r->gui_cache_valid=r->pack_cached_gui&&r->gui_cache_valid&&!r->pack_error;
        r->pack_cached_gui=0;r->pack_pending=0;r->submitted=0;r->valid=0;
    }
    return DV_OK;
}
int dv_source_renderer_destroy(void *opaque)
{
    source_renderer *r=opaque;if(!r)return DV_OK;
    if(dv_source_renderer_drain(r)!=DV_OK)return DV_BACKEND;
    if(r->backend)dv_opencl_destroy(r->backend);
    if(r->output)clReleaseMemObject(r->output);if(r->packets)clReleaseMemObject(r->packets);
    if(r->gui_pixels)clReleaseMemObject(r->gui_pixels);
    if(r->gui_cache_check)clReleaseKernel(r->gui_cache_check);
    if(r->other_output)clReleaseMemObject(r->other_output);
    if(r->kernel)clReleaseKernel(r->kernel);if(r->program)clReleaseProgram(r->program);
    if(r->full_kernel)clReleaseKernel(r->full_kernel);if(r->full_program)clReleaseProgram(r->full_program);
    if(r->resize_rows)clReleaseMemObject(r->resize_rows);
    if(r->resize_prepare)clReleaseKernel(r->resize_prepare);
    if(r->resize_kernel)clReleaseKernel(r->resize_kernel);
    if(r->resize_program)clReleaseProgram(r->resize_program);
    for(unsigned i=0;i<3;++i){if(r->overlay_kernel[i])clReleaseKernel(r->overlay_kernel[i]);if(r->overlay_program[i])clReleaseProgram(r->overlay_program[i]);}
    if(r->gui_coeff)clReleaseMemObject(r->gui_coeff);if(r->gui_error)clReleaseMemObject(r->gui_error);
    if(r->gui_prepare)clReleaseKernel(r->gui_prepare);if(r->gui_colours)clReleaseMemObject(r->gui_colours);
    if(r->gui_gray_prepare)clReleaseKernel(r->gui_gray_prepare);if(r->gui_gray)clReleaseMemObject(r->gui_gray);
    if(r->gui_tiles_prepare)clReleaseKernel(r->gui_tiles_prepare);if(r->gui_tiles)clReleaseMemObject(r->gui_tiles);
    free(r->directory);
    if(r->queue)clReleaseCommandQueue(r->queue);if(r->context)clReleaseContext(r->context);free(r);return DV_OK;
}
void *dv_source_renderer_create(uintptr_t va_display,uintptr_t egl_display,uintptr_t egl_context,
                               unsigned texture,const char *directory)
{return dv_source_renderer_create_storage(va_display,egl_display,egl_context,texture,directory,DV_SOURCE_OUTPUT_RGBA);}
void *dv_source_renderer_create_storage(uintptr_t va_display,uintptr_t egl_display,uintptr_t egl_context,
                               unsigned texture,const char *directory,unsigned storage)
{
    if(!va_display||!egl_display||!egl_context||!texture||!directory||storage>DV_SOURCE_OUTPUT_BGRA)return NULL;
    source_renderer *r=calloc(1,sizeof(*r));if(!r)return NULL;r->va=(VADisplay)va_display;
    r->directory=strdup(directory);if(!r->directory)goto fail;
    r->bgra=storage==DV_SOURCE_OUTPUT_BGRA;
    cl_platform_id platforms[16],platform=NULL;cl_uint count=0;cl_int error;
    if(clGetPlatformIDs(16,platforms,&count))goto fail;
    for(unsigned i=0;i<count&&i<16;++i){clGetDeviceIDsFromVA_APIMediaAdapterINTEL_fn get=(clGetDeviceIDsFromVA_APIMediaAdapterINTEL_fn)clGetExtensionFunctionAddressForPlatform(platforms[i],"clGetDeviceIDsFromVA_APIMediaAdapterINTEL");cl_uint n=0;
        if(get&&!get(platforms[i],CL_VA_API_DISPLAY_INTEL,r->va,CL_PREFERRED_DEVICES_FOR_VA_API_INTEL,1,&r->device,&n)&&n){platform=platforms[i];break;}}
    if(!platform)goto fail;
    r->import=(clCreateFromVA_APIMediaSurfaceINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform,"clCreateFromVA_APIMediaSurfaceINTEL");if(!r->import)goto fail;
    cl_context_properties props[]={CL_CONTEXT_PLATFORM,(cl_context_properties)platform,CL_CONTEXT_VA_API_DISPLAY_INTEL,(cl_context_properties)r->va,
        CL_CONTEXT_INTEROP_USER_SYNC,CL_TRUE,CL_GL_CONTEXT_KHR,(cl_context_properties)egl_context,CL_EGL_DISPLAY_KHR,(cl_context_properties)egl_display,0};
    r->context=clCreateContext(props,1,&r->device,NULL,NULL,&error);if(!r->context||error)goto fail;
    const char *profile=getenv("DV_SOURCE_PROFILE_ASYNC");r->profile_async=profile&&!strcmp(profile,"1");
    const char *gui_prepass=getenv("DV_OVERLAY_PREPASS");r->gui_prepass=gui_prepass&&!strcmp(gui_prepass,"1");
    const char *pixel_cache=getenv("DV_OVERLAY_PIXEL_CACHE");r->gui_pixel_cache=pixel_cache&&!strcmp(pixel_cache,"1");
    if(r->gui_pixel_cache)r->gui_prepass=1;
    const char *sparse=getenv("DV_OVERLAY_SPARSE");r->gui_sparse=r->gui_pixel_cache&&sparse&&!strcmp(sparse,"1");
    const char *cache_verify=getenv("DV_OVERLAY_CACHE_VERIFY");r->gui_cache_verify=cache_verify&&!strcmp(cache_verify,"1");
    const char *gui_gray=getenv("DV_OVERLAY_GRAY_LUT");r->gui_gray_lut=gui_gray&&!strcmp(gui_gray,"1");
    const char *gui_shared=getenv("DV_OVERLAY_SHARED");r->gui_shared=gui_shared&&!strcmp(gui_shared,"1");
    const char *gui_tiled=getenv("DV_OVERLAY_TILES");r->gui_tiled=gui_tiled&&!strcmp(gui_tiled,"1");
    if(r->gui_tiled&&(!r->gui_shared||r->gui_prepass))goto fail;
    const char *gui_profile=getenv("DV_OVERLAY_PROFILE");r->gui_profile=gui_profile&&!strcmp(gui_profile,"1");
    const char *piecewise=getenv("DV_SINGLE_PIECEWISE");r->piecewise_enabled=piecewise&&!strcmp(piecewise,"1");
    const char *nv12=getenv("DV_SINGLE_NV12");r->nv12_enabled=r->piecewise_enabled&&nv12&&!strcmp(nv12,"1");
    const char *full=getenv("DV_SINGLE_FULL_HEIGHT");r->full_enabled=r->piecewise_enabled&&full&&!strcmp(full,"1");
    const char *space2=getenv("DV_SOURCE_SPACE2");r->space2_enabled=space2&&!strcmp(space2,"1");
    r->queue=clCreateCommandQueue(r->context,r->device,(r->profile_async||r->gui_profile)?CL_QUEUE_PROFILING_ENABLE:0,&error);if(!r->queue||error)goto fail;
    r->backend=dv_opencl_create_source_at(r->context,r->device,directory);if(!r->backend)goto fail;
    r->output=clCreateFromGLTexture(r->context,CL_MEM_WRITE_ONLY,0x0de1,0,texture,&error);if(!r->output||error)goto fail;
    r->primary_texture=texture;
    cl_image_format format;size_t w,h;
    if(clGetImageInfo(r->output,CL_IMAGE_FORMAT,sizeof(format),&format,NULL)||
       clGetImageInfo(r->output,CL_IMAGE_WIDTH,sizeof(w),&w,NULL)||clGetImageInfo(r->output,CL_IMAGE_HEIGHT,sizeof(h),&h,NULL)||
       w!=3840||h!=2160||format.image_channel_order!=CL_RGBA||format.image_channel_data_type!=CL_UNORM_INT8)goto fail;
    char path[4096],source[16384];int n=snprintf(path,sizeof(path),"%s/source_tunnel.cl",directory);if(n<0||(size_t)n>=sizeof(path))goto fail;
    FILE *f=fopen(path,"rb");if(!f)goto fail;size_t length=fread(source,1,sizeof(source),f);int invalid=ferror(f)||!feof(f);fclose(f);if(invalid||!length)goto fail;
    const char *ptr=source;r->program=clCreateProgramWithSource(r->context,1,&ptr,&length,&error);if(!r->program||error)goto fail;
    char options[256];const char *order=r->bgra?"-D SOURCE_SCANOUT_BGRA":"";
    snprintf(options,sizeof(options),"-cl-std=CL1.2 -D SOURCE_GL -D SOURCE_PACK_PAIR -D SOURCE_PLACED -D SOURCE_SITED %s",order);
    if(clBuildProgram(r->program,1,&r->device,options,NULL,NULL))goto fail;
    r->kernel=clCreateKernel(r->program,"source_tunnel",&error);if(!r->kernel||error)goto fail;
    if(r->full_enabled){
        r->full_program=clCreateProgramWithSource(r->context,1,&ptr,&length,&error);if(!r->full_program||error)goto fail;
        snprintf(options,sizeof(options),"-cl-std=CL1.2 -D SOURCE_GL -D SOURCE_PACK_PAIR -D SOURCE_PLACED -D SOURCE_422 %s",order);
        if(clBuildProgram(r->full_program,1,&r->device,options,NULL,NULL))goto fail;
        r->full_kernel=clCreateKernel(r->full_program,"source_tunnel",&error);if(!r->full_kernel||error)goto fail;
    }
    const char *resize=getenv("DV_PRIVATE_RESIZE");
    if(resize&&!strcmp(resize,"1")){
        r->resize_program=clCreateProgramWithSource(r->context,1,&ptr,&length,&error);if(!r->resize_program||error)goto fail;
        snprintf(options,sizeof(options),"-cl-std=CL1.2 -D SOURCE_GL -D SOURCE_PACK_PAIR -D SOURCE_PLACED -D SOURCE_SITED -D SOURCE_SCALED -D SOURCE_RESIZE_MAPS %s",order);
        if(clBuildProgram(r->resize_program,1,&r->device,options,NULL,NULL))goto fail;
        r->resize_kernel=clCreateKernel(r->resize_program,"source_tunnel",&error);if(!r->resize_kernel||error)goto fail;
        r->resize_prepare=clCreateKernel(r->resize_program,"source_resize_prepare",&error);if(!r->resize_prepare||error)goto fail;
        r->resize_rows=clCreateBuffer(r->context,CL_MEM_READ_WRITE,2*4096*sizeof(cl_uint4),NULL,&error);if(!r->resize_rows||error)goto fail;
    }
    r->packets=clCreateBuffer(r->context,CL_MEM_READ_ONLY,640,NULL,&error);if(!r->packets||error)goto fail;return r;
fail:
    (void)dv_source_renderer_destroy(r);return NULL;
}
int dv_source_renderer_attach_output(void *opaque,unsigned texture)
{
    source_renderer *r=opaque;
    if(!r||!texture||texture==r->primary_texture)return DV_INVALID;
    if(r->other_output||r->submitted||r->acquired||r->queued||r->released)return DV_INVALID;
    cl_int error=0;cl_mem image=clCreateFromGLTexture(r->context,CL_MEM_WRITE_ONLY,0x0de1,0,texture,&error);
    if(!image||error){if(image)clReleaseMemObject(image);return DV_BACKEND;}
    cl_image_format format;size_t w,h;
    if(clGetImageInfo(image,CL_IMAGE_FORMAT,sizeof(format),&format,NULL)||
       clGetImageInfo(image,CL_IMAGE_WIDTH,sizeof(w),&w,NULL)||clGetImageInfo(image,CL_IMAGE_HEIGHT,sizeof(h),&h,NULL)||
       w!=3840||h!=2160||format.image_channel_order!=CL_RGBA||format.image_channel_data_type!=CL_UNORM_INT8){
        clReleaseMemObject(image);return DV_UNSUPPORTED;}
    r->other_output=image;r->secondary_texture=texture;return DV_OK;
}
int dv_source_renderer_select_output(void *opaque,unsigned slot)
{
    source_renderer *r=opaque;if(!r||slot>1)return DV_INVALID;
    if(r->acquired||r->queued||r->released)return DV_BACKEND;
    if(slot==r->output_slot)return DV_OK;
    if(!r->other_output)return DV_UNSUPPORTED;
    cl_mem old=r->output;r->output=r->other_output;r->other_output=old;
    r->output_slot=slot;r->valid=0;return DV_OK;
}
static uint32_t checksum(const unsigned char *data,size_t bytes)
{
    uint32_t value=UINT32_MAX;for(size_t i=0;i<bytes;++i){value^=(uint32_t)data[i]<<24;
        for(unsigned bit=0;bit<8;++bit)value=(value<<1)^((value>>31)?UINT32_C(0x04c11db7):0u);}return value;
}
static unsigned packetize(const unsigned char *dm,size_t bytes,unsigned id,unsigned char out[640])
{
    unsigned count=bytes<=119?1u:1u+(unsigned)((bytes-119u+120u)/121u);size_t offset=0;memset(out,0,640);
    for(unsigned i=0;i<count;++i){unsigned char *p=out+i*128;unsigned type=count==1?0:i==0?1:i+1==count?3:2;
        p[0]=(unsigned char)(type<<6);p[1]=(unsigned char)((id<<4)|id);size_t start=i?3:5,space=i?121:119,take=bytes-offset;
        if(take>space)take=space;if(!i){p[3]=(unsigned char)(bytes>>8);p[4]=(unsigned char)bytes;}
        memcpy(p+start,dm+offset,take);offset+=take;uint32_t crc=checksum(p,124);
        for(unsigned b=0;b<4;++b)p[124+b]=(unsigned char)(crc>>(24-b*8));}
    return count;
}
int dv_source_renderer_submit_input(void *opaque,unsigned bl_surface,unsigned el_surface,unsigned w,unsigned h,
                              const void *metadata,size_t metadata_size,uint64_t frame_id,int64_t pts,
                              const dv_source_geometry *geometry,const dv_source_input *input)
{
    source_renderer *r=opaque;if(!r||r->submitted)return DV_INVALID;r->valid=0;
    if(!input||input->size!=sizeof(*input))return DV_INVALID;
    int nv12=input->format==DV_SOURCE_INPUT_NV12;
    if((input->format!=DV_SOURCE_INPUT_P010&&!nv12)||input->bit_depth!=(nv12?8u:10u))return DV_UNSUPPORTED;
    if(nv12&&(!r->nv12_enabled||el_surface!=VA_INVALID_SURFACE))return DV_UNSUPPORTED;
    int full=r->full_enabled&&!nv12;
    const char *trace=getenv("DV_SOURCE_SUBMIT_TRACE");int tracing=trace&&!strcmp(trace,"1")&&r->submit_traces<6;
    double started=tracing?submit_clock():0,parsed=0,drained=0,sync_ms=0,import_ms=0,queued=0;
    if(!dv_source_geometry_valid(geometry)||
       geometry->width!=w||geometry->height!=h||geometry->canvas_width!=3840||geometry->canvas_height!=2160)return DV_INVALID;
    if((dv_source_destination_width(geometry)!=w||dv_source_destination_height(geometry)!=h)&&!r->resize_kernel)return DV_UNSUPPORTED;
    dv_frame_settings settings;dv_single_frame_settings general={0};unsigned char dm[512];size_t bytes;
    dv_piecewise_surface_settings surface_settings={0};
    int status=r->space2_enabled?
        dv_transport_settings_from_ffmpeg(metadata,metadata_size,w,h,(dv_identity){frame_id,pts},&settings,dm,&bytes):
        dv_settings_from_ffmpeg(metadata,metadata_size,w,h,(dv_identity){frame_id,pts},&settings,dm,&bytes);
    int single=el_surface==VA_INVALID_SURFACE,use_piecewise=0;
    int polynomial=status==DV_OK;
    if(polynomial)for(unsigned c=0;c<3;++c)if(settings.composer.component[c].mapping)polynomial=0;
    /* A decoder may still supply MEL surfaces. Validated zero residual and
     * polynomial mapping need only BL; keep MMR/FEL on their existing route. */
    if(polynomial&&!settings.composer.residual_enabled)single=1;
    /* Unsupported legacy mappings may still be valid single-layer MEL.
     * The general adapter must prove no residual before ignoring a supplied EL.
     * Preserve already-supported dual-layer MMR sampling unchanged. */
    if((single||status==DV_UNSUPPORTED)&&r->piecewise_enabled&&!polynomial){
        if(!full&&geometry->chroma_siting!=DV_SOURCE_CHROMA_CENTER_LEFT)return DV_UNSUPPORTED;
        if(full&&(dv_source_destination_width(geometry)!=w||dv_source_destination_height(geometry)!=h))return DV_UNSUPPORTED;
        status=dv_single_settings_from_ffmpeg_for_depth(metadata,metadata_size,w,h,input->bit_depth,(dv_identity){frame_id,pts},&general,dm,&bytes);
        if(status)return status;
        single=1;
        uint32_t masks[3];if(dv_single_reshape_surface_masks(&general.mapping,input->bit_depth,masks))return DV_UNSUPPORTED;
        memset(&settings,0,sizeof(settings));settings.abi_version=DV_ENGINE_ABI;
        settings.identity=general.identity;settings.source=general.source;
        memcpy(settings.active,general.active,sizeof(settings.active));use_piecewise=1;
        surface_settings.frame=general;surface_settings.layout=full?1u:0u;
        surface_settings.sampling=full?geometry->chroma_siting:0u;
    }
    if(status)return status;
    /* An NV12 declaration cannot reuse a successfully parsed legacy10bit
     * polynomial frame even if the actual surface happens to be P010. */
    if(nv12&&!use_piecewise)return DV_UNSUPPORTED;
    if(settings.source.space==2&&!r->space2_enabled)return DV_UNSUPPORTED;
    if(single&&settings.composer.residual_enabled)return DV_UNSUPPORTED;
    if(!single&&geometry->chroma_siting!=DV_SOURCE_CHROMA_CENTER_LEFT)return DV_UNSUPPORTED;
    if(dv_source_place_metadata(geometry,dm,&bytes))return DV_UNSUPPORTED;
    for(unsigned c=0;c<3;++c)if((settings.source.offset[c]&65535u)||(settings.source.offset[c]>>16)>4095)return DV_UNSUPPORTED;
    if(tracing)parsed=submit_clock();
    if(dv_source_renderer_drain(r)!=DV_OK)return DV_BACKEND;
    if(tracing)drained=submit_clock();
    VASurfaceID surfaces[2]={bl_surface,el_surface};cl_int error;status=DV_BACKEND;
    for(unsigned layer=0;layer<(single?1u:2u);++layer){double before=tracing?submit_clock():0;
        if(vaSyncSurface(r->va,surfaces[layer])!=VA_STATUS_SUCCESS)goto done;
        double synced=tracing?submit_clock():0;if(tracing)sync_ms+=(synced-before)*1e3;
        for(unsigned plane=0;plane<2;++plane){r->inputs[layer*2+plane]=r->import(r->context,CL_MEM_READ_ONLY,&surfaces[layer],plane,&error);if(!r->inputs[layer*2+plane]||error)goto done;}
        if(tracing)import_ms+=(submit_clock()-synced)*1e3;}
    if(tracing)queued=submit_clock();
    status=nv12?dv_opencl_reconstruct_piecewise_nv12(r->backend,&general,w,h,r->inputs):
        use_piecewise?dv_opencl_reconstruct_piecewise_surface(r->backend,&surface_settings,w,h,r->inputs):
        single?dv_opencl_reconstruct_single_p010(r->backend,&settings,w,h,r->inputs):
        dv_opencl_reconstruct_p010(r->backend,&settings,w,h,r->inputs,r->inputs+2);if(status!=DV_OK)goto done;
    r->settings=settings;r->geometry=*geometry;r->dm_bytes=bytes;memcpy(r->dm,dm,bytes);r->submitted=1;
    r->piecewise=use_piecewise;if(use_piecewise)r->piecewise_settings=surface_settings;
    if(tracing){++r->submit_traces;fprintf(stderr,"{\"source_submit_pts\":%lld,\"parse_ms\":%.6f,\"drain_ms\":%.6f,\"va_sync_ms\":%.6f,\"import_ms\":%.6f,\"reconstruct_submit_ms\":%.6f,\"total_ms\":%.6f}\n",
        (long long)pts,(parsed-started)*1e3,(drained-parsed)*1e3,sync_ms,import_ms,(submit_clock()-queued)*1e3,(submit_clock()-started)*1e3);}
    return DV_OK;
done:
    if(dv_source_renderer_drain(r)!=DV_OK)return DV_BACKEND;
    return status;
}
int dv_source_renderer_submit(void *opaque,unsigned bl,unsigned el,unsigned w,unsigned h,
    const void *metadata,size_t bytes,uint64_t id,int64_t pts,const dv_source_geometry *geometry)
{
    const dv_source_input input={sizeof(input),DV_SOURCE_INPUT_P010,10};
    return dv_source_renderer_submit_input(opaque,bl,el,w,h,metadata,bytes,id,pts,geometry,&input);
}
int dv_source_renderer_poll(void *opaque,uint64_t frame_id,int64_t pts,int *ready)
{
    source_renderer *r=opaque;if(!r||!ready)return DV_INVALID;
    if(!r->submitted||r->settings.identity.frame_id!=frame_id||r->settings.identity.pts!=pts)return DV_IDENTITY;
    return r->piecewise?dv_opencl_piecewise_surface_poll(r->backend,&r->piecewise_settings,r->geometry.width,r->geometry.height,ready):
        dv_opencl_source_poll(r->backend,&r->settings,r->geometry.width,r->geometry.height,ready);
}
static int renderer_source_output(source_renderer *r,unsigned w,unsigned h,cl_mem planes[3])
{
    return r->piecewise?dv_opencl_piecewise_surface_output(r->backend,&r->piecewise_settings,w,h,planes):
        dv_opencl_source_output(r->backend,&r->settings,w,h,planes);
}
int dv_source_renderer_cancel(void *opaque)
{
    source_renderer *r=opaque;if(!r)return DV_INVALID;r->valid=0;
    if(dv_source_renderer_drain(r)!=DV_OK)return DV_BACKEND;
    r->submitted=0;return DV_OK;
}
static int compile_overlay(source_renderer *r,unsigned variant)
{
    cl_int error;
    if(!r->overlay_kernel[variant]){
        char source[16384],options[4608];size_t length=0;
        if(strchr(r->directory,'"')||strchr(r->directory,'\n')||
           clGetProgramInfo(r->program,CL_PROGRAM_SOURCE,sizeof(source),source,&length)||!length)return DV_BACKEND;
        const char *ptr=source;
        cl_program program=clCreateProgramWithSource(r->context,1,&ptr,NULL,&error);
        if(!program||error){if(program)clReleaseProgram(program);return DV_BACKEND;}
        const char *layout=variant==1?"-D SOURCE_422":variant==2?
            "-D SOURCE_SITED -D SOURCE_SCALED -D SOURCE_RESIZE_MAPS":"-D SOURCE_SITED";
        int n=snprintf(options,sizeof(options),"-cl-std=CL1.2 -D SOURCE_GL -D SOURCE_PACK_PAIR -D SOURCE_PLACED -D SOURCE_OVERLAY %s %s %s %s %s %s -I\"%s\"",layout,r->bgra?"-D SOURCE_SCANOUT_BGRA":"",r->gui_prepass?"-D SOURCE_OVERLAY_CACHED":"",r->gui_gray_lut?"-D SOURCE_OVERLAY_GRAY_LUT":"",r->gui_shared?"-D SOURCE_OVERLAY_SHARED":"",r->gui_tiled?"-D SOURCE_OVERLAY_TILES":"",r->directory);
        if(r->gui_pixel_cache){
            if(n>=0&&(size_t)n+32<sizeof(options)){strcat(options," -D SOURCE_OVERLAY_PIXEL_CACHE");n=(int)strlen(options);}
            else n=-1;}
        if(r->gui_sparse){
            if(n>=0&&(size_t)n+32<sizeof(options)){strcat(options," -D SOURCE_OVERLAY_SPARSE");n=(int)strlen(options);}
            else n=-1;}
        if(n<0||(size_t)n>=sizeof(options)||clBuildProgram(program,1,&r->device,options,NULL,NULL)){
            clReleaseProgram(program);return DV_BACKEND;}
        cl_kernel kernel=clCreateKernel(program,"source_tunnel",&error);
        if(!kernel||error){if(kernel)clReleaseKernel(kernel);clReleaseProgram(program);return DV_BACKEND;}
        r->overlay_program[variant]=program;r->overlay_kernel[variant]=kernel;
    }
    return DV_OK;
}
int dv_source_renderer_prepare_overlays(void *opaque)
{
    source_renderer *r=opaque;
    if(!r||r->submitted||r->acquired||r->queued||r->released||r->gui)return DV_INVALID;
    for(unsigned variant=0;variant<3;++variant){
        if((variant==1&&!r->full_kernel)||(variant==2&&!r->resize_kernel))continue;
        int status=compile_overlay(r,variant);if(status!=DV_OK)return status;
    }
    cl_int error;
    if(!r->gui_coeff){r->gui_coeff=clCreateBuffer(r->context,CL_MEM_READ_ONLY,31*sizeof(float),NULL,&error);if(!r->gui_coeff||error)return DV_BACKEND;}
    if(!r->gui_error){r->gui_error=clCreateBuffer(r->context,CL_MEM_READ_WRITE,sizeof(int),NULL,&error);if(!r->gui_error||error)return DV_BACKEND;}
    if(r->gui_gray_lut){
        if(!r->gui_gray_prepare){r->gui_gray_prepare=clCreateKernel(r->overlay_program[0],"source_gui_gray_prepare",&error);if(!r->gui_gray_prepare||error)return DV_BACKEND;}
        if(!r->gui_gray){r->gui_gray=clCreateBuffer(r->context,CL_MEM_READ_WRITE,65536*sizeof(cl_float4),NULL,&error);if(!r->gui_gray||error)return DV_BACKEND;}
    }
    if(r->gui_tiled){
        if(!r->gui_tiles_prepare){r->gui_tiles_prepare=clCreateKernel(r->overlay_program[0],"source_gui_tiles",&error);if(!r->gui_tiles_prepare||error)return DV_BACKEND;}
        if(!r->gui_tiles){r->gui_tiles=clCreateBuffer(r->context,CL_MEM_READ_WRITE,240*135,NULL,&error);if(!r->gui_tiles||error)return DV_BACKEND;}
    }
    if(r->gui_prepass){
        if(!r->gui_prepare){r->gui_prepare=clCreateKernel(r->overlay_program[0],"source_gui_prepare",&error);if(!r->gui_prepare||error)return DV_BACKEND;}
        if(r->gui_pixel_cache&&r->gui_cache_verify&&!r->gui_cache_check){r->gui_cache_check=clCreateKernel(r->overlay_program[0],"source_gui_cache_check",&error);if(!r->gui_cache_check||error)return DV_BACKEND;}
        if(!r->gui_colours){
            size_t w=0,h=0;
            if(clGetImageInfo(r->output,CL_IMAGE_WIDTH,sizeof(w),&w,NULL)||
               clGetImageInfo(r->output,CL_IMAGE_HEIGHT,sizeof(h),&h,NULL)||!w||!h||
               w>4096||h>4096)return DV_UNSUPPORTED;
            r->gui_colours=clCreateBuffer(r->context,CL_MEM_READ_WRITE,w*h*sizeof(cl_float4),NULL,&error);
            if(!r->gui_colours||error)return DV_BACKEND;
            if(r->gui_pixel_cache){r->gui_pixels=clCreateBuffer(r->context,CL_MEM_READ_WRITE,w*h*sizeof(cl_uint),NULL,&error);if(!r->gui_pixels||error)return DV_BACKEND;}
        }
    }
    return DV_OK;
}
static int overlay_revision_matches(const source_renderer *r,unsigned texture,unsigned flip,uint64_t revision)
{
    return r->gui_pixel_cache&&r->gui_cache_valid&&!r->gui_cache_invalidate&&revision&&
        r->gui_revision==revision&&r->gui_texture==texture&&r->gui_flip==flip&&
        r->gui_width==r->geometry.canvas_width&&r->gui_height==r->geometry.canvas_height;
}
static int prepare_overlay(source_renderer *r,unsigned variant,unsigned texture,unsigned flip,uint64_t revision,
                           const double rgb_to_lms[9],double white_nits,int cached_only)
{
    if(!texture||texture==r->primary_texture||texture==r->secondary_texture||
       !rgb_to_lms||r->gui||r->acquired||r->released)return DV_INVALID;
    if(!r->overlay_kernel[variant]||!r->gui_coeff||!r->gui_error)return DV_UNSUPPORTED;
    if(r->gui_prepass&&(!r->gui_prepare||!r->gui_colours))return DV_UNSUPPORTED;
    if(r->gui_gray_lut&&(!r->gui_gray_prepare||!r->gui_gray))return DV_UNSUPPORTED;
    if(r->gui_tiled&&(!r->gui_tiles_prepare||!r->gui_tiles))return DV_UNSUPPORTED;
    dv_overlay_colour colour;
    if(dv_overlay_colour_init(&r->settings.source,rgb_to_lms,white_nits,&colour))return DV_UNSUPPORTED;
    float coeff[31];
    for(unsigned i=0;i<9;++i){coeff[i]=(float)colour.source.inverse_ycc[i];coeff[i+9]=(float)colour.source.inverse_lms[i];coeff[i+18]=(float)colour.rgb_to_lms[i];}
    for(unsigned i=0;i<3;++i)coeff[i+27]=(float)colour.source.offset[i];coeff[30]=(float)colour.white_nits;
    for(unsigned i=0;i<31;++i)if(!isfinite(coeff[i]))return DV_UNSUPPORTED;
    r->gui_cache_invalidate=!r->gui_cache_valid||memcmp(coeff,r->gui_cache_coeff,sizeof(coeff));
    r->gui_reused=overlay_revision_matches(r,texture,flip,revision);
    if(cached_only&&(!r->gui_reused||r->gui_cache_verify))return DV_UNSUPPORTED;
    memcpy(r->gui_cache_coeff,coeff,sizeof(coeff));
    r->gui_revision=revision;r->gui_texture=texture;r->gui_flip=flip;
    r->gui_width=r->geometry.canvas_width;r->gui_height=r->geometry.canvas_height;
    cl_int error;
    if(!r->gui_reused||r->gui_cache_verify){
    r->gui=clCreateFromGLTexture(r->context,CL_MEM_READ_ONLY,0x0de1,0,texture,&error);
    if(!r->gui||error)return DV_BACKEND;
    size_t w=0,h=0;cl_image_format format;
    if(clGetImageInfo(r->gui,CL_IMAGE_WIDTH,sizeof(w),&w,NULL)||
       clGetImageInfo(r->gui,CL_IMAGE_HEIGHT,sizeof(h),&h,NULL)||
       clGetImageInfo(r->gui,CL_IMAGE_FORMAT,sizeof(format),&format,NULL))return DV_BACKEND;
    if(w!=r->geometry.canvas_width||h!=r->geometry.canvas_height||
       format.image_channel_order!=CL_RGBA||format.image_channel_data_type!=CL_UNORM_INT8)return DV_UNSUPPORTED;
    }
    int clear=0;r->queued=1;
    if(clEnqueueWriteBuffer(r->queue,r->gui_coeff,CL_TRUE,0,sizeof(coeff),coeff,0,NULL,NULL)||
       clEnqueueWriteBuffer(r->queue,r->gui_error,CL_TRUE,0,sizeof(clear),&clear,0,NULL,NULL))return DV_BACKEND;
    return DV_OK;
}
static int present_internal(void *opaque,uint64_t frame_id,int64_t pts,unsigned packet_id,int refresh,
                            unsigned gui,unsigned gui_flip,const double *rgb_to_lms,double white_nits,int defer,
                            uint64_t revision,int cached_only)
{
    source_renderer *r=opaque;if(!r||packet_id>15||gui_flip>1)return DV_INVALID;
    if(r->pack_pending)return DV_INVALID;
    if(!r->submitted||r->settings.identity.frame_id!=frame_id||r->settings.identity.pts!=pts)return DV_IDENTITY;
    const char *wait_trace=getenv("DV_SOURCE_WAIT_PROFILE");
    int trace_wait=wait_trace&&!strcmp(wait_trace,"1")&&r->wait_profiles<480;
    double stamp[8]={0};if(trace_wait)stamp[0]=submit_clock();
    r->valid=0;const dv_frame_settings *settings=&r->settings;const dv_source_geometry *geometry=&r->geometry;
    unsigned w=geometry->width,h=geometry->height;unsigned char dm[512],packed[640];
    memcpy(dm,r->dm,r->dm_bytes);if(refresh)dm[1]=1;
    if(dv_source_overlay_metadata(dm,r->dm_bytes,gui!=0,r->overlay_visible))return DV_INVALID;
    unsigned count=packetize(dm,r->dm_bytes,packet_id,packed);
    unsigned dw=dv_source_destination_width(geometry),dh=dv_source_destination_height(geometry);
    int resized=dw!=w||dh!=h;
    int full=r->piecewise&&r->piecewise_settings.layout==1;
    if(full&&(resized||!r->full_kernel))return DV_UNSUPPORTED;
    cl_kernel kernel=full?r->full_kernel:resized?r->resize_kernel:r->kernel;
    cl_uint4 active,black={{0,0,0,0}};
    if(dv_source_map_active(geometry,dw,dh,settings->active,active.s))return DV_INVALID;
    for(unsigned c=0;c<3;++c)black.s[c]=settings->source.offset[c]>>16;
    int status=renderer_source_output(r,w,h,r->planes);if(status!=DV_OK)goto done;
    if(trace_wait)stamp[1]=submit_clock();
    if(gui){
        unsigned variant=full?1u:resized?2u:0u;
        double started=submit_clock();
        status=prepare_overlay(r,variant,gui,gui_flip,revision,rgb_to_lms,white_nits,cached_only);if(status!=DV_OK)goto done;
        r->gui_host_ms=(submit_clock()-started)*1000;
        kernel=r->overlay_kernel[variant];
    }
    status=DV_BACKEND;
    if(trace_wait)stamp[2]=submit_clock();
#define CL_DO(call) do{if((call)!=CL_SUCCESS)goto done;}while(0)
    r->queued=1;
    if(resized){unsigned key[5]={w,h,dw,dh,geometry->chroma_siting};
        if(!r->resize_key_valid||memcmp(key,r->resize_key,sizeof(key))){
            r->resize_key_valid=0;cl_uint2 source={{w,h}},dest={{dw,dh}};
            CL_DO(clSetKernelArg(r->resize_prepare,0,sizeof(source),&source));
            CL_DO(clSetKernelArg(r->resize_prepare,1,sizeof(dest),&dest));
            CL_DO(clSetKernelArg(r->resize_prepare,2,sizeof(geometry->chroma_siting),&geometry->chroma_siting));
            CL_DO(clSetKernelArg(r->resize_prepare,3,sizeof(r->resize_rows),&r->resize_rows));
            size_t rows=dh;CL_DO(clEnqueueNDRangeKernel(r->queue,r->resize_prepare,1,NULL,&rows,NULL,0,NULL,NULL));
            memcpy(r->resize_key,key,sizeof(key));r->resize_key_valid=1;
        }
    }
    if(trace_wait)stamp[3]=submit_clock();
    CL_DO(clEnqueueWriteBuffer(r->queue,r->packets,CL_TRUE,0,count*128u,packed,0,NULL,NULL));
    if(trace_wait)stamp[4]=submit_clock();
    cl_mem objects[2]={r->output,r->gui};cl_uint object_count=r->gui?2u:1u;
    int profile_gui=gui&&r->gui_profile&&r->gui_profiles<240;
    CL_DO(clEnqueueAcquireGLObjects(r->queue,object_count,objects,0,NULL,profile_gui?&r->gui_events[0]:NULL));r->acquired=1;
    for(unsigned i=0;i<3;++i)CL_DO(clSetKernelArg(kernel,i,sizeof(cl_mem),r->planes+i));
    CL_DO(clSetKernelArg(kernel,3,sizeof(r->packets),&r->packets));CL_DO(clSetKernelArg(kernel,4,sizeof(count),&count));
    unsigned ow=geometry->canvas_width,oh=geometry->canvas_height;
    CL_DO(clSetKernelArg(kernel,5,sizeof(ow),&ow));CL_DO(clSetKernelArg(kernel,6,sizeof(oh),&oh));
    CL_DO(clSetKernelArg(kernel,7,sizeof(active),&active));CL_DO(clSetKernelArg(kernel,8,sizeof(black),&black));
    CL_DO(clSetKernelArg(kernel,9,sizeof(r->output),&r->output));
    cl_uint4 placement={{geometry->x,geometry->y,dw,dh}};
    CL_DO(clSetKernelArg(kernel,10,sizeof(placement),&placement));
    if(!full)CL_DO(clSetKernelArg(kernel,11,sizeof(geometry->chroma_siting),&geometry->chroma_siting));
    if(resized){cl_uint2 source={{w,h}};CL_DO(clSetKernelArg(kernel,12,sizeof(source),&source));
        CL_DO(clSetKernelArg(kernel,13,sizeof(r->resize_rows),&r->resize_rows));}
    if(gui){unsigned arg=full?11u:resized?14u:12u;
        if(r->gui_tiled){
            CL_DO(clSetKernelArg(r->gui_tiles_prepare,0,sizeof(r->gui),&r->gui));
            CL_DO(clSetKernelArg(r->gui_tiles_prepare,1,sizeof(ow),&ow));
            CL_DO(clSetKernelArg(r->gui_tiles_prepare,2,sizeof(oh),&oh));
            CL_DO(clSetKernelArg(r->gui_tiles_prepare,3,sizeof(gui_flip),&gui_flip));
            CL_DO(clSetKernelArg(r->gui_tiles_prepare,4,sizeof(r->gui_tiles),&r->gui_tiles));
            size_t tile_work[2]={((ow+15)/16)*16,((oh+15)/16)*4},tile_local[2]={16,4};
            CL_DO(clEnqueueNDRangeKernel(r->queue,r->gui_tiles_prepare,2,NULL,tile_work,tile_local,0,NULL,NULL));
            CL_DO(clSetKernelArg(kernel,arg+4+(r->gui_gray_lut?1u:0u),sizeof(r->gui_tiles),&r->gui_tiles));
        }
        if(r->gui_gray_lut){
            CL_DO(clSetKernelArg(r->gui_gray_prepare,0,sizeof(r->gui_coeff),&r->gui_coeff));
            CL_DO(clSetKernelArg(r->gui_gray_prepare,1,sizeof(r->gui_gray),&r->gui_gray));
            size_t gray_work=65536;
            if(!r->gui_pixel_cache||r->gui_cache_invalidate)
                CL_DO(clEnqueueNDRangeKernel(r->queue,r->gui_gray_prepare,1,NULL,&gray_work,NULL,0,NULL,NULL));
            if(r->gui_prepass)CL_DO(clSetKernelArg(r->gui_prepare,7,sizeof(r->gui_gray),&r->gui_gray));
            else CL_DO(clSetKernelArg(kernel,arg+4,sizeof(r->gui_gray),&r->gui_gray));
        }
        if(r->gui_prepass){
            size_t gui_work[2]={ow,oh};
            if(!r->gui_reused){
            CL_DO(clSetKernelArg(r->gui_prepare,0,sizeof(r->gui),&r->gui));
            CL_DO(clSetKernelArg(r->gui_prepare,1,sizeof(r->gui_coeff),&r->gui_coeff));
            CL_DO(clSetKernelArg(r->gui_prepare,2,sizeof(gui_flip),&gui_flip));
            CL_DO(clSetKernelArg(r->gui_prepare,3,sizeof(r->gui_error),&r->gui_error));
            CL_DO(clSetKernelArg(r->gui_prepare,4,sizeof(r->gui_colours),&r->gui_colours));
            CL_DO(clSetKernelArg(r->gui_prepare,5,sizeof(ow),&ow));
            CL_DO(clSetKernelArg(r->gui_prepare,6,sizeof(oh),&oh));
            if(r->gui_pixel_cache){unsigned cache_arg=r->gui_gray_lut?8u:7u;cl_uint invalidate=(cl_uint)r->gui_cache_invalidate;
                CL_DO(clSetKernelArg(r->gui_prepare,cache_arg,sizeof(r->gui_pixels),&r->gui_pixels));
                CL_DO(clSetKernelArg(r->gui_prepare,cache_arg+1,sizeof(invalidate),&invalidate));}
            CL_DO(clEnqueueNDRangeKernel(r->queue,r->gui_prepare,2,NULL,gui_work,NULL,0,NULL,profile_gui?&r->gui_events[1]:NULL));
            }
            if(r->gui_cache_check){
                CL_DO(clSetKernelArg(r->gui_cache_check,0,sizeof(r->gui),&r->gui));
                CL_DO(clSetKernelArg(r->gui_cache_check,1,sizeof(r->gui_coeff),&r->gui_coeff));
                CL_DO(clSetKernelArg(r->gui_cache_check,2,sizeof(gui_flip),&gui_flip));
                CL_DO(clSetKernelArg(r->gui_cache_check,3,sizeof(r->gui_error),&r->gui_error));
                CL_DO(clSetKernelArg(r->gui_cache_check,4,sizeof(r->gui_colours),&r->gui_colours));
                CL_DO(clSetKernelArg(r->gui_cache_check,5,sizeof(ow),&ow));
                CL_DO(clSetKernelArg(r->gui_cache_check,6,sizeof(oh),&oh));
                CL_DO(clEnqueueNDRangeKernel(r->queue,r->gui_cache_check,2,NULL,gui_work,NULL,0,NULL,NULL));}
        }
        cl_mem gui_input=r->gui_prepass?r->gui_colours:r->gui;
        CL_DO(clSetKernelArg(kernel,arg,sizeof(gui_input),&gui_input));
        CL_DO(clSetKernelArg(kernel,arg+1,sizeof(r->gui_coeff),&r->gui_coeff));
        CL_DO(clSetKernelArg(kernel,arg+2,sizeof(gui_flip),&gui_flip));
        CL_DO(clSetKernelArg(kernel,arg+3,sizeof(r->gui_error),&r->gui_error));
        if(r->gui_sparse)CL_DO(clSetKernelArg(kernel,arg+4,sizeof(r->gui_pixels),&r->gui_pixels));}
    size_t work[2]={ow/2,oh};CL_DO(clEnqueueNDRangeKernel(r->queue,kernel,2,NULL,work,NULL,0,NULL,
        profile_gui?&r->gui_events[2]:r->profile_async&&r->pack_profiles<120?&r->pack_done:NULL));
    if(trace_wait)stamp[5]=submit_clock();
    r->pack_error=0;
    if(gui){CL_DO(clEnqueueReadBuffer(r->queue,r->gui_error,defer?CL_FALSE:CL_TRUE,0,
                                    sizeof(r->pack_error),&r->pack_error,0,NULL,NULL));
        if(!defer&&r->pack_error){status=DV_UNSUPPORTED;goto done;}}
    if(defer){
        CL_DO(clEnqueueReleaseGLObjects(r->queue,object_count,objects,0,NULL,&r->released));
        CL_DO(clFlush(r->queue));
        r->pack_pending=1;r->pack_overlay=gui!=0;r->pack_cached_gui=cached_only;
        r->count=count;memcpy(r->packed,packed,sizeof(packed));
        return DV_OK;
    }
    status=DV_OK;
done:
    if(trace_wait)stamp[6]=submit_clock();
    if(r->gui_pixel_cache)r->gui_cache_valid=gui&&status==DV_OK;
    if(status!=DV_OK)r->resize_key_valid=0;
    if(dv_source_renderer_drain(r)!=DV_OK){r->gui_cache_valid=0;r->resize_key_valid=0;return DV_BACKEND;}
    if(trace_wait&&status==DV_OK){stamp[7]=submit_clock();
        fprintf(stderr,"{\"source_wait_pts\":%lld,\"overlay\":%u,\"output_wait_ms\":%.6f,\"overlay_prepare_ms\":%.6f,\"packet_write_ms\":%.6f,\"enqueue_ms\":%.6f,\"error_read_ms\":%.6f,\"release_drain_ms\":%.6f,\"total_ms\":%.6f}\n",
            (long long)pts,gui!=0,(stamp[1]-stamp[0])*1e3,(stamp[2]-stamp[1])*1e3,
            (stamp[4]-stamp[3])*1e3,(stamp[5]-stamp[4])*1e3,(stamp[6]-stamp[5])*1e3,
            (stamp[7]-stamp[6])*1e3,(stamp[7]-stamp[0])*1e3);++r->wait_profiles;}
    r->submitted=0;
    if(status==DV_OK){r->overlay_visible=gui!=0;r->valid=1;r->count=count;memcpy(r->packed,packed,sizeof(packed));}return status;
#undef CL_DO
}
int dv_source_renderer_present(void *opaque,uint64_t id,int64_t pts,unsigned packet,int refresh)
{return present_internal(opaque,id,pts,packet,refresh,0,0,NULL,0,0,0,0);}
int dv_source_renderer_present_overlay(void *opaque,uint64_t id,int64_t pts,unsigned packet,int refresh,
                                      unsigned gui,unsigned flip,const double rgb_to_lms[9],double white_nits)
{
    if(!gui||!rgb_to_lms)return DV_INVALID;
    return present_internal(opaque,id,pts,packet,refresh,gui,flip,rgb_to_lms,white_nits,0,0,0);
}
int dv_source_renderer_present_overlay_revision(void *opaque,uint64_t id,int64_t pts,unsigned packet,int refresh,
                                      unsigned gui,unsigned flip,const double rgb_to_lms[9],double white_nits,uint64_t revision)
{
    if(!gui||!rgb_to_lms)return DV_INVALID;
    return present_internal(opaque,id,pts,packet,refresh,gui,flip,rgb_to_lms,white_nits,0,revision,0);
}
int dv_source_renderer_pack_overlay_cached(void *opaque,uint64_t id,int64_t pts,unsigned packet,int refresh,
                                      unsigned gui,unsigned flip,const double rgb_to_lms[9],double white_nits,uint64_t revision)
{
    if(!gui||!rgb_to_lms||!revision)return DV_INVALID;
    return present_internal(opaque,id,pts,packet,refresh,gui,flip,rgb_to_lms,white_nits,1,revision,1);
}
int dv_source_renderer_pack_begin(void *opaque,uint64_t id,int64_t pts,unsigned packet,int refresh,
                                 unsigned gui,unsigned flip,const double rgb_to_lms[9],double white_nits)
{
    if(gui&&!rgb_to_lms)return DV_INVALID;
    return present_internal(opaque,id,pts,packet,refresh,gui,flip,rgb_to_lms,white_nits,1,0,0);
}
int dv_source_renderer_pack_complete(void *opaque,uint64_t id,int64_t pts,int wait,int *ready)
{
    source_renderer *r=opaque;if(!r||!ready||(wait!=0&&wait!=1))return DV_INVALID;
    *ready=0;
    if(!r->pack_pending||r->settings.identity.frame_id!=id||r->settings.identity.pts!=pts)return DV_IDENTITY;
    cl_int state=1;
    if(!r->released||clGetEventInfo(r->released,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(state),&state,NULL)||state<0)
        return DV_BACKEND;
    if(!wait&&state!=CL_COMPLETE)return DV_OK;
    int overlay=r->pack_overlay;
    if(dv_source_renderer_drain(r)!=DV_OK){r->gui_cache_valid=0;r->resize_key_valid=0;return DV_BACKEND;}
    if(r->pack_error){r->resize_key_valid=0;return DV_UNSUPPORTED;}
    r->overlay_visible=overlay;r->valid=1;
    if(r->gui_pixel_cache)r->gui_cache_valid=overlay;
    *ready=1;return DV_OK;
}
int dv_source_renderer_render(void *opaque,unsigned bl_surface,unsigned el_surface,unsigned w,unsigned h,
                              const void *metadata,size_t metadata_size,uint64_t frame_id,int64_t pts,
                              unsigned packet_id,int refresh,const dv_source_geometry *geometry)
{
    source_renderer *r=opaque;if(!r)return DV_INVALID;r->valid=0;if(packet_id>15)return DV_INVALID;
    int status=dv_source_renderer_submit(r,bl_surface,el_surface,w,h,metadata,metadata_size,frame_id,pts,geometry);
    return status==DV_OK?dv_source_renderer_present(r,frame_id,pts,packet_id,refresh):status;
}
static unsigned reference_linear(const uint16_t *p,unsigned w,unsigned h,double x,double y)
{
    x=fmax(0,fmin(x,(double)w-1));y=fmax(0,fmin(y,(double)h-1));
    unsigned ix=(unsigned)floor(x),iy=(unsigned)floor(y);double value=0;
    for(unsigned dy=0;dy<2;++dy)for(unsigned dx=0;dx<2;++dx){
        unsigned xx=ix+dx<w?ix+dx:w-1,yy=iy+dy<h?iy+dy:h-1;
        value+=p[(size_t)yy*w+xx]*(dx?x-ix:1-x+ix)*(dy?y-iy:1-y+iy);}
    return (unsigned)floor(value+0.5);
}
static int check_overlay_pack(source_renderer *r,const unsigned char *rgba,size_t bytes)
{
    if(!r->gui_pixel_cache||!r->gui_cache_valid)return DV_UNSUPPORTED;
    cl_mem planes[3]={0},image=NULL;unsigned char *reference=NULL;cl_int error=0;
    unsigned variant=r->piecewise&&r->piecewise_settings.layout==1?1u:
        dv_source_destination_width(&r->geometry)!=r->geometry.width||
        dv_source_destination_height(&r->geometry)!=r->geometry.height?2u:0u;
    cl_kernel kernel=r->overlay_kernel[variant];int result=DV_BACKEND;
    if(!kernel||renderer_source_output(r,r->geometry.width,r->geometry.height,planes)!=DV_OK)goto done;
    cl_image_format format={CL_RGBA,CL_UNORM_INT8};cl_image_desc desc={0};
    desc.image_type=CL_MEM_OBJECT_IMAGE2D;desc.image_width=3840;desc.image_height=2160;
    image=clCreateImage(r->context,CL_MEM_WRITE_ONLY,&format,&desc,NULL,&error);
    reference=malloc(bytes);if(!image||error||!reference)goto done;
    for(unsigned i=0;i<3;++i)if(clSetKernelArg(kernel,i,sizeof(cl_mem),planes+i))goto done;
    if(clSetKernelArg(kernel,9,sizeof(image),&image))goto done;
    size_t work[2]={1920,2160},origin[3]={0},region[3]={3840,2160,1};
    if(clEnqueueNDRangeKernel(r->queue,kernel,2,NULL,work,NULL,0,NULL,NULL)||
       clEnqueueReadImage(r->queue,image,CL_TRUE,origin,region,0,0,reference,0,NULL,NULL))goto done;
    size_t differences=0;for(size_t i=0;i<bytes;++i)differences+=reference[i]!=rgba[i];
    fprintf(stderr,"{\"overlay_repack_verified_pts\":%lld,\"bytes\":%zu,\"byte_differences\":%zu}\n",
            (long long)r->settings.identity.pts,bytes,differences);
    result=differences?DV_SAMPLE_RANGE:DV_OK;
done:
    if(kernel&&clSetKernelArg(kernel,9,sizeof(r->output),&r->output))result=DV_BACKEND;
    if(clFinish(r->queue))result=DV_BACKEND;
    for(unsigned i=0;i<3;++i)if(planes[i])clReleaseMemObject(planes[i]);
    if(image)clReleaseMemObject(image);free(reference);return result;
}
int dv_source_renderer_check(void *opaque,const unsigned char *rgba,size_t bytes)
{
    source_renderer *r=opaque;const unsigned w=3840,h=2160;
    if(!r||!r->valid||!rgba||bytes!=(size_t)w*h*4)return DV_INVALID;
    if(r->overlay_visible)return check_overlay_pack(r,rgba,bytes);
    cl_mem buffers[3]={NULL,NULL,NULL};uint16_t *planes[3]={NULL,NULL,NULL};int result=DV_BACKEND;
    const dv_source_geometry *g=&r->geometry;unsigned sw=g->width,sh=g->height;
    unsigned dw=dv_source_destination_width(g),dh=dv_source_destination_height(g);
    int resized=sw!=dw||sh!=dh;uint32_t area[4];
    int full=r->piecewise&&r->piecewise_settings.layout==1;
    if(full&&resized)return DV_UNSUPPORTED;
    if(dv_source_map_active(g,dw,dh,r->settings.active,area))return DV_INVALID;
    if(renderer_source_output(r,sw,sh,buffers)!=DV_OK)return DV_BACKEND;
    for(unsigned c=0;c<3;++c){size_t size=(size_t)sw*sh*(c?(full?2u:1u):4u)/2u;planes[c]=malloc(size);
        if(!planes[c]||clEnqueueReadBuffer(r->queue,buffers[c],CL_TRUE,0,size,planes[c],0,NULL,NULL))goto done;}
    size_t differences=0,payload_errors=0;unsigned max_y=0,max_c=0;
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){size_t i=(size_t)y*w+x;
        unsigned lx=x-g->x,ly=y-g->y;
        unsigned yy=r->settings.source.offset[0]>>16,c=r->settings.source.offset[1+(x&1u)]>>16;
        int sampled=x>=g->x&&y>=g->y&&lx<dw&&ly<dh&&x>=area[0]&&y>=area[1]&&x<area[2]&&y<area[3];
        if(sampled){
            if(resized){double px=((double)lx+0.5)*sw/dw-0.5,py=((double)ly+0.5)*sh/dh-0.5;
                double cx=(((double)(lx/2)*2+0.5)*sw/dw-0.5)/2;
                yy=reference_linear(planes[0],sw,sh,px,py);
                c=reference_linear(planes[1+(x&1u)],sw/2,sh/2,cx,(py-(g->chroma_siting==DV_SOURCE_CHROMA_TOP_LEFT?0:0.5))/2);
            }else if(full){yy=planes[0][(size_t)ly*sw+lx];c=planes[1+(x&1u)][(size_t)ly*(sw/2)+lx/2];
            }else{
            unsigned cy=ly/2,top=(ly&1u)?cy:(cy?cy-1:0),bottom=(ly&1u)?(cy+1<sh/2?cy+1:sh/2-1):cy,weight=(ly&1u)?3u:1u;
            if(g->chroma_siting==DV_SOURCE_CHROMA_TOP_LEFT){top=cy;bottom=cy+1<sh/2?cy+1:sh/2-1;weight=(ly&1u)?2u:4u;}
            yy=planes[0][(size_t)ly*sw+lx];uint16_t *uv=planes[1+(x&1u)];
            c=(weight*uv[(size_t)top*(sw/2)+lx/2]+(4-weight)*uv[(size_t)bottom*(sw/2)+lx/2]+2)/4;}
        }
        if(resized){unsigned low=rgba[i*4+(r->bgra?0u:2u)];
            unsigned actual_y=(unsigned)rgba[i*4+1]*16+(low&15u);
            unsigned actual_c=(unsigned)rgba[i*4+(r->bgra?2u:0u)]*16+(low>>4);
            unsigned dy=actual_y>yy?actual_y-yy:yy-actual_y;
            int embedded=i<(size_t)r->count*3072;
            if(embedded){unsigned bit=((unsigned)r->packed[(i/3072)*128+(i%1024)/8]>>(7-i%8))&1u,parity=actual_c&1u;
                for(unsigned k=0;k<12;++k)parity^=(actual_y>>k)&1u;
                for(unsigned k=1;k<12;++k)parity^=(actual_c>>k)&1u;
                payload_errors+=parity!=bit;actual_c&=4094u;c&=4094u;}
            unsigned dc=actual_c>c?actual_c-c:c-actual_c;
            if(dy>max_y)max_y=dy;if(dc>max_c)max_c=dc;
            differences+=dy>(sampled?1u:0u)||dc>(sampled?(embedded?2u:1u):0u)||rgba[i*4+3]!=255;continue;
        }
        if(i<(size_t)r->count*3072){unsigned bit=((unsigned)r->packed[(i/3072)*128+(i%1024)/8]>>(7-i%8))&1u;
            unsigned parity=bit;for(unsigned k=0;k<12;++k)parity^=(yy>>k)&1u;for(unsigned k=1;k<12;++k)parity^=(c>>k)&1u;c=(c&4094u)|parity;}
        unsigned char expected[4]={(unsigned char)(c/16),(unsigned char)(yy/16),(unsigned char)((c%16)*16+yy%16),255};
        if(r->bgra){unsigned char tmp=expected[0];expected[0]=expected[2];expected[2]=tmp;}
        for(unsigned k=0;k<4;++k)differences+=expected[k]!=rgba[i*4+k];
    }
    if(resized)fprintf(stderr,"{\"private_GL_resize_verified_pts\":%lld,\"pixels\":%u,\"outside_tolerance\":%zu,\"payload_errors\":%zu,\"max_luma_delta\":%u,\"max_chroma_colour_delta\":%u}\n",(long long)r->settings.identity.pts,w*h,differences,payload_errors,max_y,max_c);
    else fprintf(stderr,"{\"private_GL_pack_verified_pts\":%lld,\"bytes\":%zu,\"byte_differences\":%zu}\n",(long long)r->settings.identity.pts,bytes,differences);
    result=differences||payload_errors?DV_SAMPLE_RANGE:DV_OK;
done:
    for(unsigned c=0;c<3;++c){free(planes[c]);if(buffers[c])clReleaseMemObject(buffers[c]);}return result;
}
