/* Private synchronous OpenCL adapter; GPU intermediates stay resident. */
#define _POSIX_C_SOURCE 200809L
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_va_api_media_sharing_intel.h>
#include "dv_gpu_backend.h"
#include "dv_backend_internal.h"
#include "mmr_prepare.h"
#include "pixel_polynomial.h"
#include "mmr64_guard.h"
#include "mmr64_lut.h"
#include "dv_colour_internal.h"
#ifdef DV_COLOUR_LUT
#include "colour_lut.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
static double host_seconds(void){struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))return 0;return (double)t.tv_sec+(double)t.tv_nsec/1e9;}
int dv_reference_luma(const dv_intel_composer_config *,uint16_t,uint16_t,int32_t *,int32_t *,uint16_t *);
enum {B=0,E=3,BP=6,EP=8,V=9,ES=10,G=12,MT=13,RT=14,P=17,O=19,M=22,R=25,S=28,ERR=31,CP=32,CS=33,CO=34,CT=37,DL=38,ELUT=39,MASK=40,PCR=41,PCT=42,PRC=43,PRN=44,FDL=45,FELUT=46,PATCH=47,UL=48,VP=50,SM=51,SL=52,CM=53,NB=55};
typedef struct {
    cl_context context;cl_command_queue queue;cl_program program,input_program;cl_kernel kernel[29];int filter_fused,verify_filter,luma_pair,mmr_pair,mmr_uniform,mmr_mulhi,mmr_dual,verify_recon;
    int direct_p010,direct_active,defer_va_release;
    cl_mem buffer[NB];size_t bytes[NB];cl_event events[32];unsigned event_kind[32],event_count;
    double kernel_seconds;size_t uploads,downloads,allocated;
    cl_device_id device;int resident_ready,luts_ready,combine_matrix,compact,sparse_repair,fast_tables,local_compact,bounded_refine,fused,shared_prepared,output_ready;unsigned width,height,chroma_phase;
    cl_kernel unpack,unpack_reference,patch,pack,single;int unpack_block;cl_event output_event,reconstruction_done;
    uint16_t single_mapping[3][1024];
    dv_intel_composer_config single_config;
    /* MT is shared with dual-layer reconstruction; host and GPU validity differ. */
    int single_cache,single_cpu_valid,single_gpu_valid;
    unsigned single_hits,single_misses;
    cl_program piecewise_program;cl_kernel piecewise_kernel,piecewise_full_kernel;
    cl_program piecewise_order2_program;cl_kernel piecewise_order2_kernel;
    unsigned piecewise_order2_hits,piecewise_order2_misses;
    cl_program nv12_program;cl_kernel nv12_kernel;
    unsigned piecewise_layout,piecewise_sampling;
    dv_single_frame_settings piecewise_frame;
    dv_single_reshape piecewise_mapping;uint16_t piecewise_luma[1024];
    int piecewise_cached,piecewise_active;
    cl_program image_program;cl_kernel image_kernel[3];int image_active,image_output;
    cl_mem owned_va[4],owned_gl[3];unsigned va_count,gl_count;cl_event va_release,gl_release;
    cl_event unpack_event[2],pack_event;unsigned unpack_event_count;
    uint16_t mapping[1024];int32_t residual_table[3][1024];uint32_t mmr_params[2][96],reconstruction_error;
    uint16_t chroma_mapping[2][1024];
    float colour_params[78];uint32_t colour_source[13],colour_candidates,colour_repair_count;
    dv_identity identity;dv_intel_composer_config config;int mmr96,mmr64,mmr_lut,recon2d;
    dv_mmr64_lut_cache mmr_lut_cache[2];
    unsigned submit_traces;
    int profile_async,profile_pending;
    unsigned profile_frames;
    cl_event piecewise_profile_event;
    unsigned piecewise_profile_frames,piecewise_profile_width,piecewise_profile_height,piecewise_profile_depth,piecewise_profile_layout;
    int64_t piecewise_profile_pts;
    int base_fused,verify_base;unsigned base_checks;
    int horizontal_fused,verify_horizontal;unsigned horizontal_checks;
} backend;
static const char *names[8]={"phase","vertical","scale_horizontal","guide","luma","mmr","colour","colour_refine"};
void dv_opencl_destroy(void *opaque)
{
    backend *b=opaque;if(!b)return;
    if(b->queue&&dv_opencl_drain(b)!=DV_OK)return; /* leak rather than free live ownership */
    if(b->single_cache)fprintf(stderr,"{\"single_table_cache_hits\":%u,\"single_table_cache_misses\":%u}\n",b->single_hits,b->single_misses);
    for(unsigned i=0;i<b->event_count;++i)clReleaseEvent(b->events[i]);
    for(unsigned i=0;i<NB;++i)if(b->buffer[i])clReleaseMemObject(b->buffer[i]);
    for(unsigned i=0;i<29;++i)if(b->kernel[i])clReleaseKernel(b->kernel[i]);
    if(b->input_program)clReleaseProgram(b->input_program);
    if(b->single)clReleaseKernel(b->single);
    if(b->piecewise_kernel)clReleaseKernel(b->piecewise_kernel);
    if(b->piecewise_full_kernel)clReleaseKernel(b->piecewise_full_kernel);
    if(b->piecewise_program)clReleaseProgram(b->piecewise_program);
    if(b->piecewise_order2_kernel){
        fprintf(stderr,"{\"piecewise_order2_selected\":%u,\"piecewise_order2_fallback\":%u}\n",b->piecewise_order2_hits,b->piecewise_order2_misses);
        clReleaseKernel(b->piecewise_order2_kernel);
    }
    if(b->piecewise_order2_program)clReleaseProgram(b->piecewise_order2_program);
    if(b->nv12_kernel)clReleaseKernel(b->nv12_kernel);
    if(b->nv12_program)clReleaseProgram(b->nv12_program);
    if(b->reconstruction_done)clReleaseEvent(b->reconstruction_done);
    if(b->unpack)clReleaseKernel(b->unpack);if(b->unpack_reference)clReleaseKernel(b->unpack_reference);if(b->patch)clReleaseKernel(b->patch);if(b->pack)clReleaseKernel(b->pack);if(b->output_event)clReleaseEvent(b->output_event);
    for(unsigned i=0;i<3;++i)if(b->image_kernel[i])clReleaseKernel(b->image_kernel[i]);
    if(b->image_program)clReleaseProgram(b->image_program);
    if(b->program)clReleaseProgram(b->program);if(b->queue)clReleaseCommandQueue(b->queue);
    if(b->context)clReleaseContext(b->context);free(b);
}
static FILE *shader_file(const char *directory,const char *name)
{
    char path[4096];int n=snprintf(path,sizeof(path),"%s/%s",directory?directory:".",name);
    return n<0||(size_t)n>=sizeof(path)?NULL:fopen(path,"rb");
}
static void *create_backend(cl_context supplied,cl_device_id supplied_device,int force_planar,const char *directory,int source_only)
{
    cl_platform_id platforms[16];cl_uint np=0;cl_device_id device=NULL;
    if(supplied){device=supplied_device;if(!device)return NULL;}
    else if(clGetPlatformIDs(16,platforms,&np)!=CL_SUCCESS)return NULL;
    for(cl_uint p=0;p<np && p<16 && !device;++p){
        cl_device_id devices[16];cl_uint nd=0;
        if(clGetDeviceIDs(platforms[p],CL_DEVICE_TYPE_GPU,16,devices,&nd)!=CL_SUCCESS)continue;
        for(cl_uint i=0;i<nd && i<16;++i){cl_uint vendor=0;
            if(clGetDeviceInfo(devices[i],CL_DEVICE_VENDOR_ID,sizeof(vendor),&vendor,NULL)==CL_SUCCESS && vendor==0x8086){device=devices[i];break;}}
    }
    if(!device){fprintf(stderr,"No Intel OpenCL GPU; refusing CPU fallback\n");return NULL;}
    backend *b=calloc(1,sizeof(*b));if(!b)return NULL;b->device=device;cl_int err;
    const char *defer=getenv("DV_DEFER_VA_RELEASE");b->defer_va_release=defer&&!strcmp(defer,"1");
    const char *profile=getenv("DV_SOURCE_PROFILE_ASYNC");b->profile_async=profile&&!strcmp(profile,"1");
    const char *base=getenv("DV_BASE_PREP_FUSED");b->base_fused=base&&!strcmp(base,"1");
    const char *single_cache=getenv("DV_SINGLE_TABLE_CACHE");b->single_cache=single_cache&&!strcmp(single_cache,"1");
    const char *verify_base=getenv("DV_VERIFY_BASE_PREP");b->verify_base=verify_base&&!strcmp(verify_base,"1");
    if(supplied){err=clRetainContext(supplied);if(err)goto fail;b->context=supplied;}else b->context=clCreateContext(NULL,1,&device,NULL,NULL,&err);if(!b->context || err)goto fail;
    b->queue=clCreateCommandQueue(b->context,device,CL_QUEUE_PROFILING_ENABLE,&err);if(!b->queue || err)goto fail;
    FILE *f=shader_file(directory,"opencl_kernels.cl");if(!f)goto fail;
    char source[65536],options[512];size_t len=fread(source,1,sizeof(source),f);int invalid=ferror(f) || !feof(f);fclose(f);if(invalid || !len)goto fail;source[len]=0;
#ifdef DV_COLOUR_LUT
    size_t expected_encode_count=1+4*(size_t)(DV_ENCODE_LAST_EXP-DV_ENCODE_FIRST_EXP+1)*DV_ENCODE_STEPS;
    if(dv_encode_lut_count()!=expected_encode_count||dv_decode_lut_count()!=4*(size_t)(DV_DECODE_LAST+1)){fprintf(stderr,"LUT object/header configuration mismatch; rebuild all objects\n");goto fail;}
    char helper[16384];f=shader_file(directory,"colour_dd.cl");if(!f)goto fail;
    size_t helper_len=fread(helper,1,sizeof(helper),f);invalid=ferror(f) || !feof(f);fclose(f);if(invalid || !helper_len)goto fail;
    const char *ptrs[2]={helper,source};size_t lengths[2]={helper_len,len};
    b->program=clCreateProgramWithSource(b->context,2,ptrs,lengths,&err);
    const char *bit_env=getenv("DV_COLOUR_BIT_INDEX");int bit_index=bit_env && strcmp(bit_env,"1")==0;
    const char *combine_env=getenv("DV_COLOUR_COMBINE_MATRIX");b->combine_matrix=combine_env && strcmp(combine_env,"1")==0;
    const char *split_env=getenv("DV_COLOUR_U32_SPLIT");int u32_split=split_env && strcmp(split_env,"1")==0;
    const char *uniform_env=getenv("DV_COLOUR_CONSTANT_UNIFORMS");int constant_uniforms=uniform_env && strcmp(uniform_env,"1")==0;
    const char *selective_env=getenv("DV_COLOUR_SELECTIVE");int selective=selective_env && strcmp(selective_env,"1")==0;
    const char *compact_env=getenv("DV_COLOUR_COMPACT");b->compact=compact_env && strcmp(compact_env,"1")==0;if(b->compact)selective=1;
    const char *repair_env=getenv("DV_COLOUR_SPARSE_REPAIR");b->sparse_repair=repair_env && strcmp(repair_env,"1")==0;
    if(b->sparse_repair && !b->compact){fprintf(stderr,"Sparse repair requires compact colour\n");goto fail;}
    const char *table_env=getenv("DV_COLOUR_FAST_TABLES");b->fast_tables=table_env && strcmp(table_env,"1")==0;
    if(b->fast_tables && !b->compact){fprintf(stderr,"Fast screening tables require compact colour\n");goto fail;}
    const char *local_env=getenv("DV_COLOUR_LOCAL_COMPACT");b->local_compact=local_env && strcmp(local_env,"1")==0;
    if(b->local_compact && !b->compact){fprintf(stderr,"Local compaction requires compact colour\n");goto fail;}
    const char *bounded_env=getenv("DV_COLOUR_BOUNDED_REFINE");b->bounded_refine=bounded_env && strcmp(bounded_env,"1")==0;
    if(b->bounded_refine && !b->compact){fprintf(stderr,"Bounded refinement requires compact colour\n");goto fail;}
    snprintf(options,sizeof(options),"-cl-std=CL1.2 -D DV_COLOUR_LUT -D DV_DECODE_LAST=%d -D DV_ENCODE_STEPS=%d%s%s%s%s%s%s",DV_DECODE_LAST,DV_ENCODE_STEPS,bit_index?" -D DV_COLOUR_BIT_INDEX":"",b->combine_matrix?" -D DV_COLOUR_COMBINE_MATRIX":"",u32_split?" -D DV_COLOUR_U32_SPLIT":"",constant_uniforms?" -D DV_COLOUR_CONSTANT_UNIFORMS":"",selective?" -D DV_COLOUR_SELECTIVE":"",b->compact?" -D DV_COLOUR_COMPACT":"");
    fprintf(stderr,"{\"colour_bit_index\":%d,\"colour_combined_matrix\":%d,\"colour_u32_split\":%d,\"colour_constant_uniforms\":%d,\"colour_selective\":%d,\"colour_compact\":%d}\n",bit_index,b->combine_matrix,u32_split,constant_uniforms,selective,b->compact);
#else
    const char *ptr=source;b->program=clCreateProgramWithSource(b->context,1,&ptr,&len,&err);
    snprintf(options,sizeof(options),"-cl-std=CL1.2");
#endif
    const char *mul_env=getenv("DV_MMR_NATIVE_MUL"),*wide_env=getenv("DV_MMR_SIGNED64");
    int native_mul=mul_env && strcmp(mul_env,"1")==0,wide=wide_env && strcmp(wide_env,"1")==0;
    const char *fused_env=getenv("DV_COLOUR_SCREEN_FUSED");int fused=fused_env && strcmp(fused_env,"1")==0;b->fused=fused;
    size_t used=strlen(options);snprintf(options+used,sizeof(options)-used,"%s%s%s%s%s%s%s",native_mul?" -D DV_MMR_NATIVE_MUL":"",wide?" -D DV_MMR_SIGNED64":"",b->sparse_repair?" -D DV_COLOUR_SPARSE_REPAIR":"",b->fast_tables?" -D DV_COLOUR_FAST_TABLES":"",b->local_compact?" -D DV_COLOUR_LOCAL_COMPACT":"",b->bounded_refine?" -D DV_COLOUR_BOUNDED_REFINE":"",fused?" -D DV_COLOUR_SCREEN_FUSED":"");
    fprintf(stderr,"{\"colour_bounded_refine\":%d}\n",b->bounded_refine);
    fprintf(stderr,"{\"colour_local_compact\":%d}\n",b->local_compact);
    fprintf(stderr,"{\"colour_fast_tables\":%d}\n",b->fast_tables);
    fprintf(stderr,"{\"mmr_native_mul\":%d,\"mmr_signed64\":%d}\n",native_mul,wide);
    const char *stream_env=getenv("DV_MMR_STREAM");int stream=stream_env && strcmp(stream_env,"1")==0;
    used=strlen(options);snprintf(options+used,sizeof(options)-used,"%s",stream?" -D DV_MMR_STREAM":"");
    fprintf(stderr,"{\"mmr_stream\":%d}\n",stream);
    const char *subgroup_env=getenv("DV_COLOUR_SUBGROUP");int subgroup=0;
    if(subgroup_env && strcmp(subgroup_env,"1")==0){
        /* CL_DEVICE_OPENCL_C_FEATURES from OpenCL 3.0. Retain CL1.2 if the
         * device does not advertise the exact subgroup language feature. */
        struct {cl_uint version;char name[64];} features[64];size_t bytes=0;
        if(clGetDeviceInfo(device,0x106F,sizeof(features),features,&bytes)==CL_SUCCESS)
            for(size_t i=0;i<bytes/sizeof(features[0]);++i)
                if(strcmp(features[i].name,"__opencl_c_subgroups")==0)subgroup=1;
        if(subgroup){memcpy(options,"-cl-std=CL3.0",strlen("-cl-std=CL3.0"));
            used=strlen(options);snprintf(options+used,sizeof(options)-used," -D DV_COLOUR_SUBGROUP");}
        else fprintf(stderr,"Subgroup feature unavailable; retaining CL1.2 colour path\n");
    }
    fprintf(stderr,"{\"colour_subgroup\":%d}\n",subgroup);
    const char *grid_env=getenv("DV_RECON_2D");b->recon2d=grid_env&&strcmp(grid_env,"1")==0;
    used=strlen(options);if(b->recon2d){const char *extra=" -D DV_RECON_2D";if(used+strlen(extra)>=sizeof(options))goto fail;strcat(options,extra);}
    fprintf(stderr,"{\"reconstruction_2d\":%d}\n",b->recon2d);
    if(!b->program || err)goto fail;
    fprintf(stderr,"{\"opencl_build_options\":\"%s\"",options);
#ifdef DV_COLOUR_LUT
    fprintf(stderr,",\"compiled_encode_steps\":%d,\"encode_lut_entries\":%zu,\"decode_lut_entries\":%zu",DV_ENCODE_STEPS,dv_encode_lut_count(),dv_decode_lut_count());
#endif
    fprintf(stderr,"}\n");
    err=clBuildProgram(b->program,1,&device,options,NULL,NULL);
    if(err){char log[8192]={0};clGetProgramBuildInfo(b->program,device,CL_PROGRAM_BUILD_LOG,sizeof(log)-1,log,NULL);fprintf(stderr,"OpenCL build: %s\n",log);goto fail;}
    for(unsigned i=0;i<7u+(unsigned)b->compact;++i){b->kernel[i]=clCreateKernel(b->program,names[i],&err);if(!b->kernel[i] || err)goto fail;}
    b->kernel[27]=clCreateKernel(b->program,"luma_equal",&err);if(!b->kernel[27]||err)goto fail;
    const char *mmr96_env=getenv("DV_MMR_96");b->mmr96=mmr96_env && strcmp(mmr96_env,"1")==0;
    if(b->mmr96){b->kernel[8]=clCreateKernel(b->program,"mmr96",&err);if(!b->kernel[8] || err)goto fail;}
    const char *mmr64_env=getenv("DV_MMR_64");b->mmr64=mmr64_env && strcmp(mmr64_env,"1")==0;
    if(b->mmr64){b->kernel[9]=clCreateKernel(b->program,"mmr64",&err);if(!b->kernel[9] || err)goto fail;}
    fprintf(stderr,"{\"mmr64_requested\":%d}\n",b->mmr64);
    const char *mmr_lut_env=getenv("DV_MMR_UNIVARIATE_LUT");b->mmr_lut=b->mmr64&&mmr_lut_env&&strcmp(mmr_lut_env,"1")==0;
    if(b->mmr_lut){b->kernel[10]=clCreateKernel(b->program,"mmr64_lut",&err);if(!b->kernel[10]||err)goto fail;}
    fprintf(stderr,"{\"mmr_univariate_lut_requested\":%d}\n",b->mmr_lut);
    const char *luma_pair=getenv("DV_LUMA_PAIR"),*mmr_pair=getenv("DV_MMR_PAIR"),*verify_recon=getenv("DV_VERIFY_RECON");
    b->luma_pair=luma_pair&&!strcmp(luma_pair,"1");b->mmr_pair=mmr_pair&&!strcmp(mmr_pair,"1");b->verify_recon=verify_recon&&!strcmp(verify_recon,"1");
    if(b->luma_pair){b->kernel[12]=clCreateKernel(b->program,"luma_pair",&err);if(!b->kernel[12]||err)goto fail;}
    if(b->mmr_pair){b->kernel[13]=clCreateKernel(b->program,"mmr64_lut_pair",&err);if(!b->kernel[13]||err)goto fail;}
    fprintf(stderr,"{\"luma_pair\":%d,\"mmr_pair\":%d,\"verify_recon\":%d}\n",b->luma_pair,b->mmr_pair,b->verify_recon);
    const char *mmr_uniform_env=getenv("DV_MMR_UNIFORM");b->mmr_uniform=mmr_uniform_env&&!strcmp(mmr_uniform_env,"1");
    if(b->mmr_uniform){b->kernel[14]=clCreateKernel(b->program,"mmr64_lut_uniform",&err);if(!b->kernel[14]||err)goto fail;}
    fprintf(stderr,"{\"mmr_uniform\":%d}\n",b->mmr_uniform);
    const char *mmr_mulhi_env=getenv("DV_MMR_MULHI");b->mmr_mulhi=mmr_mulhi_env&&!strcmp(mmr_mulhi_env,"1");
    if(b->mmr_mulhi){b->kernel[15]=clCreateKernel(b->program,"mmr64_lut_mulhi",&err);if(!b->kernel[15]||err)goto fail;}
    fprintf(stderr,"{\"mmr_mulhi\":%d}\n",b->mmr_mulhi);
    const char *mmr_dual_env=getenv("DV_MMR_DUAL");b->mmr_dual=mmr_dual_env&&!strcmp(mmr_dual_env,"1");
    if(b->mmr_dual){b->kernel[16]=clCreateKernel(b->program,"mmr64_lut_dual",&err);if(!b->kernel[16]||err)goto fail;}
    const char *horizontal=getenv("DV_MMR_HORIZONTAL_FUSED"),*verify_horizontal=getenv("DV_VERIFY_MMR_HORIZONTAL");
    b->horizontal_fused=horizontal&&!strcmp(horizontal,"1");b->verify_horizontal=verify_horizontal&&!strcmp(verify_horizontal,"1");
    if(b->horizontal_fused){b->kernel[26]=clCreateKernel(b->program,"mmr64_lut_dual_horizontal",&err);if(!b->kernel[26]||err)goto fail;}
    fprintf(stderr,"{\"mmr_dual\":%d}\n",b->mmr_dual);
    const char *filter_env=getenv("DV_FILTER_FUSED"),*verify_filter=getenv("DV_VERIFY_FILTER");
    b->filter_fused=filter_env&&!strcmp(filter_env,"1");b->verify_filter=verify_filter&&!strcmp(verify_filter,"1");
    if(b->verify_filter&&!b->filter_fused)goto fail;
    if(b->filter_fused){b->kernel[11]=clCreateKernel(b->program,"phase_vertical",&err);if(!b->kernel[11]||err)goto fail;}
    fprintf(stderr,"{\"filter_fused\":%d,\"verify_filter\":%d}\n",b->filter_fused,b->verify_filter);
    const char *unpack_env=getenv("DV_UNPACK_BLOCK"),*verify_unpack=getenv("DV_VERIFY_UNPACK");
    b->unpack_block=unpack_env&&!strcmp(unpack_env,"1");
    b->unpack=clCreateKernel(b->program,b->unpack_block?"unpack_p010_block":"unpack_p010",&err);if(!b->unpack||err)goto fail;
    if(verify_unpack&&!strcmp(verify_unpack,"1")){if(!b->unpack_block)goto fail;b->unpack_reference=clCreateKernel(b->program,"unpack_p010",&err);if(!b->unpack_reference||err)goto fail;}
    fprintf(stderr,"{\"unpack_block\":%d,\"verify_unpack\":%d}\n",b->unpack_block,b->unpack_reference!=NULL);
    const char *direct_input=getenv("DV_DIRECT_P010");b->direct_p010=!force_planar&&direct_input&&!strcmp(direct_input,"1");
    if(b->direct_p010){
        /* First qualified route uses fused EL chroma and paired BL luma.
         * Reject incompatible experiments rather than silently unpacking. */
        if(!b->filter_fused||!b->luma_pair||b->verify_recon||b->verify_filter||b->unpack_reference)goto fail;
        char input_options[768];snprintf(input_options,sizeof(input_options),"%s -D DV_INPUT_P010",options);
        /* All direct input readers are in the reconstruction-only prefix.
         * Do not compile unrelated MMR/colour/packing kernels a second time. */
        const char *input_source=source,*input_end=strstr(source,"uint2 multiply(");if(!input_end)goto fail;
        size_t input_length=(size_t)(input_end-source);
        b->input_program=clCreateProgramWithSource(b->context,1,&input_source,&input_length,&err);
        if(!b->input_program||err)goto fail;
        err=clBuildProgram(b->input_program,1,&device,input_options,NULL,NULL);
        if(err){char log[8192]={0};clGetProgramBuildInfo(b->input_program,device,CL_PROGRAM_BUILD_LOG,sizeof(log)-1,log,NULL);fprintf(stderr,"Direct P010 build: %s\n",log);goto fail;}
        const char *input_names[8]={"phase","vertical","guide","phase_vertical","luma_pair","phase_uv","phase_vertical_uv","vertical_pair"};
        for(unsigned k=0;k<8;++k){b->kernel[17+k]=clCreateKernel(b->input_program,input_names[k],&err);if(!b->kernel[17+k]||err)goto fail;}
        b->kernel[28]=clCreateKernel(b->input_program,"luma_equal",&err);if(!b->kernel[28]||err)goto fail;
        if(b->base_fused){b->kernel[25]=clCreateKernel(b->input_program,"phase_guide",&err);if(!b->kernel[25]||err)goto fail;}
    }
    fprintf(stderr,"{\"direct_p010_requested\":%d}\n",b->direct_p010);
    b->patch=clCreateKernel(b->program,"patch_colour",&err);if(!b->patch||err)goto fail;
    b->pack=clCreateKernel(b->program,"pack_r16ui",&err);if(!b->pack||err)goto fail;
    const char *image_env=getenv("DV_COLOUR_DIRECT_IMAGES");
    if(!source_only&&image_env&&strcmp(image_env,"1")==0){
        char image_options[768];snprintf(image_options,sizeof(image_options),"%s -D DV_COLOUR_OUTPUT_IMAGES",options);
#ifdef DV_COLOUR_LUT
        b->image_program=clCreateProgramWithSource(b->context,2,ptrs,lengths,&err);
#else
        b->image_program=clCreateProgramWithSource(b->context,1,&ptr,&len,&err);
#endif
        if(!b->image_program||err)goto fail;
        if(clBuildProgram(b->image_program,1,&device,image_options,NULL,NULL)){
            char log[8192]={0};clGetProgramBuildInfo(b->image_program,device,CL_PROGRAM_BUILD_LOG,sizeof(log)-1,log,NULL);fprintf(stderr,"Image OpenCL build: %s\n",log);goto fail;}
        b->image_kernel[0]=clCreateKernel(b->image_program,"colour",&err);if(!b->image_kernel[0]||err)goto fail;
        if(b->compact){b->image_kernel[1]=clCreateKernel(b->image_program,"colour_refine",&err);if(!b->image_kernel[1]||err)goto fail;}
        b->image_kernel[2]=clCreateKernel(b->image_program,"patch_colour",&err);if(!b->image_kernel[2]||err)goto fail;
    }
    const char *piecewise=getenv("DV_SINGLE_PIECEWISE");
    if(piecewise&&!strcmp(piecewise,"1")){
        char code[32768];f=shader_file(directory,"single_reshape.cl");if(!f)goto fail;
        size_t length=fread(code,1,sizeof(code),f);invalid=ferror(f)||!feof(f);fclose(f);if(invalid||!length)goto fail;
        const char *text=code;b->piecewise_program=clCreateProgramWithSource(b->context,1,&text,&length,&err);if(!b->piecewise_program||err)goto fail;
        const char *full=getenv("DV_SINGLE_FULL_HEIGHT");int full_enabled=full&&!strcmp(full,"1");
        char options[256];snprintf(options,sizeof(options),"-cl-std=CL1.2 -D DV_RESHAPE_NARROW -D DV_RESHAPE_P010 -D DV_RESHAPE_LUMA_TABLE %s",
            full_enabled?"-D DV_RESHAPE_422_EXPERIMENT -D DV_RESHAPE_LAZY_FEATURES":"");
        if(clBuildProgram(b->piecewise_program,1,&device,options,NULL,NULL)){
            char log[8192]={0};clGetProgramBuildInfo(b->piecewise_program,device,CL_PROGRAM_BUILD_LOG,sizeof(log)-1,log,NULL);fprintf(stderr,"Piecewise OpenCL build: %s\n",log);goto fail;}
        b->piecewise_kernel=clCreateKernel(b->piecewise_program,"single_reshape_p010_center_left",&err);if(!b->piecewise_kernel||err)goto fail;
        if(full_enabled){b->piecewise_full_kernel=clCreateKernel(b->piecewise_program,"single_reshape_p010_422_candidate",&err);if(!b->piecewise_full_kernel||err)goto fail;}
        const char *order2=getenv("DV_SINGLE_ORDER2_CHROMA");
        if(full_enabled&&order2&&!strcmp(order2,"1")){
            b->piecewise_order2_program=clCreateProgramWithSource(b->context,1,&text,&length,&err);
            if(!b->piecewise_order2_program||err)goto fail;
            char specialized[320];snprintf(specialized,sizeof(specialized),"%s -D DV_RESHAPE_ORDER2_CHROMA",options);
            if(clBuildProgram(b->piecewise_order2_program,1,&device,specialized,NULL,NULL)){
                char log[8192]={0};clGetProgramBuildInfo(b->piecewise_order2_program,device,CL_PROGRAM_BUILD_LOG,sizeof(log)-1,log,NULL);
                fprintf(stderr,"Order2 OpenCL build: %s\n",log);goto fail;}
            b->piecewise_order2_kernel=clCreateKernel(b->piecewise_order2_program,"single_reshape_p010_422_candidate",&err);
            if(!b->piecewise_order2_kernel||err)goto fail;
        }
        const char *nv12=getenv("DV_SINGLE_NV12");
        if(nv12&&!strcmp(nv12,"1")){
            b->nv12_program=clCreateProgramWithSource(b->context,1,&text,&length,&err);if(!b->nv12_program||err)goto fail;
            if(clBuildProgram(b->nv12_program,1,&device,"-cl-std=CL1.2 -D DV_RESHAPE_NARROW -D DV_RESHAPE_NV12 -D DV_RESHAPE_LUMA_TABLE -D DV_RESHAPE_LAZY_FEATURES",NULL,NULL)){
                char log[8192]={0};clGetProgramBuildInfo(b->nv12_program,device,CL_PROGRAM_BUILD_LOG,sizeof(log)-1,log,NULL);fprintf(stderr,"NV12 OpenCL build: %s\n",log);goto fail;}
            b->nv12_kernel=clCreateKernel(b->nv12_program,"single_reshape_nv12_center_left",&err);if(!b->nv12_kernel||err)goto fail;
        }
    }
    char name[256]={0},driver[256]={0};clGetDeviceInfo(device,CL_DEVICE_NAME,sizeof(name),name,NULL);clGetDeviceInfo(device,CL_DRIVER_VERSION,sizeof(driver),driver,NULL);
    fprintf(stderr,"Intel OpenCL GPU: %s; driver %s\n",name,driver);return b;
fail:dv_opencl_destroy(b);return NULL;
}
void *dv_opencl_create(void){return create_backend(NULL,NULL,0,NULL,0);}
void *dv_opencl_create_shared(cl_context c,cl_device_id d){if(!c||!d)return NULL;return create_backend(c,d,0,NULL,0);}
void *dv_opencl_create_shared_at(cl_context c,cl_device_id d,const char *directory)
{if(!c||!d||!directory||!directory[0])return NULL;return create_backend(c,d,0,directory,0);}
void *dv_opencl_create_source_at(cl_context c,cl_device_id d,const char *directory)
{if(!c||!d||!directory||!directory[0])return NULL;return create_backend(c,d,0,directory,1);}
void *dv_opencl_create_shared_planar_reference(cl_context c,cl_device_id d){if(!c||!d)return NULL;return create_backend(c,d,1,NULL,0);}
static int reserve(backend *b,size_t n,int diag,int omit_raw)
{
    size_t sizes[NB]={0};
    for(unsigned c=0;c<3;++c){size_t count=c?n/4:n;
        if(!omit_raw){sizes[B+c]=count*2;sizes[E+c]=count/2;}sizes[RT+c]=4096;sizes[O+c]=count*2;
        sizes[M+c]=diag?count*2:2;sizes[R+c]=sizes[S+c]=diag?count*4:4;
    }
    sizes[BP]=sizes[BP+1]=n/2;sizes[EP]=omit_raw?0:n/8;sizes[V]=n;if(omit_raw)sizes[VP]=n/4;
    sizes[ES]=sizes[ES+1]=n/2;sizes[G]=n/2;sizes[MT]=2048;
    sizes[P]=sizes[P+1]=384;sizes[ERR]=4;
    for(unsigned i=0;i<NB;++i)if(b->bytes[i]<sizes[i]){
        cl_int error;cl_mem m=clCreateBuffer(b->context,CL_MEM_READ_WRITE,sizes[i],NULL,&error);
        if(!m || error)return -1;if(b->buffer[i])clReleaseMemObject(b->buffer[i]);
        b->allocated+=sizes[i]-b->bytes[i];b->bytes[i]=sizes[i];b->buffer[i]=m;
        if(i==MT)b->single_gpu_valid=0;
    }
    return 0;
}
static int write_buffer(backend *b,unsigned slot,const void *p,size_t n)
{if(slot==MT)b->single_gpu_valid=0;b->uploads+=n;return clEnqueueWriteBuffer(b->queue,b->buffer[slot],CL_FALSE,0,n,p,0,NULL,NULL)!=CL_SUCCESS;}
static int read_reconstruction_error(backend *b)
{
    if(b->reconstruction_done){clReleaseEvent(b->reconstruction_done);b->reconstruction_done=NULL;}
    b->downloads+=sizeof(b->reconstruction_error);
    return clEnqueueReadBuffer(b->queue,b->buffer[ERR],CL_FALSE,0,sizeof(b->reconstruction_error),
        &b->reconstruction_error,0,NULL,&b->reconstruction_done)!=CL_SUCCESS;
}
/* Transient CPU repair/LUT destinations must not outlive a failed drain.
 * Blocking transfer completion owns this host-memory lifetime independently
 * of later queue errors. Reconstruction and colour uniforms use slot storage. */
static int write_buffer_sync(backend *b,unsigned slot,const void *p,size_t n)
{if(slot==MT)b->single_gpu_valid=0;b->uploads+=n;return clEnqueueWriteBuffer(b->queue,b->buffer[slot],CL_TRUE,0,n,p,0,NULL,NULL)!=CL_SUCCESS;}
static int read_buffer_sync(backend *b,unsigned slot,void *p,size_t n)
{b->downloads+=n;return clEnqueueReadBuffer(b->queue,b->buffer[slot],CL_TRUE,0,n,p,0,NULL,NULL)!=CL_SUCCESS;}
static int memarg(backend *b,unsigned k,unsigned a,unsigned slot)
{cl_mem m=b->image_active&&slot>=CO&&slot<CO+3?b->owned_gl[slot-CO]:b->buffer[slot];return clSetKernelArg(b->kernel[k],a,sizeof(m),&m)!=CL_SUCCESS;}
static int uintarg(backend *b,unsigned k,unsigned a,unsigned value)
{cl_uint v=value;return clSetKernelArg(b->kernel[k],a,sizeof(v),&v)!=CL_SUCCESS;}
static int inputarg(backend *b,unsigned k,unsigned slot,unsigned surface)
{return b->direct_active?clSetKernelArg(b->kernel[k],0,sizeof(cl_mem),&b->owned_va[surface])!=CL_SUCCESS:memarg(b,k,0,slot);}
static int inputextra(backend *b,unsigned k,unsigned start,unsigned channel)
{return b->direct_active&&(uintarg(b,k,start,channel)||memarg(b,k,start+1,ERR));}
static unsigned event_category(unsigned k)
{if(k==17||k==22||k==25)return 0;if(k==18||k==20||k==11||k==23||k==24)return 1;if(k==19)return 3;if(k==21||k==12||k==27||k==28)return 4;return k>=8?5:k;}
static int dispatch(backend *b,unsigned k,size_t count)
{
    if(b->event_count>=32)return -1;
    cl_int r=clEnqueueNDRangeKernel(b->queue,b->kernel[k],1,NULL,&count,NULL,0,NULL,&b->events[b->event_count]);
    if(r)return -1;b->event_kind[b->event_count]=event_category(k);++b->event_count;return 0;
}
static int dispatch_spatial(backend *b,unsigned k,size_t width,size_t height)
{
    if(!b->recon2d)return dispatch(b,k,width*height);
    if(b->event_count>=32)return -1;
    /* Exact logical extent: no padded global grid is legal with these kernels. */
    const size_t dimensions[2]={width,height};
    cl_int r=clEnqueueNDRangeKernel(b->queue,b->kernel[k],2,NULL,dimensions,NULL,0,NULL,&b->events[b->event_count]);
    if(r)return -1;b->event_kind[b->event_count]=event_category(k);++b->event_count;return 0;
}
static int prepare_base(backend *b,unsigned phase,unsigned w,unsigned h,int fused)
{
#define BASE_DO(call) do{if(call)return -1;}while(0)
 if(b->direct_active&&fused){
  BASE_DO(inputarg(b,25,B,0));BASE_DO(clSetKernelArg(b->kernel[25],1,sizeof(cl_mem),&b->owned_va[1]));
  BASE_DO(memarg(b,25,2,BP));BASE_DO(memarg(b,25,3,BP+1));BASE_DO(memarg(b,25,4,G));
  BASE_DO(uintarg(b,25,5,w));BASE_DO(uintarg(b,25,6,h));BASE_DO(uintarg(b,25,7,phase));BASE_DO(memarg(b,25,8,ERR));
  BASE_DO(dispatch_spatial(b,25,w/2,h/2));return 0;
 }
 unsigned phase_kernel=b->direct_active?17u:0u,guide_kernel=b->direct_active?19u:3u;
 if(b->direct_active){BASE_DO(inputarg(b,22,B+1,1));BASE_DO(memarg(b,22,1,BP));BASE_DO(memarg(b,22,2,BP+1));BASE_DO(uintarg(b,22,3,w/2));BASE_DO(uintarg(b,22,4,h/2));BASE_DO(uintarg(b,22,5,phase));BASE_DO(memarg(b,22,6,ERR));BASE_DO(dispatch_spatial(b,22,w/2,h/2));}
 else for(unsigned c=1;c<3;++c){BASE_DO(inputarg(b,phase_kernel,B+c,1));BASE_DO(memarg(b,phase_kernel,1,BP+c-1));BASE_DO(uintarg(b,phase_kernel,2,w/2));BASE_DO(uintarg(b,phase_kernel,3,h/2));BASE_DO(uintarg(b,phase_kernel,4,phase));BASE_DO(inputextra(b,phase_kernel,5,c-1));BASE_DO(dispatch_spatial(b,phase_kernel,w/2,h/2));}
 BASE_DO(inputarg(b,guide_kernel,B,0));BASE_DO(memarg(b,guide_kernel,1,G));BASE_DO(uintarg(b,guide_kernel,2,w));BASE_DO(uintarg(b,guide_kernel,3,h));BASE_DO(inputextra(b,guide_kernel,4,0));BASE_DO(dispatch_spatial(b,guide_kernel,w/2,h/2));return 0;
#undef BASE_DO
}
static int verify_base_preparation(backend *b,const dv_frame_settings *s,unsigned w,unsigned h)
{
 const unsigned slots[3]={BP,BP+1,G};size_t bytes=(size_t)w*h/2,differences=0;
 unsigned char *saved[3]={0},*reference=malloc(bytes);int failed=1;if(!reference)goto done;
 for(unsigned c=0;c<3;++c){saved[c]=malloc(bytes);if(!saved[c]||read_buffer_sync(b,slots[c],saved[c],bytes))goto done;}
 if(prepare_base(b,s->chroma_phase,w,h,0))goto done;
 for(unsigned c=0;c<3;++c){if(read_buffer_sync(b,slots[c],reference,bytes))goto done;
  for(size_t i=0;i<bytes;++i)differences+=saved[c][i]!=reference[i];}
 failed=differences!=0;++b->base_checks;
 fprintf(stderr,"{\"base_prep_verified_pts\":%lld,\"bytes\":%zu,\"differences\":%zu}\n",(long long)s->identity.pts,bytes*3,differences);
done:for(unsigned c=0;c<3;++c)free(saved[c]);free(reference);return failed;
}
/* Checked once per prepared channel, not in the coefficient hot loop.
 * |coefficient|<=2^63, every clipped feature<2^20, <=21 products plus
 * |constant|<=2^83 give |accumulator|<22*2^83<2^88. Signed96 is exact.
 * Unknown/wider prepared layouts retain the original full128 kernel. */
static int mmr96_eligible(const uint32_t p[96])
{
    if(p[0]<1 || p[0]>3 || p[1]<15 || p[1]>32)return 0;
    for(unsigned k=0;k<3;k++)if(p[2+2*k]>p[3+2*k] || p[3+2*k]>1023)return 0;
    uint32_t sign=p[10]&0x80000u?UINT32_MAX:0;
    if((p[8]&0xfffffu) || p[11]!=sign || (p[10]&0xfff00000u)!=(sign&0xfff00000u))return 0;
    for(unsigned k=0;k<7*p[0];k++){const uint32_t *v=p+12+4*k;sign=v[1]&0x80000000u?UINT32_MAX:0;
        if(v[2]!=sign || v[3]!=sign)return 0;}
    return 1;
}
static void release_events(backend *b)
{for(unsigned i=0;i<b->event_count;++i)clReleaseEvent(b->events[i]);b->event_count=0;}
static int reserve_one(backend *b,unsigned slot,size_t bytes);
static int prepare_mmr_lut(backend *b,unsigned channel,const uint32_t p[96])
{
 dv_mmr64_lut_cache *cache=&b->mmr_lut_cache[channel];
 if(dv_mmr64_lut_matches(cache,p)){
  fprintf(stderr,"{\"mmr_lut_channel\":%u,\"cache_hit\":1,\"prepare_seconds\":0,\"upload_seconds\":0,\"upload_bytes\":0}\n",channel+1);return 0;
 }
 cache->valid=0;double start=host_seconds();
 if(!dv_mmr64_build_lut(p,cache->table))return -1;
 double prepared=host_seconds();
 if(reserve_one(b,UL+channel,sizeof(cache->table))||write_buffer_sync(b,UL+channel,cache->table,sizeof(cache->table)))return -1;
 double uploaded=host_seconds();dv_mmr64_lut_commit(cache,p);
 fprintf(stderr,"{\"mmr_lut_channel\":%u,\"cache_hit\":0,\"prepare_seconds\":%.9f,\"upload_seconds\":%.9f,\"upload_bytes\":%zu}\n",channel+1,prepared-start,uploaded-prepared,sizeof(cache->table));
 return 0;
}
static int verify_recon_output(backend *b,const dv_frame_settings *s,unsigned c,size_t count,unsigned route,void **saved)
{
 size_t bytes=count*2;
 if(!route){*saved=malloc(bytes);return !*saved||read_buffer_sync(b,O+c,*saved,bytes);}
 unsigned char *reference=malloc(bytes);if(!reference)return -1;
 int error=read_buffer_sync(b,O+c,reference,bytes);size_t different=0;
 if(!error)for(size_t i=0;i<bytes;++i)different+=reference[i]!=((unsigned char *)*saved)[i];
 fprintf(stderr,"{\"recon_verified_pts\":%lld,\"channel\":%u,\"compared_bytes\":%zu,\"byte_differences\":%zu,\"read_error\":%d}\n",(long long)s->identity.pts,c,bytes,different,error);
 free(reference);free(*saved);*saved=NULL;return error||different;
}
static int reserve_one(backend *b,unsigned slot,size_t bytes);
static dv_status reconstruct_impl(void *opaque,const dv_frame_settings *s,unsigned w,unsigned h,
    const uint16_t *const bl[3],const uint16_t *const el[3],uint16_t *const out[3],const dv_observer *observer,int resident_only,int enqueue_only)
{
    backend *b=opaque;if(b){b->resident_ready=0;b->output_ready=0;}if(!b || !s || (!b->shared_prepared&&(!bl||!el)) || !out || !w || !h || w>4096 || h>4096 || w%4 || h%4 || s->chroma_phase>1)return DV_INVALID;
    if(s->composer.el_spatial_resampling_flag>1)return DV_UNSUPPORTED;
    int equal=!s->composer.el_spatial_resampling_flag;unsigned area_scale=equal?1u:4u;
    const char *trace=getenv("DV_SOURCE_SUBMIT_TRACE");int tracing=trace&&!strcmp(trace,"1")&&b->submit_traces<6;
    double started=tracing?host_seconds():0,prepared=0,uploaded=0;
    for(unsigned c=0;c<3;++c)if((!b->shared_prepared&&(!bl[c]||!el[c])) || ((!resident_only || (observer && observer->stage)) && !out[c]))return DV_INVALID;
    size_t n=(size_t)w*h;int diag=observer && observer->stage;
    dv_status status=DV_BACKEND;uint16_t *mt=b->mapping,unused;int32_t (*rt)[1024]=b->residual_table,mapped,residual;uint32_t (*params)[96]=b->mmr_params;
    void *host_m=NULL,*host_r=NULL,*host_s=NULL;
    release_events(b);b->uploads=b->downloads=0;b->kernel_seconds=0;
    for(unsigned i=0;i<1024;++i){
        if(dv_reference_luma(&s->composer,(uint16_t)i,(uint16_t)s->composer.component[0].nlq_offset,&mapped,&residual,&unused))return DV_UNSUPPORTED;
        mt[i]=(uint16_t)mapped;
        if(dv_reference_luma(&s->composer,0,(uint16_t)i,&mapped,&rt[0][i],&unused))return DV_UNSUPPORTED;
    }
    for(unsigned c=1;c<3;++c){
        if(!s->composer.component[c].mapping){
            if(dv_prepare_polynomial_component(&s->composer,c,b->chroma_mapping[c-1],rt[c]))return DV_UNSUPPORTED;
            memset(params[c-1],0,sizeof(params[c-1]));
        }else if(dv_mmr_prepare(&s->composer,c,params[c-1],rt[c])!=DV_OK)return DV_UNSUPPORTED;
    }
    if(tracing)prepared=host_seconds();
    for(unsigned c=0;!b->shared_prepared && c<3;++c){size_t count=c?n/4:n;
        uint32_t bits=0;
        for(size_t i=0;i<count;++i)bits|=bl[c][i];
        for(size_t i=0;i<count/area_scale;++i)bits|=el[c][i];
        if(bits & ~UINT32_C(1023))return DV_SAMPLE_RANGE;}
    if(reserve(b,n,diag,b->direct_active))return DV_NOMEM;
    if(equal&&!b->direct_active)for(unsigned c=0;c<3;++c)
        if(reserve_one(b,E+c,(c?n/4:n)*2))return DV_NOMEM;
#define DO(X) do{if(X)goto done;}while(0)
    /* Before reconstruction dispatch: a cache miss includes its real CPU
     * preparation and blocking upload in frame service time. On shared input,
     * this upload may also wait for already-queued VA unpack work. */
    if(b->mmr_lut)for(unsigned c=0;c<2;++c)
        if(s->composer.component[c+1].mapping&&dv_mmr64_eligible(params[c],NULL))DO(prepare_mmr_lut(b,c,params[c]));
    if(!b->shared_prepared){b->reconstruction_error=0;DO(write_buffer(b,ERR,&b->reconstruction_error,4));}DO(write_buffer(b,MT,mt,sizeof(b->mapping)));
    for(unsigned c=0;c<3;++c){size_t count=c?n/4:n;
        if(!b->shared_prepared){if(enqueue_only){DO(write_buffer(b,B+c,bl[c],count*2));DO(write_buffer(b,E+c,el[c],count*2/area_scale));}
            else {DO(write_buffer_sync(b,B+c,bl[c],count*2));DO(write_buffer_sync(b,E+c,el[c],count*2/area_scale));}}DO(write_buffer(b,RT+c,rt[c],sizeof(rt[c])));
        if(c){
            if(!s->composer.component[c].mapping){
                DO(reserve_one(b,CM+c-1,sizeof(b->chroma_mapping[c-1])));
                DO(write_buffer(b,CM+c-1,b->chroma_mapping[c-1],sizeof(b->chroma_mapping[c-1])));
            }else DO(write_buffer(b,P+c-1,params[c-1],sizeof(params[c-1])));
        }
    }
    if(tracing)uploaded=host_seconds();
    /* Both phase-adjusted BL chroma planes are needed by either MMR. */
    DO(prepare_base(b,s->chroma_phase,w,h,b->base_fused));
    if(b->direct_active&&b->base_fused&&b->verify_base&&b->base_checks<3)DO(verify_base_preparation(b,s,w,h));
    int dual=s->composer.component[1].mapping&&s->composer.component[2].mapping&&b->mmr_dual&&b->mmr_lut&&!diag&&dv_mmr64_eligible(params[0],NULL)&&dv_mmr64_eligible(params[1],NULL)&&!memcmp(params[0]+2,params[1]+2,6*sizeof(uint32_t));
    int horizontal_fused=!equal&&dual&&b->horizontal_fused&&b->direct_active&&b->filter_fused&&!b->verify_filter;
    int verify_horizontal=horizontal_fused&&b->verify_horizontal&&b->horizontal_checks<3;
    if(b->mmr_dual)fprintf(stderr,"{\"mmr_dual_pts\":%lld,\"eligible\":%d,\"diagnostics\":%d}\n",(long long)s->identity.pts,dual,diag);
    for(unsigned c=0;c<3;++c){unsigned cw=c?w/2:w,ch=c?h/2:h,input=E+c;
        if(equal){
            if(!c){unsigned k=b->direct_active?28u:27u;
                DO(inputarg(b,k,B,0));
                if(b->direct_active)DO(clSetKernelArg(b->kernel[k],1,sizeof(cl_mem),&b->owned_va[2]));
                else DO(memarg(b,k,1,E));
                unsigned slots[7]={MT,RT,M,R,S,O,ERR};for(unsigned a=0;a<7;++a)DO(memarg(b,k,a+2,slots[a]));
                DO(uintarg(b,k,9,w));DO(uintarg(b,k,10,h));DO(uintarg(b,k,11,(unsigned)diag));DO(dispatch_spatial(b,k,w,h));
                continue;
            }
            unsigned k=b->direct_active?17u:0u;
            DO(inputarg(b,k,E+c,3));DO(memarg(b,k,1,ES+c-1));
            DO(uintarg(b,k,2,cw));DO(uintarg(b,k,3,ch));DO(uintarg(b,k,4,s->chroma_phase));
            DO(inputextra(b,k,5,c-1));DO(dispatch_spatial(b,k,cw,ch));
        }else{
        if(!c||!b->filter_fused||b->verify_filter){
            if(c){DO(memarg(b,0,0,E+c));DO(memarg(b,0,1,EP));DO(uintarg(b,0,2,cw/2));DO(uintarg(b,0,3,ch/2));DO(uintarg(b,0,4,s->chroma_phase));DO(dispatch_spatial(b,0,cw/2,ch/2));input=EP;}
            unsigned vk=b->direct_active?24u:1u;DO(inputarg(b,vk,input,2));DO(memarg(b,vk,1,V));DO(uintarg(b,vk,2,cw/2));DO(uintarg(b,vk,3,ch/2));DO(uintarg(b,vk,4,c!=0));DO(inputextra(b,vk,5,0));DO(dispatch_spatial(b,vk,cw/2,b->direct_active?ch/2:ch));
        }
        if(c&&b->filter_fused){size_t bytes=(size_t)cw*ch;
            if(b->verify_filter){host_m=malloc(bytes);host_r=malloc(bytes);if(!host_m||!host_r)goto done;DO(read_buffer_sync(b,V,host_m,bytes));}
            if(b->direct_active){if(c==1){DO(inputarg(b,23,E+c,3));DO(memarg(b,23,1,V));DO(memarg(b,23,2,VP));DO(uintarg(b,23,3,cw/2));DO(uintarg(b,23,4,ch/2));DO(uintarg(b,23,5,s->chroma_phase));DO(memarg(b,23,6,ERR));DO(dispatch_spatial(b,23,cw/2,ch/2));}}
            else {unsigned fk=11;DO(inputarg(b,fk,E+c,3));DO(memarg(b,fk,1,V));DO(uintarg(b,fk,2,cw/2));DO(uintarg(b,fk,3,ch/2));DO(uintarg(b,fk,4,s->chroma_phase));DO(inputextra(b,fk,5,c-1));DO(dispatch_spatial(b,fk,cw/2,ch/2));}
            if(b->verify_filter){DO(read_buffer_sync(b,V,host_r,bytes));size_t different=0;for(size_t j=0;j<bytes;++j)different+=((unsigned char *)host_m)[j]!=((unsigned char *)host_r)[j];
                fprintf(stderr,"{\"filter_verified_pts\":%lld,\"channel\":%u,\"compared_bytes\":%zu,\"byte_differences\":%zu}\n",(long long)s->identity.pts,c,bytes,different);
                free(host_m);free(host_r);host_m=host_r=NULL;if(different)goto done;}
        }
        }
        if(!c){unsigned verify=b->verify_recon&&b->luma_pair;
            for(unsigned route=0;route<(verify?2u:1u);++route){unsigned kernel=b->direct_active?21u:b->luma_pair&&!route?12u:4u;
                unsigned slots[9]={B,V,MT,RT,M,R,S,O,ERR};DO(inputarg(b,kernel,B,0));for(unsigned a=1;a<9;++a)DO(memarg(b,kernel,a,slots[a]));
                DO(uintarg(b,kernel,9,w));DO(uintarg(b,kernel,10,h));DO(uintarg(b,kernel,11,(unsigned)diag));DO(dispatch_spatial(b,kernel,kernel==12||kernel==21?w/2:w,h));
                if(verify)DO(verify_recon_output(b,s,c,n,route,&host_m));
            }}
        else {
            if(!equal&&(!horizontal_fused||b->verify_recon||verify_horizontal)){
                DO(memarg(b,2,0,b->direct_active&&c==2?VP:V));DO(memarg(b,2,1,ES+c-1));DO(memarg(b,2,2,ERR));DO(uintarg(b,2,3,cw));DO(uintarg(b,2,4,ch));DO(dispatch_spatial(b,2,cw,ch));}
            if(!s->composer.component[c].mapping){
                /* Scaling/phase preparation above is unchanged. This generic
                 * table-add kernel preserves Q16 mapping plus FEL residual,
                 * then performs the same single final 12-bit quantization. */
                unsigned slots[9]={BP+c-1,ES+c-1,CM+c-1,RT+c,M+c,R+c,S+c,O+c,ERR};
                for(unsigned a=0;a<9;++a)DO(memarg(b,27,a,slots[a]));
                DO(uintarg(b,27,9,cw));DO(uintarg(b,27,10,ch));DO(uintarg(b,27,11,(unsigned)diag));
                DO(dispatch_spatial(b,27,cw,ch));
                b->event_kind[b->event_count-1]=5; /* Chroma work, despite shared kernel. */
                continue;
            }
            if(dual){if(c==1)continue;
                unsigned k=horizontal_fused?26u:16u;
                unsigned slots[12]={G,BP,BP+1,horizontal_fused?V:ES,horizontal_fused?VP:ES+1,P,P+1,RT+1,RT+2,O+1,O+2,ERR};for(unsigned a=0;a<12;++a)DO(memarg(b,k,a,slots[a]));
                DO(uintarg(b,k,12,(unsigned)(n/4)));DO(memarg(b,k,13,UL));DO(memarg(b,k,14,UL+1));if(horizontal_fused)DO(uintarg(b,k,15,cw));DO(dispatch(b,k,n/4));
                if(b->verify_recon||verify_horizontal)for(unsigned channel=1;channel<3;++channel){DO(verify_recon_output(b,s,channel,n/4,0,&host_m));
                    unsigned reference[11]={G,BP,BP+1,ES+channel-1,P+channel-1,RT+channel,M+channel,R+channel,S+channel,O+channel,ERR};
                    for(unsigned a=0;a<11;++a)DO(memarg(b,10,a,reference[a]));DO(uintarg(b,10,11,(unsigned)(n/4)));DO(uintarg(b,10,12,0));DO(memarg(b,10,13,UL+channel-1));DO(dispatch(b,10,n/4));
                    DO(verify_recon_output(b,s,channel,n/4,1,&host_m));}
                if(verify_horizontal)++b->horizontal_checks;
                continue;
            }
            unsigned kernel=b->mmr96 && mmr96_eligible(params[c-1])?8u:5u;
            uint64_t bound=0;int eligible64=b->mmr64 && dv_mmr64_eligible(params[c-1],&bound);
            if(eligible64)kernel=b->mmr_lut?10u:9u;
            unsigned candidate=kernel==10u?(b->mmr_mulhi?15u:b->mmr_uniform?14u:b->mmr_pair?13u:kernel):kernel;
            unsigned verify=candidate!=kernel&&b->verify_recon,original=kernel;
            for(unsigned route=0;route<(verify?2u:1u);++route){kernel=!route?candidate:original;
                unsigned slots[11]={G,BP,BP+1,ES+c-1,P+c-1,RT+c,M+c,R+c,S+c,O+c,ERR};for(unsigned a=0;a<11;++a)DO(memarg(b,kernel,a,slots[a]));
                DO(uintarg(b,kernel,11,(unsigned)(n/4)));DO(uintarg(b,kernel,12,(unsigned)diag));
                if(kernel==10u||kernel>=13u)DO(memarg(b,kernel,13,UL+c-1));DO(dispatch(b,kernel,kernel==13?(n/4+1)/2:n/4));
                if(verify)DO(verify_recon_output(b,s,c,n/4,route,&host_m));
            }
            if(b->mmr96)fprintf(stderr,"{\"mmr96_channel\":%u,\"mmr96_eligible\":%u}\n",c,kernel==8u);
            if(b->mmr64)fprintf(stderr,"{\"mmr64_channel\":%u,\"mmr64_eligible\":%d,\"mmr64_absolute_bound\":%llu,\"mmr_fallback_kernel\":%u}\n",c,eligible64,(unsigned long long)bound,kernel);
        }
    }
    DO(read_reconstruction_error(b));
    if(enqueue_only){DO(clFlush(b->queue)!=CL_SUCCESS);status=DV_OK;
        if(tracing){++b->submit_traces;fprintf(stderr,"{\"recon_submit_pts\":%lld,\"table_prepare_ms\":%.6f,\"reserve_upload_ms\":%.6f,\"dispatch_ms\":%.6f}\n",(long long)s->identity.pts,(prepared-started)*1e3,(uploaded-prepared)*1e3,(host_seconds()-uploaded)*1e3);}
        goto done;}
    DO(clFinish(b->queue)!=CL_SUCCESS);if(b->reconstruction_error){status=DV_SAMPLE_RANGE;goto done;}
    if(!resident_only){for(unsigned c=0;c<3;++c)DO(read_buffer_sync(b,O+c,out[c],(c?n/4:n)*2));DO(clFinish(b->queue)!=CL_SUCCESS);}
    if(diag){host_m=malloc(n*2);host_r=malloc(n*4);host_s=malloc(n*4);if(!host_m || !host_r || !host_s){status=DV_NOMEM;goto done;}
        for(unsigned c=0;c<3;++c){size_t count=c?n/4:n;DO(read_buffer_sync(b,M+c,host_m,count*2));DO(read_buffer_sync(b,R+c,host_r,count*4));DO(read_buffer_sync(b,S+c,host_s,count*4));DO(clFinish(b->queue)!=CL_SUCCESS);
            const void *data[4]={host_m,host_r,host_s,out[c]};unsigned bytes[4]={2,4,4,2};
            for(unsigned k=0;k<4;++k)if(observer->stage(observer->opaque,(dv_stage)k,c,data[k],count,bytes[k])){status=DV_OBSERVER;goto done;}
        }
    }
    double stage_seconds[6]={0};
    for(unsigned i=0;i<b->event_count;++i){cl_ulong start,end;
        if(clGetEventProfilingInfo(b->events[i],CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL) || clGetEventProfilingInfo(b->events[i],CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL))goto done;
        double elapsed=(double)(end-start)/1e9;b->kernel_seconds+=elapsed;stage_seconds[b->event_kind[i]]+=elapsed;
    }
    fprintf(stderr,"{\"reconstruction_gpu_stages_seconds\":{\"phase\":%.9f,\"vertical\":%.9f,\"horizontal\":%.9f,\"guide\":%.9f,\"luma\":%.9f,\"chroma_mmr\":%.9f}}\n",stage_seconds[0],stage_seconds[1],stage_seconds[2],stage_seconds[3],stage_seconds[4],stage_seconds[5]);
    fprintf(stderr,"{\"opencl_kernel_seconds\":%.9f,\"upload_bytes\":%zu,\"download_bytes\":%zu,\"retained_gpu_buffer_bytes\":%zu,\"diagnostics\":%d}\n",b->kernel_seconds,b->uploads,b->downloads,b->allocated,diag);
    status=DV_OK;
done:
    /* Even on enqueue failure, host upload pointers must remain alive until
     * the queue drains. Diagnostic side effects are not transactional. */
    if((!enqueue_only || status!=DV_OK) && clFinish(b->queue)!=CL_SUCCESS)status=DV_BACKEND;
    if(status==DV_OK){b->piecewise_active=0;b->resident_ready=1;b->width=w;b->height=h;b->chroma_phase=s->chroma_phase;b->identity=s->identity;b->config=s->composer;}
    const char *profile=getenv("DV_SOURCE_PROFILE");
    if(enqueue_only&&status==DV_OK&&profile&&!strcmp(profile,"1")){
        double ms[6]={0};if(clFinish(b->queue))status=DV_BACKEND;
        for(unsigned j=0;j<b->event_count&&status==DV_OK;++j){cl_ulong start=0,end=0;unsigned kind=b->event_kind[j];
            if(kind>=6||clGetEventProfilingInfo(b->events[j],CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL)||
               clGetEventProfilingInfo(b->events[j],CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL)||end<start){status=DV_BACKEND;break;}
            ms[kind]+=(double)(end-start)/1e6;
        }
        if(status==DV_OK)fprintf(stderr,"{\"source_profile_pts\":%lld,\"phase_ms\":%.6f,\"vertical_ms\":%.6f,\"horizontal_ms\":%.6f,\"guide_ms\":%.6f,\"luma_ms\":%.6f,\"chroma_mmr_ms\":%.6f}\n",(long long)s->identity.pts,ms[0],ms[1],ms[2],ms[3],ms[4],ms[5]);
    }
    b->profile_pending=enqueue_only&&status==DV_OK&&b->profile_async&&b->profile_frames<120;
    b->shared_prepared=0;if(!b->profile_pending)release_events(b);
    free(host_m);free(host_r);free(host_s);return status;
#undef DO
}
dv_status dv_opencl_reconstruct(void *opaque,const dv_frame_settings *s,unsigned w,unsigned h,
    const uint16_t *const bl[3],const uint16_t *const el[3],uint16_t *const out[3],const dv_observer *observer)
{return reconstruct_impl(opaque,s,w,h,bl,el,out,observer,0,0);}
dv_status dv_opencl_reconstruct_resident(void *opaque,const dv_frame_settings *s,unsigned w,unsigned h,
    const uint16_t *const bl[3],const uint16_t *const el[3],uint16_t *const out[3],const dv_observer *observer)
{backend *b=opaque;if(!b || !b->sparse_repair || !out)return DV_INVALID;
 return reconstruct_impl(opaque,s,w,h,bl,el,out,observer,!out[0] && !out[1] && !out[2],0);}
dv_status dv_opencl_reconstruct_enqueue(void *p,const dv_frame_settings *s,unsigned w,unsigned h,const uint16_t *const bl[3],const uint16_t *const el[3])
{uint16_t *out[3]={NULL,NULL,NULL};return reconstruct_impl(p,s,w,h,bl,el,out,NULL,1,1);}
static int reserve_one(backend *b,unsigned slot,size_t bytes)
{
    if(b->bytes[slot]>=bytes)return 0;cl_int error;
    cl_mem m=clCreateBuffer(b->context,CL_MEM_READ_WRITE,bytes,NULL,&error);if(!m || error)return -1;
    if(b->buffer[slot])clReleaseMemObject(b->buffer[slot]);b->buffer[slot]=m;
    if(slot==MT)b->single_gpu_valid=0;
    b->allocated+=bytes-b->bytes[slot];b->bytes[slot]=bytes;return 0;
}
static dv_status source_identity(backend *b,const dv_frame_settings *s,unsigned w,unsigned h)
{
 if(!b||!s)return DV_INVALID;
 if(b->piecewise_active||!b->resident_ready||b->width!=w||b->height!=h||b->chroma_phase!=s->chroma_phase||
    b->identity.frame_id!=s->identity.frame_id||b->identity.pts!=s->identity.pts||
    memcmp(&b->config,&s->composer,sizeof(b->config)))return DV_IDENTITY;
 if(s->source.compressed||s->source.depth!=12||(s->source.space!=0&&s->source.space!=2)||s->source.eotf!=65535||
    s->active[0]>=s->active[2]||s->active[1]>=s->active[3]||s->active[2]>w||s->active[3]>h)return DV_UNSUPPORTED;
 return b->reconstruction_done?DV_OK:DV_BACKEND;
}
static dv_status source_poll_ready(backend *b,int *ready)
{
 cl_int state=CL_QUEUED;
 if(clGetEventInfo(b->reconstruction_done,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(state),&state,NULL)||state<0)return DV_BACKEND;
 if(state!=CL_COMPLETE){*ready=0;return DV_OK;}
 if(b->reconstruction_error)return DV_SAMPLE_RANGE;
 *ready=1;return DV_OK;
}
dv_status dv_opencl_source_poll(void *opaque,const dv_frame_settings *s,unsigned w,unsigned h,int *ready)
{
 backend *b=opaque;if(!ready)return DV_INVALID;
 dv_status status=source_identity(b,s,w,h);if(status!=DV_OK)return status;
 return source_poll_ready(b,ready);
}
static dv_status source_output_ready(backend *b,cl_mem out[3])
{
 /* This event follows all reconstruction writers AND the small host range
  * check. Later queue work need not complete before the planes are readable.
  * The caller still owns this slot and drains VA leases after its consumers. */
 if(clWaitForEvents(1,&b->reconstruction_done))return DV_BACKEND;
 int ready=0;dv_status status=source_poll_ready(b,&ready);
 if(status!=DV_OK)return status;if(!ready)return DV_BACKEND;
 unsigned i=0;for(;i<3;++i)if(clRetainMemObject(b->buffer[O+i]))break;
 if(i!=3){for(unsigned j=0;j<i;++j)clReleaseMemObject(b->buffer[O+j]);return DV_BACKEND;}
 for(i=0;i<3;++i)out[i]=b->buffer[O+i];return DV_OK;
}
dv_status dv_opencl_source_output(void *opaque,const dv_frame_settings *s,unsigned w,unsigned h,cl_mem out[3])
{
 backend *b=opaque;if(!out||out[0]||out[1]||out[2])return DV_INVALID;
 dv_status status=source_identity(b,s,w,h);if(status!=DV_OK)return status;
 return source_output_ready(b,out);
}
static dv_status piecewise_frame_identity(backend *b,const dv_single_frame_settings *s,unsigned w,unsigned h)
{
 if(!b||!s)return DV_INVALID;
 if(!b->piecewise_active||!b->resident_ready||b->width!=w||b->height!=h||
    memcmp(&b->piecewise_frame,s,sizeof(*s)))return DV_IDENTITY;
 return b->reconstruction_done?DV_OK:DV_BACKEND;
}
static dv_status piecewise_identity(backend *b,const dv_single_frame_settings *s,unsigned w,unsigned h)
{
 if(!b||!s)return DV_INVALID;
 if(b->piecewise_layout||b->piecewise_sampling)return DV_IDENTITY;
 return piecewise_frame_identity(b,s,w,h);
}
dv_status dv_opencl_piecewise_poll(void *opaque,const dv_single_frame_settings *s,unsigned w,unsigned h,int *ready)
{
 backend *b=opaque;if(!ready)return DV_INVALID;
 dv_status status=piecewise_identity(b,s,w,h);if(status!=DV_OK)return status;
 return source_poll_ready(b,ready);
}
dv_status dv_opencl_piecewise_output(void *opaque,const dv_single_frame_settings *s,unsigned w,unsigned h,cl_mem out[3])
{
 backend *b=opaque;if(!out||out[0]||out[1]||out[2])return DV_INVALID;
 dv_status status=piecewise_identity(b,s,w,h);if(status!=DV_OK)return status;
 return source_output_ready(b,out);
}
static dv_status piecewise_surface_identity(backend *b,const dv_piecewise_surface_settings *s,unsigned w,unsigned h)
{
 if(!b||!s)return DV_INVALID;
 if(s->layout>1||s->sampling>1||(!s->layout&&s->sampling))return DV_UNSUPPORTED;
 if(b->piecewise_layout!=s->layout||b->piecewise_sampling!=s->sampling)return DV_IDENTITY;
 return piecewise_frame_identity(b,&s->frame,w,h);
}
dv_status dv_opencl_piecewise_surface_poll(void *opaque,const dv_piecewise_surface_settings *s,unsigned w,unsigned h,int *ready)
{
 backend *b=opaque;if(!ready)return DV_INVALID;
 dv_status status=piecewise_surface_identity(b,s,w,h);if(status!=DV_OK)return status;
 return source_poll_ready(b,ready);
}
dv_status dv_opencl_piecewise_surface_output(void *opaque,const dv_piecewise_surface_settings *s,unsigned w,unsigned h,cl_mem out[3])
{
 backend *b=opaque;if(!out||out[0]||out[1]||out[2])return DV_INVALID;
 dv_status status=piecewise_surface_identity(b,s,w,h);if(status!=DV_OK)return status;
 return source_output_ready(b,out);
}
#ifdef DV_COLOUR_LUT
static int prepare_luts(backend *b)
{
    if(b->luts_ready)return 0;
    size_t nd=dv_decode_lut_count()*2*sizeof(float),ne=dv_encode_lut_count()*2*sizeof(float);
    float *d=malloc(nd),*e=malloc(ne),*fd=NULL,*fe=NULL;int failed=1;
    if(d && e && !dv_colour_make_luts(d,e) && !reserve_one(b,DL,nd) && !reserve_one(b,ELUT,ne))
        failed=write_buffer_sync(b,DL,d,nd) || write_buffer_sync(b,ELUT,e,ne);
    if(!failed && b->fast_tables){size_t dc=dv_decode_lut_count(),ec=dv_encode_lut_count();
        fd=malloc(dc*sizeof(float));fe=malloc(ec*sizeof(float));
        if(!fd || !fe)failed=1;
        else {for(size_t i=0;i<dc/4;++i)memcpy(fd+4*i,d+8*i,4*sizeof(float));
            fe[0]=e[0];for(size_t i=0;i<(ec-1)/4;++i)memcpy(fe+1+4*i,e+2+8*i,4*sizeof(float));
            failed=reserve_one(b,FDL,dc*sizeof(float)) || reserve_one(b,FELUT,ec*sizeof(float)) || write_buffer_sync(b,FDL,fd,dc*sizeof(float)) || write_buffer_sync(b,FELUT,fe,ec*sizeof(float));}}
    if(clFinish(b->queue)!=CL_SUCCESS)failed=1;
    free(fd);free(fe);free(d);free(e);b->luts_ready=!failed;return failed;
}
static double cpu_expand(const uint16_t *p,unsigned w,unsigned h,unsigned x,unsigned y)
{
    unsigned ix=x/2,iy=y/2,next=ix+1<w?ix+1:w-1;
    unsigned top=(y&1)?iy:(iy?iy-1:0),bottom=(y&1)?(iy+1<h?iy+1:h-1):iy;
    unsigned a=(y&1)?3:1,z=4-a,left=a*p[top*w+ix]+z*p[bottom*w+ix];
    return (double)((x&1)?left+a*p[top*w+next]+z*p[bottom*w+next]:left*2)/8.0;
}
static void split_double(float *p,unsigned i,double v)
{p[2*i]=(float)v;p[2*i+1]=(float)(v-(double)p[2*i]);}
typedef struct {uint64_t key;uint16_t code[3];} correction_entry;
typedef struct {uint32_t index,codes[3],reason,padding[3];} repair_record;
_Static_assert(sizeof(repair_record)==32,"OpenCL uint8 repair record size");
static size_t correction_slot(uint64_t key)
{key^=key>>30;key*=UINT64_C(0xbf58476d1ce4e5b9);key^=key>>27;return (size_t)(key&16383);}
#endif
static dv_status colour_impl(void *opaque,const dv_frame_settings *s,unsigned w,unsigned h,
    const uint16_t *const rec[3],uint16_t *const out[3],uint64_t *excursions,int resident_output)
{
    (void)rec;backend *b=opaque;if(!b || !s || (!out&&!resident_output) || !excursions)return DV_INVALID;b->output_ready=0;b->image_output=0;
    if(b->piecewise_active || !b->resident_ready || b->width!=w || b->height!=h || b->chroma_phase!=s->chroma_phase || b->identity.frame_id!=s->identity.frame_id || b->identity.pts!=s->identity.pts || memcmp(&b->config,&s->composer,sizeof(b->config)))return DV_IDENTITY;
    for(unsigned c=0;!resident_output && c<3;++c)if(!out[c])return DV_INVALID;
    if(resident_output&&!b->sparse_repair)return DV_UNSUPPORTED;
    dv_source_colour source;dv_target_colour target;
    if(dv_source_colour_init(&s->source,s->pq_policy,&source) || dv_target_colour_init(s->target_ycc,s->target_lms,s->target_offset,&target))return DV_UNSUPPORTED;
#ifdef DV_COLOUR_LUT
    float *params=b->colour_params;uint32_t *src=b->colour_source;const size_t param_bytes=78*sizeof(*params),src_bytes=13*sizeof(*src);
    int has_rec=rec && rec[0] && rec[1] && rec[2];
    if(!has_rec && (!b->sparse_repair || (rec && (rec[0] || rec[1] || rec[2]))))return DV_INVALID;
    if(prepare_luts(b))return DV_NOMEM;
    for(unsigned k=0;k<9;++k){split_double(params,k,source.lms[k]);split_double(params,9+k,target.inverse_lms[k]);split_double(params,18+k,target.inverse_ycc[k]);src[k]=(uint32_t)s->source.ycc[k];}
    if(b->combine_matrix)for(unsigned row=0;row<3;++row)for(unsigned column=0;column<3;++column){
        long double v=0;for(unsigned k=0;k<3;++k)v+=(long double)target.inverse_lms[row*3+k]*(long double)source.lms[k*3+column];
        split_double(params,row*3+column,(double)v);
    }
    for(unsigned k=0;k<3;++k){split_double(params,27+k,target.offset[k]);src[9+k]=s->source.offset[k];}
    src[12]=(uint32_t)(b->combine_matrix!=0);
    for(unsigned row=0;row<3;row++)for(unsigned column=0;column<3;column++){
        long double v=0;for(unsigned k=0;k<3;k++)v+=(long double)target.inverse_lms[row*3+k]*source.lms[k*3+column];
        split_double(params,30+row*3+column,(double)v);
        if(b->fused && (!isfinite((double)v) || fabsl(v)>32))src[12]=1;
    }
    for(unsigned k=0;k<9;++k)if(fabs(target.inverse_lms[k])>32 || fabs(target.inverse_ycc[k])>32)src[12]=1;
    for(unsigned k=0;k<3;++k)if(fabs(target.offset[k])>2)src[12]=1;
#else
    float *params=b->colour_params;uint32_t *src=b->colour_source;const size_t param_bytes=30*sizeof(*params),src_bytes=12*sizeof(*src);
    for(unsigned k=0;k<9;++k){params[k]=(float)source.lms[k];params[9+k]=(float)target.inverse_lms[k];params[18+k]=(float)target.inverse_ycc[k];src[k]=(uint32_t)s->source.ycc[k];}
    for(unsigned k=0;k<3;++k){params[27+k]=(float)target.offset[k];src[9+k]=s->source.offset[k];}
#endif
    size_t n=(size_t)w*h,local=32,max_group=0;
    const char *group_env=getenv("DV_COLOUR_GROUP");
    if(group_env){if(strcmp(group_env,"32")==0)local=32;else if(strcmp(group_env,"64")==0)local=64;
        else if(strcmp(group_env,"128")==0)local=128;else if(strcmp(group_env,"256")==0)local=256;else return DV_INVALID;}
    size_t groups=(n+local-1)/local,global=groups*local;
    size_t capacity=n<1048576?n:1048576;uint32_t candidates=0;int dense=0;
    if(b->compact){const char *cap_env=getenv("DV_COLOUR_COMPACT_CAPACITY");if(cap_env){char *end;unsigned long cap=strtoul(cap_env,&end,10);
            if(!*cap_env || *end || !cap || cap>1048576)return DV_INVALID;capacity=cap<n?(size_t)cap:n;}
        if(reserve_one(b,PCR,capacity*16) || reserve_one(b,PCT,4))return DV_NOMEM;
    }
    if(clGetKernelWorkGroupInfo(b->kernel[6],b->device,CL_KERNEL_WORK_GROUP_SIZE,sizeof(max_group),&max_group,NULL) || max_group<local)return DV_BACKEND;
    if(reserve_one(b,CP,param_bytes) || reserve_one(b,CS,src_bytes) || reserve_one(b,CT,groups*8))return DV_NOMEM;
    for(unsigned c=0;!b->image_active&&c<3;++c)if(reserve_one(b,CO+c,n*2))return DV_NOMEM;
    uint32_t *stats=calloc(groups,8);if(!stats)return DV_NOMEM;dv_status status=DV_BACKEND;
#ifdef DV_COLOUR_LUT
    unsigned char *mask=malloc(n);size_t corrected=0,evaluated=0,reasons[5]={0},gpu_precise=0;
    size_t repair_capacity=n<16384?n:16384;uint32_t repair_count=0;repair_record *repairs=NULL;int sparse_fallback=0;
    uint32_t (*patches)[4]=NULL;
    if(b->sparse_repair){const char *cap_env=getenv("DV_COLOUR_REPAIR_CAPACITY");if(cap_env){char *end;unsigned long cap=strtoul(cap_env,&end,10);
            if(!*cap_env || *end || !cap || cap>65536){free(mask);free(stats);return DV_INVALID;}repair_capacity=cap<n?(size_t)cap:n;}
        repairs=malloc(repair_capacity*sizeof(*repairs));
        if(!repairs || reserve_one(b,PRC,repair_capacity*sizeof(*repairs)) || reserve_one(b,PRN,4)){free(repairs);free(mask);free(stats);return DV_NOMEM;}}
    correction_entry *cache=calloc(16384,sizeof(*cache));
    uint16_t *owned_rec[3]={0};const uint16_t *host_rec[3]={has_rec?rec[0]:NULL,has_rec?rec[1]:NULL,has_rec?rec[2]:NULL};
    if(!mask || !cache || reserve_one(b,MASK,n)){free(repairs);free(mask);free(cache);free(stats);return DV_NOMEM;}
#endif
    b->uploads=b->downloads=0;release_events(b);
#define TRY(X) do{if(X)goto colour_done;}while(0)
    TRY(write_buffer(b,CP,params,param_bytes));TRY(write_buffer(b,CS,src,src_bytes));
    b->colour_candidates=0;if(b->compact)TRY(write_buffer(b,PCT,&b->colour_candidates,4));
#ifdef DV_COLOUR_LUT
    b->colour_repair_count=0;if(b->sparse_repair)TRY(write_buffer(b,PRN,&b->colour_repair_count,4));
#endif
    unsigned slots[9]={O,O+1,O+2,CP,CS,CO,CO+1,CO+2,CT};for(unsigned a=0;a<9;++a)TRY(memarg(b,6,a,slots[a]));
    TRY(uintarg(b,6,9,w));TRY(uintarg(b,6,10,h));TRY(uintarg(b,6,11,s->pq_policy));
    TRY(clSetKernelArg(b->kernel[6],12,sizeof(s->active),s->active)!=CL_SUCCESS);
    fprintf(stderr,"{\"colour_work_group\":%zu}\n",local);
#ifdef DV_COLOUR_LUT
    TRY(memarg(b,6,13,DL));TRY(memarg(b,6,14,ELUT));TRY(memarg(b,6,15,MASK));
    if(b->compact){TRY(memarg(b,6,16,PCR));TRY(memarg(b,6,17,PCT));TRY(uintarg(b,6,18,(unsigned)capacity));}
    if(b->fast_tables){TRY(memarg(b,6,19,FDL));TRY(memarg(b,6,20,FELUT));}
#endif
    TRY(clEnqueueNDRangeKernel(b->queue,b->kernel[6],1,NULL,&global,&local,0,NULL,&b->events[0])!=CL_SUCCESS);b->event_count=1;
    if(b->compact){size_t work=n;
        if(b->bounded_refine){work=capacity>32768?capacity:32768;if(work>n)work=n;}
        unsigned refine_slots[9]={O,O+1,O+2,CP,CS,CO,CO+1,CO+2,PCR};for(unsigned a=0;a<9;++a)TRY(memarg(b,7,a,refine_slots[a]));
        TRY(memarg(b,7,9,PCT));TRY(uintarg(b,7,10,w));TRY(uintarg(b,7,11,h));
        TRY(clSetKernelArg(b->kernel[7],12,sizeof(s->active),s->active)!=CL_SUCCESS);
        TRY(memarg(b,7,13,DL));TRY(memarg(b,7,14,ELUT));TRY(memarg(b,7,15,MASK));
        TRY(uintarg(b,7,16,(unsigned)capacity));TRY(uintarg(b,7,17,s->pq_policy));
#ifdef DV_COLOUR_LUT
        if(b->sparse_repair){TRY(memarg(b,7,18,PRC));TRY(memarg(b,7,19,PRN));TRY(uintarg(b,7,20,(unsigned)repair_capacity));}
#endif
        size_t refine_global=(work+local-1)/local*local;
        fprintf(stderr,"{\"colour_refine_launch_items\":%zu,\"colour_frame_pixels\":%zu}\n",refine_global,n);
        TRY(clEnqueueNDRangeKernel(b->queue,b->kernel[7],1,NULL,&refine_global,&local,0,NULL,&b->events[1])!=CL_SUCCESS);b->event_count=2;
    }
    TRY(read_buffer_sync(b,CT,stats,groups*8));if(b->compact)TRY(read_buffer_sync(b,PCT,&candidates,4));
    for(unsigned c=0;!resident_output && c<3;++c)TRY(read_buffer_sync(b,CO+c,out[c],n*2));
#ifdef DV_COLOUR_LUT
    if(b->sparse_repair){TRY(read_buffer_sync(b,PRN,&repair_count,4));size_t returned=repair_count<repair_capacity?repair_count:repair_capacity;if(returned)TRY(read_buffer_sync(b,PRC,repairs,returned*sizeof(*repairs)));}
    else TRY(read_buffer_sync(b,MASK,mask,n));
#endif
    TRY(clFinish(b->queue)!=CL_SUCCESS);
    if(b->reconstruction_error){status=DV_SAMPLE_RANGE;goto colour_done;}
    uint64_t count=0;for(size_t i=0;i<groups;++i){if(stats[2*i+1]){status=DV_COLOUR_DOMAIN;goto colour_done;}count+=stats[2*i];}
    dense=b->compact && candidates>capacity;
#ifdef DV_COLOUR_LUT
    sparse_fallback=b->sparse_repair && repair_count>repair_capacity;
    if(sparse_fallback){TRY(read_buffer_sync(b,MASK,mask,n));
        if(!has_rec)for(unsigned c=0;c<3;++c){size_t count=c?n/4:n;owned_rec[c]=malloc(count*2);if(!owned_rec[c]){status=DV_NOMEM;goto colour_done;}host_rec[c]=owned_rec[c];TRY(read_buffer_sync(b,O+c,owned_rec[c],count*2));}
        TRY(clFinish(b->queue)!=CL_SUCCESS);}
    int sparse=b->sparse_repair && !sparse_fallback;
    if(resident_output){size_t pc=sparse?repair_count:n;if(pc){patches=malloc(pc*sizeof(*patches));if(!patches){status=DV_NOMEM;goto colour_done;}}}
    if(sparse)gpu_precise=dense?n:candidates;
    for(size_t job=0;job<(sparse?repair_count:n);++job){size_t i=sparse?repairs[job].index:job;
        unsigned reason=sparse?repairs[job].reason:(mask[i]&31u);
        if(sparse){TRY(i>=n || !reason || reason>31u);for(unsigned c=0;c<3;++c)TRY(repairs[job].codes[c]>32760u);}
        else gpu_precise+=(size_t)((mask[i]>>5)&1u);
        if(!reason)continue;
        for(unsigned bit=0;bit<5;++bit)reasons[bit]+=(size_t)((reason>>bit)&1u);
        unsigned x=(unsigned)(i%w),y=(unsigned)(i/w);
        double codes[3];if(sparse)for(unsigned c=0;c<3;++c)codes[c]=(double)repairs[job].codes[c]/8.0;
        else {codes[0]=host_rec[0][i];codes[1]=cpu_expand(host_rec[1],w/2,h/2,x,y);codes[2]=cpu_expand(host_rec[2],w/2,h/2,x,y);}
        uint64_t key=1+((uint64_t)(codes[0]*8.0)|((uint64_t)(codes[1]*8.0)<<15)|((uint64_t)(codes[2]*8.0)<<30));
        correction_entry *entry=&cache[correction_slot(key)];
        if(entry->key!=key){dv_source_result sr;dv_target_result tr;
            if(dv_source_colour_sample(&source,codes,&sr) || dv_target_colour_sample(&target,s->pq_policy,sr.linear_lms,&tr)){status=DV_COLOUR_DOMAIN;goto colour_done;}
            entry->key=key;memcpy(entry->code,tr.code,sizeof(entry->code));++evaluated;
        }
        uint16_t fixed[3];memcpy(fixed,entry->code,sizeof(fixed));
        if(x<s->active[0] || y<s->active[1] || x>=s->active[2] || y>=s->active[3]){fixed[0]=0;fixed[1]=fixed[2]=2048;}
        if(resident_output){patches[corrected][0]=(uint32_t)i;for(unsigned c=0;c<3;++c)patches[corrected][c+1]=fixed[c];}
        else for(unsigned c=0;c<3;++c)out[c][i]=fixed[c];
        ++corrected;
    }
    if(resident_output&&corrected){TRY(reserve_one(b,PATCH,corrected*sizeof(*patches)));TRY(write_buffer_sync(b,PATCH,patches,corrected*sizeof(*patches)));
        for(unsigned c=0;c<3;++c){cl_mem m=b->image_active?b->owned_gl[c]:b->buffer[CO+c];TRY(clSetKernelArg(b->patch,c,sizeof(m),&m)!=CL_SUCCESS);}
        TRY(clSetKernelArg(b->patch,3,sizeof(cl_mem),&b->buffer[PATCH])!=CL_SUCCESS);cl_uint np=(cl_uint)corrected;
        TRY(clSetKernelArg(b->patch,4,sizeof(np),&np)!=CL_SUCCESS);size_t work=corrected;
        if(b->image_active){cl_uint ww=w,hh=h;TRY(clSetKernelArg(b->patch,5,sizeof(ww),&ww)!=CL_SUCCESS);TRY(clSetKernelArg(b->patch,6,sizeof(hh),&hh)!=CL_SUCCESS);}
        TRY(clEnqueueNDRangeKernel(b->queue,b->patch,1,NULL,&work,NULL,0,NULL,NULL)!=CL_SUCCESS);
        /* The blocking upload owns transient host-patch lifetime independently
         * of any later queue failure. Finish also confirms GPU patch execution. */
        TRY(clFinish(b->queue)!=CL_SUCCESS);}
    fprintf(stderr,"{\"lut_cpu_corrected_pixels\":%zu,\"lut_cpu_evaluated_pixels\":%zu,\"lut_precision_experiment\":true}\n",corrected,evaluated);
    fprintf(stderr,"{\"selective_gpu_precise_pixels\":%zu}\n",gpu_precise);
    fprintf(stderr,"{\"lut_correction_reasons\":[%zu,%zu,%zu,%zu,%zu]}\n",reasons[0],reasons[1],reasons[2],reasons[3],reasons[4]);
    if(b->sparse_repair)fprintf(stderr,"{\"sparse_repair_count\":%u,\"sparse_repair_capacity\":%zu,\"sparse_repair_fallback\":%d}\n",repair_count,repair_capacity,sparse_fallback);
#endif
    double seconds[2]={0};for(unsigned k=0;k<b->event_count;++k){cl_ulong begin,end;
        TRY(clGetEventProfilingInfo(b->events[k],CL_PROFILING_COMMAND_START,sizeof(begin),&begin,NULL)!=CL_SUCCESS);
        TRY(clGetEventProfilingInfo(b->events[k],CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL)!=CL_SUCCESS);seconds[k]=(double)(end-begin)/1e9;}
    if(b->compact)fprintf(stderr,"{\"compact_candidates\":%u,\"compact_capacity\":%zu,\"compact_dense_fallback\":%d,\"colour_screen_seconds\":%.9f,\"colour_refine_seconds\":%.9f}\n",candidates,capacity,dense,seconds[0],seconds[1]);
    fprintf(stderr,"{\"experimental_gpu_colour_seconds\":%.9f,\"colour_upload_bytes\":%zu,\"colour_download_bytes\":%zu,\"retained_gpu_buffer_bytes\":%zu,\"resident_gpu_colour\":true}\n",seconds[0]+seconds[1],b->uploads,b->downloads,b->allocated);
    if(b->output_event){clReleaseEvent(b->output_event);b->output_event=NULL;}
    TRY(clEnqueueMarkerWithWaitList(b->queue,0,NULL,&b->output_event)!=CL_SUCCESS);TRY(clFlush(b->queue)!=CL_SUCCESS);
    *excursions=count;status=DV_OK;b->output_ready=1;b->image_output=b->image_active;
colour_done:if(clFinish(b->queue)!=CL_SUCCESS)status=DV_BACKEND;b->output_ready=status==DV_OK;release_events(b);free(stats);
#ifdef DV_COLOUR_LUT
    for(unsigned c=0;c<3;++c)free(owned_rec[c]);free(repairs);free(mask);free(cache);free(patches);
#endif
    return status;
#undef TRY
}
dv_status dv_opencl_colour(void *p,const dv_frame_settings *s,unsigned w,unsigned h,const uint16_t *const rec[3],uint16_t *const out[3],uint64_t *excursions)
{return colour_impl(p,s,w,h,rec,out,excursions,0);}
dv_status dv_opencl_colour_resident(void *p,const dv_frame_settings *s,unsigned w,unsigned h,uint64_t *excursions)
{return colour_impl(p,s,w,h,NULL,NULL,excursions,1);}
dv_status dv_opencl_colour_gl_resident(void *p,const dv_frame_settings *s,unsigned w,unsigned h,const cl_mem images[3],uint64_t *excursions,cl_event *event)
{backend *b=p;if(!b)return DV_INVALID;b->output_ready=0;b->image_output=0;if(!s||!images||!excursions||!event)return DV_INVALID;*event=NULL;
 if(!b->image_kernel[0]||!b->image_kernel[2])return DV_UNSUPPORTED;
 if(b->gl_count||b->image_active||!b->resident_ready||b->width!=w||b->height!=h)return DV_BACKEND;
 for(unsigned i=0;i<3;++i){cl_context c=NULL;cl_image_format f={0};size_t iw=0,ih=0;
  if(!images[i]||clGetMemObjectInfo(images[i],CL_MEM_CONTEXT,sizeof(c),&c,NULL)||c!=b->context||clGetImageInfo(images[i],CL_IMAGE_FORMAT,sizeof(f),&f,NULL)||clGetImageInfo(images[i],CL_IMAGE_WIDTH,sizeof(iw),&iw,NULL)||clGetImageInfo(images[i],CL_IMAGE_HEIGHT,sizeof(ih),&ih,NULL))return DV_INVALID;
  if(iw!=w||ih!=h||f.image_channel_order!=CL_R||f.image_channel_data_type!=CL_UNSIGNED_INT16)return DV_UNSUPPORTED;}
 for(unsigned i=0;i<3;++i){if(clRetainMemObject(images[i])){for(unsigned j=0;j<i;++j){clReleaseMemObject(b->owned_gl[j]);b->owned_gl[j]=NULL;}return DV_BACKEND;}b->owned_gl[i]=images[i];}
 double acquire_start=host_seconds();cl_int e=clEnqueueAcquireGLObjects(b->queue,3,images,0,NULL,NULL);double acquire_cpu=host_seconds()-acquire_start;
 if(e){for(unsigned i=0;i<3;++i){clReleaseMemObject(b->owned_gl[i]);b->owned_gl[i]=NULL;}return DV_BACKEND;}b->gl_count=3;
 /* A backend is serially owned by one pool worker. Swap only colour/patch
  * handles; every numerical statement remains in the same compiled source. */
 cl_kernel screen=b->kernel[6],refine=b->kernel[7],patch=b->patch;
 b->kernel[6]=b->image_kernel[0];b->kernel[7]=b->image_kernel[1];b->patch=b->image_kernel[2];b->image_active=1;
 dv_status status=colour_impl(b,s,w,h,NULL,NULL,excursions,1);
 b->image_active=0;b->kernel[6]=screen;b->kernel[7]=refine;b->patch=patch;
 if(status!=DV_OK)return status; /* caller drain recovers/quarantines ownership */
 cl_event done=NULL;double release_start=host_seconds();e=clEnqueueReleaseGLObjects(b->queue,3,images,0,NULL,&done);double release_cpu=host_seconds()-release_start;b->gl_release=done;
 fprintf(stderr,"{\"direct_colour_images\":true,\"gl_acquire_enqueue_cpu_seconds\":%.9f,\"gl_release_enqueue_cpu_seconds\":%.9f,\"co_buffer_bytes_avoided\":%zu}\n",acquire_cpu,release_cpu,(size_t)w*h*6);
 if(e||clFlush(b->queue)||clRetainEvent(done)){b->output_ready=0;return DV_BACKEND;}*event=done;return DV_OK;}
dv_status dv_opencl_output(void *p,cl_mem out[3],cl_event *event)
{backend *b=p;if(!b||!out||!event)return DV_INVALID;if(!b->output_ready||!b->output_event||b->image_output)return DV_BACKEND;
 unsigned count=0;for(;count<3;++count)if(clRetainMemObject(b->buffer[CO+count])!=CL_SUCCESS)break;
 if(count!=3){for(unsigned i=0;i<count;++i)clReleaseMemObject(b->buffer[CO+i]);return DV_BACKEND;}
 if(clRetainEvent(b->output_event)!=CL_SUCCESS){for(unsigned i=0;i<3;++i)clReleaseMemObject(b->buffer[CO+i]);return DV_BACKEND;}
 for(unsigned i=0;i<3;++i)out[i]=b->buffer[CO+i];*event=b->output_event;return DV_OK;}
dv_status dv_opencl_reconstruct_p010(void *p,const dv_frame_settings *s,unsigned w,unsigned h,const cl_mem bl[2],const cl_mem el[2])
{backend *b=p;if(b){b->resident_ready=0;b->output_ready=0;b->image_output=0;}if(!b||!s||!bl||!el||!w||!h||w>4096||h>4096||w%4||h%4)return DV_INVALID;
 if(s->composer.el_spatial_resampling_flag>1)return DV_UNSUPPORTED;
 unsigned scale=s->composer.el_spatial_resampling_flag?2u:1u;
 cl_platform_id platform=NULL;if(clGetDeviceInfo(b->device,CL_DEVICE_PLATFORM,sizeof(platform),&platform,NULL)!=CL_SUCCESS)return DV_BACKEND;
 clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn acquire=(clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform,"clEnqueueAcquireVA_APIMediaSurfacesINTEL");
 clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn release=(clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform,"clEnqueueReleaseVA_APIMediaSurfacesINTEL");
 if(!acquire||!release)return DV_UNSUPPORTED;cl_mem images[4]={bl[0],bl[1],el[0],el[1]};
 for(unsigned i=0;i<4;++i){cl_context c=NULL;cl_image_format f={0};size_t iw=0,ih=0;unsigned ew=i<2?w:w/scale,eh=i<2?h:h/scale;if(i%2){ew/=2;eh/=2;}
  if(!images[i]||clGetMemObjectInfo(images[i],CL_MEM_CONTEXT,sizeof(c),&c,NULL)||c!=b->context||clGetImageInfo(images[i],CL_IMAGE_FORMAT,sizeof(f),&f,NULL)||clGetImageInfo(images[i],CL_IMAGE_WIDTH,sizeof(iw),&iw,NULL)||clGetImageInfo(images[i],CL_IMAGE_HEIGHT,sizeof(ih),&ih,NULL))return DV_INVALID;
  /* Decoder allocations may pad coded height (e.g. 2160→2176). Kernels
   * sample only the validated logical rectangle; allocation must cover it. */
  if(iw<ew||ih<eh||f.image_channel_data_type!=CL_UNORM_INT16||f.image_channel_order!=(i%2?CL_RG:CL_R))return DV_UNSUPPORTED;}
 if(reserve(b,(size_t)w*h,0,b->direct_p010))return DV_NOMEM;b->reconstruction_error=0;if(write_buffer(b,ERR,&b->reconstruction_error,4))return DV_BACKEND;
 if(scale==1&&!b->direct_p010)for(unsigned c=0;c<3;++c)
  if(reserve_one(b,E+c,(size_t)w*h*2/(c?4u:1u)))return DV_NOMEM;
 if(b->va_count)return DV_BACKEND;
 for(unsigned i=0;i<4;++i){if(clRetainMemObject(images[i])){for(unsigned j=0;j<i;++j)clReleaseMemObject(images[j]);return DV_BACKEND;}b->owned_va[i]=images[i];}
 cl_event acquired=NULL;cl_int e=acquire(b->queue,4,images,0,NULL,&acquired);if(e){for(unsigned i=0;i<4;++i){clReleaseMemObject(b->owned_va[i]);b->owned_va[i]=NULL;}return DV_BACKEND;}b->va_count=4;
 if(b->direct_p010){
  /* In-order queue: acquire precedes all readers. Release only after the last
   * reconstruction reader; references stay owned until dv_opencl_drain. */
  clReleaseEvent(acquired);b->shared_prepared=1;b->direct_active=1;uint16_t *out[3]={NULL,NULL,NULL};
  dv_status status=reconstruct_impl(b,s,w,h,NULL,NULL,out,NULL,1,1);b->direct_active=0;
  /* Some drivers wait inside release enqueue. The caller retains the VA
   * owners; drain can return them after reconstruction has completed. */
  e=b->defer_va_release?CL_SUCCESS:release(b->queue,4,images,0,NULL,&b->va_release);
  if(e||clFlush(b->queue)||status!=DV_OK){(void)dv_opencl_drain(b);return status!=DV_OK?status:DV_BACKEND;}
  size_t raw_bytes=0;for(unsigned k=0;k<6;++k)raw_bytes+=b->bytes[k];
  fprintf(stderr,"{\"direct_p010_pts\":%lld,\"unpack_dispatches\":0,\"raw_planar_allocation_bytes\":%zu,\"raw_planar_write_bytes\":0}\n",(long long)s->identity.pts,raw_bytes);
  return DV_OK;
 }
 for(unsigned layer=0;layer<2&&!e;++layer){cl_uint ww=layer?w/scale:w,hh=layer?h/scale:h;unsigned offset=layer?E:B;
  for(unsigned i=0;i<2&&!e;++i)e=clSetKernelArg(b->unpack,i,sizeof(cl_mem),&images[layer*2+i]);
  for(unsigned i=0;i<3&&!e;++i)e=clSetKernelArg(b->unpack,i+2,sizeof(cl_mem),&b->buffer[offset+i]);
  if(!e)e=clSetKernelArg(b->unpack,5,sizeof(ww),&ww);if(!e)e=clSetKernelArg(b->unpack,6,sizeof(hh),&hh);if(!e)e=clSetKernelArg(b->unpack,7,sizeof(cl_mem),&b->buffer[ERR]);
  size_t work=(size_t)ww*hh,grid[2]={ww/2,hh/2};if(!e){e=clEnqueueNDRangeKernel(b->queue,b->unpack,b->unpack_block?2u:1u,NULL,b->unpack_block?grid:&work,NULL,1,&acquired,&b->unpack_event[b->unpack_event_count]);if(!e)++b->unpack_event_count;}
  /* Diagnostic only: compare every decoded sample against original unpack on
   * the SAME still-owned VA surfaces, before any reconstruction can mask it. */
  if(!e&&b->unpack_reference){cl_mem ref[3]={0};unsigned char *a=malloc(work*2),*r=malloc(work*2);size_t differences=0;
   if(!a||!r)e=CL_OUT_OF_HOST_MEMORY;
   for(unsigned c=0;c<3&&!e;++c){size_t bytes=(c?work/4:work)*2;ref[c]=clCreateBuffer(b->context,CL_MEM_READ_WRITE,bytes,NULL,&e);}
   for(unsigned i=0;i<2&&!e;++i)e=clSetKernelArg(b->unpack_reference,i,sizeof(cl_mem),&images[layer*2+i]);
   for(unsigned i=0;i<3&&!e;++i)e=clSetKernelArg(b->unpack_reference,i+2,sizeof(cl_mem),&ref[i]);
   if(!e)e=clSetKernelArg(b->unpack_reference,5,sizeof(ww),&ww);if(!e)e=clSetKernelArg(b->unpack_reference,6,sizeof(hh),&hh);if(!e)e=clSetKernelArg(b->unpack_reference,7,sizeof(cl_mem),&b->buffer[ERR]);
   if(!e)e=clEnqueueNDRangeKernel(b->queue,b->unpack_reference,1,NULL,&work,NULL,0,NULL,NULL);
   for(unsigned c=0;c<3&&!e;++c){size_t bytes=(c?work/4:work)*2;e=clEnqueueReadBuffer(b->queue,b->buffer[offset+c],CL_TRUE,0,bytes,a,0,NULL,NULL);if(!e)e=clEnqueueReadBuffer(b->queue,ref[c],CL_TRUE,0,bytes,r,0,NULL,NULL);if(!e)for(size_t j=0;j<bytes;++j)differences+=a[j]!=r[j];}
   for(unsigned c=0;c<3;++c)if(ref[c])clReleaseMemObject(ref[c]);free(a);free(r);
   fprintf(stderr,"{\"unpack_verified_pts\":%lld,\"layer\":%u,\"compared_bytes\":%zu,\"byte_differences\":%zu,\"cl_status\":%d}\n",(long long)s->identity.pts,layer,work*3,differences,e);
   if(differences&&!e)e=CL_INVALID_VALUE;
  }}
 cl_int re=release(b->queue,4,images,0,NULL,&b->va_release);clReleaseEvent(acquired);if(e||re){(void)dv_opencl_drain(b);return DV_BACKEND;}
 b->shared_prepared=1;uint16_t *out[3]={NULL,NULL,NULL};return reconstruct_impl(b,s,w,h,NULL,NULL,out,NULL,1,1);}
static dv_status prepare_single_mapping(backend *b,const dv_intel_composer_config *cfg)
{
 int hit=b->single_cache&&b->single_cpu_valid&&!memcmp(&b->single_config,cfg,sizeof(*cfg));
 if(!hit){uint16_t table[3][1024];if(dv_prepare_polynomial_tables(cfg,table))return DV_UNSUPPORTED;
  memcpy(b->single_mapping,table,sizeof(table));b->single_config=*cfg;
  b->single_cpu_valid=1;b->single_gpu_valid=0;++b->single_misses;
 }else ++b->single_hits;
 return DV_OK;
}
dv_status dv_opencl_reconstruct_single_p010(void *p,const dv_frame_settings *s,
 unsigned w,unsigned h,const cl_mem images[2])
{
 backend *b=p;if(b){b->resident_ready=0;b->output_ready=0;b->image_output=0;}
 if(!b||!s||!images||!w||!h||w>4096||h>4096||w%4||h%4)return DV_INVALID;
 if(b->va_count||b->gl_count)return DV_BACKEND;
 if(s->chroma_phase||s->composer.residual_enabled)return DV_UNSUPPORTED;
 if(prepare_single_mapping(b,&s->composer)!=DV_OK)return DV_UNSUPPORTED;
 for(unsigned i=0;i<2;++i){cl_context context=NULL;cl_image_format f={0};size_t iw=0,ih=0;
  if(!images[i]||clGetMemObjectInfo(images[i],CL_MEM_CONTEXT,sizeof(context),&context,NULL)||context!=b->context||
     clGetImageInfo(images[i],CL_IMAGE_FORMAT,sizeof(f),&f,NULL)||
     clGetImageInfo(images[i],CL_IMAGE_WIDTH,sizeof(iw),&iw,NULL)||
     clGetImageInfo(images[i],CL_IMAGE_HEIGHT,sizeof(ih),&ih,NULL))return DV_INVALID;
  if(iw<(i?w/2:w)||ih<(i?h/2:h)||f.image_channel_data_type!=CL_UNORM_INT16||
     f.image_channel_order!=(i?CL_RG:CL_R))return DV_UNSUPPORTED;
 }
 cl_platform_id platform=NULL;if(clGetDeviceInfo(b->device,CL_DEVICE_PLATFORM,sizeof(platform),&platform,NULL))return DV_BACKEND;
 clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn acquire=(clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform,"clEnqueueAcquireVA_APIMediaSurfacesINTEL");
 clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn release=(clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform,"clEnqueueReleaseVA_APIMediaSurfacesINTEL");
 if(!acquire||!release)return DV_UNSUPPORTED;
 cl_int error;if(!b->single){b->single=clCreateKernel(b->program,"single_polynomial_p010",&error);if(!b->single||error)return DV_BACKEND;}
 size_t n=(size_t)w*h;
 if(reserve_one(b,MT,sizeof(b->single_mapping))||reserve_one(b,ERR,4))return DV_NOMEM;
 for(unsigned c=0;c<3;++c)if(reserve_one(b,O+c,(c?n/4:n)*2))return DV_NOMEM;
 b->reconstruction_error=0;
 for(unsigned i=0;i<2;++i){if(clRetainMemObject(images[i])){for(unsigned j=0;j<i;++j)clReleaseMemObject(b->owned_va[j]);return DV_BACKEND;}b->owned_va[i]=images[i];}
 if(acquire(b->queue,2,images,0,NULL,NULL)){for(unsigned i=0;i<2;++i){clReleaseMemObject(b->owned_va[i]);b->owned_va[i]=NULL;}return DV_BACKEND;}
 b->va_count=2;dv_status status=DV_BACKEND;
#define SINGLE_DO(call) do{if(call)goto done;}while(0)
 if(!b->single_gpu_valid){SINGLE_DO(write_buffer(b,MT,b->single_mapping,sizeof(b->single_mapping)));b->single_gpu_valid=1;}
 SINGLE_DO(write_buffer(b,ERR,&b->reconstruction_error,4));
 for(unsigned i=0;i<2;++i)SINGLE_DO(clSetKernelArg(b->single,i,sizeof(cl_mem),images+i));
 unsigned slots[4]={MT,O,O+1,O+2};
 for(unsigned i=0;i<4;++i)SINGLE_DO(clSetKernelArg(b->single,i+2,sizeof(cl_mem),&b->buffer[slots[i]]));
 SINGLE_DO(clSetKernelArg(b->single,6,sizeof(w),&w));SINGLE_DO(clSetKernelArg(b->single,7,sizeof(h),&h));
 SINGLE_DO(clSetKernelArg(b->single,8,sizeof(cl_mem),&b->buffer[ERR]));
 size_t work[2]={w/2,h/2};SINGLE_DO(clEnqueueNDRangeKernel(b->queue,b->single,2,NULL,work,NULL,0,NULL,NULL));
 SINGLE_DO(read_reconstruction_error(b));
 if(!b->defer_va_release)SINGLE_DO(release(b->queue,2,images,0,NULL,&b->va_release));
 SINGLE_DO(clFlush(b->queue));
 b->piecewise_active=0;b->resident_ready=1;b->width=w;b->height=h;b->chroma_phase=s->chroma_phase;b->identity=s->identity;b->config=s->composer;
 status=DV_OK;
done:
 if(status!=DV_OK)b->single_gpu_valid=0;
 if(status!=DV_OK&&dv_opencl_drain(b)!=DV_OK)return DV_BACKEND;
 return status;
#undef SINGLE_DO
}

static dv_status reconstruct_piecewise_surface(void *p,const dv_single_frame_settings *s,
 unsigned w,unsigned h,const cl_mem images[2],unsigned layout,unsigned sampling,unsigned depth)
{
 backend *b=p;if(!b)return DV_INVALID;
 /* A rejected overlapping submission must not invalidate a pending frame or
  * overwrite host storage still borrowed by nonblocking queue transfers. */
 if(b->va_count||b->gl_count)return DV_BACKEND;
 b->resident_ready=0;b->output_ready=0;b->image_output=0;b->piecewise_active=0;
 if(!s||!images||!w||!h||w>4096||h>4096||w%4||h%4)return DV_INVALID;
 if(layout>1||sampling>1||(!layout&&sampling)||(depth!=8&&depth!=10)||(depth==8&&layout))return DV_UNSUPPORTED;
 cl_kernel kernel=depth==8?b->nv12_kernel:layout?b->piecewise_full_kernel:b->piecewise_kernel;
 if(!kernel)return DV_UNSUPPORTED;
 cl_uint4 mask={{0}};
 if(dv_single_reshape_surface_masks(&s->mapping,depth,mask.s)||memcmp(mask.s,s->narrow_mask,sizeof(s->narrow_mask))||
    s->source.compressed||s->source.depth!=12||(s->source.space!=0&&s->source.space!=2)||s->source.eotf!=65535||s->source.full_range!=1||
    s->active[0]>=s->active[2]||s->active[1]>=s->active[3]||s->active[2]>w||s->active[3]>h)return DV_UNSUPPORTED;
 if(b->piecewise_order2_kernel){
  if(layout==1&&depth==10&&dv_single_reshape_order2_chroma(&s->mapping)){
   kernel=b->piecewise_order2_kernel;++b->piecewise_order2_hits;
  }else ++b->piecewise_order2_misses;
 }
 for(unsigned i=0;i<2;++i){cl_context context=NULL;cl_image_format f={0};size_t iw=0,ih=0;
  if(!images[i]||clGetMemObjectInfo(images[i],CL_MEM_CONTEXT,sizeof(context),&context,NULL)||context!=b->context||
     clGetImageInfo(images[i],CL_IMAGE_FORMAT,sizeof(f),&f,NULL)||clGetImageInfo(images[i],CL_IMAGE_WIDTH,sizeof(iw),&iw,NULL)||
     clGetImageInfo(images[i],CL_IMAGE_HEIGHT,sizeof(ih),&ih,NULL))return DV_INVALID;
  if(iw<(i?w/2:w)||ih<(i?h/2:h)||f.image_channel_data_type!=(depth==8?CL_UNORM_INT8:CL_UNORM_INT16)||f.image_channel_order!=(i?CL_RG:CL_R))return DV_UNSUPPORTED;
 }
 int changed=!b->piecewise_cached||memcmp(&b->piecewise_mapping,&s->mapping,sizeof(s->mapping));
 if(changed){b->piecewise_cached=0;
  if(dv_single_reshape_luma_table_for_depth(&s->mapping,depth,b->piecewise_luma,(size_t)1<<depth))return DV_UNSUPPORTED;
  memcpy(&b->piecewise_mapping,&s->mapping,sizeof(s->mapping));}
 size_t n=(size_t)w*h;
 if(reserve_one(b,SM,sizeof(s->mapping))||reserve_one(b,SL,sizeof(b->piecewise_luma))||reserve_one(b,ERR,4))return DV_NOMEM;
 for(unsigned c=0;c<3;++c)if(reserve_one(b,O+c,(c?n/(layout?2u:4u):n)*2))return DV_NOMEM;
 cl_platform_id platform=NULL;if(clGetDeviceInfo(b->device,CL_DEVICE_PLATFORM,sizeof(platform),&platform,NULL))return DV_BACKEND;
 clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn acquire=(clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform,"clEnqueueAcquireVA_APIMediaSurfacesINTEL");
 clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn release=(clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform,"clEnqueueReleaseVA_APIMediaSurfacesINTEL");
 if(!acquire||!release)return DV_UNSUPPORTED;
 for(unsigned i=0;i<2;++i){if(clRetainMemObject(images[i])){for(unsigned j=0;j<i;++j){clReleaseMemObject(b->owned_va[j]);b->owned_va[j]=NULL;}return DV_BACKEND;}b->owned_va[i]=images[i];}
 if(acquire(b->queue,2,images,0,NULL,NULL)){for(unsigned i=0;i<2;++i){clReleaseMemObject(b->owned_va[i]);b->owned_va[i]=NULL;}return DV_BACKEND;}
 b->va_count=2;b->reconstruction_error=0;dv_status status=DV_BACKEND;
#define PIECEWISE_DO(call) do{if(call)goto done;}while(0)
 if(changed){PIECEWISE_DO(write_buffer(b,SM,&b->piecewise_mapping,sizeof(b->piecewise_mapping)));
  PIECEWISE_DO(write_buffer(b,SL,b->piecewise_luma,((size_t)1<<depth)*sizeof(*b->piecewise_luma)));}
 PIECEWISE_DO(write_buffer(b,ERR,&b->reconstruction_error,4));
 for(unsigned i=0;i<2;++i)PIECEWISE_DO(clSetKernelArg(kernel,i,sizeof(cl_mem),images+i));
 unsigned slots[4]={SM,O,O+1,O+2};
 for(unsigned i=0;i<4;++i)PIECEWISE_DO(clSetKernelArg(kernel,i+2,sizeof(cl_mem),&b->buffer[slots[i]]));
 PIECEWISE_DO(clSetKernelArg(kernel,6,sizeof(w),&w));PIECEWISE_DO(clSetKernelArg(kernel,7,sizeof(h),&h));
 PIECEWISE_DO(clSetKernelArg(kernel,8,sizeof(cl_mem),&b->buffer[ERR]));
 PIECEWISE_DO(clSetKernelArg(kernel,9,sizeof(mask),&mask));
 PIECEWISE_DO(clSetKernelArg(kernel,10,sizeof(cl_mem),&b->buffer[SL]));
 if(layout)PIECEWISE_DO(clSetKernelArg(kernel,11,sizeof(sampling),&sampling));
 /* Keep timing separate from P7 event categories; consume only at the normal
  * ownership drain, never add a diagnostic wait to frame submission. */
 int profile=b->profile_async&&b->piecewise_profile_frames<120;
 size_t work[2]={w/2,layout?h:h/2};PIECEWISE_DO(clEnqueueNDRangeKernel(b->queue,kernel,2,NULL,work,NULL,0,NULL,profile?&b->piecewise_profile_event:NULL));
 if(profile){b->piecewise_profile_pts=s->identity.pts;b->piecewise_profile_width=w;b->piecewise_profile_height=h;
  b->piecewise_profile_depth=depth;b->piecewise_profile_layout=layout;}
 PIECEWISE_DO(read_reconstruction_error(b));
 if(!b->defer_va_release)PIECEWISE_DO(release(b->queue,2,images,0,NULL,&b->va_release));
 PIECEWISE_DO(clFlush(b->queue));
 memcpy(&b->piecewise_frame,s,sizeof(*s));b->piecewise_cached=1;b->piecewise_active=1;
 b->piecewise_layout=layout;b->piecewise_sampling=sampling;
 b->resident_ready=1;b->width=w;b->height=h;b->identity=s->identity;status=DV_OK;
done:
 if(status!=DV_OK){b->piecewise_cached=0;if(dv_opencl_drain(b)!=DV_OK)return DV_BACKEND;}
 return status;
#undef PIECEWISE_DO
}
dv_status dv_opencl_reconstruct_piecewise_p010(void *p,const dv_single_frame_settings *s,
 unsigned w,unsigned h,const cl_mem images[2])
{return reconstruct_piecewise_surface(p,s,w,h,images,0,0,10);}
dv_status dv_opencl_reconstruct_piecewise_nv12(void *p,const dv_single_frame_settings *s,
 unsigned w,unsigned h,const cl_mem images[2])
{return reconstruct_piecewise_surface(p,s,w,h,images,0,0,8);}
dv_status dv_opencl_reconstruct_piecewise_surface(void *p,const dv_piecewise_surface_settings *s,
 unsigned w,unsigned h,const cl_mem images[2])
{return reconstruct_piecewise_surface(p,s?&s->frame:NULL,w,h,images,s?s->layout:0,s?s->sampling:0,10);}

dv_status dv_opencl_publish_gl(void *p,const cl_mem images[3],cl_event *event)
{backend *b=p;if(!b||!images||!event||!b->output_ready||b->image_output)return DV_INVALID;
 for(unsigned i=0;i<3;++i){cl_context c=NULL;cl_image_format f={0};size_t w=0,h=0;
  if(!images[i]||clGetMemObjectInfo(images[i],CL_MEM_CONTEXT,sizeof(c),&c,NULL)||c!=b->context||clGetImageInfo(images[i],CL_IMAGE_FORMAT,sizeof(f),&f,NULL)||clGetImageInfo(images[i],CL_IMAGE_WIDTH,sizeof(w),&w,NULL)||clGetImageInfo(images[i],CL_IMAGE_HEIGHT,sizeof(h),&h,NULL))return DV_INVALID;
  if(w!=b->width||h!=b->height||f.image_channel_order!=CL_R||f.image_channel_data_type!=CL_UNSIGNED_INT16)return DV_UNSUPPORTED;}
 if(b->gl_count)return DV_BACKEND;
 for(unsigned i=0;i<3;++i){if(clRetainMemObject(images[i])){for(unsigned j=0;j<i;++j)clReleaseMemObject(images[j]);return DV_BACKEND;}b->owned_gl[i]=images[i];}
 double acquire_start=host_seconds();cl_int e=clEnqueueAcquireGLObjects(b->queue,3,images,1,&b->output_event,NULL);double acquire_cpu=host_seconds()-acquire_start;if(e){for(unsigned i=0;i<3;++i){clReleaseMemObject(b->owned_gl[i]);b->owned_gl[i]=NULL;}return DV_BACKEND;}b->gl_count=3;
 for(unsigned i=0;i<3&&!e;++i)e=clSetKernelArg(b->pack,i,sizeof(cl_mem),&b->buffer[CO+i]);
 for(unsigned i=0;i<3&&!e;++i)e=clSetKernelArg(b->pack,i+3,sizeof(cl_mem),&images[i]);
 cl_uint w=b->width,h=b->height;if(!e)e=clSetKernelArg(b->pack,6,sizeof(w),&w);if(!e)e=clSetKernelArg(b->pack,7,sizeof(h),&h);size_t work=(size_t)w*h;
 if(!e)e=clEnqueueNDRangeKernel(b->queue,b->pack,1,NULL,&work,NULL,0,NULL,&b->pack_event);
 cl_event done=NULL;double release_start=host_seconds();cl_int re=clEnqueueReleaseGLObjects(b->queue,3,images,0,NULL,&done);double release_cpu=host_seconds()-release_start;b->gl_release=done;
 fprintf(stderr,"{\"gl_acquire_enqueue_cpu_seconds\":%.9f,\"gl_release_enqueue_cpu_seconds\":%.9f}\n",acquire_cpu,release_cpu);
 if(e||re||clFlush(b->queue)){(void)dv_opencl_drain(b);return DV_BACKEND;}if(clRetainEvent(done))return DV_BACKEND;*event=done;return DV_OK;}
dv_status dv_opencl_drain(void *p)
{backend *b=p;if(!b||!b->queue)return DV_INVALID;
 if(clFinish(b->queue)!=CL_SUCCESS)return DV_BACKEND;
 if(b->piecewise_profile_event){
  cl_ulong start=0,end=0;
  int valid=!clGetEventProfilingInfo(b->piecewise_profile_event,CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL)&&
   !clGetEventProfilingInfo(b->piecewise_profile_event,CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL)&&end>=start;
  fprintf(stderr,"{\"source_piecewise_profile_pts\":%lld,\"valid\":%d,\"reconstruction_ms\":%.6f,\"width\":%u,\"height\":%u,\"depth\":%u,\"layout\":%u}\n",
   (long long)b->piecewise_profile_pts,valid,valid?(double)(end-start)/1e6:0.0,
   b->piecewise_profile_width,b->piecewise_profile_height,b->piecewise_profile_depth,b->piecewise_profile_layout);
  clReleaseEvent(b->piecewise_profile_event);b->piecewise_profile_event=NULL;++b->piecewise_profile_frames;
 }
 if(b->profile_pending){
  double ms[6]={0};unsigned counts[6]={0};int valid=1;
  for(unsigned i=0;i<b->event_count;++i){cl_ulong start=0,end=0;unsigned kind=b->event_kind[i];
   if(kind>=6||clGetEventProfilingInfo(b->events[i],CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL)||
      clGetEventProfilingInfo(b->events[i],CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL)||end<start){valid=0;break;}
   ms[kind]+=(double)(end-start)/1e6;++counts[kind];}
  fprintf(stderr,"{\"source_async_profile_pts\":%lld,\"valid\":%d,\"phase_ms\":%.6f,\"vertical_ms\":%.6f,\"horizontal_ms\":%.6f,\"guide_ms\":%.6f,\"luma_ms\":%.6f,\"chroma_mmr_ms\":%.6f,\"dispatch_counts\":[%u,%u,%u,%u,%u,%u]}\n",
   (long long)b->identity.pts,valid,ms[0],ms[1],ms[2],ms[3],ms[4],ms[5],counts[0],counts[1],counts[2],counts[3],counts[4],counts[5]);
  ++b->profile_frames;b->profile_pending=0;release_events(b);
 }
 if(b->va_count){if(b->va_release){cl_int status=1;if(clGetEventInfo(b->va_release,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL)||status!=CL_COMPLETE){clReleaseEvent(b->va_release);b->va_release=NULL;return DV_BACKEND;}}
  else {cl_platform_id platform=NULL;if(clGetDeviceInfo(b->device,CL_DEVICE_PLATFORM,sizeof(platform),&platform,NULL))return DV_BACKEND;
   clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn release=(clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform,"clEnqueueReleaseVA_APIMediaSurfacesINTEL");
   if(!release||release(b->queue,b->va_count,b->owned_va,0,NULL,&b->va_release)||clFinish(b->queue))return DV_BACKEND;
   cl_int status=1;if(clGetEventInfo(b->va_release,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL)||status!=CL_COMPLETE)return DV_BACKEND;}
  clReleaseEvent(b->va_release);b->va_release=NULL;for(unsigned i=0;i<b->va_count;++i){clReleaseMemObject(b->owned_va[i]);b->owned_va[i]=NULL;}b->va_count=0;}
 if(b->gl_count){if(b->gl_release){cl_int status=1;if(clGetEventInfo(b->gl_release,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL)||status!=CL_COMPLETE){clReleaseEvent(b->gl_release);b->gl_release=NULL;return DV_BACKEND;}}
  else {if(clEnqueueReleaseGLObjects(b->queue,b->gl_count,b->owned_gl,0,NULL,&b->gl_release)||clFinish(b->queue))return DV_BACKEND;cl_int status=1;if(clGetEventInfo(b->gl_release,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL)||status!=CL_COMPLETE)return DV_BACKEND;}
  clReleaseEvent(b->gl_release);b->gl_release=NULL;for(unsigned i=0;i<b->gl_count;++i){clReleaseMemObject(b->owned_gl[i]);b->owned_gl[i]=NULL;}b->gl_count=0;}
 if(b->unpack_event_count){double seconds=0;for(unsigned i=0;i<b->unpack_event_count;++i){cl_ulong start=0,end=0;
   if(!clGetEventProfilingInfo(b->unpack_event[i],CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL)&&!clGetEventProfilingInfo(b->unpack_event[i],CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL))seconds+=(double)(end-start)/1e9;
   clReleaseEvent(b->unpack_event[i]);b->unpack_event[i]=NULL;}b->unpack_event_count=0;fprintf(stderr,"{\"p010_unpack_gpu_seconds\":%.9f,\"decoded_sample_host_upload_bytes\":0}\n",seconds);}
 if(b->pack_event){cl_ulong start=0,end=0;double seconds=0;
  if(!clGetEventProfilingInfo(b->pack_event,CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL)&&!clGetEventProfilingInfo(b->pack_event,CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL))seconds=(double)(end-start)/1e9;
  clReleaseEvent(b->pack_event);b->pack_event=NULL;fprintf(stderr,"{\"r16ui_pack_gpu_seconds\":%.9f,\"host_output_download_bytes\":0}\n",seconds);}
 return DV_OK;}
