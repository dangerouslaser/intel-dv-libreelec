/* Opt-in Intel GPU regression; independent scalar spatial/composer oracle. */
#include "dv_gpu_backend.h"
#include "dv_backend_internal.h"
#include "pixel_polynomial.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int dv_mmr_luma_guide(const uint16_t *,size_t,unsigned,unsigned,uint16_t *,size_t);
int dv_chroma_phase_left(const uint16_t *,size_t,unsigned,unsigned,unsigned,uint16_t *,size_t);
int dv_annex_b_scale2x(const uint16_t *,size_t,unsigned,unsigned,unsigned,uint16_t *,size_t,uint16_t *,size_t);
int dv_reference_chroma_batch(const dv_intel_composer_config *,unsigned,size_t,
 const uint16_t *,const uint16_t *,const uint16_t *,const uint16_t *,uint16_t *,int32_t *,int32_t *,uint16_t *);

enum {W=64,H=32,N=W*H};
static uint16_t base[3][N],enh[3][N],aligned[3][N],scaled[3][N],actual[3][N];
static uint16_t guide[N],temporary[N],scratch[W],expected[N],mapped[N];
static int32_t residual[N],sum[N];

int main(int argc,char **argv)
{
    assert(argc==2);
    cl_platform_id platforms[16];cl_uint count=0;cl_device_id device=NULL;
    assert(clGetPlatformIDs(16,platforms,&count)==CL_SUCCESS);
    for(unsigned p=0;p<count&&p<16&&!device;++p){
        cl_device_id devices[16];cl_uint n=0;
        if(clGetDeviceIDs(platforms[p],CL_DEVICE_TYPE_GPU,16,devices,&n))continue;
        for(unsigned i=0;i<n&&i<16;++i){cl_uint vendor=0;
            assert(!clGetDeviceInfo(devices[i],CL_DEVICE_VENDOR_ID,sizeof(vendor),&vendor,NULL));
            if(vendor==0x8086){device=devices[i];break;}
        }
    }
    assert(device);cl_int error;
    cl_context context=clCreateContext(NULL,1,&device,NULL,NULL,&error);assert(context&&!error);
    void *backend=dv_opencl_create_source_at(context,device,argv[1]);assert(backend);
    size_t compared=0,fel_changed=0;dv_frame_settings s;
    /* Reuse one backend across polynomial/MMR and geometry changes to exercise
     * cache/dispatch transitions as well as all four chroma method pairings. */
    for(unsigned scale=1;scale<=2;++scale)for(unsigned phase=0;phase<2;++phase)
    for(unsigned methods=0;methods<4;++methods)for(unsigned piecewise=0;piecewise<2;++piecewise){
        memset(&s,0,sizeof(s));s.abi_version=DV_ENGINE_ABI;s.chroma_phase=phase;
        dv_intel_composer_config *cfg=&s.composer;cfg->abi_version=1;
        cfg->base_depth=cfg->enhancement_depth=10;cfg->reconstruction_depth=12;
        cfg->coefficient_log2_denominator=23;cfg->residual_enabled=1;cfg->el_spatial_resampling_flag=scale==2;
        for(unsigned c=0;c<3;++c){
            dv_component_config *v=&cfg->component[c];
            v->pivot_count=2;v->pivots[1]=1023;v->polynomial_order[0]=1;v->polynomial[0][1].integer=1;
            v->nlq_offset=512;v->nlq[0].fraction=2048;v->nlq[1].fraction=1048576;
            v->nlq[2].fraction=513;
            if(c){
                v->mapping=(methods>>(c-1))&1u;
                if(v->mapping){v->mmr_order=1;v->mmr[0][c].integer=1;}
                else if(piecewise){
                    v->pivot_count=3;v->pivots[1]=512;v->pivots[2]=1023;
                    v->polynomial_order[0]=v->polynomial_order[1]=2;
                    v->polynomial[0][2].fraction=1048576;
                    v->polynomial[1][0].fraction=524288;
                    v->polynomial[1][1].fraction=7340032;
                    v->polynomial[1][2].fraction=1048576;
                }
            }
            unsigned w=c?W/2:W,h=c?H/2:H;size_t n=(size_t)w*h;
            for(size_t i=0;i<n;++i){base[c][i]=(uint16_t)(128+(i*73+c*101)%768);enh[c][i]=(uint16_t)(128+(i*37+c*91)%768);}
            if(c){
                assert(!dv_chroma_phase_left(base[c],n,w,h,phase,aligned[c],n));
                assert(!dv_chroma_phase_left(enh[c],n/(scale*scale),w/scale,h/scale,phase,temporary,n/(scale*scale)));
            }else{memcpy(aligned[c],base[c],n*2);memcpy(temporary,enh[c],n/(scale*scale)*2);}
            if(scale==2)assert(!dv_annex_b_scale2x(temporary,n/4,w/2,h/2,c!=0,scratch,w/2,scaled[c],n));
            else memcpy(scaled[c],temporary,n*2);
        }
        assert(!dv_mmr_luma_guide(base[0],N,W,H,guide,N/4));
        const uint16_t *bl[3]={base[0],base[1],base[2]},*el[3]={enh[0],enh[1],enh[2]};
        uint16_t *out[3]={actual[0],actual[1],actual[2]};
        assert(dv_opencl_reconstruct(backend,&s,W,H,bl,el,out,NULL)==DV_OK);
        for(unsigned c=0;c<3;++c){size_t n=c?N/4:N;
            if(c)assert(!dv_reference_chroma_batch(cfg,c,n,guide,aligned[1],aligned[2],scaled[c],mapped,residual,sum,expected));
            else for(size_t i=0;i<n;++i){int32_t m,r;
                assert(!dv_reference_polynomial(cfg,0,aligned[0][i],scaled[0][i],&m,&r,&expected[i]));}
            for(size_t i=0;i<n;++i){
                if(actual[c][i]!=expected[i]){
                    fprintf(stderr,"mismatch scale=%u phase=%u methods=%u piecewise=%u c=%u i=%zu GPU=%u CPU=%u\n",scale,phase,methods,piecewise,c,i,actual[c][i],expected[i]);
                    assert(0);
                }
                ++compared;
                if(c&&!cfg->component[c].mapping){int32_t m,r;uint16_t neutral;
                    assert(!dv_reference_polynomial(cfg,c,aligned[c][i],512,&m,&r,&neutral));
                    fel_changed+=(size_t)(actual[c][i]!=neutral);
                }
            }
        }
    }
    assert(fel_changed>0);
    /* Filter ringing may exceed the input-code range. Fail closed, and never
     * index the 1024-entry residual table with an out-of-range scaled sample. */
    s.chroma_phase=0;s.composer.component[1].mapping=0;s.composer.component[2].mapping=0;
    for(unsigned c=0;c<3;++c)for(unsigned i=0;i<N;++i){
        base[c][i]=512;enh[c][i]=c?(i%(W/4)<W/8?0:1023):512;
    }
    assert(!dv_annex_b_scale2x(enh[1],N/16,W/4,H/4,1,scratch,W/4,scaled[1],N/4));
    unsigned overshoots=0;for(unsigned i=0;i<N/4;++i)overshoots+=(scaled[1][i]>1023);
    assert(overshoots>0);
    const uint16_t *bl[3]={base[0],base[1],base[2]},*el[3]={enh[0],enh[1],enh[2]};
    uint16_t *out[3]={actual[0],actual[1],actual[2]};
    assert(dv_opencl_reconstruct(backend,&s,W,H,bl,el,out,NULL)==DV_SAMPLE_RANGE);
    printf("GPU polynomial/MMR FEL: 32 cases, %zu samples, zero differences; %zu chroma samples changed by FEL\n",compared,fel_changed);
    dv_opencl_destroy(backend);clReleaseContext(context);return 0;
}
