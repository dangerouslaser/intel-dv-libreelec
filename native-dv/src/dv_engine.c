#define _POSIX_C_SOURCE 200809L
#include "dv_engine.h"
#include "dv_colour_internal.h"
#include "dv_backend_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
int dv_reference_luma(const dv_intel_composer_config *,uint16_t,uint16_t,int32_t *,int32_t *,uint16_t *);
int dv_reference_chroma_batch(const dv_intel_composer_config *,unsigned,size_t,const uint16_t *,const uint16_t *,const uint16_t *,const uint16_t *,uint16_t *,int32_t *,int32_t *,uint16_t *);
int dv_mmr_luma_guide(const uint16_t *,size_t,unsigned,unsigned,uint16_t *,size_t);
int dv_chroma_phase_left(const uint16_t *,size_t,unsigned,unsigned,unsigned,uint16_t *,size_t);
int dv_annex_b_scale2x(const uint16_t *,size_t,unsigned,unsigned,unsigned,uint16_t *,size_t,uint16_t *,size_t);
struct dv_engine {int busy;dv_luma_backend luma;void *opaque;dv_chain_backend chain;void *chain_opaque;dv_chroma_backend chroma;void *chroma_opaque;int resident_guide,resident_reconstruction;dv_reconstruct_backend reconstruct;void *reconstruct_opaque;unsigned workers;dv_timings timings;dv_colour_backend colour;void *colour_opaque;};
int dv_engine_set_colour_backend(dv_engine *e,dv_colour_backend f,void *p)
{if(!e || e->busy || (f && !e->reconstruct))return -1;e->colour=f;e->colour_opaque=p;e->resident_reconstruction=0;return 0;}
static double now_seconds(void){struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))return 0;return (double)t.tv_sec+(double)t.tv_nsec/1e9;}
int dv_engine_get_timings(const dv_engine *e,dv_timings *out){if(!e || !out || e->busy)return -1;*out=e->timings;return 0;}
int dv_engine_set_colour_workers(dv_engine *e,unsigned n){if(!e || e->busy || n<1 || n>8)return -1;e->workers=n;return 0;}
int dv_engine_set_reconstruct_backend(dv_engine *e,dv_reconstruct_backend f,void *p)
{if(!e || e->busy)return -1;e->reconstruct=f;e->reconstruct_opaque=p;e->resident_reconstruction=0;if(f){e->luma=NULL;e->chain=NULL;e->chroma=NULL;e->resident_guide=0;}else{e->colour=NULL;e->colour_opaque=NULL;}return 0;}
int dv_engine_set_resident_backends(dv_engine *e,dv_reconstruct_backend r,dv_colour_backend c,void *p)
{if(!e || e->busy || !r || !c)return -1;if(dv_engine_set_reconstruct_backend(e,r,p) || dv_engine_set_colour_backend(e,c,p))return -1;e->resident_reconstruction=1;return 0;}
int dv_engine_set_chroma_backend(dv_engine *e,dv_chroma_backend f,void *p)
{if(!e || e->busy || (e->reconstruct && f))return -1;e->chroma=f;e->chroma_opaque=p;e->resident_guide=0;return 0;}
int dv_engine_set_resident_chroma_backend(dv_engine *e,dv_chroma_backend f,void *p)
{if(!e || e->busy || !e->chain || !f)return -1;e->chroma=f;e->chroma_opaque=p;e->resident_guide=1;return 0;}
int dv_engine_set_luma_backend(dv_engine *e,dv_luma_backend f,void *p)
{if(!e || e->busy || (e->reconstruct && f))return -1;e->luma=f;e->opaque=p;if(f){e->chain=NULL;e->chain_opaque=NULL;}return 0;}
int dv_engine_set_chain_backend(dv_engine *e,dv_chain_backend f,void *p)
{if(!e || e->busy || (e->reconstruct && f))return -1;e->chain=f;e->chain_opaque=p;if(f){e->luma=NULL;e->opaque=NULL;}return 0;}
typedef struct {uint64_t key;uint16_t code[3];uint8_t excursions;} entry;
static int identity(dv_identity a,dv_identity b)
{return a.frame_id==b.frame_id && a.pts==b.pts;}
static int plane_valid(const dv_plane *p,unsigned w,unsigned h)
{
    size_t row=(size_t)w*2;
    return p->data && !((uintptr_t)p->data%_Alignof(uint16_t)) &&
        p->stride_bytes>=row && !(p->stride_bytes%2) &&
        p->stride_bytes<=(SIZE_MAX-row)/(h-1u?h-1u:1u) &&
        p->size_bytes>=(size_t)(h-1)*p->stride_bytes+row &&
        p->size_bytes<=UINTPTR_MAX-(uintptr_t)p->data;
}
static int overlap(const void *a,size_t an,const void *b,size_t bn)
{return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an;}
static double expand(const uint16_t *p,unsigned w,unsigned h,unsigned x,unsigned y)
{
    unsigned ix=x/2,iy=y/2,next=ix+1<w?ix+1:ix;
    unsigned top=(y&1)?iy:(iy?iy-1:0),bottom=(y&1)?(iy+1<h?iy+1:iy):iy;
    unsigned a=(y&1)?3:1,b=4-a;
    uint32_t left=a*p[(size_t)top*w+ix]+b*p[(size_t)bottom*w+ix];
    if (!(x&1)) return (double)left/4;
    uint32_t right=a*p[(size_t)top*w+next]+b*p[(size_t)bottom*w+next];
    return (double)(left+right)/8;
}
static uint64_t hash(uint64_t x)
{x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
static uint64_t key(const double v[3])
{return 1+((uint64_t)(v[0]*8)|((uint64_t)(v[1]*8)<<15)|((uint64_t)(v[2]*8)<<30));}
typedef struct {
    const dv_frame_settings *settings;
    const dv_source_colour *source;
    const dv_target_colour *target;
    uint16_t **rec,**final;
    unsigned w,h,first,last;
    int use_cache,cache_active;
    uint64_t excursions;
    dv_status status;
} colour_job;
static void *colour_worker(void *opaque)
{
    colour_job *j=opaque;const dv_frame_settings *s=j->settings;
    entry *cache=j->use_cache?calloc(65536,sizeof(*cache)):NULL;
    if(j->use_cache && !cache){j->status=DV_NOMEM;return NULL;}
    for(unsigned y=j->first;y<j->last;++y)for(unsigned x=0;x<j->w;++x){
        double codes[3]={j->rec[0][(size_t)y*j->w+x],expand(j->rec[1],j->w/2,j->h/2,x,y),expand(j->rec[2],j->w/2,j->h/2,x,y)};
        int outside=x<s->active[0] || x>=s->active[2] || y<s->active[1] || y>=s->active[3];
        entry *p=NULL;dv_source_result sr;dv_target_result tr;
        if(cache && (j->cache_active || outside)){
            uint64_t k=key(codes);p=&cache[hash(k)&65535];if(p->key!=k){p->key=k;p->excursions=255;}
        }
        if(p && p->excursions!=255){memcpy(tr.code,p->code,sizeof(tr.code));j->excursions+=p->excursions;}
        else {
            if(dv_source_colour_sample(j->source,codes,&sr) || dv_target_colour_sample(j->target,s->pq_policy,sr.linear_lms,&tr)){
                j->status=DV_COLOUR_DOMAIN;free(cache);return NULL;
            }
            unsigned count=0;for(unsigned k=0;k<3;++k)if(sr.nonlinear[k]<0 || sr.nonlinear[k]>1)++count;
            j->excursions+=count;if(p){p->excursions=(uint8_t)count;memcpy(p->code,tr.code,sizeof(tr.code));}
        }
        if(outside){tr.code[0]=0;tr.code[1]=2048;tr.code[2]=2048;}
        for(unsigned c=0;c<3;++c)j->final[c][(size_t)y*j->w+x]=tr.code[c];
    }
    free(cache);j->status=DV_OK;return NULL;
}
dv_status dv_engine_create(dv_engine **out)
{
    if (!out) return DV_INVALID;
    dv_engine *e=calloc(1,sizeof(*e));if (!e) return DV_NOMEM;
    e->workers=1;*out=e;return DV_OK;
}
void dv_engine_destroy(dv_engine *e){free(e);}
const char *dv_status_string(dv_status s)
{
    switch(s){case DV_OK:return "success";case DV_INVALID:return "invalid buffers or settings";
    case DV_UNSUPPORTED:return "unsupported metadata or format";case DV_NOMEM:return "allocation failed";
    case DV_SAMPLE_RANGE:return "sample outside supported range";case DV_COLOUR_DOMAIN:return "colour outside declared policy";
    case DV_IDENTITY:return "layer/metadata identity mismatch";case DV_OBSERVER:return "diagnostic callback failed";case DV_BACKEND:return "accelerated backend failed";}
    return "unknown status";
}
dv_status dv_engine_process(dv_engine *e,const dv_image *bl,const dv_image *el,
    const dv_frame_settings *s,dv_image *out,dv_statistics *stats,const dv_observer *observer)
{
    if (!e || e->busy || !bl || !el || !s || !out || (e->resident_guide && (!e->chain || !e->chroma))) return DV_INVALID;
    double start_time=now_seconds(),reconstruction_end=0,colour_start=0;
    if (!identity(bl->identity,el->identity) || !identity(bl->identity,s->identity)) return DV_IDENTITY;
    unsigned w=bl->width,h=bl->height;
    unsigned scale=s->composer.el_spatial_resampling_flag?2u:1u;
    if (!w || !h || w>4096 || h>4096 || w%4 || h%4 || el->width!=w/scale || el->height!=h/scale ||
        out->width!=w || out->height!=h || bl->memory || el->memory || out->memory ||
        bl->format!=DV_PLANAR420_10 || el->format!=DV_PLANAR420_10 || out->format!=DV_PLANAR444_12)
        return DV_UNSUPPORTED;
    if (s->abi_version!=DV_ENGINE_ABI || s->pq_policy>1 || s->chroma_phase>1 || s->el_scaler!=1 ||
        s->colour_cache>2 || s->composer.el_spatial_resampling_flag>1 || s->composer.base_full_range>1 ||
        s->active[0]>=s->active[2] || s->active[1]>=s->active[3] || s->active[2]>w || s->active[3]>h ||
        s->active[0]%2 || s->active[2]%2) return DV_UNSUPPORTED;
    /* Existing whole-frame/chain backends require half-size enhancement input. */
    if(scale==1 && (e->reconstruct || e->chain))return DV_UNSUPPORTED;
    for (unsigned c=0;c<3;++c) {
        unsigned cw=c?w/2:w,ch=c?h/2:h;
        if (!plane_valid(&bl->plane[c],cw,ch) || !plane_valid(&el->plane[c],cw/scale,ch/scale) ||
            !plane_valid(&out->plane[c],w,h)) return DV_INVALID;
    }
    for (unsigned c=0;c<3;++c) {
        const dv_plane *p=&out->plane[c];
        if (overlap(p->data,p->size_bytes,s,sizeof(*s)) || overlap(p->data,p->size_bytes,bl,sizeof(*bl)) ||
            overlap(p->data,p->size_bytes,el,sizeof(*el)) || overlap(p->data,p->size_bytes,out,sizeof(*out)) ||
            (stats && overlap(p->data,p->size_bytes,stats,sizeof(*stats))) ||
            (observer && overlap(p->data,p->size_bytes,observer,sizeof(*observer)))) return DV_INVALID;
        for (unsigned k=0;k<3;++k)
            if ((c!=k && overlap(p->data,p->size_bytes,out->plane[k].data,out->plane[k].size_bytes)) ||
                overlap(p->data,p->size_bytes,bl->plane[k].data,bl->plane[k].size_bytes) ||
                overlap(p->data,p->size_bytes,el->plane[k].data,el->plane[k].size_bytes)) return DV_INVALID;
    }
    /* Primitive preflight validates logical coefficients before large work. */
    int32_t v,r,total;uint16_t code,z=0,m;
    if (dv_reference_luma(&s->composer,0,0,&v,&r,&code) ||
        dv_reference_chroma_batch(&s->composer,1,1,&z,&z,&z,&z,&m,&r,&total,&code) ||
        dv_reference_chroma_batch(&s->composer,2,1,&z,&z,&z,&z,&m,&r,&total,&code)) return DV_UNSUPPORTED;
    dv_source_colour source;dv_target_colour target;
    if (dv_source_colour_init(&s->source,s->pq_policy,&source) ||
        dv_target_colour_init(s->target_ycc,s->target_lms,s->target_offset,&target)) return DV_UNSUPPORTED;
    size_t n=(size_t)w*h,nc=n/4;
    uint16_t *base[3]={0},*enh[3]={0},*rec[3]={0},*final[3]={0};
    unsigned borrowed_base[3]={0},borrowed_enh[3]={0};
    uint16_t *raw=NULL,*ph=NULL,*scratch=NULL,*guide=NULL,*mapped=NULL;
    int32_t *residual=NULL,*sum=NULL;entry *cache=NULL;
    dv_status status=DV_OK;dv_statistics result={0};e->busy=1;
#define ALLOC(P,N,T) do{P=calloc((N),sizeof(T));if (!(P)){status=DV_NOMEM;goto done;}}while(0)
#define CHECK(C,S) do{if(C){status=(S);goto done;}}while(0)
    if(e->reconstruct){
        for(unsigned c=0;c<3;++c){size_t count=c?nc:n;unsigned cw=c?w/2:w,ch=c?h/2:h;
            if(!e->resident_reconstruction || (observer && observer->stage))ALLOC(rec[c],count,uint16_t);
            for(unsigned layer=0;layer<2;++layer){const dv_plane *p=layer?&el->plane[c]:&bl->plane[c];unsigned ww=layer?cw/2:cw,hh=layer?ch/2:ch;
                unsigned borrow=p->stride_bytes==(size_t)ww*2;
                if(layer){borrowed_enh[c]=borrow;if(borrow)enh[c]=p->data;else ALLOC(enh[c],count/4,uint16_t);}
                else {borrowed_base[c]=borrow;if(borrow)base[c]=p->data;else ALLOC(base[c],count,uint16_t);}
                uint16_t *dest=layer?enh[c]:base[c];
                for(unsigned y=0;y<hh;++y){const uint16_t *row=(const uint16_t *)((const unsigned char *)p->data+(size_t)y*p->stride_bytes);
                    uint32_t bits=0;for(unsigned x=0;x<ww;++x)bits|=row[x];
                    CHECK(bits & ~UINT32_C(1023),DV_SAMPLE_RANGE);
                    if(!borrow)memcpy(dest+(size_t)y*ww,row,(size_t)ww*sizeof(*row));}}}
        const uint16_t *bb[3]={base[0],base[1],base[2]},*ee[3]={enh[0],enh[1],enh[2]};
        status=e->reconstruct(e->reconstruct_opaque,s,w,h,bb,ee,rec,observer);CHECK(status!=DV_OK,status);
        for(unsigned c=0;c<3;++c){if(!borrowed_enh[c])free(enh[c]);enh[c]=NULL;}goto colour;
    }
    for (unsigned c=0;c<3;++c) {
        size_t count=c?nc:n;unsigned cw=c?w/2:w,ch=c?h/2:h;
        size_t enhancement_count=count/(scale*scale);
        ALLOC(base[c],count,uint16_t);ALLOC(raw,enhancement_count,uint16_t);
        for (unsigned layer=0;layer<2;++layer) {
            const dv_plane *p=layer?&el->plane[c]:&bl->plane[c];
            unsigned ww=layer?cw/scale:cw,hh=layer?ch/scale:ch;uint16_t *dest=layer?raw:base[c];
            for (unsigned y=0;y<hh;++y) {
                const uint16_t *row=(const uint16_t *)((const unsigned char *)p->data+(size_t)y*p->stride_bytes);
                uint32_t bits=0;for(unsigned x=0;x<ww;++x)bits|=row[x];
                CHECK(bits & ~UINT32_C(1023),DV_SAMPLE_RANGE);
                memcpy(dest+(size_t)y*ww,row,(size_t)ww*sizeof(*row));
            }
        }
        if(c){
            ALLOC(ph,count,uint16_t);
            CHECK(dv_chroma_phase_left(base[c],count,cw,ch,s->chroma_phase,ph,count),DV_UNSUPPORTED);
            free(base[c]);base[c]=ph;ph=NULL;
            ALLOC(ph,enhancement_count,uint16_t);
            CHECK(dv_chroma_phase_left(raw,enhancement_count,cw/scale,ch/scale,s->chroma_phase,ph,enhancement_count),DV_UNSUPPORTED);
            free(raw);raw=ph;ph=NULL;
        }
        if(scale==1 || (!c && e->chain)){enh[c]=raw;raw=NULL;}
        else {
            ALLOC(enh[c],count,uint16_t);ALLOC(scratch,cw/2,uint16_t);
            CHECK(dv_annex_b_scale2x(raw,count/4,cw/2,ch/2,c!=0,scratch,cw/2,enh[c],count),DV_UNSUPPORTED);
            free(raw);raw=NULL;free(scratch);scratch=NULL;
            for(size_t i=0;i<count;++i) CHECK(enh[c][i]>1023,DV_SAMPLE_RANGE);
        }
        ALLOC(rec[c],count,uint16_t);
    }
    if(!e->resident_guide) ALLOC(guide,nc,uint16_t);
    if(!e->chain) CHECK(dv_mmr_luma_guide(base[0],n,w,h,guide,nc),DV_UNSUPPORTED);
    for (unsigned c=0;c<3;++c) {
        size_t count=c?nc:n;int diagnostics=observer && observer->stage;
        if((c?!e->chroma:!e->chain) || diagnostics){ALLOC(mapped,count,uint16_t);ALLOC(residual,count,int32_t);ALLOC(sum,count,int32_t);}
        if(c && e->chroma){dv_status b=e->chroma(e->chroma_opaque,&s->composer,c,count,guide,base[1],base[2],enh[c],mapped,residual,sum,rec[c],diagnostics);CHECK(b!=DV_OK,b);}
        else if(c) CHECK(dv_reference_chroma_batch(&s->composer,c,count,guide,base[1],base[2],enh[c],mapped,residual,sum,rec[c]),DV_UNSUPPORTED);
        else if(e->chain){dv_status b=e->chain(e->chain_opaque,&s->composer,w,h,base[0],enh[0],guide,mapped,residual,sum,rec[c],diagnostics);CHECK(b!=DV_OK,b);}
        else if(e->luma){CHECK(e->luma(e->opaque,&s->composer,count,base[0],enh[0],mapped,residual,sum,rec[c]),DV_BACKEND);}
        else {
            uint16_t mt[1024];int32_t rt[1024];
            for(unsigned i=0;i<1024;++i){
                CHECK(dv_reference_luma(&s->composer,(uint16_t)i,(uint16_t)s->composer.component[0].nlq_offset,&v,&r,&code),DV_UNSUPPORTED);mt[i]=(uint16_t)v;
                CHECK(dv_reference_luma(&s->composer,0,(uint16_t)i,&v,&rt[i],&code),DV_UNSUPPORTED);
            }
            for(size_t i=0;i<count;++i){
                mapped[i]=mt[base[0][i]];residual[i]=rt[enh[0][i]];
                int64_t t=(int64_t)mapped[i]+residual[i];CHECK(t<INT32_MIN || t>INT32_MAX,DV_SAMPLE_RANGE);
                sum[i]=(int32_t)t;int64_t q=t+8<=0?0:(t+8)/16;rec[c][i]=(uint16_t)(q>4095?4095:q);
            }
        }
        if(observer && observer->stage){
            const void *data[4]={mapped,residual,sum,rec[c]};unsigned bytes[4]={2,4,4,2};
            for(unsigned k=0;k<4;++k) CHECK(observer->stage(observer->opaque,(dv_stage)k,c,data[k],count,bytes[k]),DV_OBSERVER);
        }
        free(mapped);mapped=NULL;free(residual);residual=NULL;free(sum);sum=NULL;free(enh[c]);enh[c]=NULL;
    }
colour:
    reconstruction_end=now_seconds();free(guide);guide=NULL;for(unsigned c=0;c<3;++c){if(!borrowed_base[c])free(base[c]);base[c]=NULL;ALLOC(final[c],n,uint16_t);}
    if(e->colour){const uint16_t *rr[3]={rec[0],rec[1],rec[2]};colour_start=now_seconds();
        status=e->colour(e->colour_opaque,s,w,h,rr,final,&result.source_pq_excursions);CHECK(status!=DV_OK,status);goto commit;}
    if(s->colour_cache) ALLOC(cache,65536,entry);
    result.cache_active=s->colour_cache==1;
    if(s->colour_cache==2){
        unsigned rows=s->active[3]-s->active[1];if(rows>32) rows=32;
        unsigned start=s->active[1]+(s->active[3]-s->active[1]-rows)/2;size_t hits=0,count=0;
        for(unsigned y=start;y<start+rows;++y) for(unsigned x=s->active[0];x<s->active[2];++x){
            double codes[3]={rec[0][(size_t)y*w+x],expand(rec[1],w/2,h/2,x,y),expand(rec[2],w/2,h/2,x,y)};
            uint64_t k=key(codes);entry *p=&cache[hash(k)&65535];++count;if(p->key==k)++hits;else p->key=k;
        }
        result.cache_active=hits*100>=count*3;memset(cache,0,65536*sizeof(*cache));
    }
    colour_start=now_seconds();
    if(e->workers>1){
        colour_job jobs[8];pthread_t threads[8];unsigned launched=0;
        for(unsigned i=0;i<e->workers;++i){
            jobs[i]=(colour_job){s,&source,&target,rec,final,w,h,h*i/e->workers,h*(i+1)/e->workers,s->colour_cache!=0,result.cache_active!=0,0,DV_BACKEND};
            if(pthread_create(&threads[i],NULL,colour_worker,&jobs[i]))break;
            ++launched;
        }
        for(unsigned i=0;i<launched;++i)(void)pthread_join(threads[i],NULL);
        CHECK(launched!=e->workers,DV_BACKEND);
        for(unsigned i=0;i<launched;++i){CHECK(jobs[i].status!=DV_OK,jobs[i].status);result.source_pq_excursions+=jobs[i].excursions;}
    }else{
    for(unsigned y=0;y<h;++y) for(unsigned x=0;x<w;++x){
        double codes[3]={rec[0][(size_t)y*w+x],expand(rec[1],w/2,h/2,x,y),expand(rec[2],w/2,h/2,x,y)};
        int outside=x<s->active[0] || x>=s->active[2] || y<s->active[1] || y>=s->active[3];
        entry *p=NULL;dv_source_result sr;dv_target_result tr;
        if(cache && (result.cache_active || outside)){
            uint64_t k=key(codes);p=&cache[hash(k)&65535];if(p->key!=k){p->key=k;p->excursions=255;}
        }
        if(p && p->excursions!=255){memcpy(tr.code,p->code,sizeof(tr.code));result.source_pq_excursions+=p->excursions;}
        else {
            CHECK(dv_source_colour_sample(&source,codes,&sr) || dv_target_colour_sample(&target,s->pq_policy,sr.linear_lms,&tr),DV_COLOUR_DOMAIN);
            unsigned count=0;for(unsigned k=0;k<3;++k) if(sr.nonlinear[k]<0 || sr.nonlinear[k]>1)++count;
            result.source_pq_excursions+=count;if(p){p->excursions=(uint8_t)count;memcpy(p->code,tr.code,sizeof(tr.code));}
        }
        if(outside){tr.code[0]=0;tr.code[1]=2048;tr.code[2]=2048;}
        for(unsigned c=0;c<3;++c)final[c][(size_t)y*w+x]=tr.code[c];
    }
    }
commit:
    for(unsigned c=0;c<3;++c)for(unsigned y=0;y<h;++y)
        memcpy((unsigned char *)out->plane[c].data+(size_t)y*out->plane[c].stride_bytes,final[c]+(size_t)y*w,(size_t)w*2);
    out->identity=s->identity;if(stats)*stats=result;
    e->timings=(dv_timings){reconstruction_end-start_time,now_seconds()-colour_start,now_seconds()-start_time};
done:
    free(raw);free(ph);free(scratch);free(guide);free(mapped);free(residual);free(sum);free(cache);
    for(unsigned c=0;c<3;++c){if(!borrowed_base[c])free(base[c]);if(!borrowed_enh[c])free(enh[c]);free(rec[c]);free(final[c]);}
    e->busy=0;return status;
#undef CHECK
#undef ALLOC
}
