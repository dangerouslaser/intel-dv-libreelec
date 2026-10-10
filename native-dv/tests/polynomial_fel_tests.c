#include "pixel_polynomial.h"
#include "mmr_prepare.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int dv_reference_chroma_batch(const dv_intel_composer_config *,unsigned,size_t,
    const uint16_t *,const uint16_t *,const uint16_t *,const uint16_t *,
    uint16_t *,int32_t *,int32_t *,uint16_t *);

static dv_intel_composer_config fixture(void)
{
    dv_intel_composer_config cfg={0};
    cfg.abi_version=1;cfg.base_depth=cfg.enhancement_depth=10;
    cfg.reconstruction_depth=12;cfg.coefficient_log2_denominator=23;
    cfg.residual_enabled=1;cfg.el_spatial_resampling_flag=1;
    for(unsigned c=0;c<3;++c){
        dv_component_config *p=&cfg.component[c];
        p->pivot_count=2;p->pivots[1]=1023;p->polynomial_order[0]=1;
        p->polynomial[0][1].integer=1;p->nlq_offset=512;
        p->nlq[0].fraction=2048;p->nlq[1].fraction=1048576;
    }
    return cfg;
}
static uint16_t finish(int32_t mapped,int32_t residual)
{
    int64_t v=(int64_t)mapped+residual+8;
    return v<0?0:v>=65536?4095:(uint16_t)(v/16);
}
int main(void)
{
    dv_intel_composer_config cfg=fixture();
    uint16_t table[1024],output;int32_t residuals[1024],m,r;
    uint32_t mmr[96];
    assert(dv_mmr_prepare(&cfg,1,mmr,residuals)==DV_UNSUPPORTED);
    /* Independently known identity polynomial and FEL deadzone arithmetic. */
    for(unsigned c=1;c<3;++c){
        assert(!dv_prepare_polynomial_component(&cfg,c,table,residuals));
        for(unsigned b=0;b<1024;++b)assert(table[b]==64*b);
        for(int e=0;e<1024;++e){
            int q=e-512;int expected=q?(2*q-(q<0?-1:1))*8:0;
            assert(residuals[e]==expected);
        }
        assert(finish(table[512],residuals[512])==2048);
        assert(finish(table[512],residuals[600])!=2048);
    }
    /* Piecewise quadratic, fractional threshold, negative coefficients and
     * endpoint/clamping cases. Exhaust every BL/EL code combination. */
    for(unsigned variant=0;variant<3;++variant){
        for(unsigned c=1;c<3;++c){
            dv_component_config *p=&cfg.component[c];
            if(variant){
                p->pivot_count=3;p->pivots[0]=16;p->pivots[1]=512;p->pivots[2]=1000;
                p->polynomial_order[0]=p->polynomial_order[1]=2;
                p->polynomial[0][0].integer=-1;
                p->polynomial[0][2].fraction=4194304;
                p->polynomial[1][0].fraction=2097152;
                p->polynomial[1][1].integer=1;
                p->polynomial[1][2].fraction=6291456;
                p->nlq[2].fraction=513;
            }
            cfg.residual_enabled=variant!=2;
            assert(!dv_prepare_polynomial_component(&cfg,c,table,residuals));
            for(unsigned b=0;b<1024;++b)for(unsigned e=0;e<1024;++e){
                assert(!dv_reference_polynomial(&cfg,c,(uint16_t)b,(uint16_t)e,&m,&r,&output));
                assert(m==table[b]&&r==residuals[e]);
                assert(output==finish(table[b],residuals[e]));
            }
            uint16_t y[4]={0,1023,400,600},cb[4]={16,511,512,1000},cr[4]={1000,512,511,16};
            uint16_t el[4]={0,511,512,1023},mapped[4],codes[4];int32_t res[4],sum[4];
            assert(!dv_reference_chroma_batch(&cfg,c,4,y,cb,cr,el,mapped,res,sum,codes));
            for(unsigned i=0;i<4;++i){
                unsigned b=c==1?cb[i]:cr[i];
                assert(mapped[i]==table[b]&&res[i]==residuals[el[i]]);
                assert(codes[i]==finish(mapped[i],res[i]));
            }
        }
    }
    memset(table,0x55,sizeof(table));memset(residuals,0x55,sizeof(residuals));
    cfg.component[1].polynomial_order[0]=3;
    assert(dv_prepare_polynomial_component(&cfg,1,table,residuals)==-1);
    for(unsigned i=0;i<1024;++i)assert(table[i]==0x5555&&residuals[i]==0x55555555);
    puts("polynomial FEL: 6291456 BL/EL combinations matched; residual retained");
    return 0;
}
