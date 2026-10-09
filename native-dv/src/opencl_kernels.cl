/* Private integer reconstruction backend. No floating-point pixel math. */
#pragma OPENCL FP_CONTRACT OFF
#ifdef DV_COLOUR_OUTPUT_IMAGES
#define DV_COLOUR_OUTPUT write_only image2d_t
#define DV_STORE_COLOUR(a,b,c,i,w,v) do { int2 point=(int2)((i)%(w),(i)/(w)); write_imageui(a,point,(uint4)((v)[0],0,0,1)); write_imageui(b,point,(uint4)((v)[1],0,0,1)); write_imageui(c,point,(uint4)((v)[2],0,0,1)); } while(0)
#else
#define DV_COLOUR_OUTPUT global ushort *
#define DV_STORE_COLOUR(a,b,c,i,w,v) do { (a)[i]=(v)[0]; (b)[i]=(v)[1]; (c)[i]=(v)[2]; } while(0)
#endif
int edge(int p,uint n){return clamp(p,0,(int)n-1);}
#ifdef DV_RECON_2D
#define GRID_X(width) ((uint)get_global_id(0))
#define GRID_Y(width) ((uint)get_global_id(1))
#else
#define GRID_X(width) ((uint)get_global_id(0)%(width))
#define GRID_Y(width) ((uint)get_global_id(0)/(width))
#endif
ushort rounded(int v,int d){v+=d/2;v=v>=0?v/d:-((-v+d-1)/d);return (ushort)clamp(v,0,65535);}
#ifdef DV_INPUT_P010
#define INPUT_PLANE read_only image2d_t
#define INPUT_EXTRA ,uint channel,global uint *error
#define INPUT_PASS ,channel,error
ushort input_sample(read_only image2d_t src,uint w,uint x,uint y,uint channel,global uint *error){
 (void)w;float4 raw=read_imagef(src,(int2)(x,y));uint q=convert_uint_rte(raw[channel]*65535.0f);
 if(q&63u)atomic_or(error,1u);return (ushort)(q>>6);
}
#define INPUT_AT(src,w,x,y) input_sample(src,w,x,y,channel,error)
#else
#define INPUT_PLANE global const ushort *
#define INPUT_EXTRA
#define INPUT_PASS
#define INPUT_AT(src,w,x,y) ((src)[(y)*(w)+(x)])
#endif
kernel void phase(INPUT_PLANE src,global ushort *dst,uint w,uint h,uint method INPUT_EXTRA){
    uint x=GRID_X(w),y=GRID_Y(w),i=y*w+x;if(i>=w*h)return;int v=0;
    if(!method)v=3*(int)INPUT_AT(src,w,x,y)+INPUT_AT(src,w,x,(uint)edge((int)y+1,h));
    else {int c[4]={-9,111,29,-3};for(int k=0;k<4;k++)v+=c[k]*(int)INPUT_AT(src,w,x,(uint)edge((int)y+k-1,h));}
    dst[i]=min(rounded(v,method?128:4),(ushort)1023);
}
kernel void vertical(INPUT_PLANE src,global ushort *dst,uint w,uint h,uint chroma INPUT_EXTRA){
    uint x=GRID_X(w),y=GRID_Y(w),i=y*w+x;if(i>=w*h*2)return;int row=(int)(y/2),v=0;
    if(chroma){int a=(y&1)?192:64,b=256-a,r0=(y&1)?row:row-1,r1=(y&1)?row+1:row;
        v=a*(int)INPUT_AT(src,w,x,(uint)edge(r0,h))+b*(int)INPUT_AT(src,w,x,(uint)edge(r1,h));}
    else {int even[4]={-3,29,111,-9},odd[4]={-9,111,29,-3};int first=(y&1)?row-1:row-2;
        for(int k=0;k<4;k++)v+=((y&1)?odd[k]:even[k])*(int)INPUT_AT(src,w,x,(uint)edge(first+k,h));}
    dst[i]=rounded(v,chroma?256:128);
}
/* Preserve phase rounding/clipping BEFORE vertical interpolation. Clamp the
 * phase row first: extending its output at the image edge is not the same as
 * extending the original input before applying the phase filter. */
ushort phase_sample(INPUT_PLANE src,uint w,uint h,uint x,int row,uint method INPUT_EXTRA){
 uint y=(uint)edge(row,h);int v=0;
 if(!method)v=3*(int)INPUT_AT(src,w,x,y)+INPUT_AT(src,w,x,(uint)edge((int)y+1,h));
 else {int taps[4]={-9,111,29,-3};for(int k=0;k<4;k++)v+=taps[k]*(int)INPUT_AT(src,w,x,(uint)edge((int)y+k-1,h));}
 return min(rounded(v,method?128:4),(ushort)1023);
}
kernel void phase_vertical(INPUT_PLANE src,global ushort *dst,uint w,uint h,uint method INPUT_EXTRA){
 uint x=GRID_X(w),y=GRID_Y(w);if(x>=w||y>=h)return;
 uint a=phase_sample(src,w,h,x,(int)y-1,method INPUT_PASS),b=phase_sample(src,w,h,x,(int)y,method INPUT_PASS),c=phase_sample(src,w,h,x,(int)y+1,method INPUT_PASS);
 dst[y*2*w+x]=(ushort)((64u*a+192u*b+128u)/256u);
 dst[(y*2+1)*w+x]=(ushort)((192u*b+64u*c+128u)/256u);
}
ushort horizontal(global const ushort *src,uint w,uint x,uint y){
    uint ix=x/2;if(!(x&1))return src[y*w+ix];int taps[8]={22,94,-524,2456,2456,-524,94,22},v=0;
    for(int k=0;k<8;k++)v+=taps[k]*(int)src[y*w+(uint)edge((int)ix+k-3,w)];return rounded(v,4096);
}
kernel void scale_horizontal(global const ushort *src,global ushort *dst,global uint *error,uint w,uint h){
    uint x=GRID_X(w),y=GRID_Y(w),i=y*w+x;if(i>=w*h)return;ushort v=horizontal(src,w/2,x,y);dst[i]=v;
    if(v>1023)atomic_or(error,1u);
}
ushort guide_sample(INPUT_PLANE src,uint w,uint h,uint gx,uint gy INPUT_EXTRA){
    uint x=gx*2,y=gy*2,a[2];
    for(uint r=0;r<2;r++){a[r]=((uint)INPUT_AT(src,w,(uint)edge((int)x-1,w),y+r)+2u*INPUT_AT(src,w,x,y+r)+INPUT_AT(src,w,(uint)edge((int)x+1,w),y+r)+2u)/4u;}
    return (ushort)((a[0]+a[1]+1u)/2u);
}
kernel void guide(INPUT_PLANE src,global ushort *dst,uint w,uint h INPUT_EXTRA){
    uint gx=GRID_X(w/2),gy=GRID_Y(w/2),i=gy*(w/2)+gx;if(i>=w*h/4)return;
    dst[i]=guide_sample(src,w,h,gx,gy INPUT_PASS);
}
kernel void luma_equal(INPUT_PLANE bl,INPUT_PLANE el,global const ushort *mt,global const int *rt,
    global ushort *mapped,global int *residual,global int *sum,global ushort *out,global uint *error,uint w,uint h,uint diag){
    uint x=GRID_X(w),y=GRID_Y(w),i=y*w+x;if(i>=w*h)return;
#ifdef DV_INPUT_P010
    uint base=input_sample(bl,w,x,y,0,error),enh=input_sample(el,w,x,y,0,error);
#else
    uint base=bl[i],enh=el[i];
#endif
    uint m=mt[base];int r=rt[enh];long total=(long)m+r;
    if(total>2147483647L || total<(-2147483647L-1L)){atomic_or(error,2u);return;}
    out[i]=(ushort)clamp(total>-8?(total+8)/16:0L,0L,4095L);
    if(diag){mapped[i]=(ushort)m;residual[i]=r;sum[i]=(int)total;}
}
kernel void luma(global const ushort *bl,global const ushort *temp,global const ushort *mt,global const int *rt,
    global ushort *mapped,global int *residual,global int *sum,global ushort *out,global uint *error,uint w,uint h,uint diag){
    uint x=GRID_X(w),y=GRID_Y(w),i=y*w+x;if(i>=w*h)return;ushort el=horizontal(temp,w/2,x,y);
    if(el>1023){atomic_or(error,1u);return;}uint m=mt[bl[i]];int r=rt[el];
    long total=(long)m+r;if(total>2147483647L || total<(-2147483647L-1L)){atomic_or(error,2u);return;}
    int t=(int)total;out[i]=(ushort)clamp(total>-8?(total+8)/16:0L,0L,4095L);if(diag){mapped[i]=(ushort)m;residual[i]=r;sum[i]=t;}
}
/* Adjacent-pixel reconstruction: makes the even passthrough / odd horizontal
 * filter explicit within each work item instead of diverging across lanes. */
kernel void luma_pair(INPUT_PLANE bl,global const ushort *temp,global const ushort *mt,global const int *rt,
    global ushort *mapped,global int *residual,global int *sum,global ushort *out,global uint *error,uint w,uint h,uint diag){
    uint sx=GRID_X(w/2),y=GRID_Y(w/2);if(sx>=w/2||y>=h)return;
    #pragma unroll
    for(uint lane=0;lane<2;++lane){uint x=sx*2+lane,i=y*w+x;ushort el=horizontal(temp,w/2,x,y);
        if(el>1023){atomic_or(error,1u);continue;}
#ifdef DV_INPUT_P010
        uint m=mt[input_sample(bl,w,x,y,0,error)];
#else
        uint m=mt[bl[i]];
#endif
        int r=rt[el];
        long total=(long)m+r;if(total>2147483647L || total<(-2147483647L-1L)){atomic_or(error,2u);continue;}
        int t=(int)total;out[i]=(ushort)clamp(total>-8?(total+8)/16:0L,0L,4095L);if(diag){mapped[i]=(ushort)m;residual[i]=r;sum[i]=t;}
    }
}
#ifdef DV_INPUT_P010
ushort2 input_uv(read_only image2d_t src,uint x,uint y,global uint *error){
 uint2 q=convert_uint2_rte(read_imagef(src,(int2)(x,y)).xy*65535.0f);
 if((q.x|q.y)&63u)atomic_or(error,1u);return convert_ushort2(q>>6);
}
ushort2 phase_uv_sample(read_only image2d_t src,uint x,int row,uint h,uint method,global uint *error){
 uint y=(uint)edge(row,h);int2 v=(int2)(0);
 if(!method)v=3*convert_int2(input_uv(src,x,y,error))+convert_int2(input_uv(src,x,(uint)edge((int)y+1,h),error));
 else {int taps[4]={-9,111,29,-3};for(int k=0;k<4;++k)v+=taps[k]*convert_int2(input_uv(src,x,(uint)edge((int)y+k-1,h),error));}
 return min((ushort2)(rounded(v.x,method?128:4),rounded(v.y,method?128:4)),(ushort2)(1023));
}
kernel void phase_uv(read_only image2d_t src,global ushort *cb,global ushort *cr,uint w,uint h,uint method,global uint *error){
 uint x=GRID_X(w),y=GRID_Y(w);if(x>=w||y>=h)return;ushort2 value=phase_uv_sample(src,x,(int)y,h,method,error);cb[y*w+x]=value.x;cr[y*w+x]=value.y;
}
kernel void phase_guide(read_only image2d_t luma,read_only image2d_t uv,
 global ushort *cb,global ushort *cr,global ushort *guide_out,uint w,uint h,uint method,global uint *error){
 uint x=GRID_X(w/2),y=GRID_Y(w/2);if(x>=w/2||y>=h/2)return;
 uint i=y*(w/2)+x;ushort2 value=phase_uv_sample(uv,x,(int)y,h/2,method,error);
 cb[i]=value.x;cr[i]=value.y;guide_out[i]=guide_sample(luma,w,h,x,y,0,error);
}
kernel void phase_vertical_uv(read_only image2d_t src,global ushort *cb,global ushort *cr,uint w,uint h,uint method,global uint *error){
 uint x=GRID_X(w),y=GRID_Y(w);if(x>=w||y>=h)return;
 uint2 a=convert_uint2(phase_uv_sample(src,x,(int)y-1,h,method,error)),b=convert_uint2(phase_uv_sample(src,x,(int)y,h,method,error)),c=convert_uint2(phase_uv_sample(src,x,(int)y+1,h,method,error));
 ushort2 even=convert_ushort2((64u*a+192u*b+128u)/256u),odd=convert_ushort2((192u*b+64u*c+128u)/256u);
 cb[y*2*w+x]=even.x;cr[y*2*w+x]=even.y;cb[(y*2+1)*w+x]=odd.x;cr[(y*2+1)*w+x]=odd.y;
}
kernel void vertical_pair(read_only image2d_t src,global ushort *dst,uint w,uint h,uint chroma,uint channel,global uint *error){
 uint x=GRID_X(w),y=GRID_Y(w);if(x>=w||y>=h)return;(void)chroma;
 int s0=input_sample(src,w,x,(uint)edge((int)y-2,h),channel,error),s1=input_sample(src,w,x,(uint)edge((int)y-1,h),channel,error),s2=input_sample(src,w,x,y,channel,error),s3=input_sample(src,w,x,(uint)edge((int)y+1,h),channel,error),s4=input_sample(src,w,x,(uint)edge((int)y+2,h),channel,error);
 dst[y*2*w+x]=rounded(-3*s0+29*s1+111*s2-9*s3,128);dst[(y*2+1)*w+x]=rounded(-9*s1+111*s2+29*s3-3*s4,128);
}
#endif
uint2 multiply(uint a,uint b){
#ifdef DV_MMR_NATIVE_MUL
    return (uint2)(a*b,mul_hi(a,b));
#else
    uint a0=a&65535u,a1=a>>16,b0=b&65535u,b1=b>>16,p=a0*b0,t=a1*b0+(p>>16),w1=(t&65535u)+a0*b1;return (uint2)((w1<<16)|(p&65535u),a1*b1+(t>>16)+(w1>>16));
#endif
}
uint4 add128(uint4 a,uint4 b){uint4 out;uint carry=0;for(uint k=0;k<4;k++){uint s=a[k]+b[k],c=(uint)(s<a[k]),v=s+carry;carry=c|(uint)(v<s);out[k]=v;}return out;}
uint4 product128(uint4 a,uint b){
#ifdef DV_MMR_SIGNED64
    /* Prepared MMR coefficients are signed int64, sign-extended to 128.
     * Two unsigned products give the 96-bit result; subtract b at word 2
     * for a negative coefficient, then sign-extend. Retain general fallback. */
    uint sign=(a.y&0x80000000u)?0xffffffffu:0u;
    if(a.z==sign && a.w==sign){
        uint2 low=multiply(a.x,b),high=multiply(a.y,b);
        uint middle=low.y+high.x,top=high.y+(uint)(middle<low.y)-(sign?b:0u);
        return (uint4)(low.x,middle,top,(top&0x80000000u)?0xffffffffu:0u);
    }
#endif
    uint4 out;uint carry=0;for(uint k=0;k<4;k++){uint2 p=multiply(a[k],b);uint v=p.x+carry;carry=p.y+(uint)(v<p.x);out[k]=v;}return out;
}
uint feature_product(uint a,uint b){uint2 p=multiply(a,b);return (p.x>>20)|(p.y<<12);}
uint4 load4(global const uint *p,uint i){return (uint4)(p[i],p[i+1],p[i+2],p[i+3]);}
uint4 accumulate(uint3 s,global const uint *params){
    for(uint k=0;k<3;k++)s[k]=clamp(s[k],params[2+2*k],params[3+2*k]);
    uint t[3][7];for(uint k=0;k<3;k++)t[0][k]=s[k]<<10;
    t[0][3]=s.x*s.y;t[0][4]=s.x*s.z;t[0][5]=s.y*s.z;t[0][6]=feature_product(t[0][3],t[0][2]);
    for(uint k=0;k<7;k++){t[1][k]=feature_product(t[0][k],t[0][k]);t[2][k]=feature_product(t[0][k],t[1][k]);}
    uint4 v=load4(params,8);for(uint order=0;order<params[0];order++)for(uint k=0;k<7;k++)v=add128(v,product128(load4(params,12+4*(order*7+k)),t[order][k]));return v;
}
uint4 accumulate_stream(uint3 s,global const uint *p){
    for(uint k=0;k<3;k++)s[k]=clamp(s[k],p[2+2*k],p[3+2*k]);
    uint xy=s.x*s.y;uint4 v=load4(p,8);
    #pragma unroll
    for(uint k=0;k<7;k++){
        uint a=k==0?s.x<<10:k==1?s.y<<10:k==2?s.z<<10:
            k==3?xy:k==4?s.x*s.z:k==5?s.y*s.z:feature_product(xy,s.z<<10);
        if(p[0]>0)v=add128(v,product128(load4(p,12+4*k),a));
        if(p[0]>1){uint b=feature_product(a,a);v=add128(v,product128(load4(p,40+4*k),b));
            if(p[0]>2)v=add128(v,product128(load4(p,68+4*k),feature_product(a,b)));}
    }
    return v;
}
kernel void mmr(global const ushort *y,global const ushort *cb,global const ushort *cr,global const ushort *el,
    global const uint *params,global const int *rt,global ushort *mapped,global int *residual,global int *sum,
    global ushort *out,global uint *error,uint count,uint diag){
    uint i=(uint)get_global_id(0);if(i>=count)return;if(el[i]>1023){atomic_or(error,1u);return;}
#ifdef DV_MMR_STREAM
    uint4 v=accumulate_stream((uint3)(y[i],cb[i],cr[i]),params);
#else
    uint4 v=accumulate((uint3)(y[i],cb[i],cr[i]),params);
#endif
    uint shift=params[1]+4,m;
    if(v.w&0x80000000u)m=0;else if(v.w || v.z || v.y>=(1u<<(shift-16)))m=65535;
    else m=shift<32?((v.x>>shift)|(v.y<<(32-shift))):(shift==32?v.y:(v.y>>(shift-32)));
    int r=rt[el[i]];long total=(long)m+r;if(total>2147483647L || total<(-2147483647L-1L)){atomic_or(error,2u);return;}
    int t=(int)total;out[i]=(ushort)clamp(total>-8?(total+8)/16:0L,0L,4095L);if(diag){mapped[i]=(ushort)m;residual[i]=r;sum[i]=t;}
}
uint3 add96(uint3 a,uint3 b){
    uint x=a.x+b.x,c=(uint)(x<a.x),t=a.y+b.y,y=t+c;
    uint carry=(uint)(t<a.y)|(uint)(y<t);
    return (uint3)(x,y,a.z+b.z+carry);
}
uint3 product96(global const uint *p,uint offset,uint b){
    uint a0=p[offset],a1=p[offset+1];
    uint2 low=multiply(a0,b),high=multiply(a1,b);
    uint middle=low.y+high.x;
    uint top=high.y+(uint)(middle<low.y)-((a1&0x80000000u)?b:0u);
    return (uint3)(low.x,middle,top);
}
uint4 accumulate96(uint3 s,global const uint *p){
    for(uint k=0;k<3;k++)s[k]=clamp(s[k],p[2+2*k],p[3+2*k]);
    uint xy=s.x*s.y;uint3 v=(uint3)(p[8],p[9],p[10]);
    #pragma unroll
    for(uint k=0;k<7;k++){
        uint a=k==0?s.x<<10:k==1?s.y<<10:k==2?s.z<<10:
            k==3?xy:k==4?s.x*s.z:k==5?s.y*s.z:feature_product(xy,s.z<<10);
        v=add96(v,product96(p,12+4*k,a));
        if(p[0]>1){uint b=feature_product(a,a);v=add96(v,product96(p,40+4*k,b));
            if(p[0]>2)v=add96(v,product96(p,68+4*k,feature_product(a,b)));}
    }
    return (uint4)(v,(v.z&0x80000000u)?0xffffffffu:0u);
}
kernel void mmr96(global const ushort *y,global const ushort *cb,global const ushort *cr,global const ushort *el,
    global const uint *params,global const int *rt,global ushort *mapped,global int *residual,global int *sum,
    global ushort *out,global uint *error,uint count,uint diag){
    uint i=(uint)get_global_id(0);if(i>=count)return;if(el[i]>1023){atomic_or(error,1u);return;}
    uint4 v=accumulate96((uint3)(y[i],cb[i],cr[i]),params);uint shift=params[1]+4,m;
    if(v.w&0x80000000u)m=0;else if(v.w || v.z || v.y>=(1u<<(shift-16)))m=65535;
    else m=shift<32?((v.x>>shift)|(v.y<<(32-shift))):(shift==32?v.y:(v.y>>(shift-32)));
    int r=rt[el[i]];long total=(long)m+r;if(total>2147483647L || total<(-2147483647L-1L)){atomic_or(error,2u);return;}
    int t=(int)total;out[i]=(ushort)clamp(total>-8?(total+8)/16:0L,0L,4095L);if(diag){mapped[i]=(ushort)m;residual[i]=r;sum[i]=t;}
}
/* Host proves signed32 coefficient words and all signed64 partial sums. */
long accumulate64(uint3 s,global const uint *p){
    for(uint k=0;k<3;k++)s[k]=clamp(s[k],p[2+2*k],p[3+2*k]);
    ulong bits=(ulong)p[8]|((ulong)p[9]<<32);long v=as_long(bits);uint xy=s.x*s.y;
    #pragma unroll
    for(uint k=0;k<7;k++){
        uint a=k==0?s.x<<10:k==1?s.y<<10:k==2?s.z<<10:
            k==3?xy:k==4?s.x*s.z:k==5?s.y*s.z:feature_product(xy,s.z<<10);
        v+=(long)as_int(p[12+4*k])*(long)a;
        if(p[0]>1){uint b=feature_product(a,a);v+=(long)as_int(p[40+4*k])*(long)b;
            if(p[0]>2)v+=(long)as_int(p[68+4*k])*(long)feature_product(a,b);}
    }
    return v;
}
kernel void mmr64(global const ushort *y,global const ushort *cb,global const ushort *cr,global const ushort *el,
    global const uint *params,global const int *rt,global ushort *mapped,global int *residual,global int *sum,
    global ushort *out,global uint *error,uint count,uint diag){
    uint i=(uint)get_global_id(0);if(i>=count)return;if(el[i]>1023){atomic_or(error,1u);return;}
    long v=accumulate64((uint3)(y[i],cb[i],cr[i]),params);uint shift=params[1]+4;
    uint m=v<0L?0u:(uint)min((ulong)v>>shift,65535UL);
    int r=rt[el[i]];long total=(long)m+r;if(total>2147483647L || total<(-2147483647L-1L)){atomic_or(error,2u);return;}
    int t=(int)total;out[i]=(ushort)clamp(total>-8?(total+8)/16:0L,0L,4095L);if(diag){mapped[i]=(ushort)m;residual[i]=r;sum[i]=t;}
}
/* Three scalar-feature tables sum all used orders, retaining exact feature
 * truncation. Original signed64 absolute-sum proof permits regrouping terms. */
long accumulate64_lut(uint3 s,global const uint *p,global const long *table){
    for(uint k=0;k<3;k++)s[k]=clamp(s[k],p[2+2*k],p[3+2*k]);
    long v=as_long((ulong)p[8]|((ulong)p[9]<<32));
    v+=table[s.x];v+=table[1024+s.y];v+=table[2048+s.z];
    uint xy=s.x*s.y;
    #pragma unroll
    for(uint k=3;k<7;k++){
        uint a=k==3?xy:k==4?s.x*s.z:k==5?s.y*s.z:feature_product(xy,s.z<<10);
        v+=(long)as_int(p[12+4*k])*(long)a;
        if(p[0]>1){uint b=feature_product(a,a);v+=(long)as_int(p[40+4*k])*(long)b;
            if(p[0]>2)v+=(long)as_int(p[68+4*k])*(long)feature_product(a,b);}
    }
    return v;
}
kernel void mmr64_lut(global const ushort *y,global const ushort *cb,global const ushort *cr,global const ushort *el,
    global const uint *params,global const int *rt,global ushort *mapped,global int *residual,global int *sum,
    global ushort *out,global uint *error,uint count,uint diag,global const long *table){
    uint i=(uint)get_global_id(0);if(i>=count)return;if(el[i]>1023){atomic_or(error,1u);return;}
    long v=accumulate64_lut((uint3)(y[i],cb[i],cr[i]),params,table);uint shift=params[1]+4;
    uint m=v<0L?0u:(uint)min((ulong)v>>shift,65535UL);
    int r=rt[el[i]];long total=(long)m+r;if(total>2147483647L || total<(-2147483647L-1L)){atomic_or(error,2u);return;}
    int t=(int)total;out[i]=(ushort)clamp(total>-8?(total+8)/16:0L,0L,4095L);if(diag){mapped[i]=(ushort)m;residual[i]=r;sum[i]=t;}
}
kernel void mmr64_lut_pair(global const ushort *y,global const ushort *cb,global const ushort *cr,global const ushort *el,
    global const uint *params,global const int *rt,global ushort *mapped,global int *residual,global int *sum,
    global ushort *out,global uint *error,uint count,uint diag,global const long *table){
    uint base=(uint)get_global_id(0)*2;
    #pragma unroll
    for(uint lane=0;lane<2;++lane){uint i=base+lane;if(i>=count)continue;if(el[i]>1023){atomic_or(error,1u);continue;}
        long v=accumulate64_lut((uint3)(y[i],cb[i],cr[i]),params,table);uint shift=params[1]+4;
        uint m=v<0L?0u:(uint)min((ulong)v>>shift,65535UL);
        int r=rt[el[i]];long total=(long)m+r;if(total>2147483647L || total<(-2147483647L-1L)){atomic_or(error,2u);continue;}
        int t=(int)total;out[i]=(ushort)clamp(total>-8?(total+8)/16:0L,0L,4095L);if(diag){mapped[i]=(ushort)m;residual[i]=r;sum[i]=t;}
    }
}
/* Same exact LUT accumulation, with genuinely uniform 384-byte parameters
 * in the constant address space. Pixel/table reads remain ordinary buffers. */
long accumulate64_lut_uniform(uint3 s,constant const uint *p,global const long *table){
    for(uint k=0;k<3;k++)s[k]=clamp(s[k],p[2+2*k],p[3+2*k]);
    long v=as_long((ulong)p[8]|((ulong)p[9]<<32));
    v+=table[s.x];v+=table[1024+s.y];v+=table[2048+s.z];
    uint xy=s.x*s.y;
    #pragma unroll
    for(uint k=3;k<7;k++){
        uint a=k==3?xy:k==4?s.x*s.z:k==5?s.y*s.z:feature_product(xy,s.z<<10);
        v+=(long)as_int(p[12+4*k])*(long)a;
        if(p[0]>1){uint b=feature_product(a,a);v+=(long)as_int(p[40+4*k])*(long)b;
            if(p[0]>2)v+=(long)as_int(p[68+4*k])*(long)feature_product(a,b);}
    }
    return v;
}
kernel void mmr64_lut_uniform(global const ushort *y,global const ushort *cb,global const ushort *cr,global const ushort *el,
    constant const uint *params,global const int *rt,global ushort *mapped,global int *residual,global int *sum,
    global ushort *out,global uint *error,uint count,uint diag,global const long *table){
    uint i=(uint)get_global_id(0);if(i>=count)return;if(el[i]>1023){atomic_or(error,1u);return;}
    long v=accumulate64_lut_uniform((uint3)(y[i],cb[i],cr[i]),params,table);uint shift=params[1]+4;
    uint m=v<0L?0u:(uint)min((ulong)v>>shift,65535UL);
    int r=rt[el[i]];long total=(long)m+r;if(total>2147483647L || total<(-2147483647L-1L)){atomic_or(error,2u);return;}
    int t=(int)total;out[i]=(ushort)clamp(total>-8?(total+8)/16:0L,0L,4095L);if(diag){mapped[i]=(ushort)m;residual[i]=r;sum[i]=t;}
}
/* Host eligibility guarantees signed32 coefficients and nonnegative features
 * below 2^20. Form the exact signed64 product from signed high/unsigned low
 * words, avoiding a generic emulated 64x64 multiply. */
long mmr_product_s32(int coefficient,uint feature){
 int high=mul_hi(coefficient,(int)feature);uint low=as_uint(coefficient)*feature;
 return as_long(((ulong)as_uint(high)<<32)|(ulong)low);
}
long accumulate64_lut_mulhi(uint3 s,global const uint *p,global const long *table){
    for(uint k=0;k<3;k++)s[k]=clamp(s[k],p[2+2*k],p[3+2*k]);
    long v=as_long((ulong)p[8]|((ulong)p[9]<<32));
    v+=table[s.x];v+=table[1024+s.y];v+=table[2048+s.z];
    uint xy=s.x*s.y;
    #pragma unroll
    for(uint k=3;k<7;k++){
        uint a=k==3?xy:k==4?s.x*s.z:k==5?s.y*s.z:feature_product(xy,s.z<<10);
        v+=mmr_product_s32(as_int(p[12+4*k]),a);
        if(p[0]>1){uint b=feature_product(a,a);v+=mmr_product_s32(as_int(p[40+4*k]),b);
            if(p[0]>2)v+=mmr_product_s32(as_int(p[68+4*k]),feature_product(a,b));}
    }
    return v;
}
kernel void mmr64_lut_mulhi(global const ushort *y,global const ushort *cb,global const ushort *cr,global const ushort *el,
    global const uint *params,global const int *rt,global ushort *mapped,global int *residual,global int *sum,
    global ushort *out,global uint *error,uint count,uint diag,global const long *table){
    uint i=(uint)get_global_id(0);if(i>=count)return;if(el[i]>1023){atomic_or(error,1u);return;}
    long v=accumulate64_lut_mulhi((uint3)(y[i],cb[i],cr[i]),params,table);uint shift=params[1]+4;
    uint m=v<0L?0u:(uint)min((ulong)v>>shift,65535UL);
    int r=rt[el[i]];long total=(long)m+r;if(total>2147483647L || total<(-2147483647L-1L)){atomic_or(error,2u);return;}
    int t=(int)total;out[i]=(ushort)clamp(total>-8?(total+8)/16:0L,0L,4095L);if(diag){mapped[i]=(ushort)m;residual[i]=r;sum[i]=t;}
}
/* Host proves both signed64 domains and identical input clamp ranges. Share
 * pixel loads and feature powers across Cb/Cr, not their coefficients/results. */
#define DUAL_ARGS global const ushort *y,global const ushort *cb,global const ushort *cr, \
    global const uint *p0,global const uint *p1,global const int *rt0,global const int *rt1, \
    global ushort *out0,global ushort *out1,global uint *error,global const long *table0,global const long *table1
#define DUAL_PASS y,cb,cr,p0,p1,rt0,rt1,out0,out1,error,table0,table1
void mmr64_dual_pixel(uint i,uint2 enhancement,DUAL_ARGS){
    uint3 s=(uint3)(y[i],cb[i],cr[i]);
    for(uint k=0;k<3;k++)s[k]=clamp(s[k],p0[2+2*k],p0[3+2*k]);
    long v0=as_long((ulong)p0[8]|((ulong)p0[9]<<32)),v1=as_long((ulong)p1[8]|((ulong)p1[9]<<32));
    v0+=table0[s.x];v0+=table0[1024+s.y];v0+=table0[2048+s.z];
    v1+=table1[s.x];v1+=table1[1024+s.y];v1+=table1[2048+s.z];uint xy=s.x*s.y;
    #pragma unroll
    for(uint k=3;k<7;k++){
        uint a=k==3?xy:k==4?s.x*s.z:k==5?s.y*s.z:feature_product(xy,s.z<<10);
        v0+=(long)as_int(p0[12+4*k])*(long)a;v1+=(long)as_int(p1[12+4*k])*(long)a;
        if(p0[0]>1||p1[0]>1){uint b=feature_product(a,a);
            if(p0[0]>1)v0+=(long)as_int(p0[40+4*k])*(long)b;
            if(p1[0]>1)v1+=(long)as_int(p1[40+4*k])*(long)b;
            if(p0[0]>2||p1[0]>2){uint c=feature_product(a,b);
                if(p0[0]>2)v0+=(long)as_int(p0[68+4*k])*(long)c;
                if(p1[0]>2)v1+=(long)as_int(p1[68+4*k])*(long)c;}
        }
    }
    #pragma unroll
    for(uint channel=0;channel<2;++channel){uint el=enhancement[channel];if(el>1023){atomic_or(error,1u);continue;}
        long v=channel?v1:v0;uint shift=(channel?p1[1]:p0[1])+4,m=v<0L?0u:(uint)min((ulong)v>>shift,65535UL);
        int r=channel?rt1[el]:rt0[el];long total=(long)m+r;
        if(total>2147483647L || total<(-2147483647L-1L)){atomic_or(error,2u);continue;}
        ushort value=(ushort)clamp(total>-8?(total+8)/16:0L,0L,4095L);if(channel)out1[i]=value;else out0[i]=value;
    }
}
kernel void mmr64_lut_dual(global const ushort *y,global const ushort *cb,global const ushort *cr,
    global const ushort *el0,global const ushort *el1,global const uint *p0,global const uint *p1,
    global const int *rt0,global const int *rt1,global ushort *out0,global ushort *out1,
    global uint *error,uint count,global const long *table0,global const long *table1){
    uint i=get_global_id(0);if(i>=count)return;
    mmr64_dual_pixel(i,(uint2)(el0[i],el1[i]),DUAL_PASS);
}
/* Identical horizontal interpolation,without materializing scaled EL planes. */
kernel void mmr64_lut_dual_horizontal(global const ushort *y,global const ushort *cb,global const ushort *cr,
    global const ushort *el0,global const ushort *el1,global const uint *p0,global const uint *p1,
    global const int *rt0,global const int *rt1,global ushort *out0,global ushort *out1,
    global uint *error,uint count,global const long *table0,global const long *table1,uint width){
    uint i=get_global_id(0);if(i>=count)return;uint x=i%width,row=i/width;
    uint2 enhancement=(uint2)(horizontal(el0,width/2,x,row),horizontal(el1,width/2,x,row));
    mmr64_dual_pixel(i,enhancement,DUAL_PASS);
}
#undef DUAL_ARGS
#undef DUAL_PASS
/* Experimental FP32 colour. Exact integer normalization/expansion/source
 * domain decisions; continuous PQ/matrix math is NOT claimed bit-exact. */
uint expand8(global const ushort *p,uint w,uint h,uint x,uint y){
    uint ix=x/2,iy=y/2,next=min(ix+1,w-1),top=(y&1)?iy:(iy?iy-1:0),bottom=(y&1)?min(iy+1,h-1):iy;
    uint a=(y&1)?3:1,b=4-a,left=a*p[top*w+ix]+b*p[bottom*w+ix];
    return (x&1)?left+a*p[top*w+next]+b*p[bottom*w+next]:left*2;
}
#ifndef DV_COLOUR_LUT
kernel void colour(global const ushort *yplane,global const ushort *cb,global const ushort *cr,
    global const float *params,global const uint *source,DV_COLOUR_OUTPUT o0,DV_COLOUR_OUTPUT o1,DV_COLOUR_OUTPUT o2,
    global uint2 *stats,uint w,uint h,uint policy,uint4 active){
    uint i=(uint)get_global_id(0),tid=(uint)get_local_id(0);uint count=0,bad=0;
    if(i<w*h){uint x=i%w,y=i/w,codes[3]={8u*yplane[i],expand8(cb,w/2,h/2,x,y),expand8(cr,w/2,h/2,x,y)};
        long shifted[3];for(uint k=0;k<3;k++)shifted[k]=(long)codes[k]*8192L-(long)source[9+k];
        float linear[3],lms[3],encoded[3];
        for(uint c=0;c<3;c++){long v=0;for(uint k=0;k<3;k++)v+=(long)as_int(source[c*3+k])*shifted[k];
            uint outside=v<0L || v>2199023255552L;count+=outside;if(!policy && outside)bad=1;
            float pq=max(convert_float(v)*0x1p-41f,0.0f),p=pow(pq,32.0f/2523.0f),d=2413.0f/128.0f-2392.0f/128.0f*p;
            if(d<=0.0f)bad=1;linear[c]=pow(max(p-3424.0f/4096.0f,0.0f)/d,16384.0f/2610.0f);if(!isfinite(linear[c]))bad=1;
        }
        for(uint c=0;c<3;c++){float v=0.0f;for(uint k=0;k<3;k++)v+=params[c*3+k]*linear[k];lms[c]=v;}
        for(uint c=0;c<3;c++){float v=0.0f;for(uint k=0;k<3;k++)v+=params[9+c*3+k]*lms[k];
            if(!isfinite(v) || (!policy && (v<0.0f || v>1.0f)))bad=1;
            float p=pow(max(v,0.0f),2610.0f/16384.0f);
            encoded[c]=pow((3424.0f/4096.0f+2413.0f/128.0f*p)/(1.0f+2392.0f/128.0f*p),2523.0f/32.0f);if(!isfinite(encoded[c]))bad=1;
        }
        ushort out[3];for(uint c=0;c<3;c++){float v=params[27+c];for(uint k=0;k<3;k++)v+=params[18+c*3+k]*encoded[k];
            if(!isfinite(v))bad=1;out[c]=convert_ushort(clamp(floor(v*4096.0f+0.5f),0.0f,4095.0f));}
        if(x<active.x || y<active.y || x>=active.z || y>=active.w){out[0]=0;out[1]=out[2]=2048;}
        DV_STORE_COLOUR(o0,o1,o2,i,w,out);
    }
    local uint2 scratch[256];scratch[tid]=(uint2)(count,bad);barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=(uint)get_local_size(0)/2u;stride;stride>>=1){if(tid<stride){scratch[tid].x+=scratch[tid+stride].x;scratch[tid].y|=scratch[tid+stride].y;}barrier(CLK_LOCAL_MEM_FENCE);}
    if(!tid)stats[get_group_id(0)]=scratch[0];
}
#else
#ifdef DV_COLOUR_CONSTANT_UNIFORMS
#define DV_UNIFORM constant
#else
#define DV_UNIFORM global
#endif
#ifdef DV_COLOUR_SELECTIVE
ushort3 screen_colour(private const long *pq,DV_UNIFORM const float *params,
    global const float *decode_table,global const float *encode_table,uint policy,uint outside,uint *uncertain){
    float linear[3],encoded[3];
    for(uint c=0;c<3;c++)linear[c]=decode_fast(pq[c],decode_table,uncertain);
#ifndef DV_COLOUR_SCREEN_FUSED
    float lms[3];
    for(uint c=0;c<3;c++){float v=0.0f,norm=0.0f;
        for(uint k=0;k<3;k++){float coefficient=du(params,c*3+k).x;
            v=fma(coefficient,linear[k],v);norm+=fabs(coefficient*linear[k]);}
        if(norm>0.0f && fabs(v)<norm*0.05f)*uncertain|=32u;
        lms[c]=v;
    }
#endif
    for(uint c=0;c<3;c++){float v=0.0f,norm=0.0f;
#ifdef DV_COLOUR_SCREEN_FUSED
        for(uint k=0;k<3;k++){float coefficient=du(params,30+c*3+k).x;
            v=fma(coefficient,linear[k],v);norm+=fabs(coefficient*linear[k]);}
#else
        for(uint k=0;k<3;k++){float coefficient=du(params,9+c*3+k).x;
            v=fma(coefficient,lms[k],v);norm+=fabs(coefficient*lms[k]);}
#endif
        if(!isfinite(v) || (!policy && (v<0.0f || v>1.0f)))*uncertain|=16u;
        if(norm>0.0f && fabs(v)<norm*0.05f)*uncertain|=32u;
        encoded[c]=encode_fast(v,encode_table,uncertain);
    }
    ushort3 out;
    for(uint c=0;c<3;c++){float v=du(params,27+c).x;
        for(uint k=0;k<3;k++)v=fma(du(params,18+c*3+k).x,encoded[k],v);
        if(!isfinite(v))*uncertain|=16u;
        float q=v*4096.0f,nearest=floor(q+0.5f);
        if(!outside && q>=-1.0f && q<=4096.0f && min(fabs(q-(nearest-0.5f)),fabs(q-(nearest+0.5f)))<0.01f)*uncertain|=64u;
        out[c]=convert_ushort(clamp(nearest,0.0f,4095.0f));
    }
    return out;
}
#endif
ushort3 compose_precise(private const long *pq,DV_UNIFORM const float *params,
    global const float *decode_table,global const float *encode_table,uint policy,uint outside,uint *uncertain){
    dd linear[3],lms[3],encoded[3];ushort3 out;
    for(uint c=0;c<3;c++)linear[c]=decode_lut(pq[c],decode_table,uncertain);
#ifndef DV_COLOUR_COMBINE_MATRIX
    for(uint c=0;c<3;c++){dd v=df(0.0f);for(uint k=0;k<3;k++)v=da(v,dm(du(params,c*3+k),linear[k]));lms[c]=v;}
#endif
    for(uint c=0;c<3;c++){dd v=df(0.0f);
#ifdef DV_COLOUR_COMBINE_MATRIX
        for(uint k=0;k<3;k++)v=da(v,dm(du(params,c*3+k),linear[k]));
#else
        for(uint k=0;k<3;k++)v=da(v,dm(du(params,9+c*3+k),lms[k]));
#endif
        if(!isfinite(v.x) || (!policy && (v.x<0.0f || v.x>1.0f)))*uncertain|=16u;
        encoded[c]=encode_lut(v,encode_table,uncertain);
    }
    for(uint c=0;c<3;c++){dd v=du(params,27+c);for(uint k=0;k<3;k++)v=da(v,dm(du(params,18+c*3+k),encoded[k]));
        if(!isfinite(v.x))*uncertain|=16u;dd q=dm(v,df(4096.0f));
        float nearest=floor(q.x+0.5f);dd lower=ds(q,df(nearest-0.5f)),upper=ds(q,df(nearest+0.5f));
        if(!outside && q.x>=-1.0f && q.x<=4096.0f && min(fabs(lower.x+lower.y),fabs(upper.x+upper.y))<0.00001f)*uncertain|=8u;
        float code=nearest;if(lower.x<0.0f || (lower.x==0.0f && lower.y<0.0f))code-=1.0f;
        else if(upper.x>0.0f || (upper.x==0.0f && upper.y>=0.0f))code+=1.0f;
        out[c]=convert_ushort(clamp(code,0.0f,4095.0f));
    }
    return out;
}
kernel void colour(global const ushort *yplane,global const ushort *cb,global const ushort *cr,
    DV_UNIFORM const float *params,DV_UNIFORM const uint *source,DV_COLOUR_OUTPUT o0,DV_COLOUR_OUTPUT o1,DV_COLOUR_OUTPUT o2,
    global uint2 *stats,uint w,uint h,uint policy,uint4 active,
    global const float *decode_table,global const float *encode_table,global uchar *uncertainty
#ifdef DV_COLOUR_COMPACT
    ,global uint4 *records,global uint *record_count,uint capacity
#endif
#ifdef DV_COLOUR_FAST_TABLES
    ,global const float *fast_decode_table,global const float *fast_encode_table
#endif
    ){
    uint i=(uint)get_global_id(0),tid=(uint)get_local_id(0),count=0,bad=0,uncertain=0;
    uint need=0;uint4 record=(uint4)(0);
    if(i<w*h){uint x=i%w,y=i/w,codes[3]={8u*yplane[i],expand8(cb,w/2,h/2,x,y),expand8(cr,w/2,h/2,x,y)};
        long shifted[3];for(uint k=0;k<3;k++)shifted[k]=(long)codes[k]*8192L-(long)source[9+k];
        long pq[3];uint outside=x<active.x || y<active.y || x>=active.z || y>=active.w;
        ushort3 out=(ushort3)(0,2048,2048);uint precise=1,screened=0;
        for(uint c=0;c<3;c++){long v=0;for(uint k=0;k<3;k++)v+=(long)as_int(source[c*3+k])*shifted[k];
            uint excursion=v<0L || v>2199023255552L;count+=excursion;if(!policy && excursion)bad=1;pq[c]=v;
        }
#ifdef DV_COLOUR_SELECTIVE
        if(!source[12]){uint confidence=0;
#ifdef DV_COLOUR_FAST_TABLES
            out=screen_colour(pq,params,fast_decode_table,fast_encode_table,policy,outside,&confidence);
#else
            out=screen_colour(pq,params,decode_table,encode_table,policy,outside,&confidence);
#endif
            precise=confidence!=0;}
        screened=precise;
#endif
#ifdef DV_COLOUR_COMPACT
        need=precise;record=(uint4)(i,codes[0],codes[1],codes[2]);screened=0;
#else
        if(precise)out=compose_precise(pq,params,decode_table,encode_table,policy,outside,&uncertain);
#endif
        if(outside){out[0]=0;out[1]=out[2]=2048;}
        DV_STORE_COLOUR(o0,o1,o2,i,w,out);uncertainty[i]=(uchar)(uncertain|(screened?32u:0u));
    }
#ifdef DV_COLOUR_COMPACT
#ifdef DV_COLOUR_LOCAL_COMPACT
    /* Each uncertain pixel claims a unique local ticket. One group leader
     * reserves global space, avoiding an eight-round prefix scan. Ordering
     * may vary, but each pixel keeps its own exact codes and output index. */
    local uint next,base;if(!tid)next=0u;barrier(CLK_LOCAL_MEM_FENCE);
    uint ticket=need?atomic_inc(&next):0u;barrier(CLK_LOCAL_MEM_FENCE);
    if(!tid)base=next?atomic_add(record_count,next):0u;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(need){uint slot=base+ticket;if(slot<capacity)records[slot]=record;}
#else
    local uint prefix[256];local uint base;prefix[tid]=need;barrier(CLK_LOCAL_MEM_FENCE);
    for(uint offset=1;offset<(uint)get_local_size(0);offset<<=1){uint add=tid>=offset?prefix[tid-offset]:0u;
        barrier(CLK_LOCAL_MEM_FENCE);prefix[tid]+=add;barrier(CLK_LOCAL_MEM_FENCE);}
    if(!tid){uint total=prefix[get_local_size(0)-1];base=total?atomic_add(record_count,total):0u;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if(need){uint slot=base+prefix[tid]-1u;if(slot<capacity)records[slot]=record;}
#endif
#endif
#ifdef DV_COLOUR_SUBGROUP
    uint subtotal=sub_group_reduce_add(count),subbad=sub_group_reduce_max(bad);
    local uint2 group_stats[256];
    if(get_sub_group_local_id()==0)group_stats[get_sub_group_id()]=(uint2)(subtotal,subbad);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(!tid){uint2 total=(uint2)(0);for(uint s=0;s<get_num_sub_groups();s++){total.x+=group_stats[s].x;total.y|=group_stats[s].y;}stats[get_group_id(0)]=total;}
#else
    local uint2 scratch[256];scratch[tid]=(uint2)(count,bad);barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=(uint)get_local_size(0)/2u;stride;stride>>=1){if(tid<stride){scratch[tid].x+=scratch[tid+stride].x;scratch[tid].y|=scratch[tid+stride].y;}barrier(CLK_LOCAL_MEM_FENCE);}
    if(!tid)stats[get_group_id(0)]=scratch[0];
#endif
}
#ifdef DV_COLOUR_COMPACT
kernel void colour_refine(global const ushort *yplane,global const ushort *cb,global const ushort *cr,
    DV_UNIFORM const float *params,DV_UNIFORM const uint *source,DV_COLOUR_OUTPUT o0,DV_COLOUR_OUTPUT o1,DV_COLOUR_OUTPUT o2,
    global const uint4 *records,global const uint *record_count,uint w,uint h,uint4 active,
    global const float *decode_table,global const float *encode_table,global uchar *uncertainty,uint capacity,uint policy
#ifdef DV_COLOUR_SPARSE_REPAIR
    ,global uint8 *repair_records,global uint *repair_count,uint repair_capacity
#endif
    ){
    uint dense=*record_count>capacity,count=dense?w*h:*record_count;
#ifdef DV_COLOUR_BOUNDED_REFINE
    /* Sparse count fits the bounded launch. Dense overflow uses a grid-stride
     * loop so every pixel is still refined, without a mid-pass CPU read. */
    for(uint j=(uint)get_global_id(0);j<count;j+=(uint)get_global_size(0)){
#else
    uint j=(uint)get_global_id(0);if(j>=count)return;
#endif
    uint i,codes[3];if(dense){i=j;uint x=i%w,y=i/w;codes[0]=8u*yplane[i];codes[1]=expand8(cb,w/2,h/2,x,y);codes[2]=expand8(cr,w/2,h/2,x,y);}
    else {uint4 r=records[j];i=r.x;codes[0]=r.y;codes[1]=r.z;codes[2]=r.w;}
    if(i>=w*h)return;uint x=i%w,y=i/w,outside=x<active.x || y<active.y || x>=active.z || y>=active.w,uncertain=0;
    long shifted[3],pq[3];for(uint k=0;k<3;k++)shifted[k]=(long)codes[k]*8192L-(long)source[9+k];
    for(uint c=0;c<3;c++){long v=0;for(uint k=0;k<3;k++)v+=(long)as_int(source[c*3+k])*shifted[k];pq[c]=v;}
    ushort3 out=compose_precise(pq,params,decode_table,encode_table,policy,outside,&uncertain);
    if(outside)out=(ushort3)(0,2048,2048);
    DV_STORE_COLOUR(o0,o1,o2,i,w,out);uncertainty[i]=(uchar)(uncertain|32u);
#ifdef DV_COLOUR_SPARSE_REPAIR
    if(uncertain&31u){uint slot=atomic_add(repair_count,1u);if(slot<repair_capacity)repair_records[slot]=(uint8)(i,codes[0],codes[1],codes[2],uncertain&31u,0u,0u,0u);}
#endif
#ifdef DV_COLOUR_BOUNDED_REFINE
    }
#endif
}
#endif
#endif
/* VA P010 imports are R/RG UNORM16, not UINT16. The 16-bit normalized
 * roundtrip recovers exact MSB-aligned codes; low six bits must be zero. */
/* One work item per native 2x2 luma / one chroma block. Same exact UNORM
 * recovery and low-bit validation as unpack_p010; no filtering or rounding
 * changes. A 2D launch avoids runtime division for image coordinates. */
__kernel void unpack_p010_block(read_only image2d_t y,read_only image2d_t uv,
 __global ushort *dy,__global ushort *du,__global ushort *dv,uint w,uint h,__global uint *error)
{uint x=get_global_id(0),cy=get_global_id(1);if(x>=w/2||cy>=h/2)return;
 int2 p=(int2)(x*2,cy*2);
 uint4 q=convert_uint4_rte((float4)(read_imagef(y,p).x,
  read_imagef(y,p+(int2)(1,0)).x,read_imagef(y,p+(int2)(0,1)).x,
  read_imagef(y,p+(int2)(1,1)).x)*65535.0f);
 uint2 v=convert_uint2_rte(read_imagef(uv,(int2)(x,cy)).xy*65535.0f);
 if((q.x|q.y|q.z|q.w|v.x|v.y)&63u)atomic_or(error,1u);
 uint i=cy*2*w+x*2,c=cy*(w/2)+x;
 vstore2(convert_ushort2(q.xy>>6),0,dy+i);
 vstore2(convert_ushort2(q.zw>>6),0,dy+i+w);
 du[c]=(ushort)(v.x>>6);dv[c]=(ushort)(v.y>>6);}
/* SINGLE_POLYNOMIAL_BEGIN */
__kernel void single_polynomial_p010(read_only image2d_t y,read_only image2d_t uv,
 __global const ushort *table,__global ushort *dy,__global ushort *du,
 __global ushort *dv,uint w,uint h,__global uint *error)
{
 uint x=get_global_id(0),cy=get_global_id(1);if(x>=w/2||cy>=h/2)return;
 int2 p=(int2)(2*x,2*cy);
 uint4 q=convert_uint4_rte((float4)(read_imagef(y,p).x,
   read_imagef(y,p+(int2)(1,0)).x,read_imagef(y,p+(int2)(0,1)).x,
   read_imagef(y,p+(int2)(1,1)).x)*65535.0f);
 uint2 c=convert_uint2_rte(read_imagef(uv,(int2)(x,cy)).xy*65535.0f);
 if((q.x|q.y|q.z|q.w|c.x|c.y)&63u)atomic_or(error,1u);
 q>>=6;c>>=6;uint i=cy*2*w+x*2,j=cy*(w/2)+x;
 dy[i]=table[q.x];dy[i+1]=table[q.y];dy[i+w]=table[q.z];dy[i+w+1]=table[q.w];
 du[j]=table[1024+c.x];dv[j]=table[2048+c.y];
}
/* SINGLE_POLYNOMIAL_END */
__kernel void unpack_p010(read_only image2d_t y,read_only image2d_t uv,
 __global ushort *dy,__global ushort *du,__global ushort *dv,uint w,uint h,__global uint *error)
{uint i=get_global_id(0);if(i>=w*h)return;int2 p=(int2)(i%w,i/w);
 uint q=convert_uint_rte(read_imagef(y,p).x*65535.0f);
 if(q&63u)atomic_or(error,1u);dy[i]=(ushort)(q>>6);
 if(i<w*h/4){int2 c=(int2)(i%(w/2),i/(w/2));uint2 v=convert_uint2_rte(read_imagef(uv,c).xy*65535.0f);
 if((v.x|v.y)&63u)atomic_or(error,1u);du[i]=(ushort)(v.x>>6);dv[i]=(ushort)(v.y>>6);}}
#ifdef DV_COLOUR_OUTPUT_IMAGES
__kernel void patch_colour(DV_COLOUR_OUTPUT y,DV_COLOUR_OUTPUT u,DV_COLOUR_OUTPUT v,
 __global const uint4 *patch,uint count,uint w,uint h)
{uint i=get_global_id(0);if(i<count){uint4 p=patch[i];if(p.x>=w*h)return;ushort3 out=convert_ushort3(p.yzw);DV_STORE_COLOUR(y,u,v,p.x,w,out);}}
#else
__kernel void patch_colour(__global ushort *y,__global ushort *u,__global ushort *v,
 __global const uint4 *patch,uint count)
{uint i=get_global_id(0);if(i<count){uint4 p=patch[i];y[p.x]=(ushort)p.y;u[p.x]=(ushort)p.z;v[p.x]=(ushort)p.w;}}
#endif
__kernel void pack_r16ui(__global const ushort *y,__global const ushort *u,__global const ushort *v,
 write_only image2d_t iy,write_only image2d_t iu,write_only image2d_t iv,uint w,uint h)
{uint i=get_global_id(0);if(i>=w*h)return;int2 p=(int2)(i%w,i/w);
 write_imageui(iy,p,(uint4)(y[i],0,0,1));write_imageui(iu,p,(uint4)(u[i],0,0,1));write_imageui(iv,p,(uint4)(v[i],0,0,1));}
