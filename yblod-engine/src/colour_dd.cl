/* Experimental float-pair arithmetic and 4-point LUT interpolation. There is
 * no unconditional accuracy proof: close rounding/domain cases are explicitly
 * marked for CPU correction, and broader validation remains required. */
#pragma OPENCL FP_CONTRACT OFF
typedef float2 dd;
dd norm(float a,float b){float s=a+b;return (dd)(s,b-(s-a));}
dd df(float a){return (dd)(a,0.0f);}
dd da(dd a,dd b){float s=a.x+b.x,v=s-a.x,t=((b.x-v)+(a.x-(s-v)))+a.y+b.y;return norm(s,t);}
dd dn(dd a){return -a;}
dd ds(dd a,dd b){return da(a,dn(b));}
dd dm(dd a,dd b){float p=a.x*b.x,e=fma(a.x,b.x,-p)+a.x*b.y+a.y*b.x+a.y*b.y;return norm(p,e);}
dd dv(dd a,dd b){float q=a.x/b.x;dd r=ds(a,dm(b,df(q)));return da(df(q),df((r.x+r.y)/b.x));}
dd dl(global const float *p,uint i){return (dd)(p[2*i],p[2*i+1]);}
#ifdef DV_COLOUR_CONSTANT_UNIFORMS
dd du(constant const float *p,uint i){return (dd)(p[2*i],p[2*i+1]);}
#else
dd du(global const float *p,uint i){return dl(p,i);}
#endif
dd cubic(global const float *p,uint base,dd t){
    float4 hi=vload4(0,p+2*base),lo=vload4(0,p+2*base+4);
    dd a=(dd)(hi.x,lo.x),b=(dd)(hi.y,lo.y),c=(dd)(hi.z,lo.z),d=(dd)(hi.w,lo.w),u=da(t,df(1.0f));
    return da(a,dm(u,da(b,dm(ds(u,df(1.0f)),da(c,dm(ds(u,df(2.0f)),d))))));
}
dd decode_lut(long v,global const float *table,uint *uncertain){
    if(v<=0L)return df(0.0f);
    if(v>((long)DV_DECODE_LAST<<25)){*uncertain|=1u;return df(0.0f);}
    if(v<(4L<<25))*uncertain|=2u;
    uint index=(uint)(v>>25);
#ifdef DV_COLOUR_U32_SPLIT
    uint rem=(uint)v&0x1ffffffu;float hi=convert_float(rem);
    dd t=(dd)(hi*0x1p-25f,convert_float((int)rem-convert_int(hi))*0x1p-25f);
#else
    long rem=v-((long)index<<25);float hi=convert_float(rem);
    dd t=(dd)(hi*0x1p-25f,convert_float(rem-(long)hi)*0x1p-25f);
#endif
    dd out=cubic(table,index*4u,t);if(out.x<0.0f)out=df(0.0f);return out;
}
dd encode_lut(dd v,global const float *table,uint *uncertain){
    if(v.x<=0.0f)return dl(table,0);
#ifdef DV_COLOUR_BIT_INDEX
    uint bits=as_uint(v.x);int exp=(int)((bits>>23)&255u)-127;
    if(exp< -80 || exp>64){*uncertain|=4u;return dl(table,0);}
    dd m=(dd)(as_float((bits&0x7fffffu)|0x3f800000u),v.y*as_float((uint)(127-exp)<<23));
#else
    int exp;frexp(v.x,&exp);--exp;
    if(exp< -80 || exp>64){*uncertain|=4u;return dl(table,0);}
    dd m=(dd)(ldexp(v.x,-exp),ldexp(v.y,-exp));
#endif
    int index=clamp(convert_int_rtn((m.x-1.0f)*(float)DV_ENCODE_STEPS),0,DV_ENCODE_STEPS-1);
    dd t=dm(ds(m,df(1.0f+(float)index/(float)DV_ENCODE_STEPS)),df((float)DV_ENCODE_STEPS));
    return cubic(table,1u+4u*((uint)(exp+80)*DV_ENCODE_STEPS+(uint)index),t);
}
#ifdef DV_COLOUR_SELECTIVE
/* Screening approximation only. Ambiguous output/domain/cancellation cases
 * MUST be re-evaluated through the existing float-pair path. Its confidence
 * margins are exploratory, NOT a conformance-grade interval proof. */
float cubic_fast(global const float *table,uint base,float t){
#ifdef DV_COLOUR_FAST_TABLES
    float4 hi=vload4(0,table+base);
#else
    float4 hi=vload4(0,table+2*base);
#endif
    float u=t+1.0f,a=hi.x,b=hi.y,c=hi.z,d=hi.w;
    return fma(u,fma(u-1.0f,fma(u-2.0f,d,c),b),a);
}
float decode_fast(long v,global const float *table,uint *uncertain){
    if(v<=0L)return 0.0f;
    if(v>((long)DV_DECODE_LAST<<25)){*uncertain|=1u;return 0.0f;}
    if(v<(4L<<25))*uncertain|=2u;
    uint index=(uint)(v>>25),rem=(uint)v&0x1ffffffu;
    return max(cubic_fast(table,index*4u,convert_float(rem)*0x1p-25f),0.0f);
}
float encode_fast(float v,global const float *table,uint *uncertain){
    if(v<=0.0f)return table[0];
    uint bits=as_uint(v);int exp=(int)((bits>>23)&255u)-127;
    if(exp< -80 || exp>64){*uncertain|=4u;return table[0];}
    float m=as_float((bits&0x7fffffu)|0x3f800000u),position=(m-1.0f)*(float)DV_ENCODE_STEPS;
    uint index=min(convert_uint_rtn(position),(uint)DV_ENCODE_STEPS-1u);
    return cubic_fast(table,1u+4u*((uint)(exp+80)*DV_ENCODE_STEPS+index),position-(float)index);
}
#endif
