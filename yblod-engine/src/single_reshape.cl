/* General tuple math plus explicit CCM center-left P010/NV12 sampling routes.
 * Host validates mapping/masks; residuals and output packing remain separate. */
typedef struct {uint method,order;long bias,coefficient[3][7];} reshape_piece;
typedef struct {uint pivot_count,pivots[9];reshape_piece piece[8];} reshape_curve;
typedef struct {uint base_depth,denominator;reshape_curve curve[3];} reshape_mapping;
uint4 reshape_add(uint4 a,uint4 b)
{
 uint4 value;uint carry=0;
 for(uint i=0;i<4;++i){uint t=a[i]+b[i],u=t+carry;carry=(uint)(t<a[i])|(uint)(u<t);value[i]=u;}
 return value;
}
uint4 reshape_product(long coefficient,uint feature)
{
 ulong bits=as_ulong(coefficient),lo=(ulong)(uint)bits*feature;
 ulong hi=(bits>>32)*feature+(lo>>32);
 uint top=(uint)(hi>>32)-(coefficient<0?feature:0);
 return (uint4)((uint)lo,(uint)hi,top,(top&0x80000000u)?0xffffffffu:0);
}
ushort3 reshape_tuple(uint3 input,global const reshape_mapping *mapping,
 uint components,global uint *error,uint4 narrow_mask)
{
 if(sizeof(reshape_mapping)!=4544){atomic_or(error,2u);return (ushort3)(0);}
 uint bits=mapping->base_depth,q=2*bits,s[3]={input.x,input.y,input.z},piece[3];
 for(uint c=0;c<3;++c){if(s[c]>=(1u<<bits)){atomic_or(error,1u);return (ushort3)(0);}
  global const reshape_curve *v=&mapping->curve[c];piece[c]=v->pivot_count-2;
  for(uint p=0;p+1<v->pivot_count;++p)if(s[c]<v->pivots[p+1]){piece[c]=p;break;}
  s[c]=clamp(s[c],v->pivots[0],v->pivots[v->pivot_count-1]);
 }
 uint feature[3][7];
#ifdef DV_RESHAPE_LAZY_FEATURES
 uint required_order=0;
 for(uint c=0;c<3;++c)if((components&(1u<<c))&&mapping->curve[c].piece[piece[c]].method)
  required_order=max(required_order,mapping->curve[c].piece[piece[c]].order);
#else
 uint required_order=3;
#endif
 for(uint c=0;c<3;++c)feature[0][c]=s[c]<<bits;
 feature[0][3]=s[0]*s[1];feature[0][4]=s[0]*s[2];feature[0][5]=s[1]*s[2];
 feature[0][6]=(uint)(((ulong)feature[0][3]*feature[0][2])>>q);
 if(required_order>=2){
  for(uint c=0;c<3;++c)feature[1][c]=s[c]*s[c];
  for(uint k=3;k<7;++k)feature[1][k]=(uint)(((ulong)feature[0][k]*feature[0][k])>>q);
 }
 if(required_order>=3)for(uint k=0;k<7;++k)feature[2][k]=(uint)(((ulong)feature[0][k]*feature[1][k])>>q);
 ushort3 result=(ushort3)(0);
 for(uint c=0;c<3;++c){if(!(components&(1u<<c)))continue;
  global const reshape_piece *p=&mapping->curve[c].piece[piece[c]];
#ifdef DV_RESHAPE_NARROW
  if(narrow_mask[c]&(1u<<piece[c])){
   long value=p->bias*(long)(1u<<q);
   if(!p->method){value+=p->coefficient[0][0]*(long)(s[c]<<bits);
    if(p->order==2)value+=p->coefficient[0][1]*(long)(s[c]*s[c]);
   }else for(uint order=0;order<p->order;++order)for(uint k=0;k<7;++k)
    value+=p->coefficient[order][k]*(long)feature[order][k];
   uint mapped=value<0?0u:(uint)min((ulong)value>>(mapping->denominator+q-16),65535UL);
   result[c]=(ushort)min((mapped+8)>>4,4095u);continue;
  }
#endif
  uint4 value=reshape_product(p->bias,1u<<q);
  if(!p->method){value=reshape_add(value,reshape_product(p->coefficient[0][0],s[c]<<bits));
   if(p->order==2)value=reshape_add(value,reshape_product(p->coefficient[0][1],s[c]*s[c]));
  }else for(uint order=0;order<p->order;++order)for(uint k=0;k<7;++k)
   value=reshape_add(value,reshape_product(p->coefficient[order][k],feature[order][k]));
  uint mapped=0;
  if(!(value.w&0x80000000u))mapped=value.w||value.z?65535u:
   (uint)min((((ulong)value.y<<32)|value.x)>>(mapping->denominator+q-16),65535UL);
  result[c]=(ushort)min((mapped+8)>>4,4095u);
 }
 return result;
}
kernel void single_reshape_aligned(global const ushort *y,global const ushort *cb,
 global const ushort *cr,global const reshape_mapping *mapping,
 global ushort *out_y,global ushort *out_cb,global ushort *out_cr,
 uint count,global uint *error
#ifdef DV_RESHAPE_NARROW
 ,uint4 narrow_mask
#endif
 )
{
 uint i=get_global_id(0);if(i>=count)return;
#ifndef DV_RESHAPE_NARROW
 uint4 narrow_mask=(uint4)(0);
#endif
 ushort3 result=reshape_tuple((uint3)(y[i],cb[i],cr[i]),mapping,7u,error,narrow_mask);
 out_y[i]=result.x;out_cb[i]=result.y;out_cr[i]=result.z;
}

#if defined(DV_RESHAPE_P010) && defined(DV_RESHAPE_NV12)
#error Select exactly one source storage format
#endif
#if defined(DV_RESHAPE_P010) || defined(DV_RESHAPE_NV12)
#ifdef DV_RESHAPE_NV12
#define RESHAPE_DEPTH 8u
#define RESHAPE_SCALE 255.0f
#define RESHAPE_SHIFT 0u
#define RESHAPE_PADDING 0u
#define RESHAPE_CENTER_ENTRY single_reshape_nv12_center_left
#define RESHAPE_FULL_ENTRY single_reshape_nv12_422_candidate
#else
#define RESHAPE_DEPTH 10u
#define RESHAPE_SCALE 65535.0f
#define RESHAPE_SHIFT 6u
#define RESHAPE_PADDING 63u
#define RESHAPE_CENTER_ENTRY single_reshape_p010_center_left
#define RESHAPE_FULL_ENTRY single_reshape_p010_422_candidate
#endif
uint4 reshape_fetch(read_only image2d_t image,int2 position)
{
 uint4 raw=convert_uint4_rte(read_imagef(image,position)*RESHAPE_SCALE);
 /* VA's R/RG views have synthetic unused channels: check only actual samples. */
 return raw;
}
uint reshape_y(read_only image2d_t image,int2 position,global uint *error)
{
 uint raw=reshape_fetch(image,position).x;
 if(raw&RESHAPE_PADDING)atomic_or(error,1u);return raw>>RESHAPE_SHIFT;
}
/* CCM 001 5.4.2.3.3: horizontal [1,2,1]/4 with row rounding,
 * then rounded mean of rows 2y and 2y+1. No BL chroma phase conversion.
 * Host must restrict this entry point to polynomial luma, matching storage and
 * this guide policy. It is not a top-left-filter conformance claim. */
kernel void RESHAPE_CENTER_ENTRY(read_only image2d_t y,read_only image2d_t uv,
 global const reshape_mapping *mapping,global ushort *out_y,
 global ushort *out_cb,global ushort *out_cr,uint w,uint h,
 global uint *error,uint4 narrow_mask
#ifdef DV_RESHAPE_LUMA_TABLE
 ,global const ushort *luma_table
#endif
 )
{
 uint cx=get_global_id(0),cy=get_global_id(1);if(cx>=w/2||cy>=h/2)return;
 if(mapping->base_depth!=RESHAPE_DEPTH){atomic_or(error,2u);return;}
 uint x=2*cx,row=2*cy;uint2 raw=reshape_fetch(uv,(int2)(cx,cy)).xy;
 if((raw.x|raw.y)&RESHAPE_PADDING)atomic_or(error,1u);uint2 chroma=raw>>RESHAPE_SHIFT;
 uint guide_rows[2];
 for(uint dy=0;dy<2;++dy){
  uint a=reshape_y(y,(int2)(x,row+dy),error);
  uint b=reshape_y(y,(int2)(x+1,row+dy),error);
  uint left=reshape_y(y,(int2)(x?x-1:0,row+dy),error);
  guide_rows[dy]=(left+2*a+b+2)>>2;
  uint i=(row+dy)*w+x;
#ifdef DV_RESHAPE_LUMA_TABLE
  out_y[i]=luma_table[a];out_y[i+1]=luma_table[b];
#else
  out_y[i]=reshape_tuple((uint3)(a,chroma),mapping,1u,error,narrow_mask).x;
  out_y[i+1]=reshape_tuple((uint3)(b,chroma),mapping,1u,error,narrow_mask).x;
#endif
 }
 uint guide=(guide_rows[0]+guide_rows[1]+1)>>1;
 ushort3 result=reshape_tuple((uint3)(guide,chroma),mapping,6u,error,narrow_mask);
 uint i=cy*(w/2)+cx;out_cb[i]=result.y;out_cr[i]=result.z;
}
#ifdef DV_RESHAPE_ORDER2_CHROMA
/* Caller validates dv_single_reshape_order2_chroma for every mapping.
 * Identical Q20 features and signed64 accumulation to the guarded path. */
ushort3 reshape_order2_chroma(uint3 input,global const reshape_mapping *mapping,
 global uint *error)
{
 if(sizeof(reshape_mapping)!=4544){atomic_or(error,2u);return (ushort3)(0);}
 if(any(input>1023u)){atomic_or(error,1u);return (ushort3)(0);}
 uint s[3];
 for(uint c=0;c<3;++c){global const reshape_curve *v=&mapping->curve[c];
  s[c]=clamp(input[c],v->pivots[0],v->pivots[v->pivot_count-1]);}
 uint f[2][7];
 for(uint c=0;c<3;++c){f[0][c]=s[c]<<10;f[1][c]=s[c]*s[c];}
 f[0][3]=s[0]*s[1];f[0][4]=s[0]*s[2];f[0][5]=s[1]*s[2];
 f[0][6]=(uint)(((ulong)f[0][3]*f[0][2])>>20);
 for(uint k=3;k<7;++k)f[1][k]=(uint)(((ulong)f[0][k]*f[0][k])>>20);
 ushort3 result=(ushort3)(0);
 #pragma unroll
 for(uint c=1;c<3;++c){global const reshape_piece *p=&mapping->curve[c].piece[0];
  long value=p->bias*1048576L;
  #pragma unroll
  for(uint o=0;o<2;++o){
   #pragma unroll
   for(uint k=0;k<7;++k)value+=p->coefficient[o][k]*(long)f[o][k];
  }
  uint mapped=value<0?0u:(uint)min((ulong)value>>(mapping->denominator+4),65535UL);
  result[c]=(ushort)min((mapped+8)>>4,4095u);
 }
 return result;
}
#endif
#ifdef DV_RESHAPE_422_EXPERIMENT
/* Experimental source interpolation before full-height chroma mapping.
 * Selected only by the experimental full-height backend/renderer flag. */
kernel void RESHAPE_FULL_ENTRY(read_only image2d_t y,read_only image2d_t uv,
 global const reshape_mapping *mapping,global ushort *out_y,
 global ushort *out_cb,global ushort *out_cr,uint w,uint h,
 global uint *error,uint4 narrow_mask
#ifdef DV_RESHAPE_LUMA_TABLE
 ,global const ushort *luma_table
#endif
 ,uint policy)
{
 uint cx=get_global_id(0),row=get_global_id(1);if(cx>=w/2||row>=h)return;
 if(mapping->base_depth!=RESHAPE_DEPTH){atomic_or(error,2u);return;}
 if(policy>1){atomic_or(error,2u);return;}
 uint cy=row/2,top,bottom,weight;
 if(policy==1){top=cy;bottom=min(cy+1,h/2-1);weight=(row&1)?2:4;}
 else if(row&1){top=cy;bottom=min(cy+1,h/2-1);weight=3;}
 else {top=cy?cy-1:0;bottom=cy;weight=1;}
 uint2 first=reshape_fetch(uv,(int2)(cx,top)).xy;
 uint2 second=reshape_fetch(uv,(int2)(cx,bottom)).xy;
 if((first.x|first.y|second.x|second.y)&RESHAPE_PADDING)atomic_or(error,1u);
 uint2 chroma=(weight*(first>>RESHAPE_SHIFT)+(4-weight)*(second>>RESHAPE_SHIFT)+2)>>2;
 uint x=2*cx,a=reshape_y(y,(int2)(x,row),error);
 uint b=reshape_y(y,(int2)(x+1,row),error);
 uint left=reshape_y(y,(int2)(x?x-1:0,row),error);
 uint guide=(left+2*a+b+2)>>2,i=row*w+x;
#ifdef DV_RESHAPE_LUMA_TABLE
 out_y[i]=luma_table[a];out_y[i+1]=luma_table[b];
#else
 out_y[i]=reshape_tuple((uint3)(a,chroma),mapping,1u,error,narrow_mask).x;
 out_y[i+1]=reshape_tuple((uint3)(b,chroma),mapping,1u,error,narrow_mask).x;
#endif
#ifdef DV_RESHAPE_ORDER2_CHROMA
 ushort3 result=reshape_order2_chroma((uint3)(guide,chroma),mapping,error);
#else
 ushort3 result=reshape_tuple((uint3)(guide,chroma),mapping,6u,error,narrow_mask);
#endif
 i=row*(w/2)+cx;out_cb[i]=result.y;out_cr[i]=result.z;
}
#endif
#endif
