/* PRIVATE experimental source420 -> RGB8 tunnel. Integer bit packing.
 * Trial vertical interpolation matches the independently checked static path.
 * Metadata is already packetized on CPU and stays bound to this frame.
 */
#ifdef SOURCE_OVERLAY
/* GUI arithmetic controls stay scoped to the helper, not movie resizing. */
#include "source_overlay.cl"
float4 source_gui_colour(read_only image2d_t gui, int x, uint y, uint w, uint h,
                  uint flip, constant float *coeff, volatile global int *error
#if defined(SOURCE_OVERLAY_GRAY_LUT) && !defined(SOURCE_OVERLAY_CACHED)
                  ,global const float4 *gray
#endif
                  )
{
 const sampler_t sampler=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE|CLK_FILTER_NEAREST;
 int2 point=(int2)(clamp(x,0,(int)w-1),(int)(flip?h-1-y:y));
 uchar4 rgba=convert_uchar4_sat_rte(read_imagef(gui,sampler,point)*255.0f);
 float4 value=(float4)(0);
#if defined(SOURCE_OVERLAY_GRAY_LUT) && !defined(SOURCE_OVERLAY_CACHED)
 if(rgba.x==rgba.y&&rgba.y==rgba.z){
  value=gray[((uint)rgba.w<<8)|rgba.x];
  if(value.w<0){atomic_or(error,1);value=(float4)(0);}
  return value;
 }
#endif
 if(overlay_source_colour(rgba,coeff,&value))atomic_or(error,1);
 return value;
}
#ifdef SOURCE_OVERLAY_GRAY_LUT
kernel void source_gui_gray_prepare(constant float *coeff, global float4 *gray)
{
 uint i=get_global_id(0);if(i>=65536)return;
 uchar v=(uchar)(i&255),a=(uchar)(i>>8);float4 value=(float4)(0);
 if(overlay_source_colour((uchar4)(v,v,v,a),coeff,&value))value.w=-1;
 gray[i]=value;
}
#endif
#ifdef SOURCE_OVERLAY_CACHED
kernel void source_gui_prepare(read_only image2d_t gui, constant float *coeff,
                              uint flip, volatile global int *error,
                              global float4 *colours, uint w, uint h
#ifdef SOURCE_OVERLAY_GRAY_LUT
                              ,global const float4 *gray
#endif
                              )
{
 uint x=get_global_id(0),y=get_global_id(1);
 if(x>=w||y>=h)return;
#ifdef SOURCE_OVERLAY_GRAY_LUT
 const sampler_t sampler=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE|CLK_FILTER_NEAREST;
 uchar4 p=convert_uchar4_sat_rte(read_imagef(gui,sampler,(int2)(x,flip?h-1-y:y))*255.0f);
 float4 value=(float4)(0);
 if(p.x==p.y&&p.y==p.z){value=gray[((uint)p.w<<8)|p.x];if(value.w<0){atomic_or(error,1);value=(float4)(0);}}
 else if(overlay_source_colour(p,coeff,&value))atomic_or(error,1);
 colours[y*w+x]=value;
#else
 colours[y*w+x]=source_gui_colour(gui,(int)x,y,w,h,flip,coeff,error);
#endif
}
float4 source_gui(global const float4 *gui, int x, uint y, uint w, uint h,
                  uint flip, constant float *coeff, volatile global int *error)
{
 return gui[y*w+(uint)clamp(x,0,(int)w-1)];
}
#else
#ifdef SOURCE_OVERLAY_GRAY_LUT
#define source_gui(g,x,y,w,h,f,c,e) source_gui_colour(g,x,y,w,h,f,c,e,gui_gray)
#else
#define source_gui source_gui_colour
#endif
#endif
#endif
#if defined(SOURCE_422) && (defined(SOURCE_SCALED) || defined(SOURCE_SITED))
#error Experimental SOURCE_422 requires native-size full-height planes, not420 siting/resize
#endif
#if defined(SOURCE_RESIZE_MAPS) && !defined(SOURCE_SCALED)
#error SOURCE_RESIZE_MAPS requires SOURCE_SCALED
#endif
#ifdef SOURCE_SCALED
#if !defined(SOURCE_PLACED) || !defined(SOURCE_SITED)
#error SOURCE_SCALED requires explicit placement and chroma siting
#endif
/* Experimental reconstruction-domain resize. Never interpolate tunnel bytes.
 * No-resize frames deliberately retain the original integer path below. */
uint source_linear(global const ushort *plane,uint w,uint h,int nx,int ny,uint dx,uint dy)
{
 /* Split integer position and fractional weight before float conversion.
  * Subtracting two large FP32 coordinates loses accuracy at 4K edges. */
 uint px=(uint)clamp(nx,0,(int)((w-1)*dx)),py=(uint)clamp(ny,0,(int)((h-1)*dy));
 uint x=px/dx,y=py/dy,x1=min(x+1,w-1),y1=min(y+1,h-1);
 float fx=(float)(px%dx)/(float)dx,fy=(float)(py%dy)/(float)dy;
 float a=(float)plane[y*w+x]*(1.0f-fx)+(float)plane[y*w+x1]*fx;
 float b=(float)plane[y1*w+x]*(1.0f-fx)+(float)plane[y1*w+x1]*fx;
 return convert_uint(floor(a*(1.0f-fy)+b*fy+0.5f));
}
#ifdef SOURCE_RESIZE_MAPS
uint4 source_axis_map(int numerator,uint denominator,uint count)
{
 uint p=(uint)clamp(numerator,0,(int)((count-1)*denominator));
 uint lo=p/denominator;
 float weight=(float)(p%denominator)/(float)denominator;
 return (uint4)(lo,min(lo+1,count-1),as_uint(weight),0);
}
/* Rebuild only when dimensions or siting change; no frame pixels are read. */
kernel void source_resize_prepare(uint2 source,uint2 dest,uint top_left,
 global uint4 *ymap)
{
 uint i=get_global_id(0);
 if(i<dest.y){int numerator=(int)((2*i+1)*source.y)-(int)dest.y;
  ymap[2*i]=source_axis_map(numerator,2*dest.y,source.y);
  ymap[2*i+1]=source_axis_map(numerator-(top_left?0:(int)dest.y),4*dest.y,source.y/2);
 }
}
uint source_vertical_mapped(global const ushort *plane,uint w,uint x,uint4 y)
{
 float fy=as_float(y.z);
 float a=(float)plane[y.x*w+x],b=(float)plane[y.y*w+x];
 return convert_uint(floor(a*(1.0f-fy)+b*fy+0.5f));
}
#endif
#endif
kernel void source_tunnel(global const ushort *yplane,
 global const ushort *cb,global const ushort *cr,
 global const uchar *packets,uint packet_count,uint w,uint h,
 uint4 active,uint4 black,
#ifdef SOURCE_GL
 write_only image2d_t output
#else
 global uint *output
#endif
#ifdef SOURCE_PLACED
 ,uint4 placement
#endif
#ifdef SOURCE_SITED
 ,uint top_left
#endif
#ifdef SOURCE_SCALED
 ,uint2 source_size
#endif
#ifdef SOURCE_RESIZE_MAPS
 ,global const uint4 *ymap
#endif
#ifdef SOURCE_OVERLAY
#ifdef SOURCE_OVERLAY_CACHED
 ,global const float4 *gui
#else
 ,read_only image2d_t gui
#endif
 ,constant float *gui_coeff,uint gui_flip,
 volatile global int *gui_error
#if defined(SOURCE_OVERLAY_GRAY_LUT) && !defined(SOURCE_OVERLAY_CACHED)
 ,global const float4 *gui_gray
#endif
#endif
 )
{
 uint sx=get_global_id(0),y=get_global_id(1);
#if defined(SOURCE_OVERLAY_SHARED) && defined(SOURCE_OVERLAY) && defined(SOURCE_PACK_PAIR)
 if(sx*2>=w||y>=h)return;
 /* Both output pixels use the same three GUI samples for chroma. */
 int gui_center=(int)(sx*2);
 float4 shared_left=source_gui(gui,gui_center-1,y,w,h,gui_flip,gui_coeff,gui_error);
 float4 shared_middle=source_gui(gui,gui_center,y,w,h,gui_flip,gui_coeff,gui_error);
 float4 shared_right=source_gui(gui,gui_center+1,y,w,h,gui_flip,gui_coeff,gui_error);
#endif
#ifdef SOURCE_PACK_PAIR
#pragma unroll
 for(uint lane=0;lane<2;++lane){uint x=sx*2+lane;
#else
 uint x=sx;
#endif
 if(x>=w||y>=h)return;
 uint i=y*w+x;
#ifdef SOURCE_PLACED
 uint dw=placement.z,dh=placement.w,lx=x-placement.x,ly=y-placement.y;
 uint sw=dw,sh=dh;
 uint inside=x>=placement.x&&y>=placement.y&&lx<dw&&ly<dh;
#ifdef SOURCE_SCALED
 sw=source_size.x;sh=source_size.y;
#endif
#else
 uint sw=w,sh=h,lx=x,ly=y,inside=1;
#endif
 uint yy=black.x,c=(x&1)?black.z:black.y;
 if(inside){
#ifdef SOURCE_SCALED
 if(sw!=dw||sh!=dh){
#ifdef SOURCE_RESIZE_MAPS
  if(sw==dw){
   yy=source_vertical_mapped(yplane,sw,lx,ymap[2*ly]);
   c=source_vertical_mapped((x&1)?cr:cb,sw/2,lx/2,ymap[2*ly+1]);
  }else
#endif
  {
  int px=(int)((2*lx+1)*sw)-(int)dw;
  int py=(int)((2*ly+1)*sh)-(int)dh;
  /* Both members of a 422 pair use the same left-sited chroma position. */
  int cx=(int)((2*(lx&~1u)+1)*sw)-(int)dw;
  int cy=py-(top_left?0:(int)dh);
  yy=source_linear(yplane,sw,sh,px,py,2*dw,2*dh);
  c=source_linear((x&1)?cr:cb,sw/2,sh/2,cx,cy,4*dw,4*dh);
  }
 }else
#endif
 {
#ifdef SOURCE_422
 /* Explicit full-height reconstructed chroma: never vertically filter again. */
 c=((x&1)?cr:cb)[ly*(sw/2)+lx/2];yy=yplane[ly*sw+lx];
#else
 uint cy=ly/2,top=(ly&1)?cy:(cy?cy-1:0),
      bottom=(ly&1)?min(cy+1,sh/2-1):cy,a=(ly&1)?3:1;
 global const ushort *chroma=(x&1)?cr:cb;
#ifdef SOURCE_SITED
 if(top_left){top=cy;bottom=min(cy+1,sh/2-1);a=(ly&1)?2:4;}
#endif
 c=(a*chroma[top*(sw/2)+lx/2]+(4-a)*chroma[bottom*(sw/2)+lx/2]+2)/4;
 yy=yplane[ly*sw+lx];
#endif
 }}
 if(x<active.x||y<active.y||x>=active.z||y>=active.w){yy=black.x;c=(x&1)?black.z:black.y;}
#ifdef SOURCE_OVERLAY
 /* GUI-only source-domain blend. Cosited chroma uses a symmetric 1:2:1
  * coverage filter; this does not replace any movie reconstruction filter. */
#if defined(SOURCE_OVERLAY_SHARED) && defined(SOURCE_PACK_PAIR)
 float4 gui_value=(x&1)?shared_right:shared_middle;
#else
 float4 gui_value=source_gui(gui,(int)x,y,w,h,gui_flip,gui_coeff,gui_error);
#endif
 if(gui_value.w>0)yy=convert_uint(floor((float)yy*(1.0f-gui_value.w)+gui_value.x*gui_value.w+0.5f));
#if defined(SOURCE_OVERLAY_SHARED) && defined(SOURCE_PACK_PAIR)
 float4 left=shared_left,middle=shared_middle,right=shared_right;
#else
 int center=(int)(x&~1u);
 float4 left=source_gui(gui,center-1,y,w,h,gui_flip,gui_coeff,gui_error);
 float4 middle=(x&1)?source_gui(gui,center,y,w,h,gui_flip,gui_coeff,gui_error):gui_value;
 float4 right=(x&1)?gui_value:source_gui(gui,center+1,y,w,h,gui_flip,gui_coeff,gui_error);
#endif
 float coverage=(left.w+2.0f*middle.w+right.w)*0.25f;
 if(coverage>0){
  float contribution=(x&1)?left.z*left.w+2.0f*middle.z*middle.w+right.z*right.w:
                           left.y*left.w+2.0f*middle.y*middle.w+right.y*right.w;
  c=convert_uint(floor((float)c*(1.0f-coverage)+contribution*0.25f+0.5f));
 }
#endif
 if(i<packet_count*3072){uint bit_index=i%1024;
  uint bit=(packets[(i/3072)*128+bit_index/8]>>(7-bit_index%8))&1;
  c=(c&4094)|(bit^((popcount(yy)+popcount(c>>1))&1));}
 uint r=c>>4,g=yy>>4,b=((c&15)<<4)|(yy&15);
#ifdef SOURCE_GL
#ifdef SOURCE_SCANOUT_BGRA
 /* Qualified N150 imported linear ARGB8888 allocation exposes raw RGBA CL
  * stores. Populate its physical BGRA bytes, not a sampled GL colour value. */
 write_imagef(output,(int2)(x,y),convert_float4((uint4)(b,g,r,255))/255.0f);
#else
 write_imagef(output,(int2)(x,y),convert_float4((uint4)(r,g,b,255))/255.0f);
#endif
#else
 output[i]=b|(g<<8)|(r<<16)|0xff000000u;
#endif
#ifdef SOURCE_PACK_PAIR
 }
#endif
}
