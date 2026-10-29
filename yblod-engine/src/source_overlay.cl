/* Coverage-alpha premultiplied sRGB8 GUI -> current source-domain colour.
 * The movie and HDMI metadata are deliberately not touched by this pass. */
float3 overlay_matrix(constant float *m, float3 v)
{
#pragma OPENCL FP_CONTRACT OFF
    return (float3)(m[0]*v.x+m[1]*v.y+m[2]*v.z,
                    m[3]*v.x+m[4]*v.y+m[5]*v.z,
                    m[6]*v.x+m[7]*v.y+m[8]*v.z);
}
int overlay_source_colour(uchar4 p, constant float *c, private float4 *out)
{
#pragma OPENCL FP_CONTRACT OFF
    if(!p.w){*out=(float4)(0);return 0;}
    if(any(p.xyz>(uchar3)(p.w)))return -1;
    float3 s=convert_float3(p.xyz)/(float)p.w;
    float3 linear=select(pow((s+0.055f)/1.055f,(float3)(2.4f)),
                          s/12.92f,s<=(float3)(0.04045f))*(c[30]/10000.0f);
    float3 common=overlay_matrix(c+18,linear);
    float3 source_linear=overlay_matrix(c+9,common);
    if(any(!isfinite(source_linear))||any(source_linear<(float3)(0))||
       any(source_linear>(float3)(1)))return -1;
    float3 power=pow(source_linear,(float3)(2610.0f/16384.0f));
    float3 pq=pow((3424.0f/4096.0f+(2413.0f/128.0f)*power)/
                   (1.0f+(2392.0f/128.0f)*power),(float3)(2523.0f/32.0f));
    float3 codes=(overlay_matrix(c,pq)+vload3(0,c+27))*4096.0f;
    if(any(!isfinite(codes))||any(codes<(float3)(0))||any(codes>(float3)(4095)))return -1;
    *out=(float4)(codes,(float)p.w/255.0f);
    return 0;
}

/* Offscreen oracle comparison only. Live packing can call the helper directly
 * without allocating full-frame float colour and per-pixel status buffers. */
#ifdef OVERLAY_COLOUR_TEST
kernel void source_overlay_colour(global const uchar4 *gui,
                                  constant float *c, uint count,
                                  global float4 *out, global int *status)
{
    uint i=get_global_id(0);
    if(i>=count)return;
    float4 value;
    int result=overlay_source_colour(gui[i],c,&value);
    if(!result)out[i]=value;
    status[i]=result;
}
#endif
