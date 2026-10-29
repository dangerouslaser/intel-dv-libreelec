/* Exact cache invalidation regression, including an intentionally stale key. */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(int argc,char **argv)
{
    assert(argc==2);cl_platform_id platforms[16];cl_uint count=0;
    assert(!clGetPlatformIDs(16,platforms,&count));cl_device_id device=NULL;
    for(unsigned i=0;i<count&&i<16&&!device;++i)
        if(clGetDeviceIDs(platforms[i],CL_DEVICE_TYPE_GPU,1,&device,NULL))device=NULL;
    assert(device);cl_int err;cl_context context=clCreateContext(NULL,1,&device,NULL,NULL,&err);
    assert(context&&!err);cl_command_queue queue=clCreateCommandQueue(context,device,0,&err);assert(queue&&!err);
    const char *source="#include \"source_tunnel.cl\"\n"
        "kernel void sparse_check(global const float4 *colours, global const uint *gui_pixels,"
        "constant float *coeff, global int *error, uint w, uint h) {"
        "uint x=get_global_id(0),y=get_global_id(1);"
        "for(int dx=-1;dx<=1;++dx){int xx=(int)x+dx;"
        "float4 a=source_gui(colours,xx,y,w,h,0,coeff,error);"
        "float4 b=colours[y*w+(uint)clamp(xx,0,(int)w-1)];"
        "if(any(as_uint4(a)!=as_uint4(b)))atomic_or(error,4);}}\n";
    cl_program program=clCreateProgramWithSource(context,1,&source,NULL,&err);assert(program&&!err);
    char options[4096];snprintf(options,sizeof(options),"-cl-std=CL1.2 -D SOURCE_OVERLAY -D SOURCE_OVERLAY_CACHED -D SOURCE_OVERLAY_PIXEL_CACHE -D SOURCE_OVERLAY_SPARSE -I%s",argv[1]);
    if(clBuildProgram(program,1,&device,options,NULL,NULL)){
        char log[8192];clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,sizeof(log),log,NULL);puts(log);return 1;}
    cl_kernel prepare=clCreateKernel(program,"source_gui_prepare",&err);assert(prepare&&!err);
    cl_kernel check=clCreateKernel(program,"source_gui_cache_check",&err);assert(check&&!err);
    cl_kernel sparse=clCreateKernel(program,"sparse_check",&err);assert(sparse&&!err);
    enum {W=16,H=8,N=W*H};unsigned char rgba[N][4];
    cl_image_format format={CL_RGBA,CL_UNORM_INT8};cl_image_desc desc={0};
    desc.image_type=CL_MEM_OBJECT_IMAGE2D;desc.image_width=W;desc.image_height=H;
    cl_mem image=clCreateImage(context,CL_MEM_READ_ONLY,&format,&desc,NULL,&err);assert(image&&!err);
    cl_mem coeff=clCreateBuffer(context,CL_MEM_READ_ONLY,31*sizeof(float),NULL,&err);assert(coeff&&!err);
    cl_mem errors=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(int),NULL,&err);assert(errors&&!err);
    cl_mem colours=clCreateBuffer(context,CL_MEM_READ_WRITE,N*sizeof(cl_float4),NULL,&err);assert(colours&&!err);
    cl_mem keys=clCreateBuffer(context,CL_MEM_READ_WRITE,N*sizeof(cl_uint),NULL,&err);assert(keys&&!err);
    float matrix[31]={0};matrix[0]=matrix[4]=matrix[8]=matrix[9]=matrix[13]=matrix[17]=matrix[18]=matrix[22]=matrix[26]=1;matrix[30]=100;
    for(unsigned i=0;i<N;++i){unsigned a=(i*37)%256;rgba[i][3]=(unsigned char)a;
        for(unsigned c=0;c<3;++c)rgba[i][c]=(unsigned char)((i*(c+3))%(a+1));}
    unsigned w=W,h=H;size_t origin[3]={0},region[3]={W,H,1},work[2]={W,H};
    for(unsigned step=0;step<9;++step){
        unsigned flip=step>=3;unsigned invalidate=step==0||step==4||step==6||step==8;
        if(step==2){rgba[7][0]=17;rgba[7][1]=12;rgba[7][2]=30;rgba[7][3]=200;}
        if(step==4||step==5)matrix[30]+=25;
        if(step==7)memset(rgba,0,sizeof(rgba));
        int status=0;
        assert(!clEnqueueWriteImage(queue,image,CL_TRUE,origin,region,0,0,rgba,0,NULL,NULL));
        assert(!clEnqueueWriteBuffer(queue,coeff,CL_TRUE,0,sizeof(matrix),matrix,0,NULL,NULL));
        assert(!clEnqueueWriteBuffer(queue,errors,CL_TRUE,0,sizeof(status),&status,0,NULL,NULL));
        cl_mem args[]={image,coeff,errors,colours};
        for(unsigned k=0;k<2;++k){cl_kernel kernel=k?check:prepare;
            assert(!clSetKernelArg(kernel,0,sizeof(cl_mem),&args[0]));
            assert(!clSetKernelArg(kernel,1,sizeof(cl_mem),&args[1]));
            assert(!clSetKernelArg(kernel,2,sizeof(flip),&flip));
            assert(!clSetKernelArg(kernel,3,sizeof(cl_mem),&args[2]));
            assert(!clSetKernelArg(kernel,4,sizeof(cl_mem),&args[3]));
            assert(!clSetKernelArg(kernel,5,sizeof(w),&w));assert(!clSetKernelArg(kernel,6,sizeof(h),&h));
            if(!k){assert(!clSetKernelArg(kernel,7,sizeof(keys),&keys));assert(!clSetKernelArg(kernel,8,sizeof(invalidate),&invalidate));}
            assert(!clEnqueueNDRangeKernel(queue,kernel,2,NULL,work,NULL,0,NULL,NULL));}
        cl_mem sparse_args[]={colours,keys,coeff,errors};
        for(unsigned k=0;k<4;++k)assert(!clSetKernelArg(sparse,k,sizeof(cl_mem),&sparse_args[k]));
        assert(!clSetKernelArg(sparse,4,sizeof(w),&w));assert(!clSetKernelArg(sparse,5,sizeof(h),&h));
        assert(!clEnqueueNDRangeKernel(queue,sparse,2,NULL,work,NULL,0,NULL,NULL));
        assert(!clEnqueueReadBuffer(queue,errors,CL_TRUE,0,sizeof(status),&status,0,NULL,NULL));
        if(step==5)assert(status==2);else assert(status==0);
    }
    puts("{\"exact_cache_steps\":8,\"stale_coefficient_negative_control\":true,\"pixel_changes\":true,\"flip\":true,\"transparent_clear\":true}");
    clReleaseMemObject(keys);clReleaseMemObject(colours);clReleaseMemObject(errors);clReleaseMemObject(coeff);clReleaseMemObject(image);
    clReleaseKernel(sparse);clReleaseKernel(check);clReleaseKernel(prepare);clReleaseProgram(program);clReleaseCommandQueue(queue);clReleaseContext(context);return 0;
}
