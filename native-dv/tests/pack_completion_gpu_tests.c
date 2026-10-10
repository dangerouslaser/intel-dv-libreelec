/* White-box lifecycle tests using real CL user events, not GL image writes. */
#include "../src/dv_source_renderer.c"
#include <assert.h>

static void pending(source_renderer *r,cl_context context)
{
    cl_int error=0;
    r->released=clCreateUserEvent(context,&error);assert(r->released&&!error);
    r->submitted=1;r->pack_pending=1;r->acquired=1;r->queued=1;r->valid=0;
    r->pack_error=0;r->pack_overlay=1;r->gui_pixel_cache=1;
    r->settings.identity.frame_id=42;r->settings.identity.pts=12345;
}
static void revision_keys(void)
{
    source_renderer r={0};r.gui_pixel_cache=1;r.gui_cache_valid=1;
    r.gui_revision=7;r.gui_texture=9;r.gui_flip=1;
    r.gui_width=r.geometry.canvas_width=3840;r.gui_height=r.geometry.canvas_height=2160;
    assert(overlay_revision_matches(&r,9,1,7));
    assert(!overlay_revision_matches(&r,9,1,0));
    assert(!overlay_revision_matches(&r,9,1,8));
    assert(!overlay_revision_matches(&r,10,1,7));
    assert(!overlay_revision_matches(&r,9,0,7));
    r.gui_cache_invalidate=1;assert(!overlay_revision_matches(&r,9,1,7));r.gui_cache_invalidate=0;
    r.gui_cache_valid=0;assert(!overlay_revision_matches(&r,9,1,7));r.gui_cache_valid=1;
    r.geometry.canvas_width=1920;assert(!overlay_revision_matches(&r,9,1,7));r.geometry.canvas_width=3840;
    r.geometry.canvas_height=1080;assert(!overlay_revision_matches(&r,9,1,7));r.geometry.canvas_height=2160;
    r.gui_pixel_cache=0;assert(!overlay_revision_matches(&r,9,1,7));
}

int main(void)
{
    revision_keys();
    cl_platform_id platforms[16];cl_uint count=0;
    assert(!clGetPlatformIDs(16,platforms,&count));cl_device_id device=NULL;
    for(unsigned i=0;i<count&&i<16&&!device;++i)
        if(clGetDeviceIDs(platforms[i],CL_DEVICE_TYPE_GPU,1,&device,NULL))device=NULL;
    assert(device);cl_int error=0;
    cl_context context=clCreateContext(NULL,1,&device,NULL,NULL,&error);assert(context&&!error);
    source_renderer r={0};
    r.queue=clCreateCommandQueue(context,device,0,&error);assert(r.queue&&!error);
    pending(&r,context);int ready=1;
    assert(dv_source_renderer_pack_complete(&r,42,12345,0,&ready)==DV_OK&&!ready);
    assert(r.pack_pending&&!r.valid&&r.submitted&&r.acquired);
    assert(dv_source_renderer_pack_complete(&r,43,12345,0,&ready)==DV_IDENTITY);
    assert(dv_source_renderer_pack_complete(&r,42,12346,0,&ready)==DV_IDENTITY);
    assert(dv_source_renderer_pack_begin(&r,42,12345,0,0,0,0,NULL,0)==DV_INVALID);
    assert(dv_source_renderer_select_output(&r,1)==DV_BACKEND);
    assert(dv_source_renderer_submit_input(&r,0,0,0,0,NULL,0,43,12346,NULL,NULL)==DV_INVALID);
    assert(!clSetUserEventStatus(r.released,CL_COMPLETE));
    assert(dv_source_renderer_pack_complete(&r,42,12345,0,&ready)==DV_OK&&ready);
    assert(r.valid&&r.gui_cache_valid&&r.overlay_visible&&!r.pack_pending&&!r.submitted&&!r.acquired);
    assert(dv_source_renderer_pack_complete(&r,42,12345,0,&ready)==DV_IDENTITY&&!ready);

    pending(&r,context);r.pack_error=2;
    assert(!clSetUserEventStatus(r.released,CL_COMPLETE));
    assert(dv_source_renderer_pack_complete(&r,42,12345,1,&ready)==DV_UNSUPPORTED&&!ready);
    assert(!r.valid&&!r.gui_cache_valid&&!r.pack_pending);

    pending(&r,context);
    assert(!clSetUserEventStatus(r.released,CL_COMPLETE));
    assert(dv_source_renderer_cancel(&r)==DV_OK);
    assert(!r.valid&&!r.gui_cache_valid&&!r.pack_pending&&!r.submitted);

    pending(&r,context);r.pack_cached_gui=1;r.gui_cache_valid=1;
    assert(!clSetUserEventStatus(r.released,CL_COMPLETE));
    assert(dv_source_renderer_cancel(&r)==DV_OK);
    assert(!r.valid&&r.gui_cache_valid&&!r.pack_pending&&!r.submitted);

    pending(&r,context);
    assert(!clSetUserEventStatus(r.released,CL_COMPLETE));
    assert(dv_source_renderer_drain(&r)==DV_OK);
    assert(!r.valid&&!r.gui_cache_valid&&!r.pack_pending&&!r.submitted);

    pending(&r,context);
    assert(!clSetUserEventStatus(r.released,-1));
    assert(dv_source_renderer_pack_complete(&r,42,12345,0,&ready)==DV_BACKEND&&!ready);
    assert(r.pack_pending&&r.acquired&&!r.valid);
    assert(dv_source_renderer_drain(&r)==DV_BACKEND&&r.pack_pending&&r.acquired);
    clReleaseEvent(r.released);clReleaseCommandQueue(r.queue);clReleaseContext(context);
    puts("{\"poll_not_ready\":true,\"identity_rejection\":true,\"bounded_submission\":true,"
         "\"publish_after_completion\":true,\"gui_error_rejection\":true,"
         "\"cancel_discards\":true,\"drain_discards\":true,\"failed_event_retains_ownership\":true}");
    return 0;
}
