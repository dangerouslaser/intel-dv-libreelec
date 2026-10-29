//! Phase 1a interop self-test (headless; no display, no playback):
//! 1. A VA P010 / NV12 surface filled through vaDeriveImage reads back through
//!    Vulkan (Y-tiled planes imported as single-plane images) with every
//!    16-bit (or 8-bit) code exact.
//! 2. A linear ARGB8888 GBM BO imported as a storage buffer: a Vulkan-written
//!    pattern reads back byte for byte through gbm_bo_map, after the render
//!    fence was attached to the BO as an implicit fence (IMPORT_SYNC_FILE).
//! 3. Costs: surface export + import, BO import.

use ash::vk;
use dv_engine_vk::device::Device;
use dv_engine_vk::gpu::{self, Bind, HostBuffer, OneShot, Pipeline, PushDescriptor};
use dv_engine_vk::interop::{dmabuf, sync, va};
use std::os::raw::{c_int, c_void};
use std::time::Instant;

#[link(name = "gbm")]
extern "C" {
    fn gbm_create_device(fd: c_int) -> *mut c_void;
    fn gbm_device_destroy(gbm: *mut c_void);
    fn gbm_bo_create(gbm: *mut c_void, w: u32, h: u32, format: u32, flags: u32) -> *mut c_void;
    fn gbm_bo_get_fd(bo: *mut c_void) -> c_int;
    fn gbm_bo_get_stride(bo: *mut c_void) -> u32;
    fn gbm_bo_map(
        bo: *mut c_void,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        flags: u32,
        stride: *mut u32,
        map_data: *mut *mut c_void,
    ) -> *mut c_void;
    fn gbm_bo_unmap(bo: *mut c_void, map_data: *mut c_void);
    fn gbm_bo_destroy(bo: *mut c_void);
}
const GBM_FORMAT_ARGB8888: u32 = 0x3432_5241;
const GBM_BO_USE_SCANOUT: u32 = 1;
const GBM_BO_USE_RENDERING: u32 = 4;
const GBM_BO_USE_LINEAR: u32 = 16;
const GBM_BO_TRANSFER_READ: u32 = 1;

const W: u32 = 3840;
const H: u32 = 2160;

fn luma_value(x: u32, y: u32, ten: bool) -> u16 {
    let v = x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13)) ^ (x >> 3).wrapping_mul(y >> 2);
    if ten { (v & 0xffff) as u16 } else { (v & 0xff) as u16 }
}
fn chroma_value(x: u32, y: u32, c: u32, ten: bool) -> u16 {
    let v = (x.wrapping_mul(31) ^ y.wrapping_mul(17)).wrapping_add(c * 0x9e37);
    if ten { (v & 0xffff) as u16 } else { (v & 0xff) as u16 }
}

fn fill_surface(display: va::VADisplay, surface: va::VASurfaceID, ten: bool) -> Result<(), String> {
    let mut image = va::VAImage::default();
    let s = unsafe { va::vaDeriveImage(display, surface, &mut image) };
    if s != 0 {
        return Err(format!("vaDeriveImage {s}"));
    }
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let s = unsafe { va::vaMapBuffer(display, image.buf, &mut ptr) };
    if s != 0 {
        unsafe { va::vaDestroyImage(display, image.image_id) };
        return Err(format!("vaMapBuffer {s}"));
    }
    let base = ptr as *mut u8;
    let bps = if ten { 2usize } else { 1 };
    unsafe {
        for y in 0..H {
            let row = base.add(image.offsets[0] as usize + y as usize * image.pitches[0] as usize);
            for x in 0..W {
                let v = luma_value(x, y, ten);
                if ten {
                    (row.add(x as usize * bps) as *mut u16).write_unaligned(v);
                } else {
                    *row.add(x as usize) = v as u8;
                }
            }
        }
        for y in 0..H / 2 {
            let row = base.add(image.offsets[1] as usize + y as usize * image.pitches[1] as usize);
            for x in 0..W / 2 {
                for c in 0..2 {
                    let v = chroma_value(x, y, c, ten);
                    let at = (2 * x as usize + c as usize) * bps;
                    if ten {
                        (row.add(at) as *mut u16).write_unaligned(v);
                    } else {
                        *row.add(at) = v as u8;
                    }
                }
            }
        }
        va::vaUnmapBuffer(display, image.buf);
        va::vaDestroyImage(display, image.image_id);
    }
    Ok(())
}

fn check_surface(d: &Device, display: va::VADisplay, ten: bool) -> Result<(), String> {
    let name = if ten { "P010" } else { "NV12" };
    let mut attribs = [
        va::VASurfaceAttrib {
            type_: va::VASurfaceAttribPixelFormat,
            flags: va::VA_SURFACE_ATTRIB_SETTABLE,
            value: va::VAGenericValue {
                type_: va::VAGenericValueTypeInteger,
                value: va::VAGenericValueUnion { i: (if ten { va::VA_FOURCC_P010 } else { va::VA_FOURCC_NV12 }) as i32 },
            },
        },
        va::VASurfaceAttrib {
            type_: va::VASurfaceAttribUsageHint,
            flags: va::VA_SURFACE_ATTRIB_SETTABLE,
            value: va::VAGenericValue {
                type_: va::VAGenericValueTypeInteger,
                value: va::VAGenericValueUnion { i: va::VA_SURFACE_ATTRIB_USAGE_HINT_DECODER },
            },
        },
    ];
    let mut surface = 0;
    let rt = if ten { va::VA_RT_FORMAT_YUV420_10 } else { va::VA_RT_FORMAT_YUV420 };
    let s = unsafe { va::vaCreateSurfaces(display, rt, W, H, &mut surface, 1, attribs.as_mut_ptr(), 2) };
    if s != 0 {
        return Err(format!("vaCreateSurfaces {s}"));
    }
    let result = (|| {
        fill_surface(display, surface, ten)?;
        unsafe { va::vaSyncSurface(display, surface) };
        let t = Instant::now();
        let imp = va::import_surface(d, display, surface, W, H).map_err(|e| e.to_string())?;
        let import_ms = t.elapsed().as_secs_f64() * 1e3;
        let fence = sync::export_sync_file(imp.fd, sync::DMA_BUF_SYNC_READ);
        let fence_state = match fence {
            Ok(fd) => {
                let s = sync::signalled(fd);
                unsafe { libc::close(fd) };
                if s { "signalled" } else { "pending" }
            }
            Err(_) => "export failed",
        };
        let pipe = Pipeline::new(
            d,
            "interop_read_planes",
            &[
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::DescriptorType::STORAGE_BUFFER,
            ],
            8,
            &[(0, ten as u32)],
        )
        .map_err(|e| e.to_string())?;
        let sampler = gpu::nearest_sampler(d).map_err(|e| e.to_string())?;
        let count = (W * H + W * H / 2) as u64;
        let out = HostBuffer::new(d, count * 2, vk::BufferUsageFlags::STORAGE_BUFFER).map_err(|e| e.to_string())?;
        let push = PushDescriptor::new(d);
        let shot = OneShot::new(d).map_err(|e| e.to_string())?;
        let cmd = shot.begin(d).map_err(|e| e.to_string())?;
        gpu::acquire_foreign_images(d, cmd, &[imp.luma.image, imp.chroma.image]);
        unsafe { d.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline) };
        push.bind(
            cmd,
            &pipe,
            &[Bind::Sampled(imp.luma.view, sampler), Bind::Sampled(imp.chroma.view, sampler), Bind::Buffer(out.buffer)],
        );
        let pc: [u32; 2] = [W, H];
        unsafe {
            d.device.cmd_push_constants(
                cmd,
                pipe.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(pc.as_ptr() as *const u8, 8),
            );
            d.device.cmd_dispatch(cmd, W.div_ceil(16), H.div_ceil(8), 1);
        }
        gpu::compute_to_host(d, cmd);
        shot.submit_wait(d, &[], &[]).map_err(|e| e.to_string())?;
        let codes: &[u16] = unsafe { std::slice::from_raw_parts(out.ptr as *const u16, count as usize) };
        let mut bad = 0usize;
        let mut first = None;
        for y in 0..H {
            for x in 0..W {
                let i = (y * W + x) as usize;
                if codes[i] != luma_value(x, y, ten) {
                    bad += 1;
                    first.get_or_insert((0, x, y, codes[i], luma_value(x, y, ten)));
                }
            }
        }
        for y in 0..H / 2 {
            for x in 0..W / 2 {
                for c in 0..2 {
                    let i = (W * H + 2 * (y * (W / 2) + x) + c) as usize;
                    if codes[i] != chroma_value(x, y, c, ten) {
                        bad += 1;
                        first.get_or_insert((1 + c, x, y, codes[i], chroma_value(x, y, c, ten)));
                    }
                }
            }
        }
        println!(
            "{name}: modifier {:#018x}, {} codes compared, {} differ{}; export+import {:.3} ms; implicit fence after vaSyncSurface: {}",
            imp.modifier,
            count,
            bad,
            first.map(|f| format!(" (first: plane {} at {},{}: got {} want {})", f.0, f.1, f.2, f.3, f.4)).unwrap_or_default(),
            import_ms,
            fence_state
        );
        shot.destroy(d);
        out.destroy(d);
        unsafe { d.device.destroy_sampler(sampler, None) };
        pipe.destroy(d);
        imp.destroy(d);
        if bad != 0 { Err(format!("{name}: {bad} codes differ")) } else { Ok(()) }
    })();
    unsafe { va::vaDestroySurfaces(display, &mut surface, 1) };
    result
}

fn check_output(d: &Device, drm_fd: c_int) -> Result<(), String> {
    let gbm = unsafe { gbm_create_device(drm_fd) };
    if gbm.is_null() {
        return Err("gbm_create_device".into());
    }
    let bo = unsafe {
        gbm_bo_create(gbm, W, H, GBM_FORMAT_ARGB8888, GBM_BO_USE_LINEAR | GBM_BO_USE_RENDERING | GBM_BO_USE_SCANOUT)
    };
    if bo.is_null() {
        unsafe { gbm_device_destroy(gbm) };
        return Err("gbm_bo_create".into());
    }
    let result = (|| {
        let fd = unsafe { gbm_bo_get_fd(bo) };
        let stride = unsafe { gbm_bo_get_stride(bo) };
        let t = Instant::now();
        let buf = dmabuf::import_buffer(d, fd, stride as u64 * H as u64).map_err(|e| e.to_string())?;
        let import_ms = t.elapsed().as_secs_f64() * 1e3;
        let pipe = Pipeline::new(d, "interop_write_output", &[vk::DescriptorType::STORAGE_BUFFER], 16, &[])
            .map_err(|e| e.to_string())?;
        let push = PushDescriptor::new(d);
        let shot = OneShot::new(d).map_err(|e| e.to_string())?;
        let done = sync::exportable_semaphore(d).map_err(|e| e.to_string())?;
        let seed = 0x5eed_1234u32;
        let cmd = shot.begin(d).map_err(|e| e.to_string())?;
        unsafe { d.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline) };
        push.bind(cmd, &pipe, &[Bind::Buffer(buf.buffer)]);
        let pc: [u32; 4] = [W, H, stride / 4, seed];
        unsafe {
            d.device.cmd_push_constants(
                cmd,
                pipe.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(pc.as_ptr() as *const u8, 16),
            );
            d.device.cmd_dispatch(cmd, W.div_ceil(64), H.div_ceil(4), 1);
        }
        // Release the buffer to the foreign queue family (KWin/KMS read it).
        let release = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .src_queue_family_index(d.queue_family)
            .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
            .buffer(buf.buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE)];
        unsafe {
            d.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().buffer_memory_barriers(&release));
        }
        shot.submit_wait(d, &[], &[done]).map_err(|e| e.to_string())?;
        let fence_fd = sync::sync_file_from_semaphore(d, done).map_err(|e| e.to_string())?;
        let attached = sync::import_sync_file(fd, fence_fd, sync::DMA_BUF_SYNC_WRITE);
        unsafe { libc::close(fence_fd) };
        let mut map_stride = 0u32;
        let mut map_data = std::ptr::null_mut();
        let ptr = unsafe { gbm_bo_map(bo, 0, 0, W, H, GBM_BO_TRANSFER_READ, &mut map_stride, &mut map_data) };
        if ptr.is_null() {
            return Err("gbm_bo_map".into());
        }
        let mut bad = 0usize;
        for y in 0..H {
            let row = unsafe { (ptr as *const u8).add(y as usize * map_stride as usize) };
            for x in 0..W {
                let v = (x.wrapping_mul(2654435761) ^ y.wrapping_mul(2246822519) ^ seed) & 0x00ff_ffff | 0xff00_0000;
                let got = unsafe { (row.add(x as usize * 4) as *const u32).read_unaligned() };
                bad += (got != v) as usize;
            }
        }
        unsafe { gbm_bo_unmap(bo, map_data) };
        println!(
            "output: linear ARGB8888 stride {stride}, {} pixels compared, {bad} differ; import {:.3} ms; IMPORT_SYNC_FILE {}",
            W * H,
            import_ms,
            if attached.is_ok() { "ok" } else { "failed" }
        );
        unsafe { d.device.destroy_semaphore(done, None) };
        shot.destroy(d);
        pipe.destroy(d);
        buf.destroy(d);
        unsafe { libc::close(fd) };
        if bad != 0 { Err(format!("output: {bad} pixels differ")) } else { attached.map_err(|e| e.to_string()) }
    })();
    unsafe {
        gbm_bo_destroy(bo);
        gbm_device_destroy(gbm);
    }
    result
}

fn main() {
    let node = std::env::args().nth(1).unwrap_or_else(|| "/dev/dri/renderD128".into());
    let path = std::ffi::CString::new(node.clone()).unwrap();
    let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
    assert!(fd >= 0, "open {node}");
    let d = Device::new(fd).unwrap_or_else(|e| panic!("device: {e}"));
    println!("device: {} ({})", d.name, d.driver);
    let display = unsafe { va::vaGetDisplayDRM(fd) };
    let (mut major, mut minor) = (0, 0);
    assert_eq!(unsafe { va::vaInitialize(display, &mut major, &mut minor) }, 0, "vaInitialize");
    let mut failures = 0;
    for ten in [true, false] {
        if let Err(e) = check_surface(&d, display, ten) {
            println!("FAIL {e}");
            failures += 1;
        }
    }
    if let Err(e) = check_output(&d, fd) {
        println!("FAIL {e}");
        failures += 1;
    }
    unsafe { va::vaTerminate(display) };
    drop(d);
    println!("{}", if failures == 0 { "PASS" } else { "FAIL" });
    std::process::exit(failures);
}
