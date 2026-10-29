//! libva: a decoded surface as dmabuf planes (Intel iHD, AMD radeonsi).
//! Hand-written declarations of the few entry points used.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]

use crate::device::{Device, Error, Result};
use crate::interop::dmabuf::{self, ImportedImage, Plane};
use ash::vk;
use std::os::raw::{c_int, c_uint, c_void};

pub type VADisplay = *mut c_void;
pub type VAStatus = c_int;
pub type VASurfaceID = c_uint;
pub const VA_INVALID_SURFACE: VASurfaceID = 0xffff_ffff;

pub const VA_FOURCC_P010: u32 = 0x3031_3050;
pub const VA_FOURCC_NV12: u32 = 0x3231_564e;
pub const VA_RT_FORMAT_YUV420: u32 = 0x1;
pub const VA_RT_FORMAT_YUV420_10: u32 = 0x100;
pub const VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2: u32 = 0x4000_0000;
pub const VA_EXPORT_SURFACE_READ_ONLY: u32 = 0x1;
pub const VA_EXPORT_SURFACE_READ_WRITE: u32 = 0x3;
pub const VA_EXPORT_SURFACE_SEPARATE_LAYERS: u32 = 0x4;
pub const VA_EXPORT_SURFACE_COMPOSED_LAYERS: u32 = 0x8;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct VADRMPRIMEObject {
    pub fd: c_int,
    pub size: u32,
    pub drm_format_modifier: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct VADRMPRIMELayer {
    pub drm_format: u32,
    pub num_planes: u32,
    pub object_index: [u32; 4],
    pub offset: [u32; 4],
    pub pitch: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct VADRMPRIMESurfaceDescriptor {
    pub fourcc: u32,
    pub width: u32,
    pub height: u32,
    pub num_objects: u32,
    pub objects: [VADRMPRIMEObject; 4],
    pub num_layers: u32,
    pub layers: [VADRMPRIMELayer; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union VAGenericValueUnion {
    pub i: c_int,
    pub f: f32,
    pub p: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAGenericValue {
    pub type_: c_int,
    pub value: VAGenericValueUnion,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VASurfaceAttrib {
    pub type_: c_int,
    pub flags: u32,
    pub value: VAGenericValue,
}
pub const VASurfaceAttribPixelFormat: c_int = 1;
pub const VASurfaceAttribUsageHint: c_int = 8;
pub const VA_SURFACE_ATTRIB_SETTABLE: u32 = 2;
pub const VAGenericValueTypeInteger: c_int = 1;
pub const VA_SURFACE_ATTRIB_USAGE_HINT_DECODER: c_int = 1;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct VAImageFormat {
    pub fourcc: u32,
    pub byte_order: u32,
    pub bits_per_pixel: u32,
    pub depth: u32,
    pub red_mask: u32,
    pub green_mask: u32,
    pub blue_mask: u32,
    pub alpha_mask: u32,
    pub va_reserved: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct VAImage {
    pub image_id: u32,
    pub format: VAImageFormat,
    pub buf: u32,
    pub width: u16,
    pub height: u16,
    pub data_size: u32,
    pub num_planes: u32,
    pub pitches: [u32; 3],
    pub offsets: [u32; 3],
    pub num_palette_entries: i32,
    pub entry_bytes: i32,
    pub component_order: [i8; 4],
    pub va_reserved: [u32; 4],
}

extern "C" {
    pub fn vaGetDisplayDRM(fd: c_int) -> VADisplay;
    pub fn vaInitialize(dpy: VADisplay, major: *mut c_int, minor: *mut c_int) -> VAStatus;
    pub fn vaTerminate(dpy: VADisplay) -> VAStatus;
    pub fn vaCreateSurfaces(
        dpy: VADisplay,
        format: c_uint,
        width: c_uint,
        height: c_uint,
        surfaces: *mut VASurfaceID,
        num: c_uint,
        attribs: *mut VASurfaceAttrib,
        num_attribs: c_uint,
    ) -> VAStatus;
    pub fn vaDestroySurfaces(dpy: VADisplay, surfaces: *mut VASurfaceID, num: c_int) -> VAStatus;
    pub fn vaSyncSurface(dpy: VADisplay, surface: VASurfaceID) -> VAStatus;
    pub fn vaDeriveImage(dpy: VADisplay, surface: VASurfaceID, image: *mut VAImage) -> VAStatus;
    pub fn vaDestroyImage(dpy: VADisplay, image: u32) -> VAStatus;
    pub fn vaMapBuffer(dpy: VADisplay, buf: u32, pbuf: *mut *mut c_void) -> VAStatus;
    pub fn vaUnmapBuffer(dpy: VADisplay, buf: u32) -> VAStatus;
    pub fn vaExportSurfaceHandle(
        dpy: VADisplay,
        surface: VASurfaceID,
        mem_type: u32,
        flags: u32,
        descriptor: *mut c_void,
    ) -> VAStatus;
}

/// A decoded 4:2:0 surface's planes as Vulkan images: luma (`R16_UNORM` or
/// `R8_UNORM`) and interleaved chroma (`R16G16_UNORM` / `R8G8_UNORM`), both on
/// the surface's single dmabuf object. Holds that object's fd for implicit sync.
pub struct SurfaceImport {
    pub luma: ImportedImage,
    pub chroma: ImportedImage,
    pub fd: c_int,
    pub modifier: u64,
    pub ten_bit: bool,
    pub inode: u64,
}

impl SurfaceImport {
    pub fn destroy(&self, d: &Device) {
        self.luma.destroy(d);
        self.chroma.destroy(d);
        unsafe { libc::close(self.fd) };
    }
}

pub fn dmabuf_inode(fd: c_int) -> u64 {
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut st) } != 0 {
        return 0;
    }
    st.st_ino
}

/// Exports `surface` (separate layers) and imports its two planes, `width` x
/// `height` the logical (decoded) size.
pub fn import_surface(
    d: &Device,
    display: VADisplay,
    surface: VASurfaceID,
    width: u32,
    height: u32,
) -> Result<SurfaceImport> {
    let mut desc = VADRMPRIMESurfaceDescriptor::default();
    let status = unsafe {
        vaExportSurfaceHandle(
            display,
            surface,
            VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
            VA_EXPORT_SURFACE_READ_ONLY | VA_EXPORT_SURFACE_SEPARATE_LAYERS,
            &mut desc as *mut _ as *mut c_void,
        )
    };
    if status != 0 {
        return Err(Error::Unsupported(format!("vaExportSurfaceHandle: {status}")));
    }
    let close_all = |desc: &VADRMPRIMESurfaceDescriptor| {
        for o in &desc.objects[..desc.num_objects as usize] {
            unsafe { libc::close(o.fd) };
        }
    };
    let ten_bit = match desc.fourcc {
        VA_FOURCC_P010 => true,
        VA_FOURCC_NV12 => false,
        f => {
            close_all(&desc);
            return Err(Error::Unsupported(format!("surface fourcc {f:#x}")));
        }
    };
    if desc.num_layers != 2 || desc.layers[0].num_planes != 1 || desc.layers[1].num_planes != 1 {
        close_all(&desc);
        return Err(Error::Unsupported("surface export layout".into()));
    }
    let (lf, cf) = if ten_bit {
        (vk::Format::R16_UNORM, vk::Format::R16G16_UNORM)
    } else {
        (vk::Format::R8_UNORM, vk::Format::R8G8_UNORM)
    };
    let plane = |l: &VADRMPRIMELayer| {
        let o = &desc.objects[l.object_index[0] as usize];
        (Plane { fd: o.fd, offset: l.offset[0] as u64, pitch: l.pitch[0] as u64 }, o.drm_format_modifier)
    };
    let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
    let (lp, lm) = plane(&desc.layers[0]);
    let (cp, cm) = plane(&desc.layers[1]);
    let luma = match dmabuf::import_plane(d, lp, lm, lf, width, height, usage) {
        Ok(i) => i,
        Err(e) => {
            close_all(&desc);
            return Err(e);
        }
    };
    let chroma = match dmabuf::import_plane(d, cp, cm, cf, width / 2, height / 2, usage) {
        Ok(i) => i,
        Err(e) => {
            luma.destroy(d);
            close_all(&desc);
            return Err(e);
        }
    };
    // Keep object 0's fd (implicit sync); the others aren't needed.
    for o in &desc.objects[1..desc.num_objects as usize] {
        unsafe { libc::close(o.fd) };
    }
    let fd = desc.objects[0].fd;
    Ok(SurfaceImport { luma, chroma, fd, modifier: lm, ten_bit, inode: dmabuf_inode(fd) })
}

/// The inode of the dmabuf behind `surface` now (None if it can't be exported).
pub fn surface_inode(display: VADisplay, surface: VASurfaceID) -> Option<u64> {
    let mut desc = VADRMPRIMESurfaceDescriptor::default();
    let status = unsafe {
        vaExportSurfaceHandle(
            display,
            surface,
            VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
            VA_EXPORT_SURFACE_READ_ONLY | VA_EXPORT_SURFACE_SEPARATE_LAYERS,
            &mut desc as *mut _ as *mut c_void,
        )
    };
    if status != 0 {
        return None;
    }
    let inode = dmabuf_inode(desc.objects[0].fd);
    for o in &desc.objects[..desc.num_objects as usize] {
        unsafe { libc::close(o.fd) };
    }
    Some(inode)
}
