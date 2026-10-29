//! dmabuf import: a plane of a decoded surface as a single-plane image with an
//! explicit DRM modifier, and a linear scanout BO as a storage buffer.

use crate::device::{Device, Error, Result};
use ash::vk;
use std::os::fd::RawFd;

/// One plane of a dmabuf: its own fd (dup'd on import), byte offset, pitch.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub fd: RawFd,
    pub offset: u64,
    pub pitch: u64,
}

pub struct ImportedImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

impl ImportedImage {
    pub fn destroy(&self, d: &Device) {
        unsafe {
            d.device.destroy_image_view(self.view, None);
            d.device.destroy_image(self.image, None);
            d.device.free_memory(self.memory, None);
        }
    }
}

pub struct ImportedBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
}

impl ImportedBuffer {
    pub fn destroy(&self, d: &Device) {
        unsafe {
            d.device.destroy_buffer(self.buffer, None);
            d.device.free_memory(self.memory, None);
        }
    }
}

fn dup(fd: RawFd) -> Result<RawFd> {
    let n = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if n < 0 {
        return Err(Error::Unsupported("dup of a dmabuf fd failed".into()));
    }
    Ok(n)
}

/// Imports memory for `requirements` from a dmabuf fd (dup'd; Vulkan owns the
/// copy on success). Dedicated to `image` or `buffer`.
fn import_memory(
    d: &Device,
    fd: RawFd,
    requirements: vk::MemoryRequirements,
    image: vk::Image,
    buffer: vk::Buffer,
) -> Result<vk::DeviceMemory> {
    let handle = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    let mut fd_props = vk::MemoryFdPropertiesKHR::default();
    unsafe { d.memory_fd.get_memory_fd_properties(handle, fd, &mut fd_props) }?;
    let bits = requirements.memory_type_bits & fd_props.memory_type_bits;
    let index = d
        .memory_type(bits, vk::MemoryPropertyFlags::empty())
        .ok_or(Error::Unsupported("no memory type for the dmabuf".into()))?;
    let owned = dup(fd)?;
    let mut import = vk::ImportMemoryFdInfoKHR::default().handle_type(handle).fd(owned);
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image).buffer(buffer);
    // The dmabuf may be larger than this image needs (another plane follows).
    let size = unsafe { libc::lseek(fd, 0, libc::SEEK_END) };
    let size = if size > 0 { (size as u64).max(requirements.size) } else { requirements.size };
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(size)
        .memory_type_index(index)
        .push_next(&mut import)
        .push_next(&mut dedicated);
    match unsafe { d.device.allocate_memory(&info, None) } {
        Ok(m) => Ok(m),
        Err(e) => {
            unsafe { libc::close(owned) };
            Err(e.into())
        }
    }
}

/// A plane of a dmabuf as a 2D image (`R16_UNORM`, `R16G16_UNORM`, `R8_UNORM`,
/// `R8G8_UNORM`, …) with the dmabuf's modifier, for storage reads.
pub fn import_plane(
    d: &Device,
    plane: Plane,
    modifier: u64,
    format: vk::Format,
    width: u32,
    height: u32,
    usage: vk::ImageUsageFlags,
) -> Result<ImportedImage> {
    let layouts = [vk::SubresourceLayout {
        offset: plane.offset,
        size: 0,
        row_pitch: plane.pitch,
        array_pitch: 0,
        depth_pitch: 0,
    }];
    let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(modifier)
        .plane_layouts(&layouts);
    let mut external =
        vk::ExternalMemoryImageCreateInfo::default().handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut explicit)
        .push_next(&mut external);
    let image = unsafe { d.device.create_image(&info, None) }?;
    let requirements = unsafe { d.device.get_image_memory_requirements(image) };
    let memory = match import_memory(d, plane.fd, requirements, image, vk::Buffer::null()) {
        Ok(m) => m,
        Err(e) => {
            unsafe { d.device.destroy_image(image, None) };
            return Err(e);
        }
    };
    let cleanup = |e: vk::Result| {
        unsafe {
            d.device.destroy_image(image, None);
            d.device.free_memory(memory, None);
        }
        Error::Vk(e)
    };
    unsafe { d.device.bind_image_memory(image, memory, 0) }.map_err(cleanup)?;
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { d.device.create_image_view(&view_info, None) }.map_err(cleanup)?;
    Ok(ImportedImage { image, view, memory, format, width, height })
}

/// A linear dmabuf (a scanout BO) as a storage buffer of `size` bytes.
pub fn import_buffer(d: &Device, fd: RawFd, size: u64) -> Result<ImportedBuffer> {
    let mut external =
        vk::ExternalMemoryBufferCreateInfo::default().handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    let buffer = unsafe { d.device.create_buffer(&info, None) }?;
    let requirements = unsafe { d.device.get_buffer_memory_requirements(buffer) };
    let memory = match import_memory(d, fd, requirements, vk::Image::null(), buffer) {
        Ok(m) => m,
        Err(e) => {
            unsafe { d.device.destroy_buffer(buffer, None) };
            return Err(e);
        }
    };
    if let Err(e) = unsafe { d.device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            d.device.destroy_buffer(buffer, None);
            d.device.free_memory(memory, None);
        }
        return Err(e.into());
    }
    Ok(ImportedBuffer { buffer, memory, size })
}

/// A whole (possibly multi-plane, e.g. CCS) image from one dmabuf, for
/// sampling: `planes` are the memory planes' (offset, pitch) on `fd`.
pub fn import_image(
    d: &Device,
    fd: RawFd,
    planes: &[(u64, u64)],
    modifier: u64,
    format: vk::Format,
    width: u32,
    height: u32,
    swizzle_alpha_one: bool,
) -> Result<ImportedImage> {
    let layouts: Vec<_> = planes
        .iter()
        .map(|&(offset, pitch)| vk::SubresourceLayout { offset, size: 0, row_pitch: pitch, array_pitch: 0, depth_pitch: 0 })
        .collect();
    let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(modifier)
        .plane_layouts(&layouts);
    let mut external =
        vk::ExternalMemoryImageCreateInfo::default().handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut explicit)
        .push_next(&mut external);
    let image = unsafe { d.device.create_image(&info, None) }?;
    let requirements = unsafe { d.device.get_image_memory_requirements(image) };
    let memory = match import_memory(d, fd, requirements, image, vk::Buffer::null()) {
        Ok(m) => m,
        Err(e) => {
            unsafe { d.device.destroy_image(image, None) };
            return Err(e);
        }
    };
    let cleanup = |e: vk::Result| {
        unsafe {
            d.device.destroy_image(image, None);
            d.device.free_memory(memory, None);
        }
        Error::Vk(e)
    };
    unsafe { d.device.bind_image_memory(image, memory, 0) }.map_err(cleanup)?;
    let a = if swizzle_alpha_one { vk::ComponentSwizzle::ONE } else { vk::ComponentSwizzle::IDENTITY };
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(vk::ComponentMapping { r: vk::ComponentSwizzle::IDENTITY, g: vk::ComponentSwizzle::IDENTITY, b: vk::ComponentSwizzle::IDENTITY, a })
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { d.device.create_image_view(&view_info, None) }.map_err(cleanup)?;
    Ok(ImportedImage { image, view, memory, format, width, height })
}
