//! Small Vulkan helpers: compute pipelines from the built-in SPIR-V, host
//! buffers, descriptor writes and one-shot command submission.

use crate::device::{Device, Error, Result};
use ash::vk;

include!(concat!(env!("OUT_DIR"), "/shaders.rs"));

pub fn spirv(name: &str) -> Result<Vec<u32>> {
    let bytes = SHADERS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, b)| *b)
        .ok_or(Error::Missing("a shader"))?;
    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

pub struct Pipeline {
    pub set_layout: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub bindings: Vec<vk::DescriptorType>,
}

impl Pipeline {
    /// `bindings[i]` is binding i of set 0; `spec` are (id, u32 value).
    pub fn new(
        d: &Device,
        shader: &str,
        bindings: &[vk::DescriptorType],
        push_bytes: u32,
        spec: &[(u32, u32)],
    ) -> Result<Self> {
        let code = spirv(shader)?;
        let module =
            unsafe { d.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None) }?;
        let b: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(i, t)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(*t)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = unsafe {
            d.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                    .bindings(&b),
                None,
            )
        }?;
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_bytes.max(4))];
        let layouts = [set_layout];
        let layout = unsafe {
            d.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges),
                None,
            )
        }?;
        let entries: Vec<_> = spec
            .iter()
            .enumerate()
            .map(|(i, (id, _))| vk::SpecializationMapEntry { constant_id: *id, offset: 4 * i as u32, size: 4 })
            .collect();
        let data: Vec<u8> = spec.iter().flat_map(|(_, v)| v.to_le_bytes()).collect();
        let spec_info = vk::SpecializationInfo::default().map_entries(&entries).data(&data);
        let mut required = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(d.subgroup_size);
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main")
            .specialization_info(&spec_info);
        if d.subgroup_size != 0 {
            stage = stage.push_next(&mut required);
        }
        let info = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
        let pipeline = unsafe { d.device.create_compute_pipelines(vk::PipelineCache::null(), &info, None) };
        unsafe { d.device.destroy_shader_module(module, None) };
        let pipeline = pipeline.map_err(|(_, e)| Error::Vk(e))?[0];
        Ok(Pipeline { set_layout, layout, pipeline, bindings: bindings.to_vec() })
    }

    pub fn destroy(&self, d: &Device) {
        unsafe {
            d.device.destroy_pipeline(self.pipeline, None);
            d.device.destroy_pipeline_layout(self.layout, None);
            d.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// One resource bound with push descriptors.
pub enum Bind {
    Buffer(vk::Buffer),
    Range(vk::Buffer, u64, u64),
    Image(vk::ImageView),
    Sampled(vk::ImageView, vk::Sampler),
}

pub struct PushDescriptor {
    pub loader: ash::khr::push_descriptor::Device,
}

impl PushDescriptor {
    pub fn new(d: &Device) -> Self {
        PushDescriptor { loader: ash::khr::push_descriptor::Device::new(&d.instance, &d.device) }
    }

    pub fn bind(&self, cmd: vk::CommandBuffer, p: &Pipeline, binds: &[Bind]) {
        let buffers: Vec<[vk::DescriptorBufferInfo; 1]> = binds
            .iter()
            .map(|b| match b {
                Bind::Buffer(buf) => [vk::DescriptorBufferInfo { buffer: *buf, offset: 0, range: vk::WHOLE_SIZE }],
                Bind::Range(buf, offset, size) => [vk::DescriptorBufferInfo { buffer: *buf, offset: *offset, range: *size }],
                _ => [vk::DescriptorBufferInfo::default()],
            })
            .collect();
        let images: Vec<[vk::DescriptorImageInfo; 1]> = binds
            .iter()
            .map(|b| match b {
                Bind::Image(v) => [vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: *v,
                    image_layout: vk::ImageLayout::GENERAL,
                }],
                Bind::Sampled(v, s) => {
                    [vk::DescriptorImageInfo { sampler: *s, image_view: *v, image_layout: vk::ImageLayout::GENERAL }]
                }
                _ => [vk::DescriptorImageInfo::default()],
            })
            .collect();
        let writes: Vec<_> = binds
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let w = vk::WriteDescriptorSet::default().dst_binding(i as u32).descriptor_type(p.bindings[i]);
                match b {
                    Bind::Buffer(_) | Bind::Range(..) => w.buffer_info(&buffers[i]),
                    _ => w.image_info(&images[i]),
                }
            })
            .collect();
        unsafe {
            self.loader.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::COMPUTE, p.layout, 0, &writes);
        }
    }
}

pub struct HostBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub ptr: *mut u8,
    pub size: u64,
}

impl HostBuffer {
    pub fn new(d: &Device, size: u64, usage: vk::BufferUsageFlags) -> Result<Self> {
        let buffer = unsafe {
            d.device.create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }?;
        let req = unsafe { d.device.get_buffer_memory_requirements(buffer) };
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let index = d.memory_type(req.memory_type_bits, flags | vk::MemoryPropertyFlags::HOST_CACHED)
            .or_else(|| d.memory_type(req.memory_type_bits, flags))
            .ok_or(Error::Missing("host-visible memory"))?;
        let memory = unsafe {
            d.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(index), None)
        }?;
        unsafe { d.device.bind_buffer_memory(buffer, memory, 0) }?;
        let ptr = unsafe { d.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }? as *mut u8;
        Ok(HostBuffer { buffer, memory, ptr, size })
    }

    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }

    pub fn destroy(&self, d: &Device) {
        unsafe {
            d.device.unmap_memory(self.memory);
            d.device.destroy_buffer(self.buffer, None);
            d.device.free_memory(self.memory, None);
        }
    }
}

pub fn nearest_sampler(d: &Device) -> Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .unnormalized_coordinates(false);
    Ok(unsafe { d.device.create_sampler(&info, None) }?)
}

/// Barrier taking imported images from the foreign (external) owner into
/// GENERAL layout, keeping their contents.
pub fn acquire_foreign_images(d: &Device, cmd: vk::CommandBuffer, images: &[vk::Image]) {
    let barriers: Vec<_> = images
        .iter()
        .map(|&image| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .dst_queue_family_index(d.queue_family)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
        })
        .collect();
    unsafe {
        d.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&barriers));
    }
}

/// Compute writes → host reads / other compute reads.
pub fn compute_to_host(d: &Device, cmd: vk::CommandBuffer) {
    let m = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST | vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags2::HOST_READ | vk::AccessFlags2::SHADER_READ)];
    unsafe { d.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&m)) };
}

/// A command pool with one buffer and a fence, for tests and set-up.
pub struct OneShot {
    pub pool: vk::CommandPool,
    pub cmd: vk::CommandBuffer,
    pub fence: vk::Fence,
}

impl OneShot {
    pub fn new(d: &Device) -> Result<Self> {
        let pool = unsafe {
            d.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(d.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }?;
        let cmd = unsafe {
            d.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1),
            )
        }?[0];
        let fence = unsafe { d.device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        Ok(OneShot { pool, cmd, fence })
    }

    pub fn begin(&self, d: &Device) -> Result<vk::CommandBuffer> {
        unsafe {
            d.device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
            d.device.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        Ok(self.cmd)
    }

    /// Ends, submits (waiting on `waits` at compute), signals `signal` if any,
    /// and blocks until done.
    pub fn submit_wait(&self, d: &Device, waits: &[vk::Semaphore], signal: &[vk::Semaphore]) -> Result<()> {
        unsafe {
            d.device.end_command_buffer(self.cmd)?;
            let w: Vec<_> = waits
                .iter()
                .map(|&s| vk::SemaphoreSubmitInfo::default().semaphore(s).stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER))
                .collect();
            let sg: Vec<_> = signal
                .iter()
                .map(|&s| vk::SemaphoreSubmitInfo::default().semaphore(s).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS))
                .collect();
            let c = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cmd)];
            let submit = [vk::SubmitInfo2::default().wait_semaphore_infos(&w).command_buffer_infos(&c).signal_semaphore_infos(&sg)];
            d.device.queue_submit2(d.queue, &submit, self.fence)?;
            d.device.wait_for_fences(&[self.fence], true, 10_000_000_000)?;
            d.device.reset_fences(&[self.fence])?;
        }
        Ok(())
    }

    pub fn destroy(&self, d: &Device) {
        unsafe {
            d.device.destroy_fence(self.fence, None);
            d.device.destroy_command_pool(self.pool, None);
        }
    }
}

/// A device-local storage buffer.
pub struct DeviceBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
}

impl DeviceBuffer {
    pub fn new(d: &Device, size: u64) -> Result<Self> {
        let buffer = unsafe {
            d.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }?;
        let req = unsafe { d.device.get_buffer_memory_requirements(buffer) };
        let index = d
            .memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .ok_or(Error::Missing("device-local memory"))?;
        let memory = unsafe {
            d.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(index), None)
        }?;
        unsafe { d.device.bind_buffer_memory(buffer, memory, 0) }?;
        Ok(DeviceBuffer { buffer, memory, size })
    }

    pub fn destroy(&self, d: &Device) {
        unsafe {
            d.device.destroy_buffer(self.buffer, None);
            d.device.free_memory(self.memory, None);
        }
    }
}

/// Compute writes → compute reads (and transfer).
pub fn compute_barrier(d: &Device, cmd: vk::CommandBuffer) {
    let m = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_READ)];
    unsafe { d.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&m)) };
}

/// Ownership of an external buffer: from the foreign owner (`acquire`) or back to it.
pub fn foreign_buffer(d: &Device, cmd: vk::CommandBuffer, buffer: vk::Buffer, acquire: bool) {
    let b = vk::BufferMemoryBarrier2::default().buffer(buffer).offset(0).size(vk::WHOLE_SIZE);
    let b = if acquire {
        b.src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
            .dst_queue_family_index(d.queue_family)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
    } else {
        b.src_queue_family_index(d.queue_family)
            .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
    };
    let bs = [b];
    unsafe { d.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().buffer_memory_barriers(&bs)) };
}

pub fn push_constants<T: Copy>(d: &Device, cmd: vk::CommandBuffer, p: &Pipeline, value: &T) {
    let bytes = unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) };
    unsafe { d.device.cmd_push_constants(cmd, p.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes) };
}
