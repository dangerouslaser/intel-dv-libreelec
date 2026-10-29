//! Implicit fences on dmabufs as sync_files, and sync_files as semaphores.

use crate::device::{Device, Error, Result};
use ash::vk;
use std::os::fd::RawFd;

pub const DMA_BUF_SYNC_READ: u32 = 1;
pub const DMA_BUF_SYNC_WRITE: u32 = 2;

#[repr(C)]
struct SyncFileArg {
    flags: u32,
    fd: i32,
}

// _IOWR('b', 2, struct dma_buf_export_sync_file), _IOW('b', 3, struct dma_buf_import_sync_file)
const DMA_BUF_IOCTL_EXPORT_SYNC_FILE: libc::c_ulong = 0xC008_6202;
const DMA_BUF_IOCTL_IMPORT_SYNC_FILE: libc::c_ulong = 0x4008_6203;

/// The fences a reader (`DMA_BUF_SYNC_READ`: the writers') or a writer
/// (`DMA_BUF_SYNC_WRITE`: all) of the dmabuf would wait for, as a sync_file.
pub fn export_sync_file(dmabuf: RawFd, flags: u32) -> Result<RawFd> {
    let mut arg = SyncFileArg { flags, fd: -1 };
    let r = unsafe { libc::ioctl(dmabuf, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &mut arg) };
    if r != 0 || arg.fd < 0 {
        return Err(Error::Unsupported(format!(
            "DMA_BUF_IOCTL_EXPORT_SYNC_FILE: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(arg.fd)
}

/// Attaches a sync_file to the dmabuf as an implicit fence (a write fence with
/// `DMA_BUF_SYNC_WRITE`), for consumers that only know implicit sync.
pub fn import_sync_file(dmabuf: RawFd, sync_file: RawFd, flags: u32) -> Result<()> {
    let mut arg = SyncFileArg { flags, fd: sync_file };
    let r = unsafe { libc::ioctl(dmabuf, DMA_BUF_IOCTL_IMPORT_SYNC_FILE, &mut arg) };
    if r != 0 {
        return Err(Error::Unsupported(format!(
            "DMA_BUF_IOCTL_IMPORT_SYNC_FILE: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}

/// True if the fence fd (sync_file) has signalled.
pub fn signalled(fd: RawFd) -> bool {
    let mut p = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
    unsafe { libc::poll(&mut p, 1, 0) == 1 }
}

/// A binary semaphore whose next wait is on `sync_file` (consumed on success).
pub fn semaphore_from_sync_file(d: &Device, semaphore: vk::Semaphore, sync_file: RawFd) -> Result<()> {
    let info = vk::ImportSemaphoreFdInfoKHR::default()
        .semaphore(semaphore)
        .flags(vk::SemaphoreImportFlags::TEMPORARY)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
        .fd(sync_file);
    unsafe { d.semaphore_fd.import_semaphore_fd(&info) }?;
    Ok(())
}

/// A sync_file for a binary semaphore's pending signal.
pub fn sync_file_from_semaphore(d: &Device, semaphore: vk::Semaphore) -> Result<RawFd> {
    let info = vk::SemaphoreGetFdInfoKHR::default()
        .semaphore(semaphore)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    Ok(unsafe { d.semaphore_fd.get_semaphore_fd(&info) }?)
}

/// A binary semaphore that can export/import sync_files.
pub fn exportable_semaphore(d: &Device) -> Result<vk::Semaphore> {
    let mut export =
        vk::ExportSemaphoreCreateInfo::default().handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    let info = vk::SemaphoreCreateInfo::default().push_next(&mut export);
    Ok(unsafe { d.device.create_semaphore(&info, None) }?)
}
