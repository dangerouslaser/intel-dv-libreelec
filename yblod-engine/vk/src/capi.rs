//! The C API (include/dv_engine_vk.h). Phase 1: the synchronous subset the
//! comparison harness uses.

use crate::engine::{Engine, FrameIn};
use crate::plan::PlanIn;
use std::os::raw::c_void;

#[repr(C)]
pub struct DvvkFrame {
    pub plan: PlanIn,
    pub bl_surface: u32,
    pub el_surface: u32,
    pub metadata: *const u8,
    pub metadata_size: usize,
    pub output: u32,
    pub flags: u32, // DVVK_FRAME_AUTO_OVERLAY
    pub release_wait: u64,
}

const DVVK_BACKEND: i32 = -8;
const DVVK_OVERLAY_FLIP_Y: u32 = 1;

fn log_error(what: &str, e: impl std::fmt::Display) {
    eprintln!("dv-engine-vk: {what}: {e}");
}

/// # Safety
/// `out` must be valid; `va_display` a VADisplay on the same device as `drm_fd`.
#[no_mangle]
pub unsafe extern "C" fn dvvk_create(drm_fd: i32, va_display: *mut c_void, out: *mut *mut Engine) -> i32 {
    match Engine::new(drm_fd, va_display) {
        Ok(e) => {
            *out = Box::into_raw(Box::new(e));
            0
        }
        Err(e) => {
            log_error("create", e);
            DVVK_BACKEND
        }
    }
}

/// # Safety
/// `e` from dvvk_create; `fd` a linear XRGB8888 3840x2160 dmabuf (dup'd).
#[no_mangle]
pub unsafe extern "C" fn dvvk_set_output(e: *mut Engine, slot: u32, fd: i32, stride: u32) -> i32 {
    match (*e).set_output(slot, fd, stride) {
        Ok(()) => 0,
        Err(err) => {
            log_error("output", err);
            DVVK_BACKEND
        }
    }
}

/// # Safety
/// `e` from dvvk_create; `frame` and its metadata valid for the call.
#[no_mangle]
pub unsafe extern "C" fn dvvk_render_sync(e: *mut Engine, frame: *const DvvkFrame, error_flags: *mut u32) -> i32 {
    let r = match (*e).render(&frame_in(&*frame)) {
        Ok(s) => s,
        Err(err) => {
            log_error("render", err);
            DVVK_BACKEND
        }
    };
    if !error_flags.is_null() {
        *error_flags = (*e).last_error_flags;
    }
    r
}

fn frame_in(f: &DvvkFrame) -> FrameIn<'_> {
    FrameIn {
        plan: f.plan,
        bl_surface: f.bl_surface,
        el_surface: f.el_surface,
        metadata: unsafe { std::slice::from_raw_parts(f.metadata, f.metadata_size) },
        output: f.output,
        auto_overlay: f.flags & 1 != 0,
        release_wait: f.release_wait,
    }
}

/// linux-drm-syncobj-v1: DRM syncobj fds of the frames' timeline (acquire
/// point = ticket) and of a release timeline (the caller's release points;
/// pass the one a buffer was last committed with as release_wait). The
/// caller owns both fds.
/// # Safety
/// `e` from dvvk_create; out pointers writable.
#[no_mangle]
pub unsafe extern "C" fn dvvk_timelines(e: *mut Engine, acquire: *mut i32, release: *mut i32) -> i32 {
    match (*e).timelines() {
        Ok((a, r)) => {
            *acquire = a;
            *release = r;
            0
        }
        Err(err) => {
            log_error("timelines", err);
            DVVK_BACKEND
        }
    }
}

/// mpv's subtitle / OSD bitmaps for the following frames (0 parts: none);
/// the bitmaps are copied before this returns.
/// # Safety
/// `e` from dvvk_create; `parts` (and their bitmaps) valid for the call.
#[no_mangle]
pub unsafe extern "C" fn dvvk_set_subtitles(e: *mut Engine, parts: *const crate::overlay::SubPart, num: u32) -> i32 {
    let slice = if num == 0 { &[][..] } else { std::slice::from_raw_parts(parts, num as usize) };
    match (*e).set_subtitles(slice) {
        Ok(()) => 0,
        Err(err) => {
            log_error("subtitles", err);
            -2
        }
    }
}

/// The shell's OSD buffer (KWin's forwarded overlay) for the following
/// frames; NULL: none. The engine dups what it keeps; the caller keeps the
/// buffer itself until dvvk_overlay_ticket(id)'s frame is done.
/// # Safety
/// `e` from dvvk_create; `buf` NULL or valid for the call.
#[no_mangle]
pub unsafe extern "C" fn dvvk_set_overlay(e: *mut Engine, buf: *const crate::overlay::Dmabuf, id: u64) -> i32 {
    dvvk_set_overlay_ex(e, buf, id, 0)
}

/// dvvk_set_overlay with flags: DVVK_OVERLAY_FLIP_Y (1), the buffer's rows
/// bottom first (a GL framebuffer's texture). Each frame that reads the
/// buffer leaves its fence on it as an implicit read fence, so the caller
/// may draw into it again at once if its renderer honours implicit sync.
/// # Safety
/// As dvvk_set_overlay.
#[no_mangle]
pub unsafe extern "C" fn dvvk_set_overlay_ex(e: *mut Engine, buf: *const crate::overlay::Dmabuf, id: u64, flags: u32) -> i32 {
    match (*e).set_shell(buf.as_ref(), id, flags & DVVK_OVERLAY_FLIP_Y != 0) {
        Ok(()) => 0,
        Err(err) => {
            log_error("overlay", err);
            -2
        }
    }
}

/// 1 if a fence could not be attached as an implicit fence (an older
/// kernel): wait for each ticket before committing the output or drawing
/// into the overlay again. 0 otherwise.
/// # Safety
/// `e` from dvvk_create.
#[no_mangle]
pub unsafe extern "C" fn dvvk_implicit_failed(e: *mut Engine) -> i32 {
    (*e).implicit_failed as i32
}

/// The ticket of the frame that read overlay `id`, once (0: not read).
/// # Safety
/// `e` from dvvk_create.
#[no_mangle]
pub unsafe extern "C" fn dvvk_overlay_ticket(e: *mut Engine, id: u64) -> u64 {
    (*e).overlay_ticket(id)
}

/// Plans and submits a frame without waiting. 0 and `*ticket`, or a dv_status.
/// The output dmabuf carries the frame's implicit write fence on return.
/// # Safety
/// `e` from dvvk_create; `frame`/metadata valid for the call; `ticket` writable.
#[no_mangle]
pub unsafe extern "C" fn dvvk_submit(e: *mut Engine, frame: *const DvvkFrame, ticket: *mut u64) -> i32 {
    match (*e).submit(&frame_in(&*frame)) {
        Ok((status, t)) => {
            *ticket = t;
            status
        }
        Err(err) => {
            log_error("submit", err);
            DVVK_BACKEND
        }
    }
}

/// Never blocks: *done 1 when the frame is finished; *flags its error flags
/// (1 range, 2 overflow, 0x100 overlay colour) once done.
/// # Safety
/// `e` from dvvk_create; out pointers writable.
#[no_mangle]
pub unsafe extern "C" fn dvvk_poll(e: *mut Engine, ticket: u64, done: *mut i32, flags: *mut u32) -> i32 {
    match (*e).poll(ticket) {
        Ok((d, f)) => {
            *done = d as i32;
            if !flags.is_null() {
                *flags = f;
            }
            0
        }
        Err(err) => {
            log_error("poll", err);
            DVVK_BACKEND
        }
    }
}

/// How many of the overlay's pixels frame `ticket` found changed since the
/// frame before (the shell's layer read by the pack, the cache kept from
/// the frame before); -1 when not known. Poll once the ticket is done.
/// # Safety
/// `e` from dvvk_create.
#[no_mangle]
pub unsafe extern "C" fn dvvk_overlay_changes(e: *mut Engine, ticket: u64) -> i64 {
    match (*e).overlay_changes(ticket) {
        Ok(Some(n)) => n as i64,
        _ => -1,
    }
}

/// Blocks up to `timeout_ns`: 1 done, 0 timed out, < 0 failure.
/// # Safety
/// `e` from dvvk_create.
#[no_mangle]
pub unsafe extern "C" fn dvvk_wait(e: *mut Engine, ticket: u64, timeout_ns: u64) -> i32 {
    match (*e).wait(ticket, timeout_ns) {
        Ok(d) => d as i32,
        Err(err) => {
            log_error("wait", err);
            DVVK_BACKEND
        }
    }
}

/// Counters: frames whose decode was waited for on the GPU (the surface's
/// write fence), and by vaSyncSurface.
/// # Safety
/// `e` from dvvk_create.
#[no_mangle]
pub unsafe extern "C" fn dvvk_decode_waits(e: *mut Engine, fence: *mut u64, vasync: *mut u64) {
    *fence = (*e).decode_fence_waits;
    *vasync = (*e).decode_vasync_waits;
}

/// # Safety
/// `e` from dvvk_create; `rgba` 2160 rows of `stride` bytes.
#[no_mangle]
pub unsafe extern "C" fn dvvk_set_overlay_rgba(e: *mut Engine, rgba: *const u8, stride: usize) {
    (*e).set_overlay_rgba(std::slice::from_raw_parts(rgba, stride * 2160), stride);
}

/// # Safety
/// `e` from dvvk_create.
#[no_mangle]
pub unsafe extern "C" fn dvvk_reset(e: *mut Engine) {
    (*e).reset();
}

/// # Safety
/// `e` from dvvk_create, not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn dvvk_destroy(e: *mut Engine) {
    if !e.is_null() {
        drop(Box::from_raw(e));
    }
}
