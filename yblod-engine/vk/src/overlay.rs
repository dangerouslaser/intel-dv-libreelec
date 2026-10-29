//! The overlay's sources in mpv: subtitle / OSD bitmaps (an atlas the CPU
//! fills, composited on the GPU) and the shell's OSD layer (KWin's forwarded
//! dmabuf, imported). The compose pass turns them into the RGBA overlay the
//! tile and colour passes prepare for the pack.

use crate::device::{Device, Error, Result};
use crate::gpu::HostBuffer;
use crate::interop::{dmabuf, va};
use ash::vk;
use std::collections::HashMap;

pub const TILE: u32 = 64;
const MAX_PARTS: usize = 4096;
const PARTS_BYTES: u64 = (MAX_PARTS * 48) as u64;      // at 0
const LISTS_OFFSET: u64 = 256 * 1024;
const LISTS_BYTES: u64 = 4 * 1024 * 1024;
const BITMAPS_OFFSET: u64 = LISTS_OFFSET + LISTS_BYTES;
const ATLAS_BYTES: u64 = 24 * 1024 * 1024;

/// One subtitle part (include/dv_engine_vk.h dvvk_sub_part).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SubPart {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
    pub dw: u32,
    pub dh: u32,
    pub stride: u32,
    pub format: u32, // 1 libass A8 + colour, 2 BGRA premultiplied
    pub color: u32,
    pub data: *const u8,
}

/// A dmabuf (include/dv_engine_vk.h dvvk_dmabuf).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Dmabuf {
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub modifier: u64,
    pub num_planes: u32,
    pub fd: [i32; 4],
    pub offset: [u32; 4],
    pub stride: [u32; 4],
}

pub struct Shell {
    pub image: dmabuf::ImportedImage,
    pub fd: i32,      // a dup of plane 0's fd: its implicit fence
    pub scaled: bool,
}

pub const SLOTS: usize = 6;   // > frames in flight: a change never waits for the GPU

pub struct Sources {
    pub atlas: Vec<HostBuffer>,
    atlas_ticket: [u64; SLOTS],
    pub subs_slot: usize,
    pub subs_parts: u32,
    pub tiles_x: u32,
    shells: HashMap<u64, Shell>,   // by dmabuf inode
    pub shell: Option<u64>,        // the current one's inode
    pub shell_id: u64,             // the caller's id for it
    pub shell_flip: bool,          // its rows bottom first (a GL framebuffer's texture)
    pub consumed: HashMap<u64, u64>, // caller id -> ticket of the frame that read it
    pub dirty: bool,
    subs_bbox: Option<[u32; 4]>,   // x0, y0, x1, y1 of the current parts
    pub dirty_rect: Option<[u32; 4]>, // what the next compose must redo; None = all
}

fn union(a: Option<[u32; 4]>, b: Option<[u32; 4]>) -> Option<[u32; 4]> {
    match (a, b) {
        (Some(a), Some(b)) => Some([a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]),
        (a, None) => a,
        (None, b) => b,
    }
}

impl Sources {
    pub fn new(d: &Device) -> Result<Self> {
        let mut atlas = Vec::new();
        for _ in 0..SLOTS {
            atlas.push(HostBuffer::new(d, ATLAS_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER)?);
        }
        Ok(Sources {
            atlas,
            atlas_ticket: [0; SLOTS],
            subs_slot: 0,
            subs_parts: 0,
            tiles_x: 0,
            shells: HashMap::new(),
            shell: None,
            shell_id: 0,
            shell_flip: false,
            consumed: HashMap::new(),
            dirty: false,
            subs_bbox: None,
            dirty_rect: Some([0, 0, 0, 0]),
        })
    }

    pub fn has_content(&self) -> bool {
        self.subs_parts > 0 || self.shell.is_some()
    }

    pub fn shell(&self) -> Option<&Shell> {
        self.shell.and_then(|k| self.shells.get(&k))
    }

    /// The parts into the free atlas slot (`wait` blocks until a ticket is
    /// done: only when both slots are still being read).
    /// What must be composited again: the old and new parts' area when only
    /// the subtitles changed (whole canvas after a shell change).
    fn mark(&mut self, rect: Option<[u32; 4]>, all: bool) {
        if all || (self.dirty && self.dirty_rect.is_none()) {
            self.dirty_rect = None;
        } else if self.dirty {
            self.dirty_rect = union(self.dirty_rect, rect);
        } else {
            self.dirty_rect = Some(rect.unwrap_or([0, 0, 0, 0]));
        }
        self.dirty = true;
    }

    pub fn set_subtitles(&mut self, parts: &[SubPart], canvas: (u32, u32), wait: impl Fn(u64)) -> Result<()> {
        let mut bbox: Option<[u32; 4]> = None;
        for p in parts {
            let x0 = p.x.clamp(0, canvas.0 as i32) as u32;
            let y0 = p.y.clamp(0, canvas.1 as i32) as u32;
            let x1 = (p.x + p.dw as i32).clamp(0, canvas.0 as i32) as u32;
            let y1 = (p.y + p.dh as i32).clamp(0, canvas.1 as i32) as u32;
            if x1 > x0 && y1 > y0 {
                bbox = union(bbox, Some([x0, y0, x1, y1]));
            }
        }
        let old = self.subs_bbox;
        self.subs_bbox = bbox;
        self.mark(union(old, bbox), false);
        if parts.is_empty() {
            self.subs_parts = 0;
            return Ok(());
        }
        if parts.len() > MAX_PARTS {
            return Err(Error::Unsupported("too many subtitle parts".into()));
        }
        let slot = (self.subs_slot + 1) % SLOTS;
        wait(self.atlas_ticket[slot]);
        let tiles_x = canvas.0.div_ceil(TILE);
        let tiles_y = canvas.1.div_ceil(TILE);
        let ntiles = (tiles_x * tiles_y) as usize;
        let buf = self.atlas[slot].bytes_mut();
        // bitmaps, packed rows
        let mut at = BITMAPS_OFFSET as usize;
        let mut table = Vec::with_capacity(parts.len() * 12);
        let mut per_tile: Vec<Vec<u32>> = vec![Vec::new(); ntiles];
        for (n, p) in parts.iter().enumerate() {
            let bpp = if p.format == 1 { 1 } else { 4 };
            let row = (p.w * bpp) as usize;
            let bytes = row * p.h as usize;
            let stride = row.next_multiple_of(4);
            if at + stride * p.h as usize > buf.len() {
                return Err(Error::Unsupported("subtitle atlas full".into()));
            }
            for y in 0..p.h as usize {
                let src = unsafe { std::slice::from_raw_parts(p.data.add(y * p.stride as usize), row) };
                buf[at + y * stride..at + y * stride + row].copy_from_slice(src);
            }
            let _ = bytes;
            // offsets relative to the bitmaps' range (the shader's binding)
            let rel = (at - BITMAPS_OFFSET as usize) as u32;
            table.extend_from_slice(&[p.x as u32, p.y as u32, p.w, p.h, p.dw, p.dh, stride as u32, rel, p.format, p.color, 0, 0]);
            at += stride * p.h as usize;
            // the tiles its destination rectangle touches
            let x0 = p.x.max(0) as u32 / TILE;
            let y0 = p.y.max(0) as u32 / TILE;
            let x1 = ((p.x + p.dw as i32).min(canvas.0 as i32) - 1).max(0) as u32 / TILE;
            let y1 = ((p.y + p.dh as i32).min(canvas.1 as i32) - 1).max(0) as u32 / TILE;
            if p.x + (p.dw as i32) <= 0 || p.y + (p.dh as i32) <= 0 || p.x >= canvas.0 as i32 || p.y >= canvas.1 as i32 {
                continue;
            }
            for ty in y0..=y1.min(tiles_y - 1) {
                for tx in x0..=x1.min(tiles_x - 1) {
                    per_tile[(ty * tiles_x + tx) as usize].push(n as u32);
                }
            }
        }
        let table_bytes = unsafe { std::slice::from_raw_parts(table.as_ptr() as *const u8, table.len() * 4) };
        buf[..table_bytes.len()].copy_from_slice(table_bytes);
        let mut lists: Vec<u32> = Vec::with_capacity(ntiles + 1 + parts.len());
        let mut offset = (ntiles + 1) as u32;
        for t in &per_tile {
            lists.push(offset);
            offset += t.len() as u32;
        }
        lists.push(offset);
        for t in &per_tile {
            lists.extend_from_slice(t);
        }
        if (lists.len() * 4) as u64 > LISTS_BYTES {
            return Err(Error::Unsupported("subtitle tile lists full".into()));
        }
        let lb = unsafe { std::slice::from_raw_parts(lists.as_ptr() as *const u8, lists.len() * 4) };
        buf[LISTS_OFFSET as usize..LISTS_OFFSET as usize + lb.len()].copy_from_slice(lb);
        self.subs_slot = slot;
        self.subs_parts = parts.len() as u32;
        self.tiles_x = tiles_x;
        Ok(())
    }

    /// The compose pass reading slot `subs_slot` was submitted as `ticket`.
    pub fn used(&mut self, ticket: u64) {
        self.atlas_ticket[self.subs_slot] = ticket;
        if self.shell.is_some() {
            self.consumed.insert(self.shell_id, ticket);
            // A caller that never asks (Kodi: a new id every frame) must not
            // grow it without bound; ids this old are long done.
            if self.consumed.len() > 1024 {
                let recent = self.shell_id.saturating_sub(1024);
                self.consumed.retain(|&id, _| id > recent);
            }
        }
    }

    pub fn ranges() -> [(u64, u64); 3] {
        [(BITMAPS_OFFSET, ATLAS_BYTES - BITMAPS_OFFSET), (0, PARTS_BYTES), (LISTS_OFFSET, LISTS_BYTES)]
    }

    /// The shell's OSD layer: a new buffer (imported once per allocation),
    /// or none.
    pub fn set_shell(&mut self, d: &Device, buf: Option<&Dmabuf>, id: u64, flip: bool, idle: impl Fn()) -> Result<()> {
        self.mark(None, true);
        let Some(b) = buf else {
            self.shell = None;
            return Ok(());
        };
        let inode = va::dmabuf_inode(b.fd[0]);
        if !self.shells.contains_key(&inode) {
            let (format, alpha_one) = match b.format {
                0x3432_5241 => (vk::Format::B8G8R8A8_UNORM, false), // AR24
                0x3432_5258 => (vk::Format::B8G8R8A8_UNORM, true),  // XR24
                0x3432_4241 => (vk::Format::R8G8B8A8_UNORM, false), // AB24
                0x3432_4258 => (vk::Format::R8G8B8A8_UNORM, true),  // XB24
                f => return Err(Error::Unsupported(format!("overlay format {f:#010x}"))),
            };
            if (1..b.num_planes as usize).any(|i| va::dmabuf_inode(b.fd[i]) != inode) {
                return Err(Error::Unsupported("overlay planes in separate buffers".into()));
            }
            let planes: Vec<(u64, u64)> = (0..b.num_planes as usize).map(|i| (b.offset[i] as u64, b.stride[i] as u64)).collect();
            if self.shells.len() >= 8 {
                // a new set of buffers (resized surface): drop the old imports
                idle();
                for (_, s) in self.shells.drain() {
                    s.image.destroy(d);
                    unsafe { libc::close(s.fd) };
                }
            }
            let image = dmabuf::import_image(d, b.fd[0], &planes, b.modifier, format, b.width, b.height, alpha_one)?;
            let fd = unsafe { libc::fcntl(b.fd[0], libc::F_DUPFD_CLOEXEC, 0) };
            self.shells.insert(inode, Shell { image, fd, scaled: b.width != 3840 || b.height != 2160 });
        }
        self.shell = Some(inode);
        self.shell_id = id;
        self.shell_flip = flip;
        Ok(())
    }

    pub fn destroy(&mut self, d: &Device) {
        for a in &self.atlas {
            a.destroy(d);
        }
        for (_, s) in self.shells.drain() {
            s.image.destroy(d);
            unsafe { libc::close(s.fd) };
        }
    }
}
