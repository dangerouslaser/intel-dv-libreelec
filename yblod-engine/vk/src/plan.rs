//! Rust view of yblod/plan.h (the CPU half of a frame, done by yblod's code).

pub const ROUTE_DUAL: u32 = 1;
pub const ROUTE_SINGLE: u32 = 2;
pub const ROUTE_PIECEWISE: u32 = 3;
pub const CHROMA_POLY: u32 = 0;
pub const CHROMA_MMR128: u32 = 1;
pub const CHROMA_MMR96: u32 = 2;
pub const CHROMA_MMR64_LUT: u32 = 3;
pub const PACK_SITED: u32 = 0;
pub const PACK_422: u32 = 1;
pub const PACK_RESIZED: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PlanIn {
    pub width: u32,
    pub height: u32,
    pub x: u32,
    pub y: u32,
    pub dw: u32,
    pub dh: u32,
    pub chroma_top_left: u32,
    pub has_el: u32,
    pub nv12: u32,
    pub el_scaler: u32,
    pub fel: u32,
    pub packet_id: u32,
    pub refresh: u32,
    pub overlay: u32,
    pub overlay_visible_before: u32,
    pub pts: i64,
    pub rgb_to_lms: [f64; 9],
    pub white_nits: f64,
}

#[repr(C)]
pub struct Plan {
    pub mmr_lut: [[[i64; 1024]; 3]; 2],
    pub reshape: [u8; 4544],
    pub residual: [[i32; 1024]; 3],
    pub mmr_params: [[u32; 96]; 2],
    pub route: u32,
    pub width: u32,
    pub height: u32,
    pub el_scale: u32,
    pub nearest: u32,
    pub chroma_phase: u32,
    pub chroma_kind: [u32; 2],
    pub dual: u32,
    pub horizontal_fused: u32,
    pub equal: u32,
    pub row_first: u32,
    pub row_last: u32,
    pub layout: u32,
    pub sampling: u32,
    pub depth: u32,
    pub order2: u32,
    pub narrow_mask: [u32; 4],
    pub pack: u32,
    pub active: [u32; 4],
    pub black: [u32; 4],
    pub placement: [u32; 4],
    pub top_left: u32,
    pub source_w: u32,
    pub source_h: u32,
    pub packet_count: u32,
    pub overlay_visible: u32,
    pub reserved: [u32; 7],
    pub gui_coeff: [f32; 32],
    pub mapping: [u16; 1024],
    pub chroma_mapping: [[u16; 1024]; 2],
    pub single_table: [[u16; 1024]; 3],
    pub luma_table: [u16; 1024],
    pub packets: [u8; 640],
}

extern "C" {
    fn dvvk_plan_frame(input: *const PlanIn, metadata: *const u8, size: usize, out: *mut Plan) -> i32;
    fn dvvk_plan_layout(out: *mut usize);
}

impl Plan {
    pub fn boxed() -> Box<Plan> {
        // ~100 KB: zeroed on the heap, never on the stack.
        unsafe { Box::new_zeroed().assume_init() }
    }

    /// yblod's planning for one frame; 0 or a dv_status.
    pub fn fill(&mut self, input: &PlanIn, metadata: &[u8]) -> i32 {
        unsafe { dvvk_plan_frame(input, metadata.as_ptr(), metadata.len(), self) }
    }
}

/// The C and Rust layouts agree.
pub fn layout_matches() -> bool {
    let mut c = [0usize; 6];
    unsafe { dvvk_plan_layout(c.as_mut_ptr()) };
    let r = [
        std::mem::size_of::<Plan>(),
        std::mem::offset_of!(Plan, residual),
        std::mem::offset_of!(Plan, route),
        std::mem::offset_of!(Plan, mapping),
        std::mem::offset_of!(Plan, packets),
        std::mem::offset_of!(Plan, packet_count),
    ];
    c == r
}
