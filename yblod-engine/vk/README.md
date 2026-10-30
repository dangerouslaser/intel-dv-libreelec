# yblod-engine/vk: the source-tunnel engine on Vulkan compute

`libdv_engine_vk.so.0`, a Rust crate (`ash`, GLSL compute shaders compiled to
SPIR-V by `glslc`) that does what `libdv_source_private.so.1` does for the
source tunnel: reconstruction, the pack, letterbox, metadata packets and the
GUI overlay. The tables and metadata come from yblod's own host C in `../src`,
compiled in by `build.rs` (`YBLOD_ENGINE_SRC`), so both engines see the same
numbers. The C API is `include/dv_engine_vk.h`.

Imported from big-fella's `dv-engine-vk` at commit 40ef696 (2026-10-27), where
it was ported from this engine and checked against OpenCL on the N150 frame by
frame: every movie byte identical on every route and both EL scalers; under
the GUI, about 0.2 % of pixels differ by at most one luma or two chroma codes
(`pow` in the sRGB/PQ conversion). big-fella's docs/vulkan-engine.md has the
design and the results. Its nearest EL scaler patch is folded into `../src`.

In this tree Kodi uses it when *Master Reference engine* is YBLOD (Vulkan)
(packages/graphics/yblod-engine-vk; Kodi patch
`kodi-9999-zzzzz-yblod-master-engines.patch`), always with `cb1_contract` set:
CB1's Master Reference processing, byte for byte. YBLOD's own processing
(contract unset) stays in the code but isn't used here; see `../README.md`.

Test tools (`tools/`): `dvvk-compare` (OpenCL vs Vulkan on decoded frames),
`dvvk-routes`, `dvvk-fence`, `vkprobe`, `regress.sh`, and the `dvvk-interop`
binary in the crate. They aren't in the image.
