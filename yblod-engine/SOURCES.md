# Source scope and references

This is a selected source export of the project's C/OpenCL implementation,
not a transfer of its development repository or all diagnostic tooling.
The build lists its translation units explicitly. The installed resources
are the five shaders named in `CMakeLists.txt`.

Implementation references and boundaries:

- `pixel_luma.c`, `pixel_chroma_final.c`, `pixel_spatial.c`, the corresponding
  GPU kernels, and `single_reshape.c` implement reconstruction operations
  described by ETSI GS CCM 001 V1.1.1, including polynomial/MMR reshaping and
  residual composition. The enhancement-filter choice is the informative
  Annex B example, not a claim about every licensed player's active filter.
- `dv_ffmpeg_metadata.c` and `single_reshape_ffmpeg.c` consume FFmpeg's parsed
  Dolby metadata. They validate structure bounds and reject unsupported
  contracts instead of interpreting arbitrary memory as metadata. The build
  uses the matching raw-extension patches distributed elsewhere in this repo.
- `dv_cm4.c` cross-checks extension grammar against quietvoid's `dovi_tool`.
  It preserves the supplied fields while changing their representation; it
  does not implement the display's content-mapping algorithm.
- `source_tunnel.cl` packs the reconstructed samples; CPU code packetizes the
  accompanying metadata. The protocol basis is ETSI sections 6.3 and 6.4.
  The source representation and frame-bound metadata must agree. Merely
  engaging a TV's Dolby Vision mode is not an accuracy test.
- `source_overlay.c`/`.cl` transform premultiplied sRGB GUI colours into the
  current source domain. The movie does not take an extra RGB round trip.
- `dv_scanout_slots.c`, geometry helpers, and renderer/backend code implement
  this project's buffer ownership, validation and OpenCL/VAAPI integration.

References:

- [ETSI GS CCM 001 V1.1.1](https://www.etsi.org/deliver/etsi_gs/CCM/001_099/001/01.01.01_60/gs_ccm001v010101p.pdf)
- [FFmpeg Dolby metadata interface](https://ffmpeg.org/doxygen/trunk/dovi__meta_8h.html)
- [dovi_tool extension metadata](https://github.com/quietvoid/dovi_tool/tree/main/dolby_vision/src/rpu/extension_metadata/blocks)

These are implementation references, not a claim of independent clean-room
development or licensed-decoder conformance. Existing source comments and
limitations are retained. Historical optional experimental paths present in
the implementation are not all qualified; only explicitly recorded build and
runtime selections should be used for a release.

No movie fixtures, device captures, credentials or external binary-module
importer are part of this directory. The Kodi/FFmpeg integration and packaged
driver dependencies have their own source and license requirements.
