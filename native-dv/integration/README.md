# Experimental Kodi adapter

`kodi-adapter.patch` exports the native engine's Kodi integration as source,
without importing another repository's history. It includes GPU-resident
VAAPI input, enhancement-layer ownership, asynchronous reconstruction,
two-buffer direct scanout, source-domain GUI composition, GUI caching and
startup/seek/end-of-file handling.

The patch targets Kodi `28ea2eac1eb7af8fdcbd2672d933ce87f594be79` **after**
this repository's existing common and Generic Kodi patches. It is deliberately
not in the automatically applied patch directory yet. The normal image build
does not select this engine or install its additional runtime dependencies.
This is inspectable/buildable integration work, not a ready-to-install release.

## Build and runtime

1. Prepare Kodi with the revision and existing patches above, using this
   repository's matching FFmpeg and libplacebo patches and staged CB1 sources.
2. From that prepared Kodi tree, apply this patch with `git apply --check`
   followed by `git apply`, or the equivalent `patch -p1` workflow.
3. Build Kodi's GBM/GLES target with `ENABLE_DVBRIDGE=ON`. The qualified
   integration build used `CB1_ENABLE_HDR10_AI=OFF`; the optional AI build
   configuration has not been qualified with this adapter.
4. Build/install the sibling native renderer using its CMake instructions,
   plus the pinned OpenCL loader, Intel compute runtime and compiler packages.
   These are separate from the VAAPI media driver. Matching FFmpeg headers
   and decoder runtime are required, not merely a library with the right name.
5. Use `runtime.env` as an explicit test configuration for the Kodi service,
   adjusting library/resource paths if the installation prefix differs. Create
   the writable cache directory before launch. Do not replace the system's
   Kodi resources with resources from a different Kodi revision.

The environment file assumes `/usr` installation. It is **not installed or
enabled automatically**. Legacy `DV_PRIVATE_*` names and the library SONAME
are retained for compatibility with the tested adapter/engine interface.
Additional profiling, readback, half-resolution GUI and GUI-tile experiments
are disabled in this configuration.

## Qualification scope

The exported code was built and exercised on an Intel N150 with the separately
CMake-built native engine and pinned Intel OpenCL runtime. The exported patch
was also applied to a clean snapshot of the prepared Kodi tree; its complete
resulting source tree matches the reviewed adapter tree. Three subsequent
comment-only license-header changes do not change the tested executable code.

- P5 and P8.4: short 4K24 playback, controls open/closed, pause, paused seek,
  resume and natural end-of-file passed.
- P7 FEL: the same checks passed with Saving Private Ryan.
- All nine measured normal-playback windows had 101 distinct successful
  display commits over about 4.2 seconds, no backwards commits and maximum
  source-PTS gaps of one frame. No route exits or logged queue skips occurred.
- Six sampled packed frames per profile (18 total) matched CPU packing
  byte-for-byte. This reference starts from the reconstructed planes: it does
  **not** independently certify reconstruction or Dolby conformance.
- A late output-admission issue that stalled P5 after its first frame is fixed:
  the renderer now refreshes its GUI/video routing when composition activates.

Full-image build/install, broader profile requalification, mode switching,
optional AI functionality and YBLOD-specific integration remain unfinished.
The source route targets 3840x2160 progressive output with admitted source
geometry and metadata; it is not a general arbitrary-resolution renderer.
HDR bitmap overlays and native composite screenshots remain unsupported in
this route. Tests here do not establish absolute GUI colour accuracy, physical
audio/video synchronization, or every display/HDMI combination.

The adapter's new native integration code uses GPL-3.0-or-later; existing Kodi
and CB1 notices remain applicable to the surrounding source. See the renderer's
LICENSE and SOURCES.md for scope and dependency notices.
