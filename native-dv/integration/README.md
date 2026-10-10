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
2. From that prepared Kodi tree, apply `kodi-adapter.patch`, followed by
   `kodi-native-optimizations.patch`. The validated sequence uses
   `patch --batch --fuzz=0 -p1 < PATCH` for each file in that order.
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

### YBLOD 0.2 synchronization

The standalone engine source now matches YBLOD 0.2. The additional adapter patch
ports movie-only asynchronous packing, GUI revision reuse, unchanged Home raster
reuse, full redraw of the separate GUI target and native-route display coordination.
The engine also contains Level 5 overlay handling and transparent-pixel skipping.
Asynchronous overlays remain disabled. Normal builds retain this repository's
existing CB1 pipeline; selecting the native adapter remains an explicit step.

Common Kodi patches include live 10-bit surface selection, audio-sink recovery,
decoder-derived live frame rates, retaining an open live MPEG-TS connection and
buffered-duration-based audio startup. Native display coordination is in the
optional adapter patch because it depends on that adapter's rate-selection state.

YBLOD's persistent DV-menu restoration and QMS-specific changes are not copied
over this repository's different display-session implementation. Neither are
YBLOD packaging, update-channel configuration, nor its LibreELEC base resync.
See [synchronization checks](SYNC-0.2-20261010.txt). YBLOD playback performance
numbers must not be presented as newly measured CB1-adapter performance.

### YBLOD pre2/pre3 applicability (2026-10-10)

At the pre3 synchronization, the standalone `native-dv/src` source matched
YBLOD's pre3 engine. Those fixes were in YBLOD's separate Kodi adapter:

- Pre2 removes YBLOD's `dvbridge.qsvmode` and `dvbridge.matchhardware`
  settings. Neither setting exists in this CB1-based adapter.
- Pre3 recreates the movie-shared GUI framebuffer when returning directly to
  YBLOD's DV-enabled menu. This adapter has no `BeginDVMenu`/`m_dvMenu` path.
  Its successful `EndDVBridge()` calls `SetGuiCompositing(0)`, which already
  cleans up the GUI framebuffer and resets its dimensions and native GUI state.
- YBLOD's earlier startup-signal fix targets its own deferred-modeset/menu
  handoff. It is not a drop-in fix for this adapter's different output-session
  handling; YBLOD playback results do not qualify the CB1 adapter.

Consequently, these changes do not require a code port here. If a persistent
DV-menu path is added later, it must invalidate the movie-shared GUI target
at the handoff and include a visual play/stop regression check: successful
DRM signalling alone did not detect YBLOD's distorted-menu defect.

See the [YBLOD pre3 fix](https://github.com/dangerouslaser/libreelec-yblod/commit/7559397e7d04688d5f1cdb761905db1151ae9721)
and its [qualification report](https://github.com/dangerouslaser/libreelec-yblod/blob/c01d3e39/native-dv/integration/PRE3-20261009.txt).
This applicability review is source inspection, not a new CB1 build or playback test.

### Existing adapter qualification

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
