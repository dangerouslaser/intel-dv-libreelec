# Experimental YBLOD Kodi adapter

This patch targets Kodi `63063c8e67b1f162761763c0cbdb1a5dcfc3a47e` after
YBLOD's common and Generic patches on the LibreELEC resync commit
`2efc53dffbdc4b2fb05e1a2e6b74f33b2479c4c5`.

It ports the previously qualified native adapter to YBLOD's own bridge. It
does not require replacing that bridge with the newer CB1 integration.
The standalone reconstruction engine is unchanged from the source published
in `dangerouslaser/intel-dv-libreelec` at
`4e0ffc4c0f4f351f44de9c02a67a39c628558f6e`.

The adapter keeps decoded VAAPI surfaces on the GPU, manages asynchronous
reconstruction and two output buffers, and composes video controls in the
source colour representation. The existing DV menu path is retained; the
legacy QSV VPP scaling option is disabled. Native movie-overlay processing and GUI reuse exclude the
standalone menu, which keeps its existing packing and appearance controls.

## Current status

The complete Generic image has built and booted on the N150. The installed
engine has passed selected current-profile controls/recovery checks, and initial
GPU timestamp baselines are available in
[QUALIFICATION-20261009.txt](QUALIFICATION-20261009.txt). The reproduced early
P7 presentation gaps were subsequently fixed and retested in
[STARTUP-HANDOFF-20261009.txt](STARTUP-HANDOFF-20261009.txt), which also records
remaining controls-open frame drops in a 59.94 Hz sample. These results apply
to the exact images identified in each report. Pre2 packaging and final smoke
results are recorded in [PRE2-20261009.txt](PRE2-20261009.txt). A subsequent
visually reproduced VM menu-return defect and the pre3 target-recreation fix
are documented in [PRE3-20261009.txt](PRE3-20261009.txt); pre2's timing/signalling
checks did not establish correctness of the visible menu image.

The unchanged standalone source passes all three Release-mode CPU contract
tests with GCC 14.2 and CMake 4.4.4: specialization admission, geometry and
output-slot ownership. These are not GPU or Kodi playback tests.

The patch is `yblod-engine/patches/kodi/kodi-9999-yblod-03-native-source.patch`.
Generic builds apply it after the existing YBLOD QSV and menu patches and
install the engine and dependencies through Kodi's package recipe. This is
integration source; use the separately packaged pre-release for installation.

The subsequent `kodi-9999-yblod-04-native-startup.patch` establishes a valid
black DV signal before releasing playback startup. The final
`kodi-9999-yblod-05-remove-legacy-settings.patch` removes Quick Sync scaling and
Match Dolby hardware levels from the settings UI. Both legacy behaviors are
forced off, including for profiles with old saved values. This does not change
native reconstruction or disable hardware video decoding.

## Build and runtime

Use YBLOD's normal Generic x86-64 image build. The package recipes select the
GBM/GLES adapter, matching modified FFmpeg and libplacebo, the native renderer,
the OpenCL loader, and pinned Intel runtime/compiler dependencies. Stock
FFmpeg headers are not interchangeable with the matching metadata extension.

The native package installs `runtime.env` as `/usr/lib/kodi/native-dv.env`
and a Kodi service drop-in. Startup creates the writable cache directory.
Optional `/storage/.config/native-dv.env` entries override the packaged values.
For a legacy-renderer diagnostic, disable `DV_PRIVATE_SOURCE`,
`DV_PRIVATE_OVERLAY` and `DV_PRIVATE_DRAIN_EOF` together, then restart Kodi.
Disabling the source alone leaves native-only composition/drain hooks active.
Use the separate base image for a strict before/after integration comparison.
VAAPI driver selection is left to the system; qualification must verify the
actual Intel driver and OpenCL libraries loaded, not just environment values.

Per-frame engine diagnostics (MMR table cache, dual-MMR eligibility, direct
P010 and GPU timing lines) are off by default because they reach the Kodi
journal on every frame; set `DV_ENGINE_FRAME_TRACE=1` to enable them for a
diagnostic run. Startup configuration lines and errors are always logged.

Kodi's Settings > Player > Dolby Vision (Expert level) controls a curated set of these
switches and applies them as environment overrides at start and whenever a setting
changes (patch `kodi-9999-yblod-21-dv-engine-settings.patch`, mapping in
`DVEngineSettings.h`). For those switches the setting wins over `native-dv.env`:
`dvbridge.engine` (DV_PRIVATE_SOURCE, DV_PRIVATE_OVERLAY, DV_PRIVATE_DRAIN_EOF),
`dvbridge.letterboxskip` (DV_RECON_ACTIVE_ROWS), `dvbridge.inlineoverlay`
(DV_OVERLAY_INLINE), `dvbridge.tablecache` (DV_SINGLE/DUAL_TABLE_CACHE),
`dvbridge.guicache` (DV_PRIVATE_GUI_CACHE), `dvbridge.homecache`
(DV_PRIVATE_GUI_CACHE_HOME), `dvbridge.packahead` (DV_PRIVATE_ASYNC_PACK),
`dvbridge.halfgui` (DV_PRIVATE_GUI_HALF), `dvbridge.verify` (the verify switches below and
DV_PRIVATE_VERIFY), `dvbridge.frametrace` (DV_ENGINE_FRAME_TRACE, DV_PRIVATE_FRAME_TIMING)
and `dvbridge.gputiming` (DV_SOURCE_PROFILE_ASYNC, DV_OVERLAY_PROFILE). The hidden setting
`yblod.build` (always true) lets skins detect YBLOD with `System.GetBool(yblod.build)`.

Byte-exact performance switches (all on in `runtime.env`; set to 0 to disable):
`DV_RECON_ACTIVE_ROWS` reconstructs only the RPU L5 active rows plus a filter
margin; `DV_OVERLAY_INLINE` lets the overlay pack check the GUI pixel cache
itself instead of running a separate colour pass; `DV_SINGLE_TABLE_CACHE` and
`DV_DUAL_TABLE_CACHE` reuse reshaping tables while the composer config is
unchanged. Their diagnostics compare against the uncached/full-frame path:
`DV_VERIFY_ACTIVE_ROWS=N`, `DV_OVERLAY_INLINE_VERIFY=N`,
`DV_SINGLE_TABLE_CACHE_VERIFY=1` and `DV_DUAL_TABLE_CACHE_VERIFY=1`.
Kodi's per-frame startup timing lines (including the first-480-frame
`Private source async pack` line) likewise require `DV_PRIVATE_FRAME_TIMING=1`.

Legacy `DV_PRIVATE_*` names and the library SONAME are retained for ABI
compatibility. Readback, half-resolution GUI and rejected GUI-tile experiments
are disabled. Keep Kodi's resources and executable from the same build.

## Qualification boundaries

The report above records packaged-image profile, timing, reference, cold-cache,
EOF, ownership/recovery and menu/QMS smoke checks. These are bounded tests, not
all-media certification. The menu-off/passthrough P7 startup fix passed the
documented A/B and packaged-boot checks; 59.94 Hz controls-open playback still
requires work.
Optical menu appearance, transition blackout and physical audio synchronization
are not proven by driver readbacks. Pre-release testing is not general certification.

The route targets 3840x2160 progressive output with admitted source geometry
and metadata. HDR bitmap overlays and native composite screenshots remain
unsupported. Retired profiles are unsupported; profile 20 is deferred.
The existing engine tests are not independent Dolby conformance tests.

`MEASUREMENTS.md` is historical context comparing the prior and newer CB1
adapters. It is **not** a measurement of this YBLOD port.

The new adapter code uses GPL-3.0-or-later; existing Kodi and bridge notices
remain applicable. See the renderer's LICENSE and SOURCES.md.
