# Native source-domain renderer — work in progress

See [LICENSE](LICENSE) and [source references](SOURCES.md) for distribution
terms, implementation scope and limitations.

This directory is a selected C/OpenCL renderer source set, not a complete Kodi
or LibreELEC image. It keeps decoded VAAPI surfaces on the GPU, reconstructs
the source signal, and packs frames with matching metadata for TV-led output.
The existing library name is retained for adapter ABI compatibility.

The standalone CMake build and short N150 playback/packing checks pass; see
[BUILD-VALIDATION.txt](BUILD-VALIDATION.txt) for the exact scope and results.
The [experimental Kodi adapter](integration/README.md) is now available as a
source patch, with build notes and runtime configuration.
This branch's Generic image now includes and selects the experimental renderer,
and has built and booted on the N150. See the adapter's qualification reports
for measured results, exact image identification and remaining issues, including
the [pre2 packaged checks](integration/PRE2-20261009.txt). The engine remains
experimental; a pre-release does not imply all-media qualification or a main merge.
`packages/graphics/yblod-engine/package.mk` packages the bundled source and its
runtime dependencies as a Generic Kodi dependency. Changes under `yblod-engine/`
participate in LibreELEC's package rebuild stamp.

Everything specific to the Dolby Vision pipeline lives in this folder: the engine
(`src/`, `tests/`, `integration/`), its patches for Kodi, FFmpeg, libplacebo and the
kernel (`patches/<package>/`, linked into LibreELEC's patch folders), tools (`tools/`),
docs (`docs/`) and licences (`licenses/`, `LICENSES/`). See `../yblod/README.md`.

Build on Linux with CMake, OpenCL headers/loader, libva, and the matching modified
FFmpeg headers from `yblod-engine/patches/ffmpeg`:

```
cmake -S yblod-engine -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --parallel 4
ctest --test-dir build-native --output-on-failure
```

CPU contract tests can be built without those playback dependencies using
`-DNATIVE_DV_BUILD_RENDERER=OFF`. These synthetic tests cover the order-two
specialization gate, geometry and output-slot ownership; they are not movie
playback, GPU accuracy or Dolby conformance tests.

If pkg-config points to stock FFmpeg headers, provide the matching include
root with `-DNATIVE_DV_FFMPEG_INCLUDE_DIR=/path/to/matched/include`. It takes
precedence for both the configuration probe and the renderer compilation.

The renderer does not silently accept unmodified FFmpeg headers: the raw RPU
extension fields are required to preserve metadata. Matching headers alone do
not establish the decoder's runtime ABI; integrated producer/consumer tests
remain necessary.

The shader bundle pins the accepted tunnel packer, without the rejected GUI
tile prototype. `DV_OVERLAY_TILES` must remain disabled. No arithmetic changes
are intended by this export. Source references and distribution terms accompany
the code. Qualification applies to the exact packaged image identified in each
report. Do not substitute an earlier test binary as proof that a different
integrated image was built or qualified.
