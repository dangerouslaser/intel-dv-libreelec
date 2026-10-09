# Native source-domain renderer — work in progress

See [LICENSE](LICENSE) and [source references](SOURCES.md) for distribution
terms, implementation scope and limitations.

This directory is a selected C/OpenCL renderer source set, not a complete Kodi
or LibreELEC image. It keeps decoded VAAPI surfaces on the GPU, reconstructs
the source signal, and packs frames with matching metadata for TV-led output.
The existing library name is retained for adapter ABI compatibility.

The standalone CMake build and short N150 playback/packing checks pass; see
[BUILD-VALIDATION.txt](BUILD-VALIDATION.txt) for the exact scope and results.
The [experimental Kodi adapter](integration/README.md) is now available as an
explicit source patch, with build notes and a runtime configuration example.
Normal LibreELEC image wiring is still being integrated.
Building this branch's image does **not yet enable this new renderer**. No
YBLOD merge or new pre-release image is included in this work-in-progress push.
`packages/graphics/native-dv/package.mk` packages the bundled source and its
runtime dependencies; it is not yet a dependency of Kodi. Changes under
`native-dv/` participate in LibreELEC's package rebuild stamp.

Build on Linux with CMake, OpenCL headers/loader, libva, and the matching modified
FFmpeg headers from `CB1/patches/ffmpeg`:

```
cmake -S native-dv -B build-native -DCMAKE_BUILD_TYPE=Release
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
the code. The remaining Kodi/YBLOD integration still requires build, feature
and publication review before an image release. Do not substitute the earlier
test binary as proof that a future integrated image was built or qualified.
