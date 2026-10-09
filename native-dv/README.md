# Native source-domain renderer — integration staging

This directory is a selected C/OpenCL renderer source set, not a complete Kodi
or LibreELEC image. It keeps decoded VAAPI surfaces on the GPU, reconstructs
the source signal, and packs frames with matching metadata for TV-led output.
The existing library name is retained for adapter ABI compatibility.

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
are intended by this export. A source/provenance and license review, a build of
this exact exported set, and integrated playback qualification are required
before publication. Do not substitute the earlier test binary as proof that
this source set was built or qualified.
