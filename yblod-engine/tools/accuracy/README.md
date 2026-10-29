# Accuracy tools

Used to measure yblod's Dolby Vision output against Dolby hardware. See yblod-engine/docs/ACCURACY.md.

| File | What it does |
|---|---|
| `cecapture.sh` | Runs on a CoreELEC box: plays a file and captures HDMI encoder output frames (VDIN1 loopback) during playback. |
| `dvtunnel.py` | Decodes Dolby Vision tunnel frames from a Kodi readback (RGBA8) or a CoreELEC capture (RGB8, 64-bit byte-reversed), and checks the embedded metadata CRC. |
| `dvlms.py` | Converts a decoded frame to Dolby LMS (PQ, 12-bit codes) using the matrices in its own metadata. |
| `dvfit.c` | Offline renderer using the system libplacebo and FFmpeg: renders exact frames with chosen chroma and enhancement-layer filters, or thumbnails for matching. Build with the LibreELEC toolchain and run on the target. |
| `match.py` | Matches reference captures to rendered frames by picture content. |
| `score.py` | Scoring helpers: tunnel packing variants, IPT encoding, offset fitting, colour-edge statistics. |

Python tools need numpy.
