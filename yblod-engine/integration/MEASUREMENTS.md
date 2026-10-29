# Short N150 integration comparison — 2026-10-09

Historical comparison of the prior and newer CB1 adapters, not measurements
of the YBLOD-specific adapter in this directory. This port remains unqualified.

Saving Private Ryan, P7 FEL, playback opened at 480 seconds. Four short runs
in new–prior–prior–new order, using separate copies of the same test profile.
Both stacks used the same CMake engine, OpenCL loader and Intel runtime.
Each used its matching Kodi/libplacebo/resources. This compares complete Kodi
integrations, not an isolated kernel or identical frame-by-frame GPU trace.
GPU frequency limits remained 200–1000 MHz; clocks were not fixed.

Two-run means; GPU values are render-engine busy percentage, not frame times:

| Window | Prior adapter GPU | Updated adapter GPU | Prior CPU* | Updated CPU* |
| --- | ---: | ---: | ---: | ---: |
| Controls closed | 62.99% | 65.46% | 64.42% | 67.93% |
| Controls open | 72.71% | 75.68% | 57.07% | 58.60% |
| Closed after seek/resume | 62.72% | 65.17% | 62.00% | 65.29% |

*CPU percentage is relative to one logical core, not the entire CPU.

All 12 approximately 4.2-second measurement windows had 101 distinct successful
commits, no backwards commits and at most a 42,001-microsecond source-PTS gap.
All four pause/paused-seek/resume tests passed, without logged queue skips or
native-route exits. Diagnostic pixel readback was disabled.

The updated integration retains measured playback cadence but adds about
2.5–3.0 percentage points of render-engine usage in this comparison. This is
not evidence of unchanged GPU cost; the extra work remains under investigation.
Short, approximately scene-matched runs do not establish performance for an
entire movie or all P7 files. Pixel accuracy checks are described separately
in the integration README.

Executable SHA256:

- Prior: `e267e49d56e68e44933c33b3ac3e976a7a2f838ef795bc9564ceeef6c7d2f284`
- Updated: `26397bfa24fdf50f783235efdcddf204b22cd821ad93046419e503714c05926a`
- Shared engine: `1349909379b213a59d4bcfae94d1eb1ec65231835338c1f4f66e7bbcade58385`

These identify test artifacts, not published release images. The complete
LibreELEC image and its default runtime wiring have not yet been qualified.
