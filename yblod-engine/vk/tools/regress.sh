#!/bin/sh
# Headless regression: every corpus clip (docs/vulkan-engine.md, "Corpus") and
# mutation, yblod's OpenCL engine vs dv-engine-vk, byte for byte.
#   regress.sh [CLIPS_DIR]      (FRAMES=N, DVVK_COMPARE=path to override)
cd "${1:-/var/tmp/dvvk/clips}" || exit 1
run() { nice "${DVVK_COMPARE:-dvvk-compare}" "$@" 2>&1 | grep -E '^\{"file' | python3 -I -c '
import json,sys
for l in sys.stdin:
    r=json.loads(l); ok=r["mismatched"]==0 and r["refused_one"]==0
    print(("PASS" if ok else "FAIL"), r["file"], "scaler", r["el_scaler"], "fel", r["fel"], "frames", r["frames"], "identical", r["identical"], "refused", r["refused_both"], "flagged", r["range_flagged"], "ov", r["overlay_pixels_differ"], r["overlay_max_luma_delta"], r["overlay_max_chroma_delta"], "ms", r["opencl_ms"], r["vulkan_ms"], sys.argv[1:])
' "$@"; }
F=${FRAMES:-24}
for c in fel-1917 fel-gladiator fel-trainspotting; do run --frames $F --el-scaler 0 $c.mkv; run --frames $F --el-scaler 1 $c.mkv; done
run --frames $F --fel 0 fel-1917.mkv
for c in mel-gemini mel-matrix mel1080-kingofnewyork p81-deadpool p86-leon p5-clearpresent p81crop-martysupreme p81hd-extraction2 p81hd-patriot p5hd-nyad p81odd-mariogalaxy; do run --frames $F $c.mkv; done
for c in fel-1917 mel-gemini p81hd-extraction2; do run --frames 8 --overlay 1 $c.mkv; done
run --frames 8 --overlay 1 --el-scaler 1 fel-trainspotting.mkv
run --frames 8 --async --sources --overlay 1 mel-gemini.mkv
run --frames 24 --async fel-1917.mkv
for m in mmr2 mmr3 nv12; do run --frames 8 --mutate $m p81-deadpool.mkv; done
for m in mixed wide equal; do run --frames 8 --mutate $m fel-1917.mkv; done
run --frames 8 --mutate mmr2 mel-matrix.mkv
