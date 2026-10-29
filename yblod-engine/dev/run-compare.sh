#!/bin/bash
# dvvk-mr-compare inside the built image: SYSTEM (with the tool, and the
# OpenCL engine + Intel compute runtime until the image carries them) imported
# as a Docker image, run on Ollie's render node, media read-only.
#   run-compare.sh [tool options] FILE      (FILE under /media-merged or /clips)
set -e
cd $(dirname $0)/../..   # the tree root; mr/ holds clips, dumps and the imported image
SYS=$(ls -t target/*.system 2>/dev/null | head -1)
TOOL=$(ls build.*/build/yblod-mr-compare-*/out/dvvk-mr-compare)
EXTRA=$(ls -d build.*/install_pkg/yblod-engine-* build.*/install_pkg/intel-igc-* build.*/install_pkg/intel-opencl-runtime-* build.*/install_pkg/opencl-icd-loader-* 2>/dev/null)
[ -n "$SYS" ] && [ -n "$TOOL" ] || { echo "no image or tool"; exit 1; }
NEWEST=$(ls -td "$SYS" "$TOOL" $EXTRA | head -1)
if [ ! -f mr/image.stamp ] || [ "$NEWEST" -nt mr/image.stamp ]; then
  rm -rf mr/root
  unsquashfs -q -n -d mr/root "$SYS" >/dev/null
  cp "$TOOL" mr/root/usr/bin/dvvk-mr-compare
  for d in $EXTRA; do cp -a "$d"/. mr/root/; done
  tar --ignore-failed-read -C mr/root -c . 2>/dev/null | docker import - cb1-vk-root:latest >/dev/null
  rm -rf mr/root
  touch mr/image.stamp
fi
exec docker run --rm --device /dev/dri/renderD128 --group-add 109 --group-add 44 \
  -v /media-merged:/media-merged:ro -v $PWD/mr/clips:/clips:ro -v "$PWD/mr/dump":/dump \
  -e LIBVA_DRIVER_NAME=iHD -e DVVK_PROFILE -e DVVK_ANNEXB_FUSED \
  cb1-vk-root:latest /usr/bin/dvvk-mr-compare "$@"
