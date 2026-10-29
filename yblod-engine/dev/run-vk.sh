#!/bin/bash
# CB1 + YBLOD Vulkan engine image, then the dvvk-mr-compare test tool; log build-vk.log.
cd $(dirname $0)/../..   # the tree root
for JOBS in 16 10 6; do
  echo "== $(date "+%F %T") $JOBS jobs" >> build-vk.log
  docker run --rm --name cb1-vk-build --memory 12g --memory-swap 12g -v "$PWD":/build -w /build \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0="*" \
    -e PROJECT=Generic -e DEVICE=Generic -e ARCH=x86_64 -e OFFICIAL=no -e CB1_HDR10_AI=yes \
    -e CONCURRENCY_MAKE_LEVEL=$JOBS -e CMAKE_BUILD_PARALLEL_LEVEL=$JOBS -e THREADCOUNT=4 \
    libreelec-dv-build bash -c "make image && scripts/build yblod-mr-compare" >> build-vk.log 2>&1 && { echo "== $(date "+%F %T") done" >> build-vk.log; exit 0; }
  tail -400 build-vk.log | grep -qE "Killed|out of memory|signal 9" || { echo "== failed" >> build-vk.log; exit 1; }
done
