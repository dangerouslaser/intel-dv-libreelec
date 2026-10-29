#!/bin/bash
# CroqueMr baseline (RC1.1) image in the libreelec-dv-build container; retries with fewer jobs on OOM.
cd $(dirname $0)/../..   # the tree root
for JOBS in 7 5 3; do
  echo "== $(date "+%F %T") $JOBS jobs" >> build-baseline.log
  docker run --rm --name cb1-vk-build --cpu-shares 256 --memory 12g --memory-swap 12g -v "$PWD":/build -w /build \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0="*" \
    -e PROJECT=Generic -e DEVICE=Generic -e ARCH=x86_64 -e OFFICIAL=no -e CB1_HDR10_AI=yes \
    -e CONCURRENCY_MAKE_LEVEL=$JOBS -e CMAKE_BUILD_PARALLEL_LEVEL=$JOBS -e THREADCOUNT=4 \
    libreelec-dv-build bash -c "make image" >> build-baseline.log 2>&1 && { echo "== $(date "+%F %T") done" >> build-baseline.log; exit 0; }
  tail -400 build-baseline.log | grep -qE "Killed|out of memory|signal 9" || { echo "== failed" >> build-baseline.log; exit 1; }
done
