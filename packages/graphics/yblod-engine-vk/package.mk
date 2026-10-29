# SPDX-License-Identifier: MIT

PKG_NAME="yblod-engine-vk"
PKG_VERSION="0.1.0"
PKG_ARCH="x86_64"
PKG_LICENSE="GPL-3.0-or-later AND MIT"
PKG_SITE="https://github.com/dangerouslaser/libreelec-yblod"
PKG_URL=""
PKG_DEPENDS_TARGET="toolchain cargo:host shaderc:host ffmpeg libva vulkan-loader vulkan-headers"
# The crate is yblod-engine/vk; it compiles yblod's host C from yblod-engine/src.
PKG_NEED_UNPACK="${ROOT}/yblod-engine/vk ${ROOT}/yblod-engine/src"
PKG_LONGDESC="yblod's Dolby Vision source-tunnel engine on Vulkan compute (libdv_engine_vk.so.0)"
PKG_TOOLCHAIN="manual"
PKG_BUILD_FLAGS="-lto"

unpack() {
  mkdir -p "${PKG_BUILD}"
  cp -a "${ROOT}/yblod-engine/vk/." "${PKG_BUILD}/"
  rm -rf "${PKG_BUILD}/target"
}

make_target() {
  export RUSTC_LINKER=${CC}
  export GLSLC="${TOOLCHAIN}/bin/glslc"
  export YBLOD_ENGINE_SRC="${ROOT}/yblod-engine/src"
  export FFMPEG_INCLUDE="${SYSROOT_PREFIX}/usr/include"
  # The library only: the test tools need GBM, EGL and the OpenCL engine.
  cargo build \
    --target ${TARGET_NAME} \
    --release \
    --locked \
    --lib
}

makeinstall_target() {
  local lib="${PKG_BUILD}/.${TARGET_NAME}/target/${TARGET_NAME}/release/libdv_engine_vk.so"

  mkdir -p "${INSTALL}/usr/lib"
  cp "${lib}" "${INSTALL}/usr/lib/libdv_engine_vk.so.0"

  # Kodi's adapter includes the C API; the library itself is dlopened.
  mkdir -p "${SYSROOT_PREFIX}/usr/include/dv-engine-vk"
  cp "${PKG_BUILD}/include/dv_engine_vk.h" "${PKG_BUILD}/include/dvvk_plan.h" \
     "${SYSROOT_PREFIX}/usr/include/dv-engine-vk/"
}
