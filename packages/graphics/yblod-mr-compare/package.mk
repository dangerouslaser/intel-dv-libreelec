# SPDX-License-Identifier: GPL-3.0-or-later

PKG_NAME="yblod-mr-compare"
PKG_VERSION="0.1.0"
PKG_ARCH="x86_64"
PKG_LICENSE="GPL-3.0-or-later"
PKG_SITE="https://github.com/dangerouslaser/intel-dv-libreelec"
PKG_URL=""
PKG_DEPENDS_TARGET="toolchain libplacebo ffmpeg libva libdrm ${OPENGLES} yblod-engine-vk"
PKG_NEED_UNPACK="${ROOT}/yblod-engine/vk/tools ${ROOT}/CB1/src"
PKG_LONGDESC="Test tool, not in the image: CB1's Master Reference against yblod's Vulkan engine, byte for byte."
PKG_TOOLCHAIN="cmake"

PKG_CMAKE_OPTS_TARGET="-DCB1_SOURCE_DIR=${ROOT}/CB1/src"

unpack() {
  mkdir -p "${PKG_BUILD}"
  cp -a "${ROOT}/yblod-engine/vk/tools" "${ROOT}/yblod-engine/vk/include" "${PKG_BUILD}/"
  cp "${PKG_BUILD}/tools/mr-compare/CMakeLists.txt" "${PKG_BUILD}/CMakeLists.txt"
}

makeinstall_target() {
  # Kept out of the image: the binary stays in the build folder.
  mkdir -p "${PKG_BUILD}/out"
  cp "${PKG_BUILD}/.${TARGET_NAME}/dvvk-mr-compare" "${PKG_BUILD}/out/"
}
