# SPDX-License-Identifier: GPL-2.0-only

PKG_NAME="intel-igc-opencl-runtime"
PKG_VERSION="2.28.4"
PKG_SHA256="9fae8175c95def354534e6d322dd1b2661eb92dec96916f50ee1f5d31c7a4f65"
PKG_ARCH="x86_64"
PKG_LICENSE="MIT Apache-2.0"
PKG_SITE="https://github.com/intel/intel-graphics-compiler"
PKG_SOURCE_NAME="intel-igc-opencl-2_${PKG_VERSION}+20760_amd64.deb"
PKG_URL="${PKG_SITE}/releases/download/v${PKG_VERSION}/${PKG_SOURCE_NAME}"
PKG_DEPENDS_TARGET="toolchain intel-igc-core-runtime zlib zstd"
PKG_LONGDESC="Intel's pinned upstream OpenCL compiler frontend binaries"
PKG_TOOLCHAIN="manual"

unpack() {
  mkdir -p "${PKG_BUILD}"
  ar p "${SOURCES}/${PKG_NAME}/${PKG_SOURCE_NAME}" data.tar.gz | tar -xz -C "${PKG_BUILD}"
}

makeinstall_target() {
  mkdir -p "${INSTALL}/usr/lib"
  cp -a "${PKG_BUILD}"/usr/local/lib/libigdfcl.so.* "${INSTALL}/usr/lib/"
  cp -a "${PKG_BUILD}/usr/local/lib/libopencl-clang2.so.16" "${INSTALL}/usr/lib/"
  # The required core package carries Intel's combined compiler notices.
}
