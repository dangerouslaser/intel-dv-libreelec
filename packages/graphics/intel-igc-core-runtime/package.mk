# SPDX-License-Identifier: GPL-2.0-only

PKG_NAME="intel-igc-core-runtime"
PKG_VERSION="2.28.4"
PKG_SHA256="3eea502b74ca57d6050e259838a91f5384805b5bb73c9fcecc055c6f8d32389f"
PKG_ARCH="x86_64"
PKG_LICENSE="MIT Apache-2.0"
PKG_SITE="https://github.com/intel/intel-graphics-compiler"
PKG_SOURCE_NAME="intel-igc-core-2_${PKG_VERSION}+20760_amd64.deb"
PKG_URL="${PKG_SITE}/releases/download/v${PKG_VERSION}/${PKG_SOURCE_NAME}"
PKG_DEPENDS_TARGET="toolchain zlib zstd"
PKG_LONGDESC="Intel's pinned upstream IGC compiler binaries and notices"
PKG_TOOLCHAIN="manual"

unpack() {
  mkdir -p "${PKG_BUILD}"
  ar p "${SOURCES}/${PKG_NAME}/${PKG_SOURCE_NAME}" data.tar.gz | tar -xz -C "${PKG_BUILD}"
}

makeinstall_target() {
  mkdir -p "${INSTALL}/usr/lib" "${INSTALL}/usr/share/licenses/${PKG_NAME}"
  cp -a "${PKG_BUILD}"/usr/local/lib/libigc.so.* "${INSTALL}/usr/lib/"
  cp -a "${PKG_BUILD}"/usr/local/lib/libiga64.so.* "${INSTALL}/usr/lib/"
  cp "${PKG_BUILD}/usr/local/lib/igc2/NOTICES.txt" "${INSTALL}/usr/share/licenses/${PKG_NAME}/"
}
