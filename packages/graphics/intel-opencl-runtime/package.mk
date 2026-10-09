# SPDX-License-Identifier: GPL-2.0-only

PKG_NAME="intel-opencl-runtime"
PKG_VERSION="26.05.37020.3"
PKG_SHA256="2129fb9ecf3634b8fccfc98e8892265f381f2b3aeb552fe3616f634f35998db4"
PKG_ARCH="x86_64"
PKG_LICENSE="MIT BSD-3-Clause SGI-B-2.0"
PKG_SITE="https://github.com/intel/compute-runtime"
PKG_SOURCE_NAME="intel-opencl-icd_${PKG_VERSION}-0_amd64.deb"
PKG_URL="${PKG_SITE}/releases/download/${PKG_VERSION}/${PKG_SOURCE_NAME}"
PKG_DEPENDS_TARGET="toolchain opencl-icd-loader intel-igc-opencl-runtime gmmlib libva zstd:host"
PKG_LONGDESC="Intel's pinned upstream OpenCL runtime with VAAPI and OpenGL sharing"
PKG_TOOLCHAIN="manual"

unpack() {
  mkdir -p "${PKG_BUILD}"
  ar p "${SOURCES}/${PKG_NAME}/${PKG_SOURCE_NAME}" data.tar.zst | tar --zstd -x -C "${PKG_BUILD}"
}

makeinstall_target() {
  mkdir -p "${INSTALL}/usr/lib/intel-opencl" "${INSTALL}/etc/OpenCL/vendors" \
           "${INSTALL}/usr/share/licenses/${PKG_NAME}"
  cp "${PKG_BUILD}/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so" \
     "${INSTALL}/usr/lib/intel-opencl/"
  cp "${PKG_DIR}/files/intel.icd" "${INSTALL}/etc/OpenCL/vendors/"
  cp "${PKG_BUILD}/usr/share/doc/intel-opencl-icd/copyright" \
     "${INSTALL}/usr/share/licenses/${PKG_NAME}/"
}
