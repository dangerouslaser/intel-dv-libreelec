# SPDX-License-Identifier: GPL-2.0-only

PKG_NAME="opencl-icd-loader"
PKG_VERSION="2025.07.22"
PKG_SHA256="dff7a0b11ad5b63a669358e3476e3dc889a4a361674e5b69b267b944d0794142"
PKG_LICENSE="Apache-2.0"
PKG_SITE="https://github.com/KhronosGroup/OpenCL-ICD-Loader"
PKG_URL="${PKG_SITE}/archive/refs/tags/v${PKG_VERSION}.tar.gz"
PKG_DEPENDS_TARGET="toolchain opencl-headers"
PKG_LONGDESC="OpenCL vendor-neutral shared dispatch library"
PKG_TOOLCHAIN="cmake"

PKG_CMAKE_OPTS_TARGET="-DOPENCL_ICD_LOADER_BUILD_SHARED_LIBS=ON \
                       -DOPENCL_ICD_LOADER_HEADERS_DIR=${SYSROOT_PREFIX}/usr/include \
                       -DENABLE_OPENCL_LAYERS=OFF \
                       -DBUILD_TESTING=OFF"

post_makeinstall_target() {
  mkdir -p "${INSTALL}/usr/share/licenses/${PKG_NAME}"
  cp "${PKG_BUILD}/LICENSE" "${INSTALL}/usr/share/licenses/${PKG_NAME}/"
}
