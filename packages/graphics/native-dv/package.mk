# SPDX-License-Identifier: MIT

PKG_NAME="native-dv"
PKG_VERSION="0.1"
PKG_ARCH="x86_64"
PKG_LICENSE="GPL-3.0-or-later AND MIT"
PKG_SITE="https://github.com/dangerouslaser/intel-dv-libreelec"
PKG_URL=""
PKG_DEPENDS_TARGET="toolchain ffmpeg libva opencl-icd-loader intel-opencl-runtime"
PKG_NEED_UNPACK="${ROOT}/native-dv"
PKG_LONGDESC="Native source-domain Dolby Vision reconstruction and transport renderer"
PKG_TOOLCHAIN="cmake"
PKG_BUILD_FLAGS="-lto"

PKG_CMAKE_OPTS_TARGET="-DNATIVE_DV_BUILD_RENDERER=ON \
                       -DNATIVE_DV_FFMPEG_INCLUDE_DIR=${SYSROOT_PREFIX}/usr/include \
                       -DBUILD_TESTING=OFF \
                       -DCMAKE_INSTALL_LIBDIR=lib"

unpack() {
  mkdir -p "${PKG_BUILD}"
  cp -a "${ROOT}/native-dv/." "${PKG_BUILD}/"
}
