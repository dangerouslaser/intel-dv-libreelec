# SPDX-License-Identifier: MIT

PKG_NAME="yblod-engine"
PKG_VERSION="0.1"
PKG_ARCH="x86_64"
PKG_LICENSE="GPL-3.0-or-later AND MIT"
PKG_SITE="https://github.com/dangerouslaser/libreelec-yblod"
PKG_URL=""
PKG_DEPENDS_TARGET="toolchain ffmpeg libva opencl-icd-loader intel-opencl-runtime"
# Sources, patches and docs live in ${ROOT}/yblod-engine; installed names stay native-dv.
PKG_NEED_UNPACK="${ROOT}/yblod-engine"
PKG_LONGDESC="Native source-domain Dolby Vision reconstruction and transport renderer"
PKG_TOOLCHAIN="cmake"
PKG_BUILD_FLAGS="-lto"

PKG_CMAKE_OPTS_TARGET="-DNATIVE_DV_BUILD_RENDERER=ON \
                       -DNATIVE_DV_FFMPEG_INCLUDE_DIR=${SYSROOT_PREFIX}/usr/include \
                       -DBUILD_TESTING=OFF \
                       -DCMAKE_INSTALL_LIBDIR=lib"

unpack() {
  mkdir -p "${PKG_BUILD}"
  cp -a "${ROOT}/yblod-engine/." "${PKG_BUILD}/"
}

post_makeinstall_target() {
  mkdir -p "${INSTALL}/usr/lib/kodi" "${INSTALL}/usr/lib/systemd/system/kodi.service.d" "${INSTALL}/usr/bin"
  cp "${PKG_BUILD}/integration/runtime.env" "${INSTALL}/usr/lib/kodi/native-dv.env"
  cp "${PKG_BUILD}/integration/20-native-dv.conf" \
     "${INSTALL}/usr/lib/systemd/system/kodi.service.d/"
  cp "${PKG_BUILD}/tools/dv-diagnostics" "${INSTALL}/usr/bin/dv-diagnostics"
  chmod 0755 "${INSTALL}/usr/bin/dv-diagnostics"
}
