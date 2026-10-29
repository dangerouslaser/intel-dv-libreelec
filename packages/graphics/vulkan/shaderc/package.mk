# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026-present Team LibreELEC (https://libreelec.tv)

PKG_NAME="shaderc"
# The glslang, SPIRV-Tools & SPIRV-Headers pkg_version/s need to match shaderc's DEPS.
# https://raw.githubusercontent.com/google/shaderc/${PKG_VERSION}/DEPS
# v2026.4 pins glslang 16.6.0 (e1b562a8) and SPIRV-Tools ef96ed76, as packaged.
PKG_VERSION="v2026.4"
PKG_SHA256="f06ce5bcca94e5df7f34e115743597d0ad2e13c5fe9213c67dc8c76031241947"
PKG_LICENSE="Apache-2.0"
PKG_SITE="https://github.com/google/shaderc"
PKG_URL="https://github.com/google/shaderc/archive/${PKG_VERSION}.tar.gz"
# glslang's unpacked tree carries the matching SPIRV-Tools and SPIRV-Headers.
PKG_DEPENDS_HOST="toolchain:host Python3:host glslang:host"
PKG_LONGDESC="glslc, a GLSL to SPIR-V compiler (for yblod-engine-vk's shaders)."

PKG_CMAKE_OPTS_HOST="-DSHADERC_SKIP_TESTS=ON \
                     -DSHADERC_SKIP_EXAMPLES=ON \
                     -DSHADERC_SKIP_COPYRIGHT_CHECK=ON \
                     -DSHADERC_ENABLE_SHARED_CRT=OFF \
                     -DSPIRV_SKIP_TESTS=ON \
                     -DSPIRV_SKIP_EXECUTABLES=ON \
                     -DSPIRV_WERROR=OFF \
                     -DENABLE_GLSLANG_BINARIES=OFF \
                     -DBUILD_EXTERNAL=OFF \
                     -DBUILD_SHARED_LIBS=OFF \
                     -Wno-dev"
PKG_MAKE_OPTS_HOST="glslc_exe"

pre_configure_host() {
  local glslang="$(get_build_dir glslang)"
  PKG_CMAKE_OPTS_HOST+=" -DSHADERC_GLSLANG_DIR=${glslang} \
                        -DSHADERC_SPIRV_TOOLS_DIR=${glslang}/External/spirv-tools \
                        -DSHADERC_SPIRV_HEADERS_DIR=${glslang}/External/spirv-tools/external/spirv-headers"
}

makeinstall_host() {
  mkdir -p ${TOOLCHAIN}/bin
  cp glslc/glslc ${TOOLCHAIN}/bin/glslc
}
