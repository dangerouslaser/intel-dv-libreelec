# SPDX-License-Identifier: GPL-2.0-only

PKG_NAME="opencl-headers"
PKG_VERSION="2025.07.22"
PKG_SHA256="98f0a3ea26b4aec051e533cb1750db2998ab8e82eda97269ed6efe66ec94a240"
PKG_LICENSE="Apache-2.0"
PKG_SITE="https://github.com/KhronosGroup/OpenCL-Headers"
PKG_URL="${PKG_SITE}/archive/refs/tags/v${PKG_VERSION}.tar.gz"
PKG_DEPENDS_TARGET="toolchain"
PKG_LONGDESC="Khronos OpenCL C API and interop headers"
PKG_TOOLCHAIN="cmake"

PKG_CMAKE_OPTS_TARGET="-DBUILD_TESTING=OFF"
