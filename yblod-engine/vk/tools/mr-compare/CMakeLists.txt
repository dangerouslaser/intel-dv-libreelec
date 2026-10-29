# dvvk-mr-compare: CB1's Master Reference against yblod's Vulkan engine.
# Copied to the package root (packages/graphics/yblod-mr-compare), next to
# tools/ and include/. CB1_SOURCE_DIR is CB1/src; the engine is dlopened.
cmake_minimum_required(VERSION 3.22)
project(dvvk_mr_compare C)
add_subdirectory(${CB1_SOURCE_DIR} cb1)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MR REQUIRED IMPORTED_TARGET gbm libavformat libavcodec libavutil libva egl glesv2 libdrm)
set(MR_TOOLS ${CMAKE_CURRENT_SOURCE_DIR}/tools)
add_executable(dvvk-mr-compare ${MR_TOOLS}/dvvk-mr-compare.c ${MR_TOOLS}/decode.c)
set_target_properties(dvvk-mr-compare PROPERTIES C_STANDARD 11)
target_include_directories(dvvk-mr-compare PRIVATE ${MR_TOOLS})
target_link_libraries(dvvk-mr-compare PRIVATE dvbridge_core PkgConfig::MR ${CMAKE_DL_LIBS} m)
install(TARGETS dvvk-mr-compare RUNTIME DESTINATION bin)
