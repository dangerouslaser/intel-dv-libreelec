cmake_minimum_required(VERSION 3.22)
project(NativeDV VERSION 0.1 LANGUAGES C)
include(GNUInstallDirs)
include(CTest)
option(NATIVE_DV_BUILD_RENDERER "Build Linux VAAPI/OpenCL renderer" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wconversion -ffp-contract=off)

if(NATIVE_DV_BUILD_RENDERER)
  find_package(PkgConfig REQUIRED)
  find_package(OpenCL REQUIRED)
  find_package(Threads REQUIRED)
  pkg_check_modules(VA REQUIRED IMPORTED_TARGET libva)
  pkg_check_modules(AVUTIL REQUIRED libavutil>=59.12.100)
  include(CheckCSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES ${AVUTIL_INCLUDE_DIRS})
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/tests/ffmpeg_metadata_fields.c" metadata_probe)
  check_c_source_compiles("${metadata_probe}" NATIVE_DV_METADATA_LAYOUT_V1)
  unset(CMAKE_REQUIRED_INCLUDES)
  if(NOT NATIVE_DV_METADATA_LAYOUT_V1)
    message(FATAL_ERROR "Matched modified FFmpeg raw metadata headers required")
  endif()
  add_library(dv_source_private SHARED
    src/dv_source_renderer.c src/source_overlay.c src/dv_scanout_slots.c
    src/source_geometry.c src/dv_ffmpeg_metadata.c src/dv_cm4.c
    src/opencl_backend.c src/mmr_prepare.c src/single_reshape.c
    src/single_reshape_ffmpeg.c src/colour_lut.c src/dv_engine.c
    src/pixel_colour.c src/pixel_luma.c src/pixel_chroma_final.c src/pixel_spatial.c)
  set_target_properties(dv_source_private PROPERTIES SOVERSION 1)
  target_compile_definitions(dv_source_private PRIVATE DV_FFMPEG_RAW_EXTENSIONS
    DV_PREPARED_CHROMA DV_ENCODE_STEPS=256 DV_COLOUR_LUT)
  target_include_directories(dv_source_private SYSTEM PRIVATE ${AVUTIL_INCLUDE_DIRS})
  target_link_libraries(dv_source_private PRIVATE OpenCL::OpenCL PkgConfig::VA Threads::Threads m)
  target_link_options(dv_source_private PRIVATE -Wl,--no-undefined -Wl,-Bsymbolic)
  install(TARGETS dv_source_private LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(FILES src/colour_dd.cl src/opencl_kernels.cl src/single_reshape.cl
    src/source_overlay.cl src/source_tunnel.cl DESTINATION ${CMAKE_INSTALL_DATADIR}/native-dv)
  install(FILES src/dv_source_renderer.h src/source_geometry.h src/source_input.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/native-dv)
endif()

if(BUILD_TESTING)
  foreach(pair "order2_gate;single_reshape" "source_geometry;source_geometry" "scanout_slots;dv_scanout_slots")
    list(GET pair 0 test_name)
    list(GET pair 1 implementation)
    add_executable(${test_name}_tests tests/${test_name}_tests.c src/${implementation}.c)
    target_include_directories(${test_name}_tests PRIVATE src)
    target_compile_options(${test_name}_tests PRIVATE -UNDEBUG)
    add_test(NAME ${test_name} COMMAND ${test_name}_tests)
  endforeach()
endif()
