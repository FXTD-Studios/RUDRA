# Third-party dependencies.
#
# vcpkg (native/vcpkg.json) is the supported way to provide them: configure with
# -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake, or use a
# preset from CMakePresets.json. Without vcpkg the two header/test libraries the
# core needs are fetched at a pinned tag, so a clean machine can still build
# and test the core and the CLI.
include(FetchContent)

find_package(nlohmann_json 3.11 CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# zlib, for the lossless ZIP and ZIPS EXR masters (1.2.6, 1.3). vcpkg and the Linux and macOS systems have it.
# The Windows package (scripts/PACKAGE_WINDOWS.ps1: Qt from aqt, no vcpkg) and this PC's build did not, so
# find_package(ZLIB) failed there and every EXR was written uncompressed whatever Deliver asked (24 Nov 2026: the
# ZIP tests skipped, "this build has no zlib"). Without one it is fetched at a pinned release and linked static.
find_package(ZLIB QUIET)
if(NOT TARGET ZLIB::ZLIB)
  set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
  FetchContent_Declare(zlib
    URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
    URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(zlib)
  add_library(ZLIB::ZLIB ALIAS zlibstatic)
  message(STATUS "zlib 1.3.1 fetched and linked static (no system zlib)")
endif()

# OpenEXR (1.3.2, 28 Nov 2026), for the EXR compressions RUDRA's own writer does not do: RLE, PIZ, PXR24, B44, B44A,
# DWAA and DWAB (deliver/src/exr_openexr.cpp; None, ZIP and ZIPS stay RUDRA's own, byte for byte). vcpkg
# (native/vcpkg.json), Homebrew (openexr) and the Linux systems (libopenexr-dev) have it; without one it is fetched at a
# pinned release and linked static, as zlib above, so every package offers every compression. RUDRA_WITH_OPENEXR=OFF
# builds without it: Deliver then lists ZIP, ZIPS and None only.
# The packages (scripts/native_app.sh, PACKAGE_WINDOWS.ps1) set RUDRA_OPENEXR_FETCH: the pinned copy, static, even where
# the system has one, so nothing in a package points at a Homebrew or system OpenEXR.
option(RUDRA_WITH_OPENEXR "Every OpenEXR compression in Deliver (the OpenEXR library)" ON)
option(RUDRA_OPENEXR_FETCH "Build the pinned OpenEXR, static, even when the system has one (the packages)" OFF)
if(RUDRA_WITH_OPENEXR)
  if(NOT RUDRA_OPENEXR_FETCH)
    find_package(OpenEXR 3 CONFIG QUIET)
  endif()
  if(NOT TARGET OpenEXR::OpenEXR)
    set(OPENEXR_INSTALL OFF CACHE BOOL "" FORCE)
    set(OPENEXR_INSTALL_TOOLS OFF CACHE BOOL "" FORCE)
    set(OPENEXR_INSTALL_PKG_CONFIG OFF CACHE BOOL "" FORCE)
    set(OPENEXR_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(OPENEXR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(OPENEXR_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(IMATH_INSTALL OFF CACHE BOOL "" FORCE)
    set(IMATH_INSTALL_PKG_CONFIG OFF CACHE BOOL "" FORCE)
    set(CMAKE_DEBUG_POSTFIX "" CACHE STRING "")   # OpenEXR would set "_d" for every library and module, the plug-in's too
    # Its own Imath and libdeflate, static too: a system one would leave the package pointing at the system's libraries.
    set(OPENEXR_FORCE_INTERNAL_IMATH ON CACHE BOOL "" FORCE)
    set(OPENEXR_FORCE_INTERNAL_DEFLATE ON CACHE BOOL "" FORCE)
    FetchContent_Declare(openexr
      GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openexr.git
      GIT_TAG v3.3.2   # 28 Nov 2026; it fetches Imath 3.1 and libdeflate itself when the system has none
      GIT_SHALLOW TRUE)
    # Static (as a subproject it would otherwise build shared libraries a package would have to carry); RUDRA's own
    # BUILD_SHARED_LIBS, if any, is put back after.
    if(DEFINED BUILD_SHARED_LIBS)
      set(_rudra_shared "${BUILD_SHARED_LIBS}")
    endif()
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(openexr)
    if(DEFINED _rudra_shared)
      set(BUILD_SHARED_LIBS "${_rudra_shared}")
      unset(_rudra_shared)
    else()
      unset(BUILD_SHARED_LIBS)
    endif()
    message(STATUS "OpenEXR 3.3.2 fetched and linked static (no system OpenEXR)")
  endif()
endif()

# A Windows executable must carry the runtime matching the headers/import
# library it was built with. PATH cannot override Windows' System32 copy, which
# can expose an older C API even when the correct SDK is first on PATH.
function(rudra_stage_onnxruntime target)
  if(WIN32 AND RUDRA_WITH_ONNXRUNTIME)
    find_file(RUDRA_ORT_RUNTIME_DLL NAMES onnxruntime.dll
              PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/bin"
              NO_DEFAULT_PATH NO_CACHE REQUIRED)
    get_filename_component(_ort_runtime_dir "${RUDRA_ORT_RUNTIME_DLL}" DIRECTORY)
    file(GLOB _ort_runtime_dlls CONFIGURE_DEPENDS "${_ort_runtime_dir}/*.dll")
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${_ort_runtime_dlls} "$<TARGET_FILE_DIR:${target}>"
      COMMENT "Stage the configured ONNX Runtime beside ${target}" VERBATIM)
  endif()
endfunction()

if(RUDRA_BUILD_TESTS)
  find_package(GTest 1.14 CONFIG QUIET)
  if(NOT GTest_FOUND)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    # Pinned by tag. CI installs GoogleTest from the platform package manager
    # (apt, Homebrew, vcpkg), so this fallback only runs on a bare machine.
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.15.2
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(googletest)
  endif()
endif()

set(RUDRA_TORCH_ROOT "" CACHE PATH
    "LibTorch or pip torch folder (include/, lib/), imported without TorchConfig")
if(RUDRA_WITH_LIBTORCH)
  if(RUDRA_TORCH_ROOT)
    # TorchConfig of a CUDA build calls enable_language(CUDA): it needs the
    # CUDA toolkit, at torch's exact version, integrated with this compiler,
    # to build a program that never compiles a line of CUDA. The runtime does
    # not need any of it, so the libraries are imported directly here and the
    # CUDA library is loaded on demand (infer/src/libtorch_backend.cpp).
    find_path(RUDRA_TORCH_INCLUDE torch/script.h
              PATHS "${RUDRA_TORCH_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
    set(_torch_libs "")
    foreach(_lib torch torch_cpu c10)
      find_library(RUDRA_TORCH_LIB_${_lib} NAMES ${_lib} PATHS "${RUDRA_TORCH_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
      list(APPEND _torch_libs "${RUDRA_TORCH_LIB_${_lib}}")
    endforeach()
    add_library(rudra_torch INTERFACE)
    target_include_directories(rudra_torch SYSTEM INTERFACE
      "${RUDRA_TORCH_INCLUDE}" "${RUDRA_TORCH_INCLUDE}/torch/csrc/api/include")
    target_link_libraries(rudra_torch INTERFACE ${_torch_libs})
    if(NOT MSVC)
      set(RUDRA_TORCH_CXX11_ABI 1 CACHE STRING "torch.compiled_with_cxx11_abi() of that torch")
      target_compile_definitions(rudra_torch INTERFACE _GLIBCXX_USE_CXX11_ABI=${RUDRA_TORCH_CXX11_ABI})
    endif()
    set(TORCH_LIBRARIES rudra_torch)
    message(STATUS "LibTorch imported from ${RUDRA_TORCH_ROOT} (no TorchConfig, no CUDA toolkit)")
  else()
    # LibTorch: the official archive, or the pip torch package
    # (python -c "import torch; print(torch.utils.cmake_prefix_path)").
    find_package(Torch REQUIRED CONFIG)
  endif()
endif()

if(RUDRA_WITH_OCIO)
  # The ACES v2 checkpoints pin OCIO 2.5.2 (built-in cg-config-v4.0.0_aces-v2.0_ocio-v2.5).
  # Another version is refused at run time too (aces_v2::make_ocio_view).
  find_package(OpenColorIO CONFIG REQUIRED)
  set(RUDRA_OCIO_ALLOW_VERSION_MISMATCH OFF CACHE BOOL "Configure even if OpenColorIO is not 2.5.2 (research only)")
  if(NOT OpenColorIO_VERSION VERSION_EQUAL 2.5.2 AND NOT RUDRA_OCIO_ALLOW_VERSION_MISMATCH)
    message(FATAL_ERROR "RUDRA_WITH_OCIO needs OpenColorIO 2.5.2, found ${OpenColorIO_VERSION}")
  endif()
endif()

if(RUDRA_WITH_ONNXRUNTIME)
  if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "RUDRA_WITH_ONNXRUNTIME needs -DONNXRUNTIME_ROOT=<path to the ORT package>")
  endif()
  find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
            PATHS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/include/onnxruntime"
            NO_DEFAULT_PATH REQUIRED)
  find_library(ORT_LIBRARY NAMES onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}")
endif()
