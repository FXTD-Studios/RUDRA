# ofx: RUDRA as an OFX image effect (roadmap 5.1, 20 Oct 2026).
#
# rudra_ofx_process (the picture, no OFX) is always built and tested. The
# plug-in itself, RUDRA.ofx.bundle, needs the OpenFX SDK (BSD): set
# RUDRA_OFX_SDK_DIR to a checkout (include/, Support/, HostSupport/), or leave
# it empty to fetch the pinned release. Built with -DRUDRA_BUILD_OFX=ON.
rudra_library(rudra_ofx_process SOURCES ofx_process.cpp)
target_include_directories(rudra_ofx_process PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rudra_ofx_process PUBLIC rudra::rudra_deliver rudra::rudra_core)

if(NOT RUDRA_BUILD_OFX)
  return()
endif()

set(RUDRA_OFX_SDK_DIR "" CACHE PATH "OpenFX SDK checkout (include/, Support/, HostSupport/); empty: fetch it")
if(NOT RUDRA_OFX_SDK_DIR)
  include(FetchContent)
  FetchContent_Declare(openfx
    GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openfx.git
    GIT_TAG OFX_Release_1.4_TAG
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR do-not-add)   # headers and sources only, not its own build
  FetchContent_MakeAvailable(openfx)
  set(RUDRA_OFX_SDK_DIR ${openfx_SOURCE_DIR})
endif()
if(NOT EXISTS ${RUDRA_OFX_SDK_DIR}/include/ofxImageEffect.h OR NOT EXISTS ${RUDRA_OFX_SDK_DIR}/Support/Library/ofxsImageEffect.cpp)
  message(FATAL_ERROR "RUDRA_OFX_SDK_DIR (${RUDRA_OFX_SDK_DIR}) is not an OpenFX SDK checkout")
endif()

# A shared library links the static RUDRA libraries: they must be position-independent.
foreach(t rudra_platform rudra_core rudra_deliver rudra_ofx_process)
  set_property(TARGET ${t} PROPERTY POSITION_INDEPENDENT_CODE ON)
endforeach()

set(_ofxs ${RUDRA_OFX_SDK_DIR}/Support/Library)
set(_ofxs_sources ${_ofxs}/ofxsCore.cpp ${_ofxs}/ofxsImageEffect.cpp ${_ofxs}/ofxsInteract.cpp ${_ofxs}/ofxsLog.cpp
                  ${_ofxs}/ofxsMultiThread.cpp ${_ofxs}/ofxsParams.cpp ${_ofxs}/ofxsProperty.cpp
                  ${_ofxs}/ofxsPropertyValidation.cpp)
set_source_files_properties(${_ofxs_sources} PROPERTIES COMPILE_OPTIONS "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")
add_library(rudra_ofx_plugin MODULE plugin.cpp ${_ofxs_sources})
target_include_directories(rudra_ofx_plugin PRIVATE ${RUDRA_OFX_SDK_DIR}/include ${RUDRA_OFX_SDK_DIR}/Support/include ${_ofxs})
# Link-only: rudra_ofx_process carries C++20 in its usage requirements, and the
# plug-in builds as C++14 (below); it needs nothing from it but the plain interface.
target_include_directories(rudra_ofx_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rudra_ofx_plugin PRIVATE $<LINK_ONLY:rudra_ofx_process>)
# The 1.4 SDK's Support library has dynamic exception specifications, which
# C++17 removed: the plug-in and the library build as C++14 (RUDRA itself is
# reached through rudra_ofx_process's plain interface).
set_target_properties(rudra_ofx_plugin PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS ON)   # gnu++14: the SDK tests the `linux` macro
# The bundle layout every OFX host scans: RUDRA.ofx.bundle/Contents/<arch>/RUDRA.ofx.
if(WIN32)
  set(_ofx_arch Win64)
elseif(APPLE)
  set(_ofx_arch MacOS)
else()
  set(_ofx_arch Linux-x86-64)
endif()
set(RUDRA_OFX_BUNDLE ${CMAKE_BINARY_DIR}/RUDRA.ofx.bundle CACHE INTERNAL "")
set_target_properties(rudra_ofx_plugin PROPERTIES
  PREFIX "" SUFFIX ".ofx" OUTPUT_NAME "RUDRA"
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
  LIBRARY_OUTPUT_DIRECTORY ${RUDRA_OFX_BUNDLE}/Contents/${_ofx_arch}
  LIBRARY_OUTPUT_DIRECTORY_RELEASE ${RUDRA_OFX_BUNDLE}/Contents/${_ofx_arch}
  LIBRARY_OUTPUT_DIRECTORY_DEBUG ${RUDRA_OFX_BUNDLE}/Contents/${_ofx_arch})
# Only the two OFX entry points leave the plug-in: RUDRA's own symbols must not
# meet another plug-in's in the host's process.
if(APPLE)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/exports.txt "_OfxGetNumberOfPlugins\n_OfxGetPlugin\n")
  target_link_options(rudra_ofx_plugin PRIVATE "LINKER:-exported_symbols_list,${CMAKE_CURRENT_BINARY_DIR}/exports.txt")
elseif(NOT WIN32)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/exports.map "{ global: OfxGetNumberOfPlugins; OfxGetPlugin; local: *; };\n")
  target_link_options(rudra_ofx_plugin PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/exports.map" "LINKER:--exclude-libs,ALL")
endif()
if(APPLE)
  configure_file(Info.plist.in ${RUDRA_OFX_BUNDLE}/Contents/Info.plist @ONLY)
endif()

# A test host (the SDK's HostSupport, which needs expat) that loads the bundle
# and renders a frame through it: tests/conform/test_ofx_host.cpp.
find_package(EXPAT QUIET)
if(RUDRA_BUILD_TESTS AND EXPAT_FOUND AND EXISTS ${RUDRA_OFX_SDK_DIR}/HostSupport/src/ofxhHost.cpp)
  file(GLOB _ofxh_sources ${RUDRA_OFX_SDK_DIR}/HostSupport/src/*.cpp)
  set_source_files_properties(${_ofxh_sources} PROPERTIES COMPILE_OPTIONS "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")
  add_library(rudra_ofx_hostsupport STATIC ${_ofxh_sources})
  set_target_properties(rudra_ofx_hostsupport PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS ON)   # gnu++14: the SDK tests the `linux` macro
  target_include_directories(rudra_ofx_hostsupport PUBLIC ${RUDRA_OFX_SDK_DIR}/include ${RUDRA_OFX_SDK_DIR}/HostSupport/include)
  target_link_libraries(rudra_ofx_hostsupport PUBLIC EXPAT::EXPAT ${CMAKE_DL_LIBS})
  set(RUDRA_OFX_HOST_TEST ON CACHE INTERNAL "")
endif()
