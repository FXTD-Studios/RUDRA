# infer: InferenceBackend and its runtimes. Runtime headers (torch, onnxruntime)
# appear only in src/*_backend.cpp; the public headers stay runtime-free.
set(_sources src/tiler.cpp src/self_test.cpp)
if(RUDRA_WITH_LIBTORCH)
  list(APPEND _sources src/libtorch_backend.cpp)
endif()
if(RUDRA_WITH_ONNXRUNTIME)
  list(APPEND _sources src/onnxruntime_backend.cpp)
endif()
# ACES v2 staged pipeline runtime pieces (core/aces_v2.hpp).
if(RUDRA_WITH_LIBTORCH)
  list(APPEND _sources src/aces_g1_libtorch.cpp)
endif()
if(RUDRA_WITH_OCIO)
  list(APPEND _sources src/aces_ocio_view.cpp)
endif()
rudra_library(rudra_infer SOURCES ${_sources})
target_link_libraries(rudra_infer PUBLIC rudra::rudra_core PRIVATE nlohmann_json::nlohmann_json)

if(RUDRA_WITH_LIBTORCH)
  target_compile_definitions(rudra_infer PRIVATE RUDRA_HAVE_LIBTORCH)
  target_link_libraries(rudra_infer PRIVATE ${TORCH_LIBRARIES})
  # LibTorch's own headers are noisy under -Wconversion/-Wshadow.
  # Keep MSVC's conforming front end: /permissive triggers incomplete-type
  # trait errors in current Torch headers. Suppress third-party warnings only.
  set_source_files_properties(src/libtorch_backend.cpp src/aces_g1_libtorch.cpp PROPERTIES COMPILE_OPTIONS
    "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-conversion;-Wno-shadow;-Wno-pedantic;-Wno-unused-parameter>;$<$<CXX_COMPILER_ID:MSVC>:/external:W0;/wd4244;/wd4267;/wd4251;/wd4275;/wd4100>")
  if(CMAKE_DL_LIBS)
    target_link_libraries(rudra_infer PRIVATE ${CMAKE_DL_LIBS})
  endif()
endif()
if(RUDRA_WITH_ONNXRUNTIME)
  target_compile_definitions(rudra_infer PRIVATE RUDRA_HAVE_ONNXRUNTIME)
  target_link_libraries(rudra_infer PRIVATE onnxruntime::onnxruntime)
endif()

if(RUDRA_WITH_OCIO)
  target_compile_definitions(rudra_infer PUBLIC RUDRA_HAVE_OCIO)
  target_link_libraries(rudra_infer PRIVATE OpenColorIO::OpenColorIO)
  # Bit parity with the Python reference (tools/export_native_v2.py), needs LibTorch too:
  #   aces_v2_parity in.f32 package/rudra_g1.ts outdir ; then tools/aces_v2_parity/parity_check.py
  if(RUDRA_WITH_LIBTORCH)
    add_executable(aces_v2_parity ../tools/aces_v2_parity/aces_v2_parity.cpp)
    target_link_libraries(aces_v2_parity PRIVATE rudra::rudra_infer rudra::rudra_core)
    add_executable(aces_v2_constraint_parity ../tools/aces_v2_parity/aces_v2_constraint_parity.cpp)
    target_link_libraries(aces_v2_constraint_parity PRIVATE rudra::rudra_infer rudra::rudra_core)
  endif()
endif()
