# Roadmap 4.3, 5.2, 5.4 and 5.5 (20 Oct 2026): shot keyframes, the correction
# log, conform by shot and the Dolby Vision RPU, in an executable of their own.
add_executable(rudra_conform_tests test_shot_grades.cpp test_conform.cpp test_correction_log.cpp test_dolby_vision.cpp)
target_link_libraries(rudra_conform_tests PRIVATE rudra::rudra_core rudra::rudra_deliver rudra::rudra_video
                      nlohmann_json::nlohmann_json GTest::gtest_main)
target_include_directories(rudra_conform_tests BEFORE PRIVATE
                           $<TARGET_PROPERTY:GTest::gtest,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(rudra_conform_tests PRIVATE RUDRA_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")
if(RUDRA_BUILD_CLI)
  add_dependencies(rudra_conform_tests rudra-native)
  target_compile_definitions(rudra_conform_tests PRIVATE RUDRA_CLI_PATH="$<TARGET_FILE:rudra-native>")
endif()
if(RUDRA_WITH_OPENCV)   # the movie export
  target_sources(rudra_conform_tests PRIVATE test_shot_export.cpp test_conform_movie.cpp)
endif()
rudra_warnings(rudra_conform_tests)
include(GoogleTest)
gtest_discover_tests(rudra_conform_tests DISCOVERY_MODE PRE_TEST)

# The OFX plug-in's picture (roadmap 5.1), and, with the plug-in built and the
# SDK's HostSupport available, the plug-in in a test host.
target_sources(rudra_conform_tests PRIVATE test_ofx_process.cpp)
target_link_libraries(rudra_conform_tests PRIVATE rudra_ofx_process)
if(RUDRA_OFX_HOST_TEST)
  add_executable(rudra_ofx_host_tests test_ofx_host.cpp)
  set_target_properties(rudra_ofx_host_tests PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS ON)   # gnu++14: the SDK tests the `linux` macro
  target_link_libraries(rudra_ofx_host_tests PRIVATE rudra_ofx_hostsupport $<LINK_ONLY:rudra_ofx_process> GTest::gtest_main)
  target_include_directories(rudra_ofx_host_tests PRIVATE ${PROJECT_SOURCE_DIR}/ofx)
  target_include_directories(rudra_ofx_host_tests BEFORE PRIVATE $<TARGET_PROPERTY:GTest::gtest,INTERFACE_INCLUDE_DIRECTORIES>)
  target_compile_definitions(rudra_ofx_host_tests PRIVATE RUDRA_OFX_BUNDLE_DIR="${CMAKE_BINARY_DIR}")
  add_dependencies(rudra_ofx_host_tests rudra_ofx_plugin)
  gtest_discover_tests(rudra_ofx_host_tests DISCOVERY_MODE PRE_TEST)
endif()
