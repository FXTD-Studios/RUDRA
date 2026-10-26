add_executable(rudra_tests test_platform.cpp test_core.cpp test_source_curve.cpp test_reference_fit.cpp test_masks.cpp test_anchor_view.cpp test_anchor_target.cpp test_composite.cpp test_clip_fill.cpp test_exposure.cpp test_delivery.cpp test_qc.cpp test_queue.cpp test_sequence.cpp test_viewer.cpp test_viewport.cpp test_engine.cpp test_theme.cpp test_actions.cpp test_session.cpp test_project.cpp test_scope_draw.cpp test_readouts.cpp test_compare.cpp test_render_plan.cpp test_master_job.cpp test_model_catalog.cpp test_copy_texts.cpp test_video.cpp test_video_predict.cpp test_video_encode.cpp test_video_qc.cpp test_sequence_encode.cpp test_ffmpeg_check.cpp test_log_encode.cpp test_master_provenance.cpp test_still_metadata.cpp test_aces_v2.cpp)
add_executable(rudra_headroom_test_producer headroom_test_producer.cpp)
add_executable(rudra_headroom_frames_bench EXCLUDE_FROM_ALL ../bench/headroom_frames.cpp)
target_link_libraries(rudra_headroom_frames_bench PRIVATE rudra::rudra_video nlohmann_json::nlohmann_json)
rudra_warnings(rudra_headroom_frames_bench)
add_executable(rudra_headroom_presentation_bench EXCLUDE_FROM_ALL ../bench/headroom_presentation.cpp)
target_link_libraries(rudra_headroom_presentation_bench PRIVATE rudra::rudra_engine nlohmann_json::nlohmann_json)
rudra_warnings(rudra_headroom_presentation_bench)
rudra_stage_onnxruntime(rudra_headroom_presentation_bench)
target_sources(rudra_tests PRIVATE test_video_decode_float.cpp test_headroom_baseline.cpp test_headroom_render.cpp test_headroom_project.cpp test_headroom_frames.cpp test_headroom_session.cpp test_headroom_preview.cpp test_headroom_export.cpp)
add_dependencies(rudra_tests rudra_headroom_test_producer)
target_compile_definitions(rudra_tests PRIVATE RUDRA_HEADROOM_PRODUCER_PATH="$<TARGET_FILE:rudra_headroom_test_producer>")
target_link_libraries(rudra_tests PRIVATE rudra::rudra_core rudra::rudra_deliver rudra::rudra_media rudra::rudra_video rudra::rudra_engine
                      nlohmann_json::nlohmann_json GTest::gtest_main)
# Prefer the headers of the linked GTest over unrelated dependency prefixes
# (e.g. a Conda nlohmann_json whose include dir also holds another GTest),
# as app/CMakeLists.txt does for rudra_app_tests.
target_include_directories(rudra_tests BEFORE PRIVATE
                           $<TARGET_PROPERTY:GTest::gtest,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(rudra_tests PRIVATE RUDRA_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden"
                           RUDRA_THEME_DIR="${RUDRA_THEME_DIR}" RUDRA_THEME_CSS="${RUDRA_THEME_CSS}"
                           RUDRA_APP_DIR="${PROJECT_SOURCE_DIR}/app" RUDRA_CMAKE_COMMAND="${CMAKE_COMMAND}")
add_dependencies(rudra_tests rudra_theme)
if(RUDRA_BUILD_CLI)   # test_sequence_encode runs `rudra-native deliver`
  add_dependencies(rudra_tests rudra-native)
  target_compile_definitions(rudra_tests PRIVATE RUDRA_CLI_PATH="$<TARGET_FILE:rudra-native>")
endif()   # test_theme reads the generated sheet
rudra_warnings(rudra_tests)
rudra_stage_onnxruntime(rudra_tests)
if(RUDRA_WITH_OPENCV)
  target_sources(rudra_tests PRIVATE test_decode.cpp test_video_master.cpp test_video_graded.cpp)
endif()

include(GoogleTest)
# Listed when ctest runs, not after the link: on Windows the runtimes' DLLs are
# copied beside the executable after the build, so running it at link time fails.
gtest_discover_tests(rudra_tests DISCOVERY_MODE PRE_TEST)

# Gate A in C++: point at an exported package to run it as a test
# (cmake -DRUDRA_TEST_PACKAGE=dist/models/sdr2hdr_shadow_v1).
set(RUDRA_TEST_PACKAGE "" CACHE PATH "Model package for the golden parity test")
if(RUDRA_TEST_PACKAGE AND RUDRA_BUILD_CLI)
  add_test(NAME model_package_parity COMMAND rudra-native diff "${RUDRA_TEST_PACKAGE}" --runtime all)
  if(RUDRA_WITH_OPENCV)
    # Phase 2 step 10: the frame path with the real model over real images.
    target_sources(rudra_tests PRIVATE test_frame_engine_real.cpp test_video_predict_real.cpp test_video_convert_real.cpp test_video_queue_real.cpp)
    target_link_libraries(rudra_tests PRIVATE rudra::rudra_infer)
    target_compile_definitions(rudra_tests PRIVATE RUDRA_TEST_PACKAGE_DIR="${RUDRA_TEST_PACKAGE}")
    # Phase 1 step 7: a Studio-identical master with no Python.
    add_test(NAME master_parity COMMAND rudra-native master-check "${RUDRA_TEST_PACKAGE}"
             "${CMAKE_CURRENT_SOURCE_DIR}/golden/master")
  endif()
endif()
