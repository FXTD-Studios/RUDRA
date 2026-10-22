# cli: rudra-native. Core, infer, media and deliver; must link without Qt.
add_executable(rudra-native main.cpp deliver.cpp ffmpeg_check.cpp)
target_link_libraries(rudra-native PRIVATE rudra::rudra_infer rudra::rudra_deliver rudra::rudra_video nlohmann_json::nlohmann_json)
if(RUDRA_WITH_OPENCV)
  # `master`, `master-check` and `video` need still decode (the spool is PNG).
  target_sources(rudra-native PRIVATE master.cpp video.cpp batch.cpp bench_clip_fill.cpp)
  target_link_libraries(rudra-native PRIVATE rudra::rudra_media)
endif()
rudra_warnings(rudra-native)
rudra_stage_onnxruntime(rudra-native)

# clip_fill_probe (17 Oct 2026): the shipped clip fill as a file-in/file-out tool, for
# PeakNet's data and native scoring (training/peaknet/). Core only, no Qt, no OpenCV.
add_executable(clip_fill_probe ../tools/clip_fill_probe.cpp)
target_link_libraries(clip_fill_probe PRIVATE rudra::rudra_core)
rudra_warnings(clip_fill_probe)
