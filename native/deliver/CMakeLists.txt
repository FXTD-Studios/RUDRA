# deliver: what leaves the app. EXR/ACES/OCIO writers and the metadata
# sidecars now (byte-identical ports of rudra/delivery); presets, the ffmpeg
# pipe, QC and the queue follow in Phase 1 steps 8-9 and Phase 4.
rudra_library(rudra_deliver SOURCES src/exr.cpp src/sidecars.cpp src/qc.cpp src/queue.cpp src/master.cpp src/video_master.cpp src/video_encode.cpp src/video_qc.cpp src/sequence_encode.cpp src/shot_grades.cpp src/conform.cpp src/correction_log.cpp src/dolby_vision.cpp)
target_link_libraries(rudra_deliver PUBLIC rudra::rudra_core rudra::rudra_platform PRIVATE nlohmann_json::nlohmann_json)
# Provenance (roadmap 1.3): every master records the build that wrote it.
target_compile_definitions(rudra_deliver PRIVATE RUDRA_VERSION_LABEL="${RUDRA_VERSION_LABEL}"
                           RUDRA_BUILD_COMMIT="${RUDRA_BUILD_COMMIT}")
