#pragma once
// rudra-native video: rudra/video.py's command line, on a model package.
#include <string>
#include <vector>

namespace rudra {
// args after "video": <package> <input> --output <file> [video.py's options] [--runtime ...] [--device ...]
int cmd_video(const std::vector<std::string>& args);
// args after "conform" (roadmap 5.4): <movie> (--edl FILE | --fcpxml FILE | --detect) --output <shots.json>
// [--base <settings.json>] [--record-start HH:MM:SS:FF] [--fps RATE] [--threshold X] [--key-each-shot]
int cmd_conform(const std::vector<std::string>& args);
}  // namespace rudra
