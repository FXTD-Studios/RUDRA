#pragma once
#include <string>
#include <vector>
// rudra-native aces-v2 <package> <image|folder> --out <dir> [--runtime libtorch|onnxruntime] [--device cpu|cuda|directml] [--ocio-threads N] [--no-smoothing]
int cmd_aces_v2(const std::vector<std::string>& args);
