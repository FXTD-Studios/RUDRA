#pragma once
#include <string>
#include <vector>
// rudra-native aces-v2-serve <package> [...]: the ACES v2 master as a long-running
// engine, JSON lines on stdin/stdout (aces_v2_serve.cpp).
int cmd_aces_v2_serve(const std::vector<std::string>& args);
