#pragma once
// `rudra-native bench-clip-fill <bench>`: the clip fill (core/clip_fill.hpp)
// as a bench tree, beside the export's baseline/ and model trees, so
// training/score_regions.py scores it like any other (12 Oct 2026).

#include <string>
#include <vector>

namespace rudra {

int cmd_bench_clip_fill(const std::vector<std::string>& args);

}  // namespace rudra
