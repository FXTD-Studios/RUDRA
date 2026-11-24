#include "rudra/core/block_grid.hpp"

#include <cstdlib>

namespace rudra {

double block_grid(const std::uint8_t* luma, int width, int height, std::ptrdiff_t stride) noexcept {
    if (!luma || width < 16 || height < 16) return 1.0;
    double edge_sum = 0.0, other_sum = 0.0;
    std::size_t edge_n = 0, other_n = 0;
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* row = luma + std::ptrdiff_t(y) * stride;
        for (int x = 0; x + 1 < width; ++x) {   // the step between x and x + 1: a block edge when x is 7, 15, ...
            const int d = std::abs(int(row[x + 1]) - int(row[x]));
            if ((x & 7) == 7) edge_sum += d, ++edge_n;
            else other_sum += d, ++other_n;
        }
        if (y + 1 < height) {
            const std::uint8_t* next = row + stride;
            const bool edge = (y & 7) == 7;
            for (int x = 0; x < width; ++x) {
                const int d = std::abs(int(next[x]) - int(row[x]));
                if (edge) edge_sum += d, ++edge_n;
                else other_sum += d, ++other_n;
            }
        }
    }
    if (!edge_n || !other_n) return 1.0;
    const double edge = edge_sum / double(edge_n), other = other_sum / double(other_n);
    return other > 1e-9 ? edge / other : 1.0;
}

int choose_cleanup_qp(std::span<const int> tried, std::span<const double> measured, double target) noexcept {
    if (tried.empty() || tried.size() != measured.size()) return 0;
    for (std::size_t i = 0; i < tried.size(); ++i)
        if (measured[i] <= target) return tried[i];
    return tried.back();
}

}  // namespace rudra
