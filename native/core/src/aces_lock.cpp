// The lock (stage 6b). Mirrors training/aces_partial_tighten.py operation by operation,
// in float64 like the NumPy, so tests/golden/aces_lock holds it to the Python oracle.

#include "rudra/core/aces_lock.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace rudra::aces_v2 {
namespace {

using clock_type = std::chrono::steady_clock;
double ms_since(clock_type::time_point t) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t).count();
}

// f(y) for y in [0, h) on every core, in contiguous blocks of at least `min_items`; the
// work per item is independent, so the result is the same for any thread count
// (RUDRA_SMOOTH_THREADS=1 forces one).
template <class F>
void parallel_rows(int h, F&& f, int min_items = 8) {
    unsigned t = std::max(1u, std::thread::hardware_concurrency());
    if (const char* e = std::getenv("RUDRA_SMOOTH_THREADS")) t = unsigned(std::max(1, std::atoi(e)));
    t = std::min<unsigned>(t, unsigned(std::max(1, h / min_items)));
    if (t <= 1) {
        for (int y = 0; y < h; ++y) f(y);
        return;
    }
    std::vector<std::thread> pool;
    const int step = (h + int(t) - 1) / int(t);
    for (int y0 = 0; y0 < h; y0 += step)
        pool.emplace_back([&f, y0, y1 = std::min(h, y0 + step)] {
            for (int y = y0; y < y1; ++y) f(y);
        });
    for (auto& th : pool) th.join();
}

// scipy.ndimage.laplace(a, mode='reflect'), one row: per axis the symmetric [1, -2, 1] correlation
// x*-2 + (prev + next), axis 0 first, then += axis 1; the border repeats the edge sample (d c b a | a b c d).
inline void laplace_row(const double* a, double* o, int y, int h, int w) {
    const double* row = a + std::size_t(y) * w;
    const double* up = a + std::size_t(y > 0 ? y - 1 : 0) * w;
    const double* dn = a + std::size_t(y < h - 1 ? y + 1 : h - 1) * w;
    auto px = [&](int x, double l, double r) {
        const double c = row[x];
        const double dy = c * -2.0 + (up[x] + dn[x]);
        const double dx = c * -2.0 + (l + r);
        o[x] = dy + dx;
    };
    if (w == 1) {
        px(0, row[0], row[0]);
        return;
    }
    px(0, row[0], row[1]);
    for (int x = 1; x < w - 1; ++x) {   // no edge tests inside: the compiler vectorises it
        const double c = row[x];
        const double dy = c * -2.0 + (up[x] + dn[x]);
        const double dx = c * -2.0 + (row[x - 1] + row[x + 1]);
        o[x] = dy + dx;
    }
    px(w - 1, row[w - 2], row[w - 1]);
}

// A reusable barrier for a fixed set of threads (std::barrier needs macOS 11 in Apple's libc++).
class Barrier {
public:
    explicit Barrier(int n) : n_(n) {}
    void wait() {
        std::unique_lock<std::mutex> l(m_);
        const long gen = gen_;
        if (++count_ == n_) {
            count_ = 0;
            ++gen_;
            cv_.notify_all();
        } else {
            cv_.wait(l, [&] { return gen_ != gen; });
        }
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    int n_, count_ = 0;
    long gen_ = 0;
};

// scipy.ndimage.uniform_filter1d(size 5, mode='nearest') along one line, as scipy's C
// does it: a running sum, divided per sample.
void uniform5_line(const double* in, std::size_t stride, int n, double* out, std::size_t ostride) {
    constexpr int size = 5, s1 = 2;
    auto at = [&](int i) { return in[std::size_t(std::clamp(i, 0, n - 1)) * stride]; };
    double tmp = 0.0;
    for (int l = 0; l < size; ++l) tmp += at(l - s1);
    out[0] = tmp / size;
    for (int l = 1; l < n; ++l) {
        tmp += at(l + size - 1 - s1) - at(l - 1 - s1);
        out[std::size_t(l) * ostride] = tmp / size;
    }
}

// scipy.ndimage.uniform_filter(a, 5, mode='nearest') in 2-D: axis 0, then axis 1.
std::vector<double> uniform5(const std::vector<double>& a, int h, int w) {
    std::vector<double> t(a.size()), o(a.size());
    for (int x = 0; x < w; ++x) uniform5_line(a.data() + x, std::size_t(w), h, t.data() + x, std::size_t(w));
    for (int y = 0; y < h; ++y) uniform5_line(t.data() + std::size_t(y) * w, 1, w, o.data() + std::size_t(y) * w, 1);
    return o;
}

// numpy.linalg.pinv(a, rcond=1e-10) for a 3x3 matrix (row major), by one-sided Jacobi SVD.
std::array<double, 9> pinv3(const std::array<double, 9>& a, double rcond = 1e-10) {
    double u[3][3], v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) u[i][j] = a[std::size_t(i * 3 + j)];
    for (int sweep = 0; sweep < 60; ++sweep) {
        bool rotated = false;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                double al = 0, be = 0, ga = 0;
                for (int i = 0; i < 3; ++i) {
                    al += u[i][p] * u[i][p];
                    be += u[i][q] * u[i][q];
                    ga += u[i][p] * u[i][q];
                }
                // Converged, or one column is numerically null (its singular value is far
                // below the cutoff either way): rotating it only churns denormals.
                if (ga == 0.0 || std::fabs(ga) <= 1e-15 * std::sqrt(al * be) || std::min(al, be) <= 1e-30 * std::max(al, be))
                    continue;
                rotated = true;
                const double zeta = (be - al) / (2.0 * ga);
                const double t = (zeta >= 0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t), s = c * t;
                for (int i = 0; i < 3; ++i) {
                    const double up = u[i][p], uq = u[i][q];
                    u[i][p] = c * up - s * uq;
                    u[i][q] = s * up + c * uq;
                    const double vp = v[i][p], vq = v[i][q];
                    v[i][p] = c * vp - s * vq;
                    v[i][q] = s * vp + c * vq;
                }
            }
        if (!rotated) break;
    }
    double sig[3];
    double smax = 0.0;
    for (int j = 0; j < 3; ++j) {
        double n2 = 0;
        for (int i = 0; i < 3; ++i) n2 += u[i][j] * u[i][j];
        sig[j] = std::sqrt(n2);
        smax = std::max(smax, sig[j]);
    }
    std::array<double, 9> out{};
    const double cutoff = rcond * smax;
    for (int j = 0; j < 3; ++j) {
        if (!(sig[j] > cutoff)) continue;
        const double inv2 = 1.0 / (sig[j] * sig[j]);   // u col j is sigma_j * (unit left vector)
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) out[std::size_t(r * 3 + c)] += v[r][j] * u[c][j] * inv2;
    }
    return out;
}

// The view on packed float64 pixels, through float32 as the Python's fwd does.
void forward64(const AcesView& view, const std::vector<double>& x, std::vector<double>& d, std::vector<float>& scratch) {
    scratch.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) scratch[i] = static_cast<float>(x[i]);
    view.forward(scratch.data(), x.size() / 3);
    d.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) d[i] = static_cast<double>(scratch[i]);
}

}  // namespace

Dequantized dequantize_recorded(const PlanarBuffer& source, const std::vector<std::uint8_t>* region, int iters,
                                double max_lap, double half_bin, double clip_ceiling) {
    const int H = source.height(), W = source.width();
    const std::size_t n = source.plane_size();
    Dequantized out;
    out.values.resize(3 * n);
    out.used.assign(3 * n, 0);
    for (int c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < n; ++i) out.values[c * n + i] = static_cast<double>(source.plane(c)[i]);
    int y0 = 0, y1 = H, x0 = 0, x1 = W;
    if (region) {
        int ymin = H, ymax = -1, xmin = W, xmax = -1;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if ((*region)[std::size_t(y) * W + x]) {
                    ymin = std::min(ymin, y), ymax = std::max(ymax, y);
                    xmin = std::min(xmin, x), xmax = std::max(xmax, x);
                }
        if (ymax < 0) return out;
        constexpr int m = 24;
        y0 = std::max(ymin - m, 0), y1 = std::min(ymax + m + 1, H);
        x0 = std::max(xmin - m, 0), x1 = std::min(xmax + m + 1, W);
    }
    const int h = y1 - y0, w = x1 - x0;
    const std::size_t bn = std::size_t(h) * w;
    struct Plane {
        std::vector<double> codes, lo, hi, u, yv, l1;
        std::vector<std::uint8_t> rec;
    };
    std::array<Plane, 3> P;
    for (int c = 0; c < 3; ++c) {
        Plane& q = P[std::size_t(c)];
        q.codes.resize(bn), q.lo.resize(bn), q.hi.resize(bn), q.rec.resize(bn), q.l1.resize(bn);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const std::size_t k = std::size_t(y) * w + x;
                q.codes[k] = static_cast<double>(source.at(c, y0 + y, x0 + x)) * 255.0;
                q.rec[k] = q.codes[k] < 254.5;
                q.lo[k] = q.rec[k] ? q.codes[k] - half_bin : 254.5;
                q.hi[k] = q.rec[k] ? q.codes[k] + half_bin : 255.0 + clip_ceiling;
            }
        q.u = q.codes;
        q.yv = q.codes;
    }
    // The accelerated projected gradient on ||L u||^2 (||L|| <= 8: a 1/64 step is nonexpansive), all
    // three channels, on threads started once (8 Nov 2026: a thread start per step cost ~4 s a frame on
    // Windows). Each thread owns a band of rows; per step: L(y) on its rows, a barrier, then L(L(y)) and
    // the update on its rows, a barrier. Every pixel's arithmetic is unchanged, so any thread count
    // gives the same result.
    unsigned nt_threads = std::max(1u, std::thread::hardware_concurrency());
    if (const char* e = std::getenv("RUDRA_SMOOTH_THREADS")) nt_threads = unsigned(std::max(1, std::atoi(e)));
    const int T = int(std::min<unsigned>(nt_threads, unsigned(std::max(1, h / 4))));
    Barrier barrier(T);
    auto worker = [&](int ti) {
        const int r0 = int(std::int64_t(h) * ti / T), r1 = int(std::int64_t(h) * (ti + 1) / T);
        std::vector<double> lrow(static_cast<std::size_t>(w));
        double t = 1.0;
        for (int it = 0; it < iters; ++it) {
            for (auto& q : P)
                for (int y = r0; y < r1; ++y) laplace_row(q.yv.data(), q.l1.data() + std::size_t(y) * w, y, h, w);
            barrier.wait();
            const double nt = (1.0 + std::sqrt(1.0 + 4.0 * t * t)) / 2.0;
            const double mom = (t - 1.0) / nt;
            for (auto& q : P)
                for (int y = r0; y < r1; ++y) {
                    laplace_row(q.l1.data(), lrow.data(), y, h, w);
                    const std::size_t k0 = std::size_t(y) * w;
                    for (int x = 0; x < w; ++x) {
                        const std::size_t k = k0 + std::size_t(x);
                        const double nu = std::min(std::max(q.yv[k] - lrow[std::size_t(x)] / 64.0, q.lo[k]), q.hi[k]);
                        q.yv[k] = nu + mom * (nu - q.u[k]);
                        q.u[k] = nu;
                    }
                }
            barrier.wait();
            t = nt;
        }
    };
    if (iters > 0) {
        std::vector<std::thread> pool;
        for (int ti = 1; ti < T; ++ti) pool.emplace_back(worker, ti);
        worker(0);
        for (auto& th : pool) th.join();
    }
    for (int c = 0; c < 3; ++c) {
        const Plane& q = P[std::size_t(c)];
        const auto& codes = q.codes;
        const auto& rec = q.rec;
        const auto& u = q.u;
        // Smooth content: mean |code Laplacian| over 5x5 (interior, fully recorded crosses only).
        std::vector<double> lapabs(bn, 0.0), okd(bn, 0.0);
        for (int y = 1; y < h - 1; ++y)
            for (int x = 1; x < w - 1; ++x) {
                const std::size_t k = std::size_t(y) * w + x;
                const double lap = codes[k - w] + codes[k + w] + codes[k - 1] + codes[k + 1] - 4.0 * codes[k];
                const bool ok = rec[k] && rec[k - w] && rec[k + w] && rec[k - 1] && rec[k + 1];
                lapabs[k] = std::fabs(lap) * (ok ? 1.0 : 0.0);
                okd[k] = ok ? 1.0 : 0.0;
            }
        const std::vector<double> num = uniform5(lapabs, h, w), den = uniform5(okd, h, w);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const std::size_t k = std::size_t(y) * w + x;
                const bool use = rec[k] && den[k] > 0.2 && num[k] <= max_lap * den[k];
                const std::size_t o = c * n + std::size_t(y0 + y) * W + std::size_t(x0 + x);
                out.values[o] = (use ? u[k] : codes[k]) / 255.0;
                out.used[o] = use;
            }
    }
    return out;
}

PlanarBuffer lock_observed(const PlanarBuffer& source, const PlanarBuffer& constrained, const PlanarBuffer& weight,
                           const AcesView& view, const LockSettings& st, LockStats* stats) {
    constexpr double kEps = 2e-3;
    const auto t0 = clock_type::now();
    LockStats s;
    PlanarBuffer out = constrained;
    const std::size_t n = source.plane_size();
    const float full_f = static_cast<float>(kClipCode);
    std::vector<std::uint8_t> sup(n);
    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < n; ++i) {
        const float a = source.plane(0)[i], b = source.plane(1)[i], c = source.plane(2)[i];
        sup[i] = weight.plane(0)[i] > 0.0f && std::max(std::max(a, b), c) > 0.5f && std::min(std::min(a, b), c) < full_f;
        if (sup[i]) idx.push_back(i);
    }
    s.support_pixels = idx.size();
    if (idx.empty()) {
        if (stats) *stats = s;
        return out;
    }
    const std::size_t m = idx.size();
    std::vector<double> src(m * 3), x(m * 3), target(m * 3);
    std::vector<std::uint8_t> rec(m * 3);
    for (std::size_t k = 0; k < m; ++k)
        for (int c = 0; c < 3; ++c) {
            src[k * 3 + c] = static_cast<double>(source.plane(c)[idx[k]]);
            x[k * 3 + c] = static_cast<double>(constrained.plane(c)[idx[k]]);
            rec[k * 3 + c] = src[k * 3 + c] < kClipCode;
        }
    if (st.dequant) {
        const Dequantized dq = dequantize_recorded(source, &sup, st.dq_iters, st.dq_max_lap);
        for (std::size_t k = 0; k < m; ++k)
            for (int c = 0; c < 3; ++c) {
                target[k * 3 + c] = dq.values[c * n + idx[k]];
                s.dequant_channels += dq.used[c * n + idx[k]];
            }
    } else {
        target = src;
    }
    s.ms_dequant = ms_since(t0);
    const auto t1 = clock_type::now();

    std::vector<float> scratch;
    std::vector<double> d, dp, xp(m * 3), xpos(m * 3);
    const int iters = std::max(3, std::min(st.iters, 6));
    for (int it = 0; it < iters; ++it) {
        forward64(view, x, d, scratch);
        std::vector<double> J(m * 9);   // J[k][row][col]
        std::vector<double> h(m * 3);
        for (int c = 0; c < 3; ++c) {
            xp = x;
            for (std::size_t k = 0; k < m; ++k) {
                h[k * 3 + c] = kEps * std::max(std::fabs(x[k * 3 + c]), 1e-3);
                xp[k * 3 + c] += h[k * 3 + c];
            }
            forward64(view, xp, dp, scratch);
            for (std::size_t k = 0; k < m; ++k)
                for (int r = 0; r < 3; ++r) J[k * 9 + r * 3 + c] = (dp[k * 3 + r] - d[k * 3 + r]) / h[k * 3 + c];
        }
        // Per pixel and independent: blocks of 4096 pixels on every core (same result for any count).
        const int blocks = int((m + 4095) / 4096);
        parallel_rows(blocks, [&](int b) {
            const std::size_t k0 = std::size_t(b) * 4096, k1 = std::min(m, k0 + 4096);
            for (std::size_t k = k0; k < k1; ++k) {
            std::array<double, 9> jr;
            double xq[3], res[3];
            for (int c = 0; c < 3; ++c) {
                xq[c] = std::max(x[k * 3 + c], 1e-6);
                res[c] = rec[k * 3 + c] ? target[k * 3 + c] - d[k * 3 + c] : 0.0;
            }
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    jr[std::size_t(r * 3 + c)] = J[k * 9 + r * 3 + c] * xq[c] * (rec[k * 3 + r] ? 1.0 : 0.0);
            const auto p = pinv3(jr);
            for (int r = 0; r < 3; ++r) {
                const double du = p[std::size_t(r * 3)] * res[0] + p[std::size_t(r * 3 + 1)] * res[1] +
                                  p[std::size_t(r * 3 + 2)] * res[2];
                x[k * 3 + r] = xq[r] * std::exp(std::clamp(du, -2.0, 2.0));
            }
            }
        }, 1);
    }
    forward64(view, x, d, scratch);
    std::vector<std::uint8_t> ok(m);
    std::vector<std::size_t> oks;
    for (std::size_t k = 0; k < m; ++k) {
        double err = 0.0, near = 0.0;
        bool finite = true, unclipped = false;
        for (int c = 0; c < 3; ++c) {
            const std::size_t q = k * 3 + c;
            finite = finite && std::isfinite(x[q]);
            if (rec[q]) {
                err = std::max(err, std::fabs(d[q] - target[q]));
                near = std::max(near, std::fabs(d[q] - src[q]));
            } else if (d[q] < kClipCode) {
                unclipped = true;
            }
        }
        err = std::max(err * 255.0 - 0.0, 0.0);
        ok[k] = finite && err <= 0.5 && near * 255.0 <= 1.0 && !unclipped;
        if (ok[k]) {
            oks.push_back(k);
            s.max_recorded_error_codes = std::max(s.max_recorded_error_codes, err);
        }
    }
    s.changed_pixels = oks.size();
    s.kept_pixels = m - oks.size();

    if (st.strength < 1.0) {
        // The appearance guard: the weakest strength, lambda up to the full lock, whose
        // geometric blend shows every recorded channel within 1 code and keeps clipped ones clipped.
        const double lam = st.strength;
        const double steps[5] = {lam, lam + (1 - lam) * .25, lam + (1 - lam) * .5, lam + (1 - lam) * .75, 1.0};
        auto powr = [](double b, double e) { return e == 0.5 ? std::sqrt(b) : e == 1.0 ? b : std::pow(b, e); };
        std::vector<std::size_t> todo = oks, next;
        std::vector<double> blend, db;
        for (double lk : steps) {
            if (todo.empty()) break;
            blend.resize(todo.size() * 3);
            for (std::size_t j = 0; j < todo.size(); ++j)
                for (int c = 0; c < 3; ++c) {
                    const double raw = static_cast<double>(constrained.plane(c)[idx[todo[j]]]);
                    blend[j * 3 + c] = powr(std::max(raw, 1e-9), 1.0 - lk) * powr(std::max(x[todo[j] * 3 + c], 1e-9), lk);
                }
            forward64(view, blend, db, scratch);
            next.clear();
            for (std::size_t j = 0; j < todo.size(); ++j) {
                const std::size_t k = todo[j];
                double e = 0.0;
                bool unclipped = false;
                for (int c = 0; c < 3; ++c) {
                    if (rec[k * 3 + c]) e = std::max(e, std::fabs(db[j * 3 + c] - src[k * 3 + c]));
                    else if (db[j * 3 + c] < kClipCode) unclipped = true;
                }
                if ((e * 255.0 <= 1.0 && !unclipped) || lk >= 1.0) {
                    for (int c = 0; c < 3; ++c) out.plane(c)[idx[k]] = static_cast<float>(blend[j * 3 + c]);
                    if (lk > lam + 1e-9) ++s.strength_raised_pixels;
                } else {
                    next.push_back(k);
                }
            }
            todo.swap(next);
        }
    } else {
        for (std::size_t k : oks)
            for (int c = 0; c < 3; ++c) out.plane(c)[idx[k]] = static_cast<float>(x[k * 3 + c]);
    }
    s.ms_solve = ms_since(t1);
    if (stats) *stats = s;
    return out;
}

}  // namespace rudra::aces_v2
