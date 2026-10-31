// Deterministic child process for cancellation/late-failure tests. Not installed.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    // A pinned stand-in ffprobe, fed the real fixture's saved probe JSON.
    // A marker proves the requested blocking phase has actually been reached.
    bool probe = false, frames = false;
    for (int i = 1; i < argc; ++i) {
        probe |= std::string(argv[i]) == "-show_streams";
        frames |= std::string(argv[i]) == "-show_frames";
    }
    if (probe) {
        const std::string path = argv[argc - 1];
        const bool stall = !std::filesystem::exists(path + ".release-probe") &&
            ((path.find("probe-wait-stream") != std::string::npos && !frames) ||
             (path.find("probe-wait-frame") != std::string::npos && frames));
        const auto text = path + (frames ? ".frames.json" : ".streams.json");
        if (!stall || path.find("after") != std::string::npos) {
            std::ifstream in(text, std::ios::binary);
            const std::string data((std::istreambuf_iterator<char>(in)), {});
            std::fwrite(data.data(), 1, data.size(), stdout); std::fflush(stdout);
        }
        if (stall && path.find("after") != std::string::npos) {
            std::fclose(stdout); std::fclose(stderr);
        }
        { std::ofstream marker(path + ".probe-calls", std::ios::app); marker << (frames ? "frames\n" : "streams\n"); }
        if (stall) std::this_thread::sleep_for(std::chrono::seconds(10));
        return 0;
    }

    std::string source, filter;
    int count = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "wait-before-eof" || arg == "wait-after-eof") {
            if (arg == "wait-after-eof") std::fclose(stdout);
            std::this_thread::sleep_for(std::chrono::seconds(10)); return 0;
        }
        if (i + 1 < argc) {
            if (arg == "-i") source = argv[i + 1];
            if (arg == "-vf") filter = argv[i + 1];
            if (arg == "-frames:v") count = std::stoi(argv[i + 1]);
        }
    }
    if (source.find("stall") != std::string::npos && filter.starts_with("trim=start_frame=0:"))
        std::this_thread::sleep_for(std::chrono::seconds(10));
    std::vector<float> pixels(std::size_t(count) * 16 * 4 * 3, 1.25f);
    pixels[0] = -.125f;
    if (source.find("nonfinite") != std::string::npos) pixels.back() = std::numeric_limits<float>::infinity();
    auto bytes = pixels.size() * sizeof(float);
    if (source.find("truncated") != std::string::npos) --bytes;
    if (source.find("read-wait") != std::string::npos && count > 1) {
        const std::size_t first = 16 * 4 * 3 * sizeof(float);
        std::fwrite(pixels.data(), 1, first, stdout); std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(10));
        std::fwrite(reinterpret_cast<const char*>(pixels.data()) + first, 1, bytes - first, stdout);
    } else std::fwrite(pixels.data(), 1, bytes, stdout);
    if (source.find("extra") != std::string::npos) std::fputc(0, stdout);
    if (source.find("delayed-late-failure") != std::string::npos) {
        std::fflush(stdout); std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    if (source.find("finish-wait-") != std::string::npos) {
        std::fflush(stdout);
        if (source.find("finish-wait-after") != std::string::npos) std::fclose(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
    return source.find("late-failure") != std::string::npos ? 7 : 0;
}
