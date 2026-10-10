#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <cerrno>
#include <cstdlib>

#include "rudra/platform/hash.hpp"
#include "rudra/platform/io_error.hpp"
#include "rudra/platform/png8.hpp"
#include "rudra/platform/npy.hpp"
#include "rudra/platform/process.hpp"
#include "rudra/platform/tools.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace rudra;

namespace {
std::span<const std::byte> bytes(const std::string& s) { return std::as_bytes(std::span(s.data(), s.size())); }
}  // namespace

TEST(Xxh64, ReferenceVectors) {
    // From the reference implementation (python-xxhash).
    EXPECT_EQ(xxh64(bytes("")), 0xef46db3751d8e999ULL);
    EXPECT_EQ(xxh64(bytes("a")), 0xd24ec4f1a98c6e5bULL);
    EXPECT_EQ(xxh64(bytes("abc")), 0x44bc2cf5ad770999ULL);
    std::string long_input;
    for (int r = 0; r < 3; ++r)
        for (int i = 0; i < 256; ++i) long_input.push_back(static_cast<char>(i));
    EXPECT_EQ(xxh64(bytes(long_input), 7), 0xb1e10f6c5294cd6bULL);   // exercises the 32-byte stripes
}

TEST(Sha1, Fips180Vectors) {
    EXPECT_EQ(sha1_hex(bytes("")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    EXPECT_EQ(sha1_hex(bytes("abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
    EXPECT_EQ(sha1_hex(bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
              "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
}

TEST(Sha256, Fips180Vectors) {
    EXPECT_EQ(sha256_hex(bytes("")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256_hex(bytes("abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(sha256_hex(bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    Sha256 h;   // one million 'a', fed in uneven chunks
    const std::string chunk(997, 'a');
    std::size_t fed = 0;
    while (fed < 1000000) {
        const std::size_t n = std::min<std::size_t>(chunk.size(), 1000000 - fed);
        h.update(std::as_bytes(std::span(chunk.data(), n)));
        fed += n;
    }
    const auto d = h.finish();
    EXPECT_EQ(to_hex(d), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Result, CarriesValueOrError) {
    Result<int> ok = 3;
    ASSERT_TRUE(ok);
    EXPECT_EQ(*ok, 3);
    Result<int> bad = make_error(ErrorCode::Unsupported, "no", "why");
    ASSERT_FALSE(bad);
    EXPECT_EQ(bad.error().code, ErrorCode::Unsupported);
    Result<void> v;
    EXPECT_TRUE(v);
}

TEST(Npy, ReadsPythonWrittenFloat32) {
    auto a = read_npy(std::filesystem::path(RUDRA_GOLDEN_DIR) / "core" / "in_image.npy");
    ASSERT_TRUE(a) << a.error().detail;
    ASSERT_EQ(a->shape, (std::vector<std::int64_t>{1, 3, 24, 32}));
    EXPECT_EQ(a->data.size(), 1u * 3 * 24 * 32);
    for (float v : a->data) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LT(v, 1.0f);
    }
}

TEST(Npy, RefusesWhatItCannotRead) {
    auto missing = read_npy("does/not/exist.npy");
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().code, ErrorCode::NotFound);
}

// Processes: cmake -E is the one program every build machine has.
namespace {
namespace fs = std::filesystem;
fs::path scratch(const std::string& name) {
    const fs::path d = fs::temp_directory_path() / ("rudra-process-test-" + name);
    fs::create_directories(d);
    return d;
}
}  // namespace

TEST(Process, FindExecutable) {
    EXPECT_TRUE(find_executable(RUDRA_CMAKE_COMMAND).has_value());   // a path is taken as it is
    EXPECT_FALSE(find_executable("rudra-no-such-program-xyz").has_value());
    EXPECT_FALSE(find_executable((scratch("find") / "missing").string()).has_value());
}

TEST(Process, RunCapturesOutputAndExit) {
    auto r = run_process({RUDRA_CMAKE_COMMAND, "-E", "echo", "hello world", "a\"b", ""});
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r->exit_code, 0);
    EXPECT_EQ(r->out.substr(0, r->out.find_last_not_of("\r\n") + 1), "hello world a\"b ");   // quoting survives
    auto bad = run_process({RUDRA_CMAKE_COMMAND, "-E", "cat", (scratch("run") / "missing.txt").string()});
    ASSERT_TRUE(bad.ok());
    EXPECT_NE(bad->exit_code, 0);
    EXPECT_FALSE(bad->err.empty());
    EXPECT_FALSE(run_process({"rudra-no-such-program-xyz"}).ok());
    EXPECT_FALSE(run_process({}).ok());
}

TEST(Process, StreamsLargeStdoutAndLogsStderr) {
    const fs::path dir = scratch("stream");
    std::string data(3 * 1024 * 1024 + 17, '\0');   // larger than any pipe buffer
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<char>((i * 131 + 7) & 0xff);
    { std::ofstream(dir / "big.bin", std::ios::binary).write(data.data(), std::streamsize(data.size())); }
    auto p = Process::start({RUDRA_CMAKE_COMMAND, "-E", "cat", (dir / "big.bin").string()}, dir / "err.log");
    ASSERT_TRUE(p.ok());
    std::vector<std::uint8_t> got;
    std::vector<std::uint8_t> buf(1 << 16);
    for (;;) {
        const std::size_t n = (*p)->read(buf);
        got.insert(got.end(), buf.begin(), buf.begin() + std::ptrdiff_t(n));
        if (n < buf.size()) break;
    }
    EXPECT_EQ((*p)->wait(), 0);
    EXPECT_FALSE((*p)->running());
    ASSERT_EQ(got.size(), data.size());
    EXPECT_TRUE(std::equal(got.begin(), got.end(), reinterpret_cast<const std::uint8_t*>(data.data())));

    auto e = Process::start({RUDRA_CMAKE_COMMAND, "-E", "cat", (dir / "nope.bin").string()}, dir / "err2.log");
    ASSERT_TRUE(e.ok());
    std::uint8_t one[1];
    EXPECT_EQ((*e)->read(one), 0u);
    EXPECT_NE((*e)->wait(), 0);
    EXPECT_GT(fs::file_size(dir / "err2.log"), 0u);   // stderr went to the log
}

TEST(Process, KillEndsARunningProgram) {
    auto p = Process::start({RUDRA_CMAKE_COMMAND, "-E", "sleep", "30"});
    ASSERT_TRUE(p.ok());
    EXPECT_TRUE((*p)->running());
    (*p)->kill();
    (*p)->wait();
    EXPECT_FALSE((*p)->running());
}

// A failed write says why (9 Oct 2026: a master failed with only "The EXR file
// could not be written." and nothing to tell a full disk from a blocked folder).
TEST(IoError, BytesReadLikeAPerson) {
    EXPECT_EQ(human_bytes(17), "17 bytes");
    EXPECT_EQ(human_bytes(812'000), "812 KB");
    EXPECT_EQ(human_bytes(52'000'000), "52.0 MB");
    EXPECT_EQ(human_bytes(1'500'000'000), "1.5 GB");
}

TEST(IoError, TheReasonNamesWhatToFix) {
    const auto tmp = std::filesystem::temp_directory_path() / "rudra-io-error";
    std::filesystem::create_directories(tmp);
    const auto target = tmp / "master.exr";
    // Full: from errno, or because the file is bigger than the free space.
    EXPECT_EQ(io_failure_reason(target, ENOSPC, 1000).rfind("the disk is full: ", 0), 0u) << io_failure_reason(target, ENOSPC);
    const std::string huge = io_failure_reason(target, 0, std::uintmax_t(1) << 62);
    EXPECT_EQ(huge.rfind("the disk is full: ", 0), 0u) << huge;
    EXPECT_NE(huge.find(" free on "), std::string::npos) << huge;
    EXPECT_NE(huge.find(" needed"), std::string::npos) << huge;
    EXPECT_EQ(io_failure_reason(target, EACCES).rfind("access denied", 0), 0u);
    EXPECT_EQ(io_failure_reason(tmp / "gone" / "x.exr", ENOENT).rfind("the folder does not exist", 0), 0u);
    EXPECT_FALSE(io_failure_reason(target, EIO).empty());
    EXPECT_FALSE(io_failure_reason(target, 0).empty());   // never an empty reason
    std::filesystem::remove_all(tmp);
}

TEST(IoError, AWriterThatFailsSaysWhy) {
    // A "folder" that is a file: the open fails, and the message carries a reason.
    const auto tmp = std::filesystem::temp_directory_path() / "rudra-io-error-writer";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    std::ofstream(tmp / "not-a-folder") << "x";
    Png8 img;
    img.width = img.height = 2;
    img.channels = 1;
    img.data.assign(4, 0);
    auto r = write_png8(tmp / "not-a-folder" / "mask.png", img);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().message.rfind("The PNG could not be written (", 0), 0u) << r.error().message;
    EXPECT_EQ(r.error().message.back(), '.');
    std::filesystem::remove_all(tmp);
}

// Product item 2: the package's own ffmpeg goes first on the PATH.
namespace {
struct KeepEnv {
    std::string name;
    std::optional<std::string> old;
    explicit KeepEnv(std::string n) : name(std::move(n)) {
        if (const char* v = std::getenv(name.c_str())) old = v;
    }
    ~KeepEnv() {
#ifdef _WIN32
        _putenv_s(name.c_str(), old ? old->c_str() : "");
#else
        if (old) setenv(name.c_str(), old->c_str(), 1);
        else unsetenv(name.c_str());
#endif
    }
};
void set_env(const char* name, const std::string& v) {
#ifdef _WIN32
    _putenv_s(name, v.c_str());
#else
    setenv(name, v.c_str(), 1);
#endif
}
void touch_tools(const std::filesystem::path& d) {
    std::filesystem::create_directories(d);
#ifdef _WIN32
    std::ofstream(d / "ffmpeg.exe") << "";
    std::ofstream(d / "ffprobe.exe") << "";
#else
    std::ofstream(d / "ffmpeg") << "";
    std::ofstream(d / "ffprobe") << "";
#endif
}
}  // namespace

TEST(Tools, TheExecutablesFolderIsWhereTheTestsRun) {
    const auto d = rudra::executable_dir();
    ASSERT_FALSE(d.empty());
    EXPECT_TRUE(std::filesystem::is_directory(d));
}

TEST(Tools, APackagesOwnFfmpegGoesFirst) {
    KeepEnv path("PATH"), chosen("RUDRA_FFMPEG_DIR");
    set_env("RUDRA_FFMPEG_DIR", "");
    const auto root = std::filesystem::temp_directory_path() / "rudra-tools-test";
    std::filesystem::remove_all(root);
    // No tools: nothing changes.
    std::filesystem::create_directories(root / "app");
    const std::string before = std::getenv("PATH") ? std::getenv("PATH") : "";
    EXPECT_TRUE(rudra::bundled_tools_dir(root / "app").empty());
    EXPECT_FALSE(rudra::use_bundled_tools(root / "app"));
    EXPECT_EQ(std::string(std::getenv("PATH") ? std::getenv("PATH") : ""), before);
    // Half a pair is not a pair.
    std::filesystem::create_directories(root / "app" / "ffmpeg");
    std::ofstream(root / "app" / "ffmpeg" / "ffmpeg.exe") << "";
    std::ofstream(root / "app" / "ffmpeg" / "ffmpeg") << "";
    EXPECT_TRUE(rudra::bundled_tools_dir(root / "app").empty());
    // Windows and Linux: <exe>/ffmpeg.
    touch_tools(root / "app" / "ffmpeg");
    EXPECT_EQ(rudra::bundled_tools_dir(root / "app"), root / "app" / "ffmpeg");
    const auto used = rudra::use_bundled_tools(root / "app");
    ASSERT_TRUE(used);
    EXPECT_EQ(*used, root / "app" / "ffmpeg");
    const std::string after = std::getenv("PATH");
    EXPECT_EQ(after.rfind((root / "app" / "ffmpeg").string(), 0), 0u) << after;
    // macOS: Contents/Resources/ffmpeg beside Contents/MacOS.
    touch_tools(root / "RUDRA.app" / "Contents" / "Resources" / "ffmpeg");
    std::filesystem::create_directories(root / "RUDRA.app" / "Contents" / "MacOS");
    EXPECT_EQ(rudra::bundled_tools_dir(root / "RUDRA.app" / "Contents" / "MacOS"),
              root / "RUDRA.app" / "Contents" / "Resources" / "ffmpeg");
    // RUDRA_FFMPEG_DIR: another folder instead, or "path" for the user's own.
    touch_tools(root / "mine");
    set_env("RUDRA_FFMPEG_DIR", (root / "mine").string());
    EXPECT_EQ(rudra::use_bundled_tools(root / "app"), root / "mine");
    set_env("RUDRA_FFMPEG_DIR", "path");
    EXPECT_FALSE(rudra::use_bundled_tools(root / "app"));
    std::filesystem::remove_all(root);
}

#ifdef _WIN32
// cmake/utf8.manifest: the executables run in the UTF-8 code page, so narrow
// paths carry any letter (Windows 10 1903 and later; the CI runners are).
TEST(Tools, WindowsRunsTheTestsInTheUtf8CodePage) {
    EXPECT_EQ(GetACP(), 65001u);
    const std::filesystem::path p(u8"C:/Users/Jos\u00e9/\u65e5\u672c/ffmpeg.exe");
    EXPECT_EQ(std::filesystem::path(p.string()), p);   // through a narrow string and back
}
#endif
