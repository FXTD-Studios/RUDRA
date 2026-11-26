#pragma once
// Running a program (Phase 4): ffmpeg and ffprobe, as the Python runs them
// with subprocess. The arguments are passed as a vector, never through a
// shell. Two shapes: run() to completion with stdout and stderr captured,
// and Process to stream a large stdout (decoded frames) while stderr goes to
// a file, as the Python's Popen(stdout=PIPE, stderr=log) does.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include "rudra/platform/stop_token.hpp"
#include <string>
#include <vector>

#include "rudra/platform/result.hpp"

namespace rudra {

// shutil.which: the program on PATH (with .exe on Windows), or nullopt.
std::optional<std::filesystem::path> find_executable(const std::string& name);

struct ProcessOutput {
    int exit_code = 0;
    std::string out, err;
};

// Runs argv[0] with the rest as its arguments and waits. An error only when
// it could not be started or was cancelled; a non-zero exit is in exit_code.
// Cancellation terminates/reaps the direct child, including pipe/exit waits.
// Intended for leaf tools such as ffprobe, not shells that spawn descendants.
Result<ProcessOutput> run_process(const std::vector<std::string>& argv, rudra::stop_token stop = {});

// 1.3 (24 Nov 2026, audit): the child `pid` ends with this program, also when this program crashes or
// is ended from Task Manager. Windows: a job object that kills its processes when its last handle closes,
// which the system does when this process ends (the AI fill's Python held about 16 GB of VRAM after a
// crash). The child's own children join the job with it. False when it cannot be done (no rights, a
// Windows without nested jobs); elsewhere a no-op that returns false.
bool bind_child_to_this_process(std::int64_t pid);

// video.executable: shutil.which or "<name> is required on PATH".
Result<std::filesystem::path> require_executable(const std::string& name);
// video.run: stdout, or "<program> failed: <last 4000 bytes of stderr>".
Result<std::string> run_tool(const std::vector<std::string>& argv, rudra::stop_token stop = {});

// A running program whose stdout is read as it comes.
class Process {
public:
    ~Process();
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;

    // stderr goes to `stderr_file` (truncated), or is discarded when empty.
    static Result<std::unique_ptr<Process>> start(const std::vector<std::string>& argv,
                                                  const std::filesystem::path& stderr_file = {});
    // A program fed on stdin (an encoder reading raw frames, as the Python's
    // Popen(stdin=PIPE) is); its stdout is discarded, stderr to the file.
    static Result<std::unique_ptr<Process>> start_writer(const std::vector<std::string>& argv,
                                                         const std::filesystem::path& stderr_file = {});
    // Writes all of `in`; false when the program has closed its input (a broken pipe).
    bool write(std::span<const std::uint8_t> in);
    // Closes stdin so the program sees the end of its input.
    void close_input();
    // A program talked to both ways (25 Nov 2026, the OFX plug-in's learned engine): requests on its
    // stdin, answers read from its stdout, stderr to the file. `path_prepend` (folders separated as PATH
    // separates them) goes in front of the child's PATH only; this process's own PATH is not touched
    // (the plug-in runs inside Resolve or Nuke).
    static Result<std::unique_ptr<Process>> start_duplex(const std::vector<std::string>& argv,
                                                         const std::filesystem::path& stderr_file = {},
                                                         const std::string& path_prepend = {});
    // Reads up to out.size() bytes, looping until the buffer is full or the
    // stream ends; returns how many were read (less only at the end).
    std::size_t read(std::span<std::uint8_t> out);
    // Reads what is there, at least one byte, waiting for it; 0 only at the end of the stream.
    std::size_t read_some(std::span<std::uint8_t> out);
    // Waits for the program to end; its exit code.
    int wait();
    // Ends it now (the Python's kill on an error path).
    void kill();
    bool running() const;

private:
    Process() = default;
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace rudra
