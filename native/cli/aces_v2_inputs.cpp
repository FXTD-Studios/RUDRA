#include "aces_v2_inputs.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>

#include <nlohmann/json.hpp>

#include "rudra/core/timecode.hpp"
#include "rudra/media/still.hpp"

namespace fs = std::filesystem;

namespace rudra {

namespace {
std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}
Result<AcesFrameIn> from_raw(const RawFrame16& raw, const fs::path& video, const VideoSource& vs, int index) {
    AcesFrameIn f;
    f.codes = rgb_from_frame(raw).buffer();
    f.bits = aces_video_bits(vs.stream.pix_fmt);
    if (f.bits == 8) {
        // An 8-bit video's code values, as the model was made for (the stills and
        // the Python's rgb24 decode): the YCbCr-to-RGB conversion lands between the
        // 8-bit codes, and some of those values (e.g. 252.9, 255, 255) are outside
        // what the ACES 2.0 SDR view can show, so even the exact inverse misses them.
        for (float& v : f.codes.span()) v = std::nearbyint(v * 255.0f) / 255.0f;
    }
    f.source = video.filename().string() + " frame " + std::to_string(index);
    f.source_path = video.string();
    f.exr = aces_video_exr_name(video, index);
    f.frame = index;
    f.fps = aces_video_fps(vs);
    const AcesSourceTimecode tc = aces_source_timecode(vs.streams_json);
    f.timecode = timecode_at(tc.start, index, f.fps);
    f.reel = tc.reel.empty() ? video.stem().string() : tc.reel;
    return f;
}
Result<AcesFrameIn> from_still(const fs::path& p) {
    auto still = decode_sdr_file(p);
    if (!still) return still.error();
    AcesFrameIn f;
    f.codes = still->rgb.buffer();
    f.bits = still->bits;
    f.source = p.filename().string();
    f.source_path = p.string();
    f.exr = p.stem().string() + ".exr";
    return f;
}
}  // namespace

AcesSourceTimecode aces_source_timecode(const std::string& streams_json) {
    AcesSourceTimecode out;
    const auto j = nlohmann::json::parse(streams_json, nullptr, false);
    if (j.is_discarded() || !j.contains("streams") || !j["streams"].is_array()) return out;
    std::vector<const nlohmann::json*> tagged;
    for (const auto& s : j["streams"]) tagged.push_back(&s);
    if (j.contains("format") && j["format"].is_object()) tagged.push_back(&j["format"]);   // MXF keeps it here
    for (const auto* sp : tagged) {
        const auto& s = *sp;
        if (!s.contains("tags") || !s["tags"].is_object()) continue;
        for (const auto& [k, v] : s["tags"].items()) {
            if (!v.is_string()) continue;
            std::string key = lower(k);
            if (key == "timecode" && out.start.empty() && parse_timecode(v.get<std::string>())) out.start = v.get<std::string>();
            if (key == "reel_name" && out.reel.empty()) out.reel = v.get<std::string>();
        }
    }
    return out;
}

double aces_video_fps(const VideoSource& vs) {
    const std::string& f = vs.clock.fps;
    const auto slash = f.find('/');
    try {
        const double v = slash == std::string::npos ? std::stod(f) : std::stod(f.substr(0, slash)) / std::stod(f.substr(slash + 1));
        if (v > 0) return v;
    } catch (...) {
    }
    return 24.0;
}

std::vector<fs::path> aces_expand_input(const fs::path& input) {
    std::vector<fs::path> out;
    std::error_code ec;
    if (fs::is_directory(input, ec)) {
        for (const auto& e : fs::directory_iterator(input, ec))
            if (e.is_regular_file(ec) && aces_is_still(e.path())) out.push_back(e.path());
        std::sort(out.begin(), out.end());
    } else if (fs::is_regular_file(input, ec) && (aces_is_still(input) || aces_is_video(input))) {
        out.push_back(input);
    }
    return out;
}

Result<std::unique_ptr<AcesInputs>> AcesInputs::open(const std::vector<fs::path>& sources, const fs::path& decode_log) {
    std::unique_ptr<AcesInputs> in(new AcesInputs());
    in->log_ = decode_log;
    std::set<std::string> names;
    auto claim = [&](const std::string& name, const fs::path& from) -> Result<void> {
        if (!names.insert(lower(name)).second)
            return make_error(ErrorCode::InvalidArgument, "Inputs map to the same EXR name; rename them before export.",
                              from.string());
        return {};
    };
    for (const auto& s : sources) {
        Item it{s, aces_is_video(s), {}};
        if (it.video) {
            auto vs = aces_open_video(s);
            if (!vs) return vs.error();
            if (vs->clock.frames <= 0) return make_error(ErrorCode::InvalidArgument, "The video has no frames.", s.string());
            it.source = *vs;
            for (int i = 0; i < vs->clock.frames; ++i)
                if (auto c = claim(aces_video_exr_name(s, i), s); !c) return c.error();
            in->total_ += static_cast<std::size_t>(vs->clock.frames);
        } else if (aces_is_still(s)) {
            if (auto c = claim(s.stem().string() + ".exr", s); !c) return c.error();
            in->total_ += 1;
        } else {
            return make_error(ErrorCode::InvalidArgument, "Not a supported still or video.", s.string());
        }
        in->items_.push_back(std::move(it));
    }
    return in;
}

double AcesInputs::fps() const {
    for (const auto& it : items_) {
        if (!it.video) continue;
        const std::string& f = it.source.clock.fps;
        const auto slash = f.find('/');
        try {
            const double v = slash == std::string::npos ? std::stod(f) : std::stod(f.substr(0, slash)) / std::stod(f.substr(slash + 1));
            if (v > 0) return v;
        } catch (...) {
        }
    }
    return 24.0;
}

Result<void> AcesInputs::set_range(int first, int last) {
    if (items_.size() != 1 || !items_.front().video)
        return make_error(ErrorCode::InvalidArgument, "A frame range is for one video.");
    const int frames = items_.front().source.clock.frames;
    if (first < 0 || last < first || last >= frames)
        return make_error(ErrorCode::InvalidArgument, "That frame range is not in the video.",
                          std::to_string(first) + "-" + std::to_string(last) + " of " + std::to_string(frames));
    first_ = first;
    count_ = last - first + 1;
    total_ = std::size_t(count_);
    return {};
}

AcesInputs::~AcesInputs() {
    if (decoder_) decoder_->close();
}

Result<AcesFrameIn> AcesInputs::next() {
    while (item_ < items_.size()) {
        const Item& it = items_[item_];
        if (!it.video) {
            ++item_;
            return from_still(it.path);
        }
        if (!decoder_) {
            auto d = count_ >= 0 ? aces_open_video_decoder_range(it.path, it.source, first_, count_, log_)
                                 : aces_open_video_decoder(it.path, it.source, log_);
            if (!d) return d.error();
            decoder_ = std::move(*d);
            frame_ = count_ >= 0 ? first_ : 0;
        }
        if (frame_ < (count_ >= 0 ? first_ + count_ : it.source.clock.frames)) {
            auto raw = decoder_->next();
            if (!raw) return raw.error();
            return from_raw(*raw, it.path, it.source, frame_++);
        }
        auto done = decoder_->finish();
        decoder_.reset();
        ++item_;
        if (!done) return done.error();
    }
    return make_error(ErrorCode::InvalidArgument, "No more frames.");
}

Result<AcesFrameIn> aces_read_one(const fs::path& source, int frame, const fs::path& decode_log) {
    if (!aces_is_video(source)) return from_still(source);
    auto vs = aces_open_video(source);
    if (!vs) return vs.error();
    auto raw = aces_read_video_frame(source, *vs, frame, decode_log);
    if (!raw) return raw.error();
    return from_raw(*raw, source, *vs, frame);
}

}  // namespace rudra
