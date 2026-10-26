#include "aces_v2_inputs.hpp"

#include <algorithm>
#include <cctype>
#include <set>

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
    f.source = video.filename().string() + " frame " + std::to_string(index);
    f.source_path = video.string();
    f.exr = aces_video_exr_name(video, index);
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
            auto d = aces_open_video_decoder(it.path, it.source, log_);
            if (!d) return d.error();
            decoder_ = std::move(*d);
            frame_ = 0;
        }
        if (frame_ < it.source.clock.frames) {
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
