#pragma once
// An edit decision list into the batch queue (28 Oct 2026): a CMX 3600 EDL
// exported from Resolve, Premiere or Avid, matched to the source movies in a
// folder by clip name, then reel, then timecode, so only the frames each edit
// uses (with handles) are rendered, named by event, and keep their source
// timecode for the relink.
//
// Movies only in this version: an image-sequence event is reported as unmatched.

#include <QString>
#include <QStringList>

#include <filesystem>
#include <vector>

#include "batch_queue.hpp"

namespace rudra::app {

struct EdlEvent {
    int number = 0;
    QString reel, track, transition;
    QString src_in, src_out, rec_in, rec_out;   // as written ("01:00:10:00", ";" for drop frame)
    QString clip_name, source_file;             // from "* FROM CLIP NAME:" / "* SOURCE FILE:"
    double speed = 0;                           // an M2 motion effect: the source's frames a second (0: none; < 0 reverse)
    bool video() const { return track.contains('V', Qt::CaseInsensitive) || track.compare("B", Qt::CaseInsensitive) == 0; }
};

// The events of a CMX 3600 EDL, in order, with its TITLE. Lines that are not
// events, M2 motion effects or known comments are ignored; an event without four timecodes too.
std::vector<EdlEvent> parse_edl(const QString& text, QString* title = nullptr);

// A source movie as the matching sees it.
struct EdlMedia {
    std::filesystem::path path;
    QString start_tc, reel;     // "" when the movie has none
    double fps = 24.0;
    long long frames = 0;
};
// ffprobe's view of a movie (streams and format, no frame scan); false when it cannot read it.
bool edl_media_info(const std::filesystem::path& movie, EdlMedia& out, QString* why = nullptr);
// Every movie in a folder (and its sub-folders, one level), probed.
std::vector<EdlMedia> edl_media_in(const std::filesystem::path& folder);

struct EdlShot {
    BatchShot shot;
    int first = 0, last = 0;    // frames of the movie, handles included
    QString name;               // "<event>_<reel>"
    int event = 0;
    QString rec_in, rec_out, src_in;   // the record (timeline) timecode of the edit, for the conform
    QStringList warnings;       // a speed change, handles cut short by the movie's ends (28 Oct 2026 review)
};
struct EdlMatch {
    std::vector<EdlShot> shots;
    QStringList unmatched;      // "004 B002C001: no movie with this clip name, reel or timecode"
    QStringList warnings;       // "007 A001: speed 200% (M2 48 fps) ..." for every shot that has one
};
// Video events only; `handles` frames either side, clamped to the movie.
EdlMatch edl_match(const std::vector<EdlEvent>& events, const std::vector<EdlMedia>& media, int handles);

}  // namespace rudra::app
