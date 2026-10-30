#pragma once
// The batch queue (28 Oct 2026): shots (movies, numbered image sequences,
// stills) each with its own Deliver choice, name and render folder, rendered
// one after another through the learned ACES 2.0 model kept loaded
// (aces_engine.hpp, the same `master` request as the Deliver tab). The queue is
// a JSON file, saved on every change, so it survives a crash or a restart: a
// job that was running comes back as pending ("interrupted").
//
// The analytic source's movie exports already have their own queue (the
// Export sheet and the Queue window); the batch runs the learned model only.

#include <QJsonObject>
#include <QString>

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "deliver_choice.hpp"

namespace rudra::app {

class AcesEngine;

// A shot: a movie (one file), a numbered sequence (its frames, in order) or a still.
struct BatchShot {
    QString title;                              // "clip.mov", "SH010", "plate.png"
    QString kind = "still";                     // "movie", "sequence" or "still"
    std::vector<std::filesystem::path> sources; // the movie, the frames, or the still
    bool operator==(const BatchShot&) const = default;
};

// What a drop or an Add brings: each movie is a shot; stills are grouped by
// their numbered name (SH010.1001.png, SH010_1002.png ...) into sequences;
// a lone still is a shot of its own. A folder is read the same way, with its
// sub-folders (one level) as shots of their own. Order: as given, then by name.
std::vector<BatchShot> batch_shots_from(const std::vector<std::filesystem::path>& paths);

struct BatchJob {
    QString id;
    BatchShot shot;
    DeliverChoice choice;          // the learned source's choice (fitted to the shot when it runs)
    QString render_dir, name;
    int start = 1001;
    double strength = 1.0, saturation = 1.0, peak_nits = 0.0;   // the colorist's trims (core/aces_v2_trims.hpp)
    double exposure_ev = 0.0, contrast = 1.0;
    QString timecode, reel;        // "" = the movie's own (stills: none) / the movie's or file's name
    QString source_display = "bt1886";   // the display the SDR was made for: bt1886, srgb or gamma22
    QJsonObject edl;               // the EDL event it came from (event, reel, rec_in, rec_out, src_in, warnings): provenance
    int first = -1, last = -1;     // a movie's frames to render (an EDL event with handles); -1 = all
    double maxcll = 0, maxfall = 0;   // what the render measured (nits, scene-referred)
    QString status = "pending";    // pending, running, done, failed, stopped
    int done = 0, total = 0;       // frames
    QString out, error, note;      // the folder written; why it failed; "interrupted"
    QJsonObject to_json() const;
    static std::optional<BatchJob> from_json(const QJsonObject& o);
};

// The choice this job writes: the learned source, fitted to its shot, no AI fill.
DeliverChoice batch_effective(const BatchJob& j);
// What a job writes, in a few words ("ACES 2065-1 EXR, float32", "RGB 16-bit MOV, PQ").
QString batch_output_text(const BatchJob& j);
// The engine's `master` request for the job, with a new folder <render_dir>/<name>
// (else _2, _3 ...); empty with the reason when it cannot run.
QJsonObject batch_request(const BatchJob& j, QString* why = nullptr);
// A render name from a title: letters, digits, space, '.', '_' and '-'.
QString batch_clean_name(const QString& title);

// The trims go with each job as they are when it is added; apply() changes them.
struct BatchTrims {
    double strength = 1.0, saturation = 1.0, peak_nits = 0.0;
    double exposure_ev = 0.0, contrast = 1.0;
};

class BatchQueue {
public:
    explicit BatchQueue(std::filesystem::path file);
    const std::filesystem::path& file() const { return file_; }
    const std::vector<BatchJob>& jobs() const { return jobs_; }
    BatchJob* find(const QString& id);

    // Adds shots with this choice, folder and start frame; returns their ids.
    std::vector<QString> add(const std::vector<BatchShot>& shots, const DeliverChoice& choice, const QString& render_dir,
                             int start, BatchTrims trims = {});
    void remove(const std::vector<QString>& ids);
    void move(const QString& id, int delta);        // -1 up, +1 down
    // The choice (and folder and start, when given) on these jobs (all when none).
    void apply(const std::vector<QString>& ids, const DeliverChoice& choice, const QString& render_dir, int start,
               std::optional<BatchTrims> trims = std::nullopt);
    void rename(const QString& id, const QString& name);
    // A movie's frames first..last (inclusive); -1, -1 for all. False when it is not a movie or not a range.
    bool set_range(const QString& id, int first, int last);
    void retry(const std::vector<QString>& ids);
    // The AI fill on or off for these jobs (all waiting ones when none); EXR stills and sequences only run it.
    void set_ai_fill(const std::vector<QString>& ids, bool on);     // failed or stopped (or done) back to pending
    void clear_finished();                           // drops the done jobs
    std::optional<QString> next_pending() const;
    void save() const;
    // Set when the queue file could not be read (it is kept aside): for the window to show.
    QString load_warning() const { return load_warning_; }
    // Called after every change (the window refreshes).
    std::function<void()> changed;
    void touch();                                    // save, then changed

private:
    void load();
    QString load_warning_;
    std::filesystem::path file_;
    std::vector<BatchJob> jobs_;
};

// Runs the queue's pending jobs, one at a time, through the engine.
class BatchRunner {
public:
    BatchRunner(BatchQueue& queue, std::function<AcesEngine*()> engine);
    ~BatchRunner();
    // Starts with the next pending job; false with the reason when it cannot.
    bool start(QString* why = nullptr);
    void stop_after_current();                       // the job in hand finishes
    void stop_now();                                 // the job in hand stops after its frame
    bool running() const { return running_; }
    bool stopping() const { return stop_after_; }
    QString current() const { return current_; }     // the running job's id
    std::function<void()> changed;                   // progress, a job ended, the run ended
    std::function<void(const QString&)> log;         // one line per event
    // The AI fill of a finished job's master (the window's genfill job): false with
    // `why` when it cannot start; else `done` is called when it ends.
    std::function<bool(const BatchJob&, QString* why, std::function<void(bool ok, const QString& text)> done)> fill;

private:
    void run_next();
    void finished(const QString& id, const QJsonObject& reply);
    BatchQueue& queue_;
    std::function<AcesEngine*()> engine_;
    bool running_ = false, stop_after_ = false, stop_now_ = false;
    QString current_;
    int request_id_ = 0;   // the engine's master request for the running job
    std::shared_ptr<bool> alive_ = std::make_shared<bool>(true);
};

}  // namespace rudra::app
