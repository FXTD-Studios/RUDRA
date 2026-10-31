// RUDRA: the Qt application.
//
// The window is app/main_window.cpp: the Studio's menubar and actions
// (engine/actions, Phase 3 step 2) around the QRhi viewer (render/, ADR-010)
// and the frame engine. Its look is app/theme.cpp, from ui/theme.css (step 1).
//
//   RUDRA [package [still-or-folder]]   no package: the first-run check once, then the last or picked one
//   RUDRA --theme-check out.json      the fonts, weights and style as resolved here
//   RUDRA --grab out.png [...]        the window as drawn, then quit
//   RUDRA --tab grade|deliver ...     open on that inspector tab
//   RUDRA --splash-grab out.png       the start screen (splash.hpp), rendered at 2x, then quit
//   RUDRA --workflow-check report.json (--analytic | --package P) --frames DIR [--backend libtorch/cuda] [--out DIR] [--movie CLIP] [--timeout-ms N]
//                                     the Studio workflow, scripted, in this window (Phase 3 step 12)
//   RUDRA --headroom-check report.json --movie CLIP [--timeout-ms N]
//                                     the experimental headroom preview, end to end (developer acceptance, Windows)

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QLockFile>
#include <QPixmap>
#include <QSettings>
#include <QStandardPaths>
#include <QSysInfo>
#include <QTimer>

#include <algorithm>
#include <memory>

#include "main_window.hpp"
#include "splash.hpp"
#include "startup.hpp"
#include "rudra/platform/tools.hpp"
#include "headroom_check.hpp"
#include "workflow_check.hpp"
#include "theme.hpp"

int main(int argc, char** argv) {
    // Before QApplication, which reads them to pick its platform plugin: a
    // packaged RUDRA ignores Qt settings other software left in the
    // environment (startup.hpp).
    const std::vector<std::string> ignored_env =
        rudra::app::ignore_foreign_qt_environment(rudra::app::shipped_platforms_dir(rudra::app::executable_dir()));
    QApplication app(argc, argv);
    QApplication::setApplicationName("RUDRA");
    QApplication::setOrganizationName("FXTD Studios");
#ifdef RUDRA_VERSION_LABEL
    QApplication::setApplicationVersion(QStringLiteral(RUDRA_VERSION_LABEL));
#endif
    // The log file, so a failure leaves a record: on Windows
    // %LOCALAPPDATA%\FXTD Studios\RUDRA\logs\rudra.log, on macOS
    // ~/Library/Application Support/FXTD Studios/RUDRA/logs/rudra.log.
    const QString log_file = rudra::app::install_app_log(
        QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + QStringLiteral("/logs"));
    if (!log_file.isEmpty()) {
        rudra::app::app_log_line(QStringLiteral("RUDRA %1 started on %2, Qt %3; arguments: %4")
                                     .arg(QApplication::applicationVersion(), QSysInfo::prettyProductName(),
                                          QString::fromLatin1(qVersion()), QApplication::arguments().join(' ')));
    }
    for (const auto& line : ignored_env) rudra::app::app_log_line(QString::fromStdString(line));
    // The package's own ffmpeg and ffprobe, first on the PATH (platform/tools.hpp).
    const auto tools = rudra::use_bundled_tools(rudra::executable_dir());
    const QString tools_line = tools ? "ffmpeg: " + QDir::toNativeSeparators(QString::fromStdString(tools->string()))
                                     : QStringLiteral("ffmpeg: from the PATH");
    rudra::app::app_log_line(tools_line);
    const rudra::app::ThemeReport theme = rudra::app::apply_theme(app);
    // RUDRA --theme-check out.json: the look as this machine resolves it
    // (fonts, weights, style), for CI and the gates; exit 1 on a problem.
    const QStringList args = QApplication::arguments();
    if (const qsizetype i = args.indexOf("--theme-check"); i >= 0) {
        QFile f(i + 1 < args.size() ? args[i + 1] : QString("theme-check.json"));
        if (f.open(QIODevice::WriteOnly | QIODevice::Truncate)) f.write(rudra::app::theme_report_json(theme));
        return theme.ok() ? 0 : 1;
    }
    // RUDRA --splash-grab out.png: the start screen as drawn (2x), for the review.
    if (const qsizetype i = args.indexOf("--splash-grab"); i >= 0) {
        rudra::app::Splash s;
        s.set_status(QStringLiteral("Loading the learned ACES 2.0 engine"));
        s.set_progress(0.62);
        s.settle();
        QPixmap pm(QSize(rudra::app::Splash::kWidth, rudra::app::Splash::kHeight) * 2);
        pm.setDevicePixelRatio(2.0);
        pm.fill(Qt::transparent);
        s.render(&pm, QPoint(), QRegion(), QWidget::DrawChildren);
        return pm.save(i + 1 < args.size() ? args[i + 1] : QString("rudra-splash.png")) ? 0 : 1;
    }
    // RUDRA --grab out.png [package [source]]: the window as drawn, for the
    // side-by-side review against the Studio and the Pro-direction boards.
    QString grab_to;
    QStringList rest = args.mid(1);
    if (const qsizetype i = rest.indexOf("--grab"); i >= 0) {
        grab_to = i + 1 < rest.size() ? rest[i + 1] : QString("rudra-window.png");
        rest.remove(i, std::min<qsizetype>(2, rest.size() - i));
    }
    // The scripted workflow: a window of its own settings, shown, driven, closed.
    if (const qsizetype i = args.indexOf("--workflow-check"); i >= 0) {
        auto value = [&](const char* flag, const QString& fallback = {}) {
            const qsizetype k = args.indexOf(flag);
            return k >= 0 && k + 1 < args.size() ? args[k + 1] : fallback;
        };
        rudra::app::WorkflowArgs wa;
        wa.report = i + 1 < args.size() ? args[i + 1] : QString("rudra-workflow.json");
        wa.package = value("--package");
        wa.frames = value("--frames");
        wa.backend = value("--backend");
        wa.out = value("--out", QDir::temp().filePath("rudra-workflow-masters"));
        wa.movie = value("--movie");
        wa.analytic = args.contains("--analytic");
        wa.timeout_ms = value("--timeout-ms", "120000").toInt();
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, QDir::temp().filePath("rudra-workflow-settings"));
        QSettings().clear();
        QSettings().setValue("firstRun/done", true);
        rudra::app::MainWindow w;
        w.show();
        return rudra::app::run_workflow_check(w, wa);
    }
    // Developer acceptance of the experimental headroom preview (E1): the only
    // entry point that opts a window into the capability; not a user setting.
    if (const qsizetype i = args.indexOf("--headroom-check"); i >= 0) {
        auto value = [&](const char* flag, const QString& fallback = {}) {
            const qsizetype k = args.indexOf(flag);
            return k >= 0 && k + 1 < args.size() ? args[k + 1] : fallback;
        };
        rudra::app::HeadroomCheckArgs ha;
        ha.report = i + 1 < args.size() ? args[i + 1] : QString("rudra-headroom.json");
        ha.movie = value("--movie");
        ha.timeout_ms = value("--timeout-ms", "120000").toInt();
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, QDir::temp().filePath("rudra-headroom-settings"));
        QSettings().clear();
        QSettings().setValue("firstRun/done", true);
        rudra::app::MainWindow w(true, true);
        w.show();
        return rudra::app::run_headroom_check(w, ha);
    }
    // --grab-delay ms: time for a model and a frame to arrive (default 800).
    int grab_delay = 800;
    if (const qsizetype i = rest.indexOf("--grab-delay"); i >= 0 && i + 1 < rest.size()) {
        grab_delay = rest[i + 1].toInt();
        rest.remove(i, 2);
    }
    // --dialog welcome|export|models: grab that sheet instead of the window (the reviews).
    QString dialog;
    if (const qsizetype i = rest.indexOf("--dialog"); i >= 0 && i + 1 < rest.size()) {
        dialog = rest[i + 1];
        rest.remove(i, 2);
    }
    // The start screen while the window is made (splash.hpp); not for grabs, and
    // off with ui/splash = false.
    std::unique_ptr<rudra::app::Splash> splash;
    if (grab_to.isEmpty() && QSettings().value("ui/splash", true).toBool()) {
        splash = std::make_unique<rudra::app::Splash>();
        splash->show();
        splash->set_status(QStringLiteral("Starting"));
        splash->set_progress(0.12);
    }
    auto step = [&splash](const QString& what, double at) {
        if (splash) {
            splash->set_status(what);
            splash->set_progress(at);
        }
    };
    // Keep super-whites, turned on once (Reconstruct menu): the window opens
    // with headroom support so its projects reopen.
    step(QStringLiteral("Preparing the viewer"), 0.3);
    rudra::app::MainWindow w(true, QSettings().value("experimental/superWhites", false).toBool());
    step(QStringLiteral("Restoring your workspace"), 0.6);
    for (const auto& line : ignored_env) w.log(QString::fromStdString(line));
    if (!log_file.isEmpty()) w.log("log file: " + QDir::toNativeSeparators(log_file));
    {   // A new user starts in Simple (29 Oct 2026): Look, Check, Deliver. Full is one click away.
        QSettings st;
        if (!st.contains("window/workspace") && !st.value("firstRun/done", false).toBool())
            st.setValue("window/workspace", "simple");
    }
    w.restore_settings();   // the window, rails, tab, container and Render fields of the last run
    if (const qsizetype t = rest.indexOf("--tab"); t >= 0 && t + 1 < rest.size()) {   // for the review grabs
        w.show_tab(rest[t + 1]);
        rest.remove(t, 2);
    }
    step(QStringLiteral("Loading the model"), 0.8);
    w.show();
    // A package named on the command line, else the bare start: the first-run
    // check once, then the package used last or the catalog's pick (step 10).
    // A project on the command line (a double-clicked .rudra) opens after the
    // bare start, with its own model.
    const bool project_arg = rest.size() > 0 && rest[0].endsWith(".rudra", Qt::CaseInsensitive);
    if (project_arg) {
        w.open_project(std::filesystem::path(rest[0].toStdU16String()));
        if (!w.loading_model() && w.model_package().empty()) w.boot();   // its model is missing: the usual one
    } else if (rest.size() > 0) {
        w.open_package(rest[0]);
    } else {
        w.boot();
    }
    if (!project_arg && rest.size() > 1) w.open_source(rest[1]);
    // The learned ACES 2.0 model loads in the background (Deliver > Learned ACES 2.0 master…).
    if (grab_to.isEmpty()) w.preload_aces_engine();
    if (splash) {
        step(QStringLiteral("Ready"), 1.0);
        splash->finish(&w);
    }
    // The session is autosaved; the one the last run left is offered by File >
    // Reopen last session. One running RUDRA owns the autosave: a second one
    // (a .rudra double-clicked while the first is open) does without.
    const QString autosave_dir = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + "/autosave";
    QDir().mkpath(autosave_dir);
    QLockFile autosave_lock(autosave_dir + "/autosave.lock");
    if (grab_to.isEmpty()) {
        if (autosave_lock.tryLock(0))
            w.set_autosave_path(std::filesystem::path((autosave_dir + "/last-session.rudra").toStdU16String()));
        else
            w.log("another RUDRA is running and keeps the autosave; this window does not autosave");
        // Once a day at most; Help > Check for updates any time.
        w.maybe_check_for_updates();
        // The five steps once, after the first-run check has been through
        // (the check shows them itself when it closes).
        if (QSettings().value("firstRun/done", false).toBool()) w.show_getting_started_once();
    }
    if (!grab_to.isEmpty()) {
        QTimer::singleShot(grab_delay, &w, [&w, grab_to, dialog] {
            if (!dialog.isEmpty()) {
                if (dialog == "welcome") w.open_first_run();
                else if (dialog == "export") w.open_export_sheet();
                else w.open_model_manager();
                QTimer::singleShot(400, &w, [&w, grab_to, dialog] {
                    const char* id = dialog == "welcome" ? "firstRun" : dialog == "export" ? "exportSheet" : "modelManager";
                    if (auto* d = w.findChild<QWidget*>(id)) d->grab().save(grab_to);
                    QApplication::exit(0);
                });
                return;
            }
            w.grab().save(grab_to);
            QApplication::exit(0);
        });
    }
    return QApplication::exec();
}
