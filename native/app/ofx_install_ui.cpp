// Help > Install the OFX plug-in for Resolve and Nuke (26 Nov 2026): the window's side of ofx_install.hpp.
#include "main_window.hpp"

#include <fstream>
#include <sstream>

#include <QCoreApplication>
#include <QDir>
#include <QMessageBox>
#include <QProcess>

#include "aces_engine.hpp"
#include "engine_client.hpp"   // ofx/: engine_config_path, the file the plug-in reads
#include "genfill_job.hpp"
#include "ofx_install.hpp"
#include "paths.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

namespace rudra::app {

namespace fs = std::filesystem;

namespace {

// The running processes' names, for the hosts that would hold the plug-in open.
std::vector<std::string> process_names() {
    std::vector<std::string> names;
    QProcess p;
#ifdef _WIN32
    p.start("tasklist", {"/FO", "CSV", "/NH"});
#else
    p.start("ps", {"-A", "-o", "comm="});
#endif
    if (!p.waitForFinished(10000)) return names;
    for (const QString& line : QString::fromLocal8Bit(p.readAllStandardOutput()).split('\n', Qt::SkipEmptyParts)) {
        QString n = line.trimmed();
#ifdef _WIN32
        n = n.section(',', 0, 0).remove('"');   // "Nuke15.1.exe","1234",...
#endif
        if (!n.isEmpty()) names.push_back(n.toStdString());
    }
    return names;
}

// Can this user write `dir` (a folder of their own, as the tests use)? Then no administrator is needed.
bool writable(const fs::path& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return false;
    const fs::path probe = dir / ".rudra_write_probe";
    {
        std::ofstream f(probe);
        if (!f) return false;
    }
    fs::remove(probe, ec);
    return true;
}

// Runs the copy as an administrator and waits; "" when it worked, else why not.
QString run_elevated(const std::vector<std::string>& argv, const fs::path& error_file) {
    std::error_code ec;
    fs::remove(error_file, ec);
#ifdef _WIN32
    std::wstring params;
    for (std::size_t i = 1; i < argv.size(); ++i) {
        if (!params.empty()) params += L' ';
        params += L'"' + QString::fromStdString(argv[i]).toStdWString() + L'"';   // the script holds no double quote
    }
    const std::wstring file = QString::fromStdString(argv[0]).toStdWString();
    SHELLEXECUTEINFOW info{};
    info.cbSize = sizeof info;
    info.fMask = SEE_MASK_NOCLOSEPROCESS;
    info.lpVerb = L"runas";
    info.lpFile = file.c_str();
    info.lpParameters = params.c_str();
    info.nShow = SW_HIDE;
    if (!ShellExecuteExW(&info)) {
        const DWORD e = GetLastError();
        return e == ERROR_CANCELLED ? QStringLiteral("Windows' administrator prompt was declined")
                                    : QStringLiteral("the administrator copy did not start (error %1)").arg(e);
    }
    WaitForSingleObject(info.hProcess, 120000);
    DWORD code = 1;
    GetExitCodeProcess(info.hProcess, &code);
    CloseHandle(info.hProcess);
    if (code == 0) return {};
    std::ifstream f(error_file);
    std::stringstream ss;
    ss << f.rdbuf();
    const QString why = QString::fromStdString(ss.str()).trimmed();
    return why.isEmpty() ? QStringLiteral("the copy failed (exit %1)").arg(code) : why.section('\n', 0, 0);
#else
    QStringList args;
    for (std::size_t i = 1; i < argv.size(); ++i) args << QString::fromStdString(argv[i]);
    QProcess p;
    p.start(QString::fromStdString(argv[0]), args);
    if (!p.waitForStarted(10000)) return QStringLiteral("%1 did not start (is it installed?)").arg(QString::fromStdString(argv[0]));
    p.waitForFinished(300000);
    if (p.exitStatus() == QProcess::NormalExit && p.exitCode() == 0) return {};
    const QString err = QString::fromLocal8Bit(p.readAllStandardError()).trimmed();
    return err.isEmpty() ? QStringLiteral("the copy failed or was cancelled (exit %1)").arg(p.exitCode()) : err.section('\n', 0, 0);
#endif
}

fs::path nuke_home() {
    if (const QString t = qEnvironmentVariable("RUDRA_NUKE_HOME"); !t.isEmpty()) return to_path(t);   // tests
    return to_path(QDir::homePath()) / ".nuke";
}

}  // namespace

QString MainWindow::install_ofx(bool interactive) {
    // RUDRA_OFX_APP_DIR: where the tests put a package's ofx/ folder.
    const QString app_env = qEnvironmentVariable("RUDRA_OFX_APP_DIR");
    const fs::path app_dir = to_path(app_env.isEmpty() ? QCoreApplication::applicationDirPath() : app_env);
    const fs::path bundle = shipped_ofx_bundle(app_dir);
    auto finish = [&](const QString& text, bool ok) {
        log("OFX plug-in: " + text);
        if (interactive) {
            if (ok) QMessageBox::information(this, QStringLiteral("OFX plug-in"), text);
            else QMessageBox::warning(this, QStringLiteral("OFX plug-in"), text);
        }
        return text;
    };
    if (bundle.empty())
        return finish(QStringLiteral("This RUDRA carries no OFX plug-in (the packages carry it from 1.3.2; a development "
                                     "build installs it with tmp\\BUILD_OFX.ps1 -Install)."),
                      false);
    if (const auto hosts = ofx_hosts_running(process_names()); !hosts.empty()) {
        QStringList h;
        for (const auto& n : hosts) h << QString::fromStdString(n);
        return finish(QStringLiteral("Close %1 first: a host that has the plug-in loaded keeps it locked.").arg(h.join(", ")), false);
    }

    // What the plug-in will use: this RUDRA's engine and learned model, and AI fill v5 where it is set up.
    OfxEngineSettings s;
    QString learned, aifill;
    if (const auto c = aces_engine_config(model_roots())) {
        s.cli = c->cli, s.package = c->package, s.runtime = c->runtime, s.device = c->device;
        learned = QStringLiteral("ready (%1, %2)").arg(from_path(c->package.filename()), QString::fromStdString(c->device));
    } else {
        learned = QStringLiteral("not yet: install the learned ACES 2.0 model (Help › Check this computer…), then "
                                 "install the plug-in again");
    }
    QString why;
    if (const auto g = genfill_setup(&why); g && g->v5) {
        s.genfill = OfxEngineSettings::GenFill{g->python, g->repo, g->base, g->lora, genfill_florence_dir(*g)};
        aifill = QStringLiteral("ready (AI fill v5)");
    } else {
        aifill = g ? QStringLiteral("not yet: it needs AI fill v5 (File › Model packages)")
                   : QStringLiteral("not set up (Reconstruct › AI generation › Setup…)");
    }

    const fs::path dest = system_ofx_dir();
    const fs::path installed = dest / "RUDRA.ofx.bundle";
    const QString version = QString::fromStdString(ofx_bundle_version(bundle));
    const OfxInstalled state = ofx_installed(bundle, installed);
    const bool copy = state != OfxInstalled::Same;
    if (interactive) {
        const QString what = state == OfxInstalled::None    ? QStringLiteral("Install RUDRA SDR to HDR %1 into %2?")
                             : state == OfxInstalled::Older ? QStringLiteral("Update the installed plug-in to %1, in %2?")
                             : state == OfxInstalled::Same  ? QStringLiteral("RUDRA SDR to HDR %1 is installed in %2. Point it at this RUDRA again?")
                                                            : QStringLiteral("A newer plug-in is installed in %2. Replace it with %1?");
        const QString text = what.arg(version.isEmpty() ? QStringLiteral("(this RUDRA's)") : version, from_path(dest)) +
                             QStringLiteral("\n\nLearned ACES 2.0: %1\nAI fill: %2\n\n").arg(learned, aifill) +
                             (copy ? QStringLiteral("Your computer asks for an administrator to copy it. Close Resolve and Nuke first.")
                                   : QStringLiteral("Nothing is copied; the plug-in's settings are written again."));
        if (QMessageBox::question(this, QStringLiteral("OFX plug-in for Resolve and Nuke"), text,
                                  QMessageBox::Ok | QMessageBox::Cancel) != QMessageBox::Ok)
            return {};
    }

    // 1. The bundle into the system's OFX folder.
    if (copy) {
        std::error_code ec;
        if (writable(dest)) {
            fs::remove_all(installed, ec);
            fs::copy(bundle, installed, fs::copy_options::recursive, ec);
            if (ec) return finish(QStringLiteral("Could not copy the plug-in into %1: %2").arg(from_path(dest), QString::fromStdString(ec.message())), false);
        } else {
            const fs::path err = fs::temp_directory_path(ec) / "rudra_ofx_install_error.txt";
            if (const QString failed = run_elevated(elevated_copy_command(bundle, dest, err), err); !failed.isEmpty())
                return finish(QStringLiteral("The plug-in was not installed: %1.").arg(failed), false);
        }
        if (ofx_installed(bundle, installed) != OfxInstalled::Same)
            return finish(QStringLiteral("The plug-in is not in %1 after the copy.").arg(from_path(dest)), false);
    }
    // 2. Where this RUDRA's engine, model and AI fill are.
    const fs::path cfg = ofx_engine::engine_config_path();
    {
        std::error_code ec;
        fs::create_directories(cfg.parent_path(), ec);
        std::ofstream f(cfg, std::ios::binary | std::ios::trunc);
        f << ofx_engine_json(s);
        if (!f) return finish(QStringLiteral("The plug-in is installed, but its settings could not be written to %1.").arg(from_path(cfg)), false);
    }
    // 3. Nuke's toolbar icon (no administrator: the user's own ~/.nuke).
    QString nuke;
    if (const fs::path menu = shipped_nuke_menu(app_dir); !menu.empty()) {
        std::error_code ec;
        const fs::path home = nuke_home(), to = home / "RUDRA";
        fs::create_directories(to, ec);
        for (const auto& e : fs::directory_iterator(menu, ec))
            if (e.is_regular_file()) fs::copy_file(e.path(), to / e.path().filename(), fs::copy_options::overwrite_existing, ec);
        std::string init;
        {
            std::ifstream in(home / "init.py");
            std::stringstream ss;
            ss << in.rdbuf();
            init = ss.str();
        }
        if (!nuke_init_has_rudra(init)) std::ofstream(home / "init.py", std::ios::app) << "\nimport nuke\n" << nuke_init_line() << "\n";
        nuke = ec ? QStringLiteral(" Nuke's toolbar icon could not be copied (%1).").arg(QString::fromStdString(ec.message()))
                  : QStringLiteral(" Nuke's toolbar gets the RUDRA icon (%1).").arg(from_path(to));
    }
    return finish(QStringLiteral("RUDRA SDR to HDR %1 is in %2. In Resolve: Color page › OpenFX › RUDRA; in a color-managed "
                                 "project keep Enable OpenFX Color Management on (Project Settings › Color Management), else use a "
                                 "DaVinci YRGB project with the timeline in Rec.709 Gamma 2.4. In Nuke: Tab › RUDRA SDR to HDR, the "
                                 "clip read with Raw Data on.%3\nLearned ACES 2.0: %4\nAI fill: %5")
                      .arg(version, from_path(dest), nuke, learned, aifill),
                  true);
}

}  // namespace rudra::app
