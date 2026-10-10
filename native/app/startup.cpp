#include "startup.hpp"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMutex>
#include <QMutexLocker>
#include <QTextStream>
#include <QtGlobal>

#include <cctype>
#include <cstdio>
#include <memory>

#include "rudra/platform/tools.hpp"

namespace rudra::app {

namespace fs = std::filesystem;

fs::path executable_dir() { return rudra::executable_dir(); }

fs::path shipped_platforms_dir(const fs::path& exe_dir) {
    if (exe_dir.empty()) return {};
    std::error_code ec;
    for (const fs::path& p : {exe_dir / "platforms", exe_dir.parent_path() / "PlugIns" / "platforms"})
        if (fs::is_directory(p, ec)) return p;
    return {};
}

namespace {

// "windows:darkmode=2" -> "windows".
std::string plugin_name(const QByteArray& value) {
    const QByteArray v = value.trimmed();
    const qsizetype colon = v.indexOf(':');
    return (colon >= 0 ? v.left(colon) : v).toLower().toStdString();
}

bool ships_plugin(const fs::path& dir, const std::string& name) {
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        std::string stem = e.path().stem().string();
        for (char& c : stem) c = char(std::tolower(static_cast<unsigned char>(c)));
        if (stem == "q" + name || stem == "libq" + name) return true;
    }
    return false;
}

}  // namespace

std::vector<std::string> ignore_foreign_qt_environment(const fs::path& platforms_dir) {
    std::vector<std::string> ignored;
    if (platforms_dir.empty()) return ignored;
    for (const char* var : {"QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"}) {
        if (!qEnvironmentVariableIsSet(var)) continue;
        ignored.push_back(std::string("ignored ") + var + "=" + qgetenv(var).toStdString() +
                          ": this build uses the plugins it ships with");
        qunsetenv(var);
    }
    if (qEnvironmentVariableIsSet("QT_QPA_PLATFORM")) {
        const QByteArray value = qgetenv("QT_QPA_PLATFORM");
        const std::string name = plugin_name(value);
        if (name.empty() || !ships_plugin(platforms_dir, name)) {
            ignored.push_back("ignored QT_QPA_PLATFORM=" + value.toStdString() + ": this build has no '" + name +
                              "' platform plugin");
            qunsetenv("QT_QPA_PLATFORM");
        }
    }
    return ignored;
}

namespace {

struct LogState {
    QMutex mutex;
    std::unique_ptr<QFile> file;
    QtMessageHandler previous = nullptr;
    bool installed = false;
};

LogState& state() {
    static LogState s;
    return s;
}

void write_line(const QString& level, const QString& text) {
    LogState& s = state();
    QMutexLocker lock(&s.mutex);
    if (!s.file) return;
    const QByteArray line = (QDateTime::currentDateTime().toString(Qt::ISODateWithMs) + " " + level + " " + text + "\n").toUtf8();
    s.file->write(line);
    s.file->flush();   // a crash must not take the last lines with it
}

void message_handler(QtMsgType type, const QMessageLogContext& context, const QString& message) {
    const char* level = type == QtDebugMsg      ? "debug"
                        : type == QtInfoMsg     ? "info "
                        : type == QtWarningMsg  ? "warn "
                        : type == QtCriticalMsg ? "error"
                                                : "fatal";
    write_line(QString::fromLatin1(level), message);
    if (QtMessageHandler prev = state().previous) prev(type, context, message);
    else std::fprintf(stderr, "%s\n", qPrintable(qFormatLogMessage(type, context, message)));
}

}  // namespace

QString install_app_log(const QString& dir, qint64 max_bytes, int keep) {
    if (dir.isEmpty() || !QDir().mkpath(dir)) return {};
    const QString path = QDir(dir).filePath("rudra.log");
    if (QFileInfo(path).size() > max_bytes) {
        QFile::remove(QDir(dir).filePath(QStringLiteral("rudra.%1.log").arg(keep)));
        for (int i = keep - 1; i >= 1; --i)
            QFile::rename(QDir(dir).filePath(QStringLiteral("rudra.%1.log").arg(i)),
                          QDir(dir).filePath(QStringLiteral("rudra.%1.log").arg(i + 1)));
        QFile::rename(path, QDir(dir).filePath("rudra.1.log"));
    }
    auto file = std::make_unique<QFile>(path);
    if (!file->open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) return {};
    {
        LogState& s = state();
        QMutexLocker lock(&s.mutex);
        s.file = std::move(file);
    }
    if (!state().installed) {
        state().previous = qInstallMessageHandler(message_handler);
        state().installed = true;
    }
    return path;
}

void app_log_line(const QString& line) { write_line(QStringLiteral("app  "), line); }

QString app_log_path() {
    LogState& s = state();
    QMutexLocker lock(&s.mutex);
    return s.file ? s.file->fileName() : QString();
}

}  // namespace rudra::app
