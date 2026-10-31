// Help > About and the model downloads' place in the window (29 Oct 2026).
//
// About: the mark, the version and whether it is current, the model in use,
// what it runs on, the viewer, the build; links to www.fxtdstudios.com, the
// user guide, the release notes, the licences, issues and the update check;
// Copy details for a bug report.

#include <QApplication>
#include <QClipboard>
#include <QDesktopServices>
#include <QDir>
#include <QDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QNetworkAccessManager>
#include <QPixmap>
#include <QPushButton>
#include <QSettings>
#include <QStatusBar>
#include <QSysInfo>
#include <QUrl>
#include <QVBoxLayout>

#include "aces_engine.hpp"
#include "aces_master.hpp"
#include "main_window.hpp"

#ifdef RUDRA_APP_VIEWER
#include "rudra/render/viewer_window.hpp"
#endif

namespace rudra::app {

namespace {

QString q(const std::string& s) { return QString::fromStdString(s); }

QLabel* text(const QString& t, const QString& id, const QString& role, QWidget* parent) {
    auto* l = new QLabel(t, parent);
    if (!id.isEmpty()) l->setObjectName(id);
    if (!role.isEmpty()) l->setProperty("role", role);
    return l;
}

}  // namespace

ModelDownloads* MainWindow::model_downloads() {
    if (!downloads_) {
        if (!net_) net_ = new QNetworkAccessManager(this);
        downloads_ = std::make_unique<ModelDownloads>(net_, this);
        downloads_->installed = [this](const RemoteModel& m, const QString& folder) {
            const auto root = std::filesystem::path(downloads_->root().toStdU16String());
            add_model_root(root);
            rescan_models();
            have_aces_package_.reset();
            log(QStringLiteral("model downloaded and checked: %1 %2 → %3").arg(m.title, m.version, folder));
            if (statusBar()) statusBar()->showMessage(m.title + QStringLiteral(" is downloaded and ready"), 6000);
            // The AI fill's LoRA (with brightness.json beside it): what Deliver > AI fill uses from now on.
            if (m.kind == "genfill") {
                const auto loras = QDir(folder).entryList({QStringLiteral("*.safetensors")}, QDir::Files, QDir::Name);
                if (!loras.isEmpty()) {
                    QSettings().setValue("genfill/lora", QDir(folder).filePath(loras.front()));
                    log(QStringLiteral("AI fill: using the downloaded LoRA ") + QDir(folder).filePath(loras.front()));
                }
            }
            // The learned model loads at once when nothing is loaded yet.
            if (m.kind == "aces_v2" && (!aces_engine_ || aces_engine_->state() != AcesEngine::State::Ready))
                use_aces_package(std::filesystem::path(folder.toStdU16String()));
            sync_ui();
        };
    }
    return downloads_.get();
}

bool MainWindow::use_aces_package(const std::filesystem::path& package, QString* why) {
    auto say = [&](const QString& s) {
        log(s);
        if (why) *why = s;
        return false;
    };
    if (find_aces_packages({package}).empty()) return say(QStringLiteral("not a learned ACES 2.0 package: ") + QString::fromStdU16String(package.u16string()));
    QSettings().setValue("aces/package", QString::fromStdU16String(package.u16string()));
    have_aces_package_.reset();
    const auto c = aces_engine_config(model_roots());
    if (!c) return say(QStringLiteral("The ACES engine (rudra-native) is not beside RUDRA: the package is kept for when it is"));
    if (!aces_engine()->start(*c)) return say(AcesEngine::busy_text());
    sync_ui();
    return true;
}

bool MainWindow::have_aces_package() {
    if (!have_aces_package_) have_aces_package_ = aces_engine_config(model_roots()).has_value();
    return *have_aces_package_;
}

void MainWindow::open_about() {
    if (sheet_) sheet_->close();
    auto* d = new QDialog(this);
    d->setAttribute(Qt::WA_DeleteOnClose);
    d->setObjectName("aboutDialog");
    d->setWindowTitle("About RUDRA");
    d->setMinimumWidth(520);
    auto* v = new QVBoxLayout(d);
    v->setContentsMargins(30, 26, 30, 22);
    v->setSpacing(6);
    auto* mark = text({}, "aboutMark", {}, d);
    QPixmap px(QStringLiteral(":/assets/rudra-mark-large.png"));
    if (!px.isNull()) {
        const qreal dpr = d->devicePixelRatioF();
        px = px.scaled(int(88 * dpr), int(88 * dpr), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        px.setDevicePixelRatio(dpr);
        mark->setPixmap(px);
    }
    v->addWidget(mark, 0, Qt::AlignHCenter);
    v->addWidget(text("RUDRA", "aboutName", {}, d), 0, Qt::AlignHCenter);
    v->addWidget(text("SDR to HDR reconstruction · by FXTD Studios", "aboutTag", "note", d), 0, Qt::AlignHCenter);
    {
        auto* pills = new QWidget(d);
        auto* ph = new QHBoxLayout(pills);
        ph->setContentsMargins(0, 8, 0, 4);
        ph->setSpacing(8);
        const QString ver = QCoreApplication::applicationVersion();
        ph->addWidget(text(QStringLiteral("Version ") + (ver.isEmpty() ? QStringLiteral("(development build)") : ver), "aboutVersion", "pill", pills));
        if (!update_state_.isEmpty()) {
            auto* up = text(update_state_, "aboutUpdate", "pill", pills);
            up->setProperty("state", update_state_.startsWith("Up to date") ? "ok" : "warn");
            ph->addWidget(up);
        }
        v->addWidget(pills, 0, Qt::AlignHCenter);
    }
    // What runs here.
    QString model = QStringLiteral("—");
    if (aces_engine_ && aces_engine_->state() == AcesEngine::State::Ready)
        model = QStringLiteral("Learned ACES 2.0 · ") + QString::fromStdU16String(aces_engine_->config().package.filename().u16string());
    else if (manifest_) model = q(manifest_->name);
    else model = QStringLiteral("Analytic (no model loaded)");
    QString viewer = QStringLiteral("CPU");
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer = QStringLiteral("GPU · QRhi %1").arg(q(viewer_->status().backend));
#endif
    const std::vector<std::pair<QString, QString>> rows = {
        {"Model", model},
        {"Running on", device_->text()},
        {"Viewer", viewer},
        {"Build", QStringLiteral("%1 · %2 · Qt %3").arg(QSysInfo::prettyProductName(), QSysInfo::currentCpuArchitecture(), QString::fromLatin1(qVersion()))},
    };
    auto* card = new QWidget(d);
    card->setObjectName("aboutRows");
    card->setProperty("role", "card");
    card->setAttribute(Qt::WA_StyledBackground, true);
    auto* g = new QGridLayout(card);
    g->setContentsMargins(16, 12, 16, 12);
    g->setHorizontalSpacing(18);
    g->setVerticalSpacing(6);
    int r = 0;
    QString details = QStringLiteral("RUDRA %1\n").arg(QCoreApplication::applicationVersion());
    for (const auto& [k, val] : rows) {
        auto* kl = text(k, {}, "key", card);
        auto* vl = text(val, {}, "value", card);
        vl->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        vl->setWordWrap(true);
        vl->setTextInteractionFlags(Qt::TextSelectableByMouse);
        g->addWidget(kl, r, 0);
        g->addWidget(vl, r, 1);
        ++r;
        details += k + ": " + val + "\n";
    }
    g->setColumnStretch(1, 1);
    v->addSpacing(10);
    v->addWidget(card);
    v->addSpacing(8);
    // The links.
    struct Link { const char* id; const char* title; const char* sub; QString url; };
    const std::vector<Link> links = {
        {"aboutWebsite", "www.fxtdstudios.com", "FXTD Studios ↗", QStringLiteral("https://www.fxtdstudios.com")},
        {"aboutGuide", "User guide", "Docs and tutorials ↗", QStringLiteral("https://www.fxtdstudios.com/rudra-docs/")},
        {"aboutNotes", "What’s new", "Release notes ↗", QStringLiteral("https://github.com/FXTD-Studios/RUDRA/releases")},
        {"aboutLicences", "Licences", "Apache 2.0 · models · third party ↗", QStringLiteral("https://github.com/FXTD-Studios/RUDRA/blob/main/NOTICE")},
        {"aboutIssue", "Report an issue", "GitHub ↗", QStringLiteral("https://github.com/FXTD-Studios/RUDRA/issues")},
        {"aboutUpdates", "Check for updates", "", QString()},
    };
    auto* grid = new QWidget(d);
    auto* lg = new QGridLayout(grid);
    lg->setContentsMargins(0, 0, 0, 0);
    lg->setSpacing(8);
    int i = 0;
    for (const auto& l : links) {
        auto* b = new QPushButton(QString::fromUtf8(l.title) + "\n" +
                                      (l.url.isEmpty() ? (update_state_.isEmpty() ? QStringLiteral("Not checked yet") : update_state_)
                                                       : QString::fromUtf8(l.sub)),
                                  grid);
        b->setObjectName(l.id);
        b->setProperty("role", "about-link");
        b->setCursor(Qt::PointingHandCursor);
        if (!l.url.isEmpty()) {
            b->setToolTip(l.url);
            const QString url = l.url;
            connect(b, &QPushButton::clicked, d, [url] { QDesktopServices::openUrl(QUrl(url)); });
        } else {
            connect(b, &QPushButton::clicked, this, [this] { run("check-updates"); });
        }
        lg->addWidget(b, i / 3, i % 3);
        ++i;
    }
    v->addWidget(grid);
    v->addSpacing(10);
    auto* foot = new QWidget(d);
    auto* fh = new QHBoxLayout(foot);
    fh->setContentsMargins(0, 0, 0, 0);
    fh->addWidget(text(QStringLiteral("© 2026 FXTD Studios · Radiance Research\nCode: Apache License 2.0 · models: their own licences"),
                       "aboutCopyright", "note", foot));
    fh->addStretch(1);
    auto* copy = new QPushButton("Copy details", foot);
    copy->setObjectName("aboutCopy");
    copy->setToolTip("The version, the model and what it runs on, for a bug report");
    connect(copy, &QPushButton::clicked, d, [details, this] {
        QApplication::clipboard()->setText(details);
        log("About details copied");
    });
    auto* close = new QPushButton("Close", foot);
    close->setObjectName("aboutClose");
    close->setProperty("role", "primary");
    close->setDefault(true);
    connect(close, &QPushButton::clicked, d, &QDialog::accept);
    fh->addWidget(copy);
    fh->addWidget(close);
    v->addWidget(foot);
    sheet_ = d;
    d->show();
}

}  // namespace rudra::app
