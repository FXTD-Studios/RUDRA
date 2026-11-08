// The AI fill's prompt suggestions (6 Nov 2026): rudra.genfill.prompts on the frame on
// screen, shown as buttons at the top of the AI fill card and in a bar over the viewer
// while the AI fill is previewed or shown, so a prompt is one click away where the fill
// is judged. The "Suggest a prompt…" button runs it too and also opens the menu (the
// suggestions, then every preset).

#include <QCoreApplication>
#include <QCursor>
#include <QDir>
#include <QFile>
#include <QBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QProcess>
#include <QSizePolicy>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QSignalBlocker>
#include <QStatusBar>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

#include "genfill_job.hpp"
#include "main_window.hpp"
#include "rudra/platform/png8.hpp"

namespace rudra::app {

namespace {

// The chips' text: the label, short; the prompt itself in the tooltip.
QString chip_text(const PromptSuggestion& p) {
    QString t = p.label.isEmpty() ? p.text : p.label;
    if (t.size() > 34) t = t.left(32) + QStringLiteral("…");
    return (p.region == "shadow" ? QStringLiteral("Shadow: ") : QString()) + t;
}

// Every child of a chips row goes; the row is rebuilt.
void clear_row(QWidget* row) {
    if (!row || !row->layout()) return;
    while (auto* item = row->layout()->takeAt(0)) {
        if (auto* w = item->widget()) w->deleteLater();
        delete item;
    }
}

}  // namespace

void MainWindow::use_prompt_suggestion(const PromptSuggestion& p) {
    const bool shadow = p.region == "shadow";
    if (auto* le = findChild<QLineEdit*>(shadow ? "deliverAiPromptShadow" : "deliverAiPromptHighlight")) le->setText(p.text);
    set_deliver(shadow ? "ai_prompt_shadow" : "ai_prompt_highlight", p.text);
    log("AI fill prompt: " + p.text + (p.reason.isEmpty() ? QString() : " (" + p.reason + ")"));
    statusBar()->showMessage(QStringLiteral("AI fill %1 prompt: %2 (Preview AI fill to see it)")
                                 .arg(shadow ? QStringLiteral("shadow") : QStringLiteral("highlight"), p.text),
                             6000);
    sync_ai_prompt_bar();
}

void MainWindow::suggest_genfill_prompt() { run_prompt_suggestions(true); }

void MainWindow::run_prompt_suggestions(bool popup, bool quiet) {
    auto say = [this, quiet](const QString& t) {
        log(t);
        if (!quiet) statusBar()->showMessage(t, 8000);
    };
    if (prompt_suggest_running_) return;
    QString why;
    const auto setup = genfill_setup(&why);
    if (!setup) return say(QStringLiteral("Suggest a prompt: the AI fill is not set up (") + why + ")");
    // The frame on screen as the SDR codes the fill's regions are made from: the analytic
    // route's frame, else the learned route's SDR (it has no analytic frame: 6 Nov 2026).
    const SdrImage* sdr = current_frame_ ? &current_frame_->sdr : aces_sdr_img_ ? aces_sdr_img_.get() : nullptr;
    if (!sdr || sdr->width() <= 0) return say(QStringLiteral("Suggest a prompt: open a shot first"));
    Png8 png{sdr->width(), sdr->height(), 3, std::vector<std::uint8_t>(std::size_t(sdr->width()) * std::size_t(sdr->height()) * 3)};
    for (int y = 0; y < sdr->height(); ++y)
        for (int x = 0; x < sdr->width(); ++x)
            for (int c = 0; c < 3; ++c)
                png.data[(std::size_t(y) * std::size_t(sdr->width()) + std::size_t(x)) * 3 + std::size_t(c)] =
                    std::uint8_t(std::lround(std::clamp(sdr->buffer().at(c, y, x), 0.0f, 1.0f) * 255.0f));
    const QString file = QDir(QDir::tempPath()).filePath(QStringLiteral("rudra_suggest_%1.png").arg(QCoreApplication::applicationPid()));
    if (auto w = write_png8(file.toStdString(), png); !w) return say(QString::fromStdString(w.error().message));
    auto* proc = new QProcess(this);
    auto env = QProcessEnvironment::systemEnvironment();
    const QString old = env.value("PYTHONPATH");
    env.insert("PYTHONPATH", QString::fromStdString(setup->repo.string()) + (old.isEmpty() ? QString() : QDir::listSeparator() + old));
    proc->setProcessEnvironment(env);
    proc->setWorkingDirectory(QString::fromStdString(setup->repo.string()));
    prompt_suggest_running_ = true;
    for (auto* b : findChildren<QPushButton*>(QStringLiteral("aiSuggestRun"))) b->setEnabled(false);
    if (auto* b = findChild<QPushButton*>("deliverAiSuggest")) b->setEnabled(false);
    const int frame = current_;
    connect(proc, &QProcess::finished, this, [this, proc, file, popup, frame, quiet](int, QProcess::ExitStatus) {
        QFile::remove(file);
        prompt_suggest_running_ = false;
        for (auto* b : findChildren<QPushButton*>(QStringLiteral("aiSuggestRun"))) b->setEnabled(true);
        if (auto* b = findChild<QPushButton*>("deliverAiSuggest")) b->setEnabled(true);
        proc->deleteLater();
        QString why2;
        const auto r = parse_prompt_suggestions(proc->readAllStandardOutput(), &why2);
        if (!r) {
            log("Suggest a prompt: " + why2 + " " + QString::fromUtf8(proc->readAllStandardError()).right(400));
            if (!quiet) statusBar()->showMessage("Suggest a prompt: " + why2, 8000);
            return;
        }
        prompt_suggestions_ = *r;
        prompt_suggest_frame_ = frame;
        refresh_prompt_chips();
        if (!popup) return;
        auto* menu = new QMenu(this);
        menu->setAttribute(Qt::WA_DeleteOnClose);
        menu->setToolTipsVisible(true);
        auto add = [this](QMenu* m, const PromptSuggestion& p) {
            auto* a = m->addAction(p.label + (p.cached ? QStringLiteral("  · ready") : QString()) + QStringLiteral("  — ") + p.text);
            a->setToolTip(p.reason.isEmpty() ? p.text : p.reason);
            connect(a, &QAction::triggered, this, [this, p] { use_prompt_suggestion(p); });
        };
        if (r->suggested.empty()) {
            menu->addAction(QStringLiteral("Nothing fully clipped in this frame: nothing for the fill to draw"))->setEnabled(false);
        } else {
            menu->addSection(QStringLiteral("Suggested for this frame"));
            for (const auto& p : r->suggested) add(menu, p);
        }
        auto* all = menu->addMenu(QStringLiteral("All presets"));
        all->setToolTipsVisible(true);
        for (const auto& p : r->presets) add(all, p);
        auto* b = findChild<QPushButton*>("deliverAiSuggest");
        menu->popup(b && b->isVisible() ? b->mapToGlobal(QPoint(0, b->height())) : QCursor::pos());
    });
    proc->start(QString::fromStdString(setup->python.string()), genfill_suggest_arguments(*setup, deliver_effective().ai, {file}));
}

void MainWindow::refresh_prompt_chips() {
    for (auto* row : findChildren<QWidget*>(QStringLiteral("aiSuggestChips"))) {
        clear_row(row);
        auto* h = static_cast<QBoxLayout*>(row->layout());   // a row over the viewer, a column in the card
        if (!prompt_suggestions_) {
            h->addWidget(new QLabel(QStringLiteral("Suggestions appear here for the frame on screen"), row));
            h->addStretch(1);
            continue;
        }
        const auto& r = *prompt_suggestions_;
        if (r.suggested.empty()) {
            auto* l = new QLabel(QStringLiteral("Nothing fully clipped on frame %1: nothing for the fill to draw").arg(prompt_suggest_frame_ + 1), row);
            l->setProperty("class", "note");
            h->addWidget(l);
        }
        int shown = 0;
        for (const auto& p : r.suggested) {
            if (shown++ >= 4) break;
            auto* b = new QPushButton(chip_text(p), row);
            b->setObjectName("aiSuggestChip");
            b->setProperty("class", "chip");
            b->setToolTip(p.text + (p.reason.isEmpty() ? QString() : QStringLiteral("\n") + p.reason));
            connect(b, &QPushButton::clicked, this, [this, p] { use_prompt_suggestion(p); });
            h->addWidget(b);
        }
        if (!r.presets.empty()) {   // the rest, one click away
            auto* more = new QPushButton(QStringLiteral("Presets ▾"), row);
            more->setObjectName("aiSuggestPresets");
            auto* m = new QMenu(more);
            m->setToolTipsVisible(true);
            for (const auto& p : r.presets) {
                auto* a = m->addAction(p.label + QStringLiteral("  — ") + p.text);
                a->setToolTip(p.text);
                connect(a, &QAction::triggered, this, [this, p] { use_prompt_suggestion(p); });
            }
            more->setMenu(m);
            h->addWidget(more);
        }
        h->addStretch(1);
    }
}

void MainWindow::build_ai_prompt_bar(QVBoxLayout* v, QWidget* parent) {
    ai_prompt_bar_ = new QWidget(parent);
    ai_prompt_bar_->setObjectName("viewerAiPromptBar");
    auto* h = new QHBoxLayout(ai_prompt_bar_);
    h->setContentsMargins(12, 4, 12, 4);
    h->setSpacing(6);
    auto* head = new QLabel(QStringLiteral("AI fill prompt"), ai_prompt_bar_);
    head->setProperty("class", "note");
    h->addWidget(head);
    auto* le = new QLineEdit(ai_prompt_bar_);
    le->setObjectName("viewerAiPrompt");
    le->setPlaceholderText(QStringLiteral("highlight prompt (optional): pick a suggestion or type one"));
    le->setMinimumWidth(220);
    connect(le, &QLineEdit::editingFinished, this, [this, le] {
        if (le->text() == deliver_.ai.prompt_highlight) return;
        if (auto* d = findChild<QLineEdit*>("deliverAiPromptHighlight")) d->setText(le->text());
        set_deliver("ai_prompt_highlight", le->text());
    });
    connect(le, &QLineEdit::returnPressed, this, [this, le] {   // Return previews with the prompt (7 Nov 2026)
        if (le->text() != deliver_.ai.prompt_highlight) {   // returnPressed comes before editingFinished
            if (auto* d = findChild<QLineEdit*>("deliverAiPromptHighlight")) d->setText(le->text());
            set_deliver("ai_prompt_highlight", le->text());
        }
        if (!genfill_preview_busy_) preview_genfill();
    });
    h->addWidget(le, 2);
    auto* chips = new QWidget(ai_prompt_bar_);
    chips->setObjectName("aiSuggestChips");
    auto* ch = new QHBoxLayout(chips);
    ch->setContentsMargins(0, 0, 0, 0);
    ch->setSpacing(4);
    // The chips never widen the window (8 Nov 2026): a row of suggestions, or the long "Suggestions
    // appear here..." note, set the viewer's minimum width and pushed the right rail (Grade, Deliver)
    // off a 1080p-wide screen. They now get what is left, and are clipped at the edge.
    chips->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    chips->setMinimumWidth(0);
    h->addWidget(chips, 3);
    auto* run = new QPushButton(QStringLiteral("Suggest"), ai_prompt_bar_);
    run->setObjectName("aiSuggestRun");
    run->setToolTip(QStringLiteral("Suggest prompts from this frame's fully clipped areas"));
    connect(run, &QPushButton::clicked, this, [this] { run_prompt_suggestions(false); });
    h->addWidget(run);
    auto* again = new QPushButton(QStringLiteral("Preview again"), ai_prompt_bar_);
    again->setObjectName("viewerAiPreviewAgain");
    again->setToolTip(QStringLiteral("Fill the frame at the playhead again with this prompt"));
    connect(again, &QPushButton::clicked, this, [this] { preview_genfill(); });
    h->addWidget(again);
    ai_prompt_bar_->hide();
    v->addWidget(ai_prompt_bar_);
    // The AI fill card's chips are built after the viewer: both get their first state then.
    QTimer::singleShot(0, this, [this] { refresh_prompt_chips(); });
}

void MainWindow::sync_ai_prompt_bar() {
    if (!ai_prompt_bar_) return;
    // Over the viewer while the AI fill is being previewed, or shown (the preview's or a render's),
    // and (7 Nov 2026) whenever AI fill is on in Deliver and set up, so the first prompt and the first
    // preview start here too.
    const bool on = !frames_.empty() && (genfill_preview_busy_ || genfill_preview_.has_value() || fill_view_on_ ||
                                         (deliver_.ai_fill && genfill_ready_));
    ai_prompt_bar_->setVisible(on);
    if (!on) return;
    if (auto* le = findChild<QLineEdit*>("viewerAiPrompt"); le && !le->hasFocus() && le->text() != deliver_.ai.prompt_highlight) {
        const QSignalBlocker b(le);
        le->setText(deliver_.ai.prompt_highlight);
    }
    if (auto* again = findChild<QPushButton*>("viewerAiPreviewAgain")) {
        again->setText(genfill_preview_busy_ ? QStringLiteral("Stop preview")
                       : genfill_preview_      ? QStringLiteral("Preview again")
                                               : QStringLiteral("Preview AI fill"));
        again->setEnabled(genfill_ready_ || genfill_preview_busy_);   // either route (7 Nov 2026)
    }
}

}  // namespace rudra::app
