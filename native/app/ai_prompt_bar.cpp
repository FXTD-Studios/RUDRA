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
        // 12 Nov 2026: in the card the chips are stacked, and a button there may be squeezed below
        // its text when the card is fuller than the panel (they showed as slivers). Each keeps the
        // height of its text.
        const bool column = h->direction() == QBoxLayout::TopToBottom || h->direction() == QBoxLayout::BottomToTop;
        auto keep_height = [column](QWidget* w) {
            if (column) w->setMinimumHeight(std::max(24, w->sizeHint().height()));
        };
        if (!prompt_suggestions_ && column && deliver_.ai.rerender) continue;   // v4 has no Suggest: nothing to announce
        if (!prompt_suggestions_) {
            h->addWidget(new QLabel(column ? QStringLiteral("Suggest a prompt\u2026 lists them here")
                                           : QStringLiteral("Suggestions appear here for the frame on screen"),
                                    row));
            h->addStretch(1);
            continue;
        }
        // 12 Nov 2026: in v4 only the presets. The suggestions are made from v3's clipped specks
        // ("specular glints" drew speckles over a whole door).
        const bool v4 = deliver_.ai.rerender;
        PromptSuggestions shown_set = *prompt_suggestions_;
        if (v4) shown_set.suggested.clear();
        const auto& r = shown_set;
        if (column) {
            // 12 Nov 2026: in the Deliver card the suggestions are one button with a menu. A stack of
            // chips there was squeezed into overlapping slivers, and it repeated the row over the viewer.
            auto* pick = new QPushButton(r.suggested.empty() ? QStringLiteral("Prompt presets \u25BE")
                                                             : QStringLiteral("%1 suggested \u25BE").arg(r.suggested.size()),
                                         row);
            pick->setObjectName("aiSuggestPresets");
            pick->setFixedHeight(28);
            auto* m = new QMenu(pick);
            m->setToolTipsVisible(true);
            auto add = [this, m](const PromptSuggestion& p, const QString& text) {
                auto* a = m->addAction(text);
                a->setToolTip(p.text + (p.reason.isEmpty() ? QString() : QStringLiteral("\n") + p.reason));
                connect(a, &QAction::triggered, this, [this, p] { use_prompt_suggestion(p); });
            };
            if (!r.suggested.empty()) m->addSection(QStringLiteral("Suggested for frame %1").arg(prompt_suggest_frame_ + 1));
            for (const auto& p : r.suggested) add(p, chip_text(p));
            if (!r.presets.empty()) m->addSection(QStringLiteral("Presets"));
            for (const auto& p : r.presets) add(p, p.label + QStringLiteral("  \u2014 ") + p.text);
            pick->setMenu(m);
            pick->setEnabled(!m->isEmpty());
            h->addWidget(pick);
            continue;
        }
        if (r.suggested.empty() && !v4) {
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
            keep_height(b);
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
            keep_height(more);
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
    // 12 Nov 2026: which fill a preview runs, here where the previews are started. v3 fills only the
    // fully clipped pixels; v4 (experimental) draws the whole bright region again. The same setting as
    // Deliver > AI fill > Re-render bright regions.
    auto* mode = new QPushButton(QStringLiteral("Fill: v3"), ai_prompt_bar_);
    mode->setObjectName("viewerAiRerender");
    mode->setCheckable(true);
    mode->setToolTip(QStringLiteral("AI fill v3 draws only inside fully clipped pixels (often under 1 % of the frame). "
                                    "v4, experimental, draws the whole bright region again from the source: a window, "
                                    "a door, a sky parked just under white. Click to switch; the next preview uses it"));
    connect(mode, &QPushButton::clicked, this, [this] {
        set_deliver("ai_rerender", deliver_.ai.rerender ? "0" : "1");
        sync_ai_prompt_bar();
    });
    h->addWidget(mode);
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
    // v4 (region re-render) was tested with no prompt, and the suggestions are made from v3's
    // clipped specks ("specular glints" drew speckles over a whole door, 12 Nov 2026): in v4 the
    // prompt, the chips and Suggest rest, and the fill runs as it was tested.
    const bool v4 = deliver_.ai.rerender;
    if (auto* le = findChild<QLineEdit*>("viewerAiPrompt")) {
        // 12 Nov 2026: v4 takes the highlight prompt (the 09V reference was rendered with one)
        le->setEnabled(true);
        le->setPlaceholderText(v4 ? QStringLiteral("EMPTY: type what is in the bright region (a sky, a window, a street)")
                                  : QStringLiteral("highlight prompt (optional): pick a suggestion or type one"));
        le->setToolTip(v4 ? QStringLiteral("AI fill v4 draws the whole bright region again: say what is there (a sky, a window, "
                                           "a street). Empty, it draws texture with no subject")
                          : QString());
    }
    for (auto* row : findChildren<QWidget*>(QStringLiteral("aiSuggestChips"))) row->setEnabled(true);
    for (auto* b : findChildren<QPushButton*>(QStringLiteral("aiSuggestRun"))) b->setVisible(!v4);
    if (auto* mode = findChild<QPushButton*>("viewerAiRerender")) {
        mode->setChecked(deliver_.ai.rerender);
        mode->setText(deliver_.ai.rerender ? QStringLiteral("Fill: v4 re-render") : QStringLiteral("Fill: v3"));
        mode->setEnabled(!genfill_preview_busy_);
    }
    if (auto* again = findChild<QPushButton*>("viewerAiPreviewAgain")) {
        again->setText(genfill_preview_busy_ ? QStringLiteral("Stop preview")
                       : genfill_preview_      ? QStringLiteral("Preview again")
                                               : QStringLiteral("Preview AI fill"));
        again->setEnabled(genfill_ready_ || genfill_preview_busy_);   // either route (7 Nov 2026)
    }
}

}  // namespace rudra::app
