// The AI fill's prompt suggestions (6 Nov 2026): rudra.genfill.prompts on the frame on
// screen, shown at the top of Reconstruct's AI generation card (21 Nov 2026: the bar over
// the viewer is gone; the viewer keeps one AI fill button, sync_ai_generation). The
// "Suggest a prompt…" button runs it too and also opens the menu (the suggestions, then
// every preset).

#include <QCoreApplication>
#include <QCursor>
#include <QDir>
#include <QFile>
#include <QBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
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
    sync_ai_generation();
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
    if (!genfill_florence_dir(*setup).empty()) env.insert("HF_HUB_OFFLINE", "1");   // 1.2.6: Florence-2 from its folder
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
        if (!r->describe_error.isEmpty())   // 1.2.6: why there is no "From this shot"
            log(QStringLiteral("Suggest a prompt: the shot could not be described (Florence-2): ") + r->describe_error);
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
    proc->start(QString::fromStdString(setup->python.string()), genfill_suggest_arguments(*setup, deliver_effective().ai, {file}, /*describe=*/!quiet));
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
        if (!prompt_suggestions_ && column) continue;   // 1.2.6: the card's Suggest button says it already
        if (!prompt_suggestions_) {
            h->addWidget(new QLabel(QStringLiteral("Suggestions appear here for the frame on screen"), row));
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

void MainWindow::sync_ai_generation() {
    // 21 Nov 2026 (user feedback): the AI fill in three places, each with one job. The viewer has ONE
    // button, AI fill, that shows the generated fill or the master in one click; Reconstruct > AI
    // generation holds the area, the prompt, Preview and every setting; Deliver has one checkbox.
    // The prompt bar that sat over the viewer (6 Nov 2026) is gone: its prompt, suggestions, v3 / v4.3
    // switch, progress and note are all on the card.
    const bool have_fill = genfill_preview_.has_value() || !genfill_last_out_.isEmpty();
    if (auto* b = findChild<QPushButton*>(QStringLiteral("vtAiFill"))) {
        // Offered while AI generation is on, or while there is a fill to show (a preview, a finished render).
        b->setVisible(!frames_.empty() && (ai_generation_ || have_fill || fill_view_on_));
        if (b->isChecked() != fill_view_on_) {
            const QSignalBlocker block(b);
            b->setChecked(fill_view_on_);
        }
        b->setToolTip(fill_view_on_ ? QStringLiteral("Showing the AI fill. Click for the master without it")
                                    : have_fill ? QStringLiteral("Showing the master. Click for the AI fill (the preview's "
                                                                 "frames or the last finished fill); Changes shows what it invented")
                                                : QStringLiteral("No AI fill to show yet: Reconstruct \u203a AI generation \u203a "
                                                                 "Preview AI fill makes one"));
    }
    if (auto* pv = findChild<QPushButton*>(QStringLiteral("deliverAiPreview")))
        pv->setText(genfill_preview_busy_ ? QStringLiteral("Stop preview")
                    : genfill_preview_    ? QStringLiteral("Preview again")
                                          : QStringLiteral("Preview AI fill"));
}

}  // namespace rudra::app
