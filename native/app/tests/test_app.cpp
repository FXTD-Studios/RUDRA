// The app's window against the Studio page, offscreen (Phase 3).
//
// Step 2: the menubar the window builds is engine/actions' menus exactly
// (titles, entries, separators, submenus), and for the page's menus the
// Studio's labels, key hints and check states (tests/golden/actions). Every
// key the page maps reaches its action and no two actions share a key.
// Running the actions changes what the page's handlers change. Step 1: the
// look resolves (fonts, weights, the sheet) from the static library.

#include <gtest/gtest.h>

#include <condition_variable>
#include <mutex>
#include <thread>

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QAbstractButton>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSlider>
#include <QStackedWidget>
#include <QImage>
#include <QColorSpace>
#include <QMouseEvent>
#include <QComboBox>
#include <QFile>
#include <QPlainTextEdit>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QStandardPaths>
#include <QProcess>
#include <QSettings>
#include <QClipboard>
#include <QNetworkAccessManager>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QPointer>
#include <QUrl>
#include <QTest>
#include <QTreeWidget>

#include "paths.hpp"
#include "json_lines.hpp"
#include "rudra/core/hdr10.hpp"
#include "scope_widgets.hpp"
#include "genfill_review.hpp"
#include "export_sheet.hpp"
#include "source_panel.hpp"
#include "queue_window.hpp"
#include "rudra/deliver/queue.hpp"
#include "rudra/platform/process.hpp"
#include "rudra/media/video_probe.hpp"

TEST(AppSourceInspection, ReportsTagsAndClearsWarningsWhenSourceChanges) {
    rudra::app::SourcePanel panel;
    auto* label = panel.findChild<QLabel*>("sourceInspection");
    ASSERT_NE(label, nullptr);
    rudra::VideoStreamInfo stream;
    stream.width = 1920;
    stream.height = 1080;
    stream.color_range = "tv";
    stream.color_transfer = "smpte2084";
    panel.inspect_source(true, true, stream);
    EXPECT_TRUE(label->text().contains("1920 x 1080"));
    EXPECT_TRUE(label->text().contains("not reported"));
    EXPECT_TRUE(label->text().contains("Limited range"));
    EXPECT_TRUE(label->text().contains("HDR source detected"));
    stream.color_range = "pc";
    stream.color_transfer = "bt709";
    panel.inspect_source(true, true, stream);
    EXPECT_FALSE(label->text().contains("Limited range"));
    EXPECT_FALSE(label->text().contains("HDR source detected"));
    panel.inspect_source(true, true, {});
    EXPECT_TRUE(label->text().contains("unavailable"));
    panel.inspect_source(false, false, {});
    EXPECT_TRUE(label->text().contains("Open a source"));
}

TEST(AppSourceInspection, StillProfileAndFallbackAreExplicitAndPlainText) {
    rudra::app::SourcePanel panel;
    auto* label = panel.findChild<QLabel*>("sourceInspection"); ASSERT_NE(label, nullptr);
    rudra::StillColour colour;
    colour.tagged = true; colour.primaries = rudra::Primaries::P3D65;
    colour.tag = "icc"; colour.profile = "<b>Display P3</b>"; colour.orientation = 6;
    panel.inspect_source(true, false, {}, colour);
    EXPECT_EQ(label->textFormat(), Qt::PlainText);
    EXPECT_TRUE(label->text().contains("Display P3"));
    EXPECT_TRUE(label->text().contains("EXIF orientation 6 applied"));
    colour.recognised = false;
    panel.inspect_source(true, false, {}, colour);
    EXPECT_TRUE(label->text().contains("not one RUDRA knows"));
    EXPECT_TRUE(label->text().contains("read as Rec.709"));
    panel.inspect_source(true, false, {}, rudra::StillColour{});
    EXPECT_TRUE(label->text().contains("no colour profile"));
    EXPECT_FALSE(label->text().contains("Display P3"));
    panel.inspect_source(true, true, {});
    EXPECT_FALSE(label->text().contains("no colour profile"));
}

#include <cstdio>
#include <functional>
#include <QTextDocumentFragment>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>
#include <cctype>
#include <cmath>
#include <limits>

#include <fstream>
#include <sstream>
#include <optional>
#include <map>
#include <set>

#include <nlohmann/json.hpp>

#include "main_window.hpp"
#include "model_dialogs.hpp"
#include "workflow_check.hpp"
#include "startup.hpp"
#include "updates.hpp"
#include "playback.hpp"
#include <QElapsedTimer>
#include "rudra/engine/project.hpp"
#include <QStatusBar>
#include "rudra/deliver/master.hpp"
#include "region_editor.hpp"
#include "scope_widgets.hpp"
#include "widgets.hpp"
#include "rudra/core/copy_texts.hpp"
#include "rudra/engine/actions.hpp"
#include "theme.hpp"

using nlohmann::json;
using namespace rudra;

namespace {

app::ThemeReport* g_theme = nullptr;

const json& golden() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/actions/actions.json");
        return json::parse(f);
    }();
    return g;
}

QString qs(std::string_view s) { return QString::fromUtf8(s.data(), qsizetype(s.size())); }

std::string id_of(const QAction* a) {
    const QString n = a->objectName();
    return n.startsWith("act:") ? n.mid(4).toStdString() : std::string();
}

// A menu as the spec writes it: ids, "-", ">Title" and "<".
void flatten(const QMenu* m, std::vector<std::string>& out) {
    for (const QAction* a : m->actions()) {
        if (a->isSeparator()) out.push_back("-");
        else if (a->menu()) {
            out.push_back(">" + a->menu()->title().toStdString());
            flatten(a->menu(), out);
            out.push_back("<");
        } else if (a->objectName() != "showTips") {   // Help > Show tips: the app's own (simple_ui.cpp), not a page action
            out.push_back(id_of(a));
        }
    }
}

// The page's hint as Qt prints the shortcut.
QString hint_text(const std::string& hint) {
    if (hint == "Space") return "Space";
    return QString::fromStdString(hint);
}

}  // namespace

TEST(AppLook, ResolvesFromTheLibrary) {
    ASSERT_NE(g_theme, nullptr);
    EXPECT_TRUE(g_theme->ok()) << g_theme->problems.join("; ").toStdString();
    EXPECT_EQ(g_theme->fonts_loaded.size(), 12);   // Geist and Geist Mono, and Plex for the scope labels
    EXPECT_FALSE(qApp->styleSheet().isEmpty());
}

TEST(AppActions, MenubarIsTheSpec) {
    app::MainWindow w(false);
    const auto menus = w.menuBar()->actions();
    ASSERT_EQ(std::size_t(menus.size()), menu_specs().size());
    for (std::size_t i = 0; i < menu_specs().size(); ++i) {
        const auto& spec = menu_specs()[i];
        const QMenu* m = menus[qsizetype(i)]->menu();
        ASSERT_NE(m, nullptr);
        EXPECT_EQ(m->title(), qs(spec.title));
        std::vector<std::string> got, want;
        flatten(m, got);
        for (auto e : spec.entries) want.emplace_back(e);
        EXPECT_EQ(got, want) << m->title().toStdString();
    }
}

TEST(AppActions, PageMenusLabelsHintsChecks) {
    app::MainWindow w(false);
    for (const auto& gm : golden()["menus"]) {
        for (const auto& item : gm["items"]) {
            if (item.contains("separator")) continue;
            const std::string id = item["act"];
            QAction* a = w.action(id);
            ASSERT_NE(a, nullptr) << id;
            EXPECT_EQ(a->text().toStdString(), std::string(item["label"])) << id;
            if (!item["hint"].is_null()) {
                EXPECT_EQ(a->shortcut().toString(QKeySequence::PortableText), hint_text(item["hint"])) << id;
            }
            EXPECT_EQ(a->isCheckable(), !item["check"].is_null()) << id;
        }
    }
    // One choice of several: the modes, the containers, the zooms.
    for (const char* g : {"mode-all", "container-aces", "zoom-fit"}) {
        ASSERT_NE(w.action(g)->actionGroup(), nullptr) << g;
        EXPECT_TRUE(w.action(g)->actionGroup()->isExclusive()) << g;
    }
    EXPECT_EQ(w.action("preserve")->actionGroup(), nullptr);
}

TEST(AppActions, EveryPageKeyReachesItsAction) {
    app::MainWindow w(false);
    std::map<QString, std::string> owner;   // one owner per key sequence
    for (const auto& spec : action_specs()) {
        for (const QKeySequence& k : w.action(spec.id)->shortcuts()) {
            const QString t = k.toString(QKeySequence::PortableText);
            const auto [it, fresh] = owner.emplace(t, std::string(spec.id));
            EXPECT_TRUE(fresh) << t.toStdString() << " is on " << it->second << " and " << spec.id;
        }
    }
    for (const auto& [key, act] : golden()["keys"].items()) {
        QString t;
        if (key == " ") t = "Space";
        else if (key.size() == 1) t = QString::fromStdString(key).toUpper();
        else t = QString::fromStdString(key);
        ASSERT_TRUE(owner.count(t)) << "key '" << key << "' reaches nothing";
        EXPECT_EQ(owner[t], std::string(act)) << "key '" << key << "'";
    }
}

TEST(AppActions, KeyPressRunsTheAction) {
    app::MainWindow w(false);
    w.show();
    w.activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(&w));
    QTest::keyClick(&w, Qt::Key_2);
    EXPECT_EQ(w.composite().mode, RecoveryMode::Highlights);
    EXPECT_TRUE(w.action("mode-highlights")->isChecked());
    EXPECT_FALSE(w.action("mode-all")->isChecked());
    QTest::keyClick(&w, Qt::Key_BracketRight);
    EXPECT_FLOAT_EQ(w.composite().strength, 1.1f);
    QTest::keyClick(&w, Qt::Key_P);
    EXPECT_FALSE(w.composite().preserve_outside);
    EXPECT_FALSE(w.action("preserve")->isChecked());
}

TEST(AppActions, HandlersDoWhatThePagesDo) {
    app::MainWindow w(false);
    // nudgeStrength: 0.1 a press, rounded to hundredths, clamped to 0..2.
    for (int i = 0; i < 25; ++i) w.run("strength-up");
    EXPECT_FLOAT_EQ(w.composite().strength, 2.0f);
    for (int i = 0; i < 7; ++i) w.run("strength-down");
    EXPECT_FLOAT_EQ(w.composite().strength, 1.3f);
    for (int i = 0; i < 30; ++i) w.run("strength-down");
    EXPECT_FLOAT_EQ(w.composite().strength, 0.0f);
    w.run("mode-off");
    w.run("preserve");
    EXPECT_EQ(w.composite().mode, RecoveryMode::Off);
    EXPECT_FALSE(w.composite().preserve_outside);
    // reset-recon: mode all, strength 1, preserve on.
    w.run("reset-recon");
    EXPECT_EQ(w.composite().mode, RecoveryMode::All);
    EXPECT_FLOAT_EQ(w.composite().strength, 1.0f);
    EXPECT_TRUE(w.composite().preserve_outside);
    EXPECT_TRUE(w.action("mode-all")->isChecked());
    EXPECT_TRUE(w.action("preserve")->isChecked());
    // setContainer.
    EXPECT_EQ(w.container(), "aces");
    w.run("container-linear");
    EXPECT_EQ(w.container(), "linear");
    EXPECT_TRUE(w.action("container-linear")->isChecked());
    EXPECT_FALSE(w.action("container-aces")->isChecked());
}

TEST(AppActions, FTriggersFitButStillTypesInTextFields) {
    app::MainWindow w(false);
    w.show();
    w.activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(&w));
    int fits = 0;
    QObject::connect(w.action("zoom-fit"), &QAction::triggered, &w, [&] { ++fits; });
    EXPECT_EQ(w.action("zoom-fit")->shortcut(), QKeySequence(Qt::Key_F));
    QTest::keyClick(&w, Qt::Key_F);
    EXPECT_EQ(fits, 1);
    auto* name = w.findChild<QLineEdit*>("renderName");
    ASSERT_NE(name, nullptr);
    name->setText("");
    name->setFocus();
    QTest::keyClick(name, Qt::Key_F);
    EXPECT_EQ(name->text(), "f");
    EXPECT_EQ(fits, 1);
}

TEST(AppActions, EnabledAsThePageEnablesThem) {
    app::MainWindow w(false);
    w.refresh_enabled();
    // No frames: refreshMenu turns these off.
    for (const char* id : {"close", "first", "prev", "next", "last", "play", "master", "undo", "redo",
                           "copy-metrics", "copy-scopes", "copy-delivery", "remeasure"})
        EXPECT_FALSE(w.action(id)->isEnabled()) << id;
    for (const char* id : {"open", "mode-all", "preserve", "strength-up", "reset-recon", "container-aces",
                           "shortcuts", "about", "open-package", "quit"})
        EXPECT_TRUE(w.action(id)->isEnabled()) << id;
    // What waits on a later step says so.
    EXPECT_TRUE(w.action("rail-left")->isEnabled());
    // Step 11 made the last of the page's actions live: none waits on a
    // later step; the only ones off are the viewer's, in a window without one.
    for (const auto& spec : action_specs()) {
        const QString why = w.pending_reason(spec.id);
        if (spec.id == "reopen-session") {   // off until a run has left a session to reopen
            EXPECT_EQ(why, "No earlier session was saved.");
            continue;
        }
        EXPECT_TRUE(why.isEmpty() || why == "This build has no viewer.") << spec.id << ": " << why.toStdString();
    }
    for (const char* id : {"master", "copy-metrics", "copy-scopes", "copy-delivery", "models", "recent-clear"})
        EXPECT_TRUE(w.pending_reason(id).isEmpty()) << id;
    // A disabled action does not run from its key either.
    w.run("undo");
    EXPECT_EQ(w.session().undo_depth(), 0u);
}

TEST(AppSession, UndoAndRedoThroughTheMenus) {
    app::MainWindow w(false);
    w.run("strength-up");
    w.run("mode-shadows");
    w.refresh_enabled();
    EXPECT_TRUE(w.action("undo")->isEnabled());
    EXPECT_FALSE(w.action("redo")->isEnabled());
    w.action("undo")->trigger();
    EXPECT_EQ(w.composite().mode, RecoveryMode::All);
    EXPECT_TRUE(w.action("mode-all")->isChecked());
    EXPECT_TRUE(w.action("redo")->isEnabled());   // refreshed after the change
    w.action("undo")->trigger();
    EXPECT_FLOAT_EQ(w.composite().strength, 1.0f);
    EXPECT_FALSE(w.action("undo")->isEnabled());
    w.action("redo")->trigger();
    w.action("redo")->trigger();
    EXPECT_EQ(w.session().grade.mode, "shadows");
    EXPECT_FLOAT_EQ(w.composite().strength, 1.1f);
    // Region EV reset is live now, and undoable.
    w.session().region_press(0, 0.0);
    w.session().region_move(50.0, false);
    w.session().region_release();
    EXPECT_DOUBLE_EQ(w.session().grade.regions[0].ev, 0.5);
    w.run("reset-regions");
    EXPECT_DOUBLE_EQ(w.session().grade.regions[0].ev, 0.0);
    w.run("undo");
    EXPECT_DOUBLE_EQ(w.session().grade.regions[0].ev, 0.5);
    // The wipe key is the session's.
    w.run("wipe");
    ASSERT_TRUE(w.session().wipe.has_value());
    EXPECT_TRUE(w.action("wipe")->isChecked());
}

// ---- step 4: the layout of the page -------------------------------------

namespace {

const json& layout() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/layout/layout.json");
        return json::parse(f);
    }();
    return g;
}

// textContent with whitespace collapsed, as the golden records it.
QString norm(QString t) {
    if (t.contains('<')) t = QTextDocumentFragment::fromHtml(t).toPlainText();
    for (QChar& c : t)
        if (c.isSpace() || c == QChar(0x200A)) c = ' ';
    return t.simplified();
}

// A widget of one of the app's own classes by name (they have no Q_OBJECT).
template <class T>
T* find(const QWidget& w, const QString& name) {
    return dynamic_cast<T*>(w.findChild<QWidget*>(name));
}

// The page's ids the native window has no widget for, and why.
const std::map<std::string, std::string> kNotWidgets = {
    {"menubar", "the Qt menubar (its box is checked with the frame)"},
    {"gl", "the viewer draws with QRhi in its own window (ADR-010)"},
    {"plate", "the viewer's own window"},
    {"plateLabel", "drawn over the picture by the viewer: step 7"},
    {"plateHint", "drawn over the picture by the viewer: step 7"},
    {"peakBadge", "drawn over the picture by the viewer: step 7"},
    {"fcLegend", "drawn over the picture by the viewer: step 7"},
    {"probeBox", "the floating probe: step 7"},
    {"scrubHead", "painted by the scrub bar"},
};
// Texts that are the machine's, not the page's (the harness has no model).
const std::set<std::string> kLiveText = {"ckpt", "log", "menubar", "device"};

struct Window {
    app::MainWindow w{false};
    explicit Window(const std::string& state) {
        w.resize(1600, 1000);
        w.show();
        (void)QTest::qWaitForWindowExposed(&w);
        if (state == "simple") w.findChild<QPushButton*>("wsSimple")->click();
        if (state == "rails_hidden")
            for (const char* a : {"rail-left", "rail-right", "scopes"}) w.run(a);
        if (state == "tab_grade") w.findChild<QPushButton*>("tabGrade")->click();
        if (state == "tab_deliver") w.findChild<QPushButton*>("tabDeliver")->click();
        QApplication::processEvents();
    }
    QWidget* get(const std::string& id) const { return w.findChild<QWidget*>(QString::fromStdString(id)); }
    bool shown(const QWidget* x) const { return x && x->isVisibleTo(&w) && x->width() > 0 && x->height() > 0; }
    std::array<int, 4> box(const QWidget* x) const {
        const QPoint p = x->mapTo(&w, QPoint(0, 0));
        return {p.x(), p.y(), x->width(), x->height()};
    }
};

}  // namespace

TEST(AppLayout, EveryPageElementHasItsWidget) {
    Window win("full");
    for (const auto& [id, e] : layout()["states"]["full"]["ids"].items()) {
        if (kNotWidgets.count(id)) continue;
        EXPECT_NE(win.get(id), nullptr) << "no widget named " << id;
    }
}

// Where the Pro boards put a page element somewhere else, and why.
const std::map<std::string, std::string> kProPlaced = {
    {"btnMaster", "on the Deliver tab (the board's Delivery inspector), and the toolbar's Export sheet"},
    {"btnReprocess", "on the Deliver tab, beside Master EXR"},
    {"renderDir", "the Render card of the Deliver tab"},
    {"renderMode", "the Render card of the Deliver tab"},
    {"renderName", "the Render card of the Deliver tab"},
    {"renderStart", "the Render card of the Deliver tab"},
    {"measA", "the This frame card's rows arrive with the first measurement"},
    {"ckpt", "the sidebar's model card (the toolbar's machine pill stays when the sidebar hides)"},
    {"itabs", "the toolbar's segmented tabs, which stay when the inspector hides"},
    {"tabRec", "the toolbar's segmented tabs"},
    {"tabGrade", "the toolbar's segmented tabs"},
    {"tabDeliver", "the toolbar's segmented tabs"},
    {"measB", "the This frame card's rows arrive with the first measurement"},
};
// The page's panel labels as the boards name their sections and cards.
const std::map<std::string, std::string> kProLabels = {{"Media", "Shots"}, {"Frame", "This frame"}};

TEST(AppLayout, ShownAndHiddenAsThePageIsInEveryState) {
    for (const auto& [state, snap] : layout()["states"].items()) {
        // The page's Simple (scopes and notes hidden) became Look, Check, Deliver
        // (29 Oct 2026): AppSimple tests it.
        if (state == "simple") continue;
        Window win(state);
        for (const auto& [id, e] : snap["ids"].items()) {
            if (kNotWidgets.count(id) || kProPlaced.count(id)) continue;
            QWidget* x = win.get(id);
            if (!x) continue;
            EXPECT_EQ(win.shown(x), e["shown"].get<bool>()) << state << ": " << id;
        }
        // The page's panel labels: each is a section, a card head or a label
        // of the window, shown where the page shows it (the scopes' with the
        // scopes, Region EV with the Grade tab).
        for (const auto& l : snap["panel_labels"]) {
            const std::string page = l["text"];
            const auto it = kProLabels.find(page);
            const QString want = QString::fromStdString(it == kProLabels.end() ? page : it->second);
            bool any = false, shown = false;
            for (QLabel* q : win.w.findChildren<QLabel*>())
                if (norm(q->text()) == want) {
                    any = true;
                    shown = shown || win.shown(q);
                }
            EXPECT_TRUE(any) << state << ": no label " << want.toStdString();
            if (page == "Region EV" || page == "Waveform" || page == "Histogram" || page == "Vectorscope") {
                EXPECT_EQ(shown, l["shown"].get<bool>()) << state << ": " << page;
            }
        }
        std::vector<std::pair<std::string, bool>> nwant, ngot;
        for (const auto& l : snap["notes"]) nwant.emplace_back(l["text"], l["shown"]);
        // In the page's order: the Reconstruct, Grade and Deliver panels.
        for (QLabel* l : win.w.findChild<QWidget*>("ipanels")->findChildren<QLabel*>())
            if (l->property("role").toString() == "note-p") ngot.emplace_back(norm(l->text()).toStdString(), win.shown(l));
        ASSERT_EQ(ngot.size(), nwant.size()) << state;
        for (std::size_t i = 0; i < ngot.size(); ++i) {
            EXPECT_EQ(ngot[i].second, nwant[i].second) << state << ": note " << i;
            if (i < 2) { EXPECT_EQ(ngot[i].first, nwant[i].first) << state << ": note " << i; }   // the Deliver note is the board's
        }
    }
}

// The words the Pro boards changed on purpose (the Studio page's are in the
// golden); everything else is still the page's word for word.
const std::map<std::string, std::string> kProWords = {
    {"seqOpen text", "Open"},                                     // the sidebar's field: "Open"
    {"seqPath title", "A folder of frames, read where it sits: nothing is copied."},   // no server behind it
    {"seqPath placeholder", "Open a folder of frames\u2026"},     // the board's search field
    {"iMedia title", "Show or hide the sidebar"},                 // the toolbar's sidebar toggle
    {"wsSimple title", "Look, Check, Deliver: the controls most shots need"},   // Simple, 29 Oct 2026
    // The pipeline strip says what the input is read as (Source display, 28 Oct 2026 review) and what the
    // analytic master is (32-bit float EXR, not half).
    {"pipeIn text", "Rec.709 \u00b7 BT.1886"},
    {"pipeMaster text", "ACES 2065-1 EXR, float32"},
};
// The Signal card's rows the 28 Oct 2026 review reworded: RUDRA's white convention said as such.
const std::map<std::string, std::string> kProFields = {{"White", "1.0 = diffuse white (RUDRA's 203 nits)"}};

std::string pro_or(const std::string& key, const std::string& page) {
    const auto it = kProWords.find(key);
    return it == kProWords.end() ? page : it->second;
}

TEST(AppLayout, TheWordsAreThePages) {
    Window win("full");
    const auto& snap = layout()["states"]["full"];
    for (const auto& [id, e] : snap["ids"].items()) {
        QWidget* x = win.get(id);
        if (!x || kNotWidgets.count(id) || kLiveText.count(id)) continue;
        if (!e["buttons"].is_null()) {
            std::vector<std::string> got;
            for (auto* b : x->findChildren<QPushButton*>(QString(), Qt::FindDirectChildrenOnly))
                if (!b->isHidden()) got.push_back(norm(b->text()).toStdString());   // Simple's Check tab is hidden in Full
            EXPECT_EQ(got, e["buttons"].get<std::vector<std::string>>()) << id;
            continue;
        }
        if (e["text"].is_null()) continue;
        QString text;
        if (auto* l = qobject_cast<QLabel*>(x)) text = l->text();
        else if (auto* b = qobject_cast<QAbstractButton*>(x)) text = b->text();
        else continue;   // a container: its children are checked on their own
        EXPECT_EQ(norm(text).toStdString(), pro_or(id + " text", e["text"].get<std::string>())) << id;
    }
    for (const auto& [id, title] : snap["titles"].items()) {
        QWidget* x = win.get(id);
        if (!x) continue;
        EXPECT_EQ(x->toolTip().toStdString(), pro_or(id + " title", title.get<std::string>())) << id;
    }
    for (const auto& [id, ph] : snap["placeholders"].items())
        EXPECT_EQ(win.w.findChild<QLineEdit*>(QString::fromStdString(id))->placeholderText().toStdString(),
                  pro_or(id + " placeholder", ph.get<std::string>())) << id;
    // The page's toolbar labels (Compare, Layer, Zoom): the board's compare
    // bar has none, its segments name themselves.
    for (QLabel* l : win.w.findChildren<QLabel*>())
        EXPECT_NE(l->property("role").toString(), "vlabel") << l->text().toStdString();
    EXPECT_FALSE(snap["toolbar_labels"].empty());
    for (const auto& c : snap["checks"]) {
        auto* row = find<app::CheckRow>(win.w, QString::fromStdString(c["id"].get<std::string>()));
        ASSERT_NE(row, nullptr) << c["id"];
        EXPECT_EQ(row->label()->text().toStdString(), c["label"].get<std::string>());
        EXPECT_EQ(row->hint()->text().toStdString(), c["hint"].get<std::string>());
        EXPECT_EQ(row->on(), c["on"].get<bool>()) << c["id"];
    }
    for (const auto& ic : snap["icon_titles"]) {
        auto* b = find<app::IconButton>(win.w, QString::fromStdString(ic["id"].get<std::string>()));
        ASSERT_NE(b, nullptr);
        EXPECT_EQ(b->toolTip().toStdString(), pro_or(ic["id"].get<std::string>() + " title", ic["title"].get<std::string>()));
        EXPECT_EQ(b->on(), ic["on"].get<bool>()) << ic["id"];
    }
    // The sliders: label, value, unit and the page's range in its steps.
    const auto& sl = snap["sliders"];
    auto* strength = win.w.findChild<QSlider*>("strength");
    auto* peak = win.w.findChild<QSlider*>("peak");
    EXPECT_EQ(sl[0]["value"], win.w.findChild<QLabel*>("strengthVal")->text().toStdString());
    EXPECT_EQ(sl[1]["value"], win.w.findChild<QLabel*>("peakVal")->text().toStdString());
    EXPECT_DOUBLE_EQ(strength->minimum() / 20.0, std::stod(sl[0]["min"].get<std::string>()));
    EXPECT_DOUBLE_EQ(strength->maximum() / 20.0, std::stod(sl[0]["max"].get<std::string>()));
    EXPECT_DOUBLE_EQ(1 / 20.0, std::stod(sl[0]["step"].get<std::string>()));
    EXPECT_DOUBLE_EQ(peak->minimum() / 2.0, std::stod(sl[1]["min"].get<std::string>()));
    EXPECT_DOUBLE_EQ(peak->maximum() / 2.0, std::stod(sl[1]["max"].get<std::string>()));
    // Fields.
    std::vector<std::vector<std::string>> fwant, fgot;
    for (const auto& f : snap["fields"]) {
        auto row = f.get<std::vector<std::string>>();
        if (const auto it = kProFields.find(row[0]); it != kProFields.end()) row[1] = it->second;
        fwant.push_back(row);
    }
    for (QWidget* f : win.w.findChildren<QWidget*>()) {
        if (f->property("role").toString() != "field") continue;
        const auto labels = f->findChildren<QLabel*>();
        fgot.push_back({labels[0]->text().toStdString(), labels[1]->text().toStdString()});
    }
    EXPECT_EQ(fgot, fwant);
}

// The frame is the Pro board's ("Pro direction: main window"): the unified
// toolbar across the top, the 240 px sidebar, the 320 px inspector, and in
// the middle the compare bar, the 24 px compare line (beta 3: what RUDRA
// changed and how much of it the view shows), the viewer, the 30 px colour
// pipeline and the 128 px timeline; a rail that is hidden gives its width to
// the viewer.
TEST(AppLayout, TheFrameIsTheProBoards) {
    for (const std::string state : {"full", "rails_hidden"}) {   // Simple's frame: AppSimple
        Window win(state);
        auto box = [&](const char* id) {
            QWidget* x = win.get(id);
            EXPECT_NE(x, nullptr) << id;
            return x ? win.box(x) : std::array<int, 4>{};
        };
        const auto mb = win.box(win.w.menuBar());
        const auto tb = box("toolbar"), side = box("railLeft"), centre = box("centre"), insp = box("railRight");
        const auto tools = box("viewerTools"), viewer = box("viewer"), pipe = box("pipe"), time = box("transport");
        const int W = win.w.centralWidget()->width();
        EXPECT_EQ(tb[1], mb[1] + mb[3]) << state;
        EXPECT_EQ(tb[3], 52) << state;
        EXPECT_EQ(tb[2], W) << state;
        const bool rails = state != "rails_hidden";
        // The scopes' own column beside the viewer (29 Oct 2026), 300 wide when shown.
        const int dock = win.shown(win.get("scopesDock")) ? 300 : 0;
        if (dock) { EXPECT_EQ(box("scopesDock")[0], centre[0] + centre[2]) << state; }
        EXPECT_EQ(win.shown(win.get("railLeft")), rails) << state;
        EXPECT_EQ(win.shown(win.get("railRight")), rails) << state;
        if (rails) {
            EXPECT_EQ(side[2], 240) << state;
            EXPECT_EQ(side[0], 0) << state;
            EXPECT_EQ(insp[2], 320) << state;
            EXPECT_EQ(insp[0] + insp[2], W) << state;
            EXPECT_EQ(centre[0], 240) << state;
            EXPECT_EQ(centre[2], W - 240 - 320 - dock) << state;
        } else {
            EXPECT_EQ(centre[0], 0) << state;
            EXPECT_EQ(centre[2], W - dock) << state;
        }
        EXPECT_EQ(side[1], tb[1] + 52) << state;
        EXPECT_EQ(tools[3], 52) << state;
        const auto line = box("compareBar");
        EXPECT_EQ(line[1], tools[1] + tools[3]) << state;
        EXPECT_EQ(line[3], 24) << state;
        EXPECT_EQ(viewer[1], line[1] + line[3]) << state;
        EXPECT_EQ(pipe[1], viewer[1] + viewer[3]) << state;
        EXPECT_EQ(pipe[3], 30) << state;
        EXPECT_EQ(time[1], pipe[1] + pipe[3]) << state;
        EXPECT_EQ(time[3], 128) << state;
        EXPECT_EQ(time[1] + time[3], centre[1] + centre[3]) << state;
        // The compare bar sits in the middle of the surround when the badges
        // leave it room, and between them always.
        const auto hud = box("hud"), hdr = box("hdrBadge"), cll = box("cllBadgeWrap");
        if (!rails) { EXPECT_LE(std::abs((hud[0] + hud[2] / 2) - (tools[0] + tools[2] / 2)), 2) << state; }
        EXPECT_GE(hud[0], hdr[0] + hdr[2]) << state;
        EXPECT_LE(hud[0] + hud[2], cll[0]) << state;
    }
}

TEST(AppLayout, ShellControlsDoWhatShellJsDoes) {
    Window win("full");
    auto& w = win.w;
    // Workspace: Simple hides the scopes and the notes, and the Scopes icon goes off.
    w.findChild<QPushButton*>("wsSimple")->click();
    EXPECT_EQ(w.workspace(), "simple");
    EXPECT_FALSE(win.shown(win.get("scopes")));
    EXPECT_FALSE(find<app::IconButton>(w, "iScopes")->on());
    w.findChild<QPushButton*>("wsFull")->click();
    EXPECT_TRUE(win.shown(win.get("scopes")));
    // The icon rail drives the Window menu's actions, so their checks follow.
    find<app::IconButton>(w, "iMedia")->click();
    EXPECT_FALSE(win.shown(win.get("railLeft")));
    EXPECT_FALSE(w.action("rail-left")->isChecked());
    find<app::IconButton>(w, "iMedia")->click();
    EXPECT_TRUE(win.shown(win.get("railLeft")));
    // A menu item for a control on a hidden tab brings the tab forward.
    w.show_tab("grade");
    w.menuBar()->actions()[4 - 1]->menu()->triggered(w.action("mode-shadows"));   // Reconstruct
    EXPECT_EQ(w.tab(), "rec");
    // The controls move the session, and the session moves them.
    find<app::Seg>(w, "mode")->button("highlights")->click();
    EXPECT_EQ(w.session().grade.mode, "highlights");
    w.run("strength-up");
    EXPECT_EQ(w.findChild<QLabel*>("strengthVal")->text(), "1.10");
    EXPECT_EQ(w.findChild<QSlider*>("strength")->value(), 22);
    w.findChild<QSlider*>("peak")->setValue(5);   // 2.5 EV
    EXPECT_EQ(w.findChild<QLabel*>("peakVal")->text(), "1,148");
    find<app::CheckRow>(w, "preserve")->clicked();
    EXPECT_EQ(w.findChild<QLabel*>("preserveHint")->text(), "raw prediction");
    w.run("container-linear");
    EXPECT_EQ(w.findChild<QLabel*>("containerField")->text().toStdString(), "OpenEXR \u2014 linear Rec.2020");
    EXPECT_EQ(w.findChild<QLabel*>("primariesField")->text(), "Rec.2020");
    w.findChild<QPushButton*>("wipeBtn")->click();
    EXPECT_TRUE(w.session().wipe.has_value());
    find<app::Seg>(w, "viewMode")->button("baseline")->click();
    EXPECT_FALSE(w.session().wipe.has_value());
    EXPECT_EQ(w.session().show, "baseline");
    find<app::Seg>(w, "viewLayer")->button("1")->click();
    EXPECT_EQ(w.session().view_layer, 1);
    EXPECT_TRUE(w.action("view-false-colour")->isChecked());
    // Undo takes the controls back too.
    w.run("undo");
    w.run("undo");
    w.run("undo");
    EXPECT_EQ(w.findChild<QLabel*>("strengthVal")->text(), "1.00");
    EXPECT_EQ(w.findChild<QLabel*>("preserveHint")->text(), "do-no-harm");    // The master exposure (13 Oct 2026): the slider, its label, the master request, the menu, double-click for 0.
    auto* exposure = w.findChild<QSlider*>("exposure");
    ASSERT_NE(exposure, nullptr);
    exposure->setValue(20);
    EXPECT_DOUBLE_EQ(w.session().grade.exposure_ev, 1.0);
    EXPECT_EQ(w.findChild<QLabel*>("exposureVal")->text(), "+1.00");
    EXPECT_DOUBLE_EQ(w.master_request().exposure_ev, 1.0);
    w.run("exposure-down");
    EXPECT_NEAR(w.session().grade.exposure_ev, 2.0 / 3.0, 1e-12);
    EXPECT_EQ(exposure->value(), 13);
    QTest::mouseDClick(exposure, Qt::LeftButton);
    EXPECT_EQ(w.session().grade.exposure_ev, 0.0);
    EXPECT_EQ(exposure->value(), 0);
}

// ---- step 6: the panels driven as the page's are, against its golden -------

namespace {

const json& session_golden() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/session/scripts.json");
        return json::parse(f);
    }();
    return g;
}

void send_mouse(QWidget* target, QEvent::Type type, double x, Qt::KeyboardModifiers mods = Qt::NoModifier) {
    const QPointF local(x, 5.0);
    const QPointF global = target->mapToGlobal(local);
    const Qt::MouseButtons held = type == QEvent::MouseButtonRelease ? Qt::NoButton : Qt::LeftButton;
    QMouseEvent e(type, local, local, global, Qt::LeftButton, held, mods);
    QApplication::sendEvent(target, &e);
}

// One gesture of the golden, made on the window's own widgets.
void gesture(app::MainWindow& w, const json& op) {
    const std::string k = op[0];
    if (k == "start") return;
    if (k == "mode") {
        find<app::Seg>(w, "mode")->button(QString::fromStdString(op[1].get<std::string>()))->click();
    } else if (k == "preserve") {
        QTest::mouseClick(w.findChild<QWidget*>("preserve"), Qt::LeftButton);
    } else if (k == "strength" || k == "peak") {
        auto* s = w.findChild<QSlider*>(QString::fromStdString(k));
        const double v = std::stod(op[1].get<std::string>());
        const int n = int(std::lround(v * (k == "strength" ? 20.0 : 2.0)));
        if (k == "strength") s->setSliderDown(true);   // the pointerdown
        s->setValue(n);
        if (k == "strength") s->setSliderDown(false);
    } else if (k == "region") {
        auto* ev = find<app::RegionEditor>(w, "regions")->value(op[1].get<int>());
        double x = 5.0;
        const auto mods = op[3].get<bool>() ? Qt::ShiftModifier : Qt::NoModifier;
        send_mouse(ev, QEvent::MouseButtonPress, x);
        for (const auto& dx : op[2]) {
            x += dx.get<double>();
            send_mouse(ev, QEvent::MouseMove, x, mods);
        }
        send_mouse(ev, QEvent::MouseButtonRelease, x);
    } else if (k == "region0") {
        auto* ev = find<app::RegionEditor>(w, "regions")->value(op[1].get<int>());
        const double x = 5.0;
        // A double click: press, release, then the second press as a double click.
        send_mouse(ev, QEvent::MouseButtonPress, x);
        send_mouse(ev, QEvent::MouseButtonRelease, x);
        send_mouse(ev, QEvent::MouseButtonDblClick, x);
        send_mouse(ev, QEvent::MouseButtonRelease, x);
    } else if (k == "act") {
        w.action(op[1].get<std::string>())->trigger();
    } else if (k == "key") {
        const std::string key = op[1];
        Qt::KeyboardModifiers mods = op[2].get<bool>() ? Qt::ShiftModifier : Qt::NoModifier;
        int code = 0;
        if (key == "ArrowLeft") code = Qt::Key_Left;
        else if (key == "ArrowRight") code = Qt::Key_Right;
        else if (key == "Escape") code = Qt::Key_Escape;
        else if (key == " ") code = Qt::Key_Space;
        else if (key == "[") code = Qt::Key_BracketLeft;
        else if (key == "]") code = Qt::Key_BracketRight;
        else if (key.size() == 1 && std::isdigit(static_cast<unsigned char>(key[0]))) code = Qt::Key_0 + (key[0] - '0');
        else if (key.size() == 1 && std::isalpha(static_cast<unsigned char>(key[0]))) {
            code = Qt::Key_A + (std::toupper(static_cast<unsigned char>(key[0])) - 'A');
            if (std::isupper(static_cast<unsigned char>(key[0]))) mods |= Qt::ShiftModifier;   // "P" is Shift+P
        }
        ASSERT_NE(code, 0) << key;
        QTest::keyClick(&w, Qt::Key(code), mods);
    } else {
        FAIL() << "unknown gesture " << k;
    }
    QApplication::processEvents();
}

}  // namespace

TEST(AppPanels, EveryGestureOnTheWidgetsGivesThePagesState) {
    for (const auto& [name, steps] : session_golden()["scripts"].items()) {
        app::MainWindow w(false);
        w.resize(1600, 1000);
        w.show();
        w.activateWindow();
        ASSERT_TRUE(QTest::qWaitForWindowActive(&w));
        w.show_tab("grade");   // the Region EV rows must be laid out to be pressed
        // The page has no source picker: it is the ACES inverse, "unknown"
        // (the app starts on BT.1886 since 13 Oct 2026).
        w.session().grade.source = "unknown";
        QApplication::processEvents();
        int i = 0;
        for (const auto& st : steps) {
            gesture(w, st["op"]);
            const std::string where = name + " step " + std::to_string(i++) + " " + st["op"].dump();
            const Session& s = w.session();
            ASSERT_EQ(s.params_json(), st["params"].get<std::string>()) << where;
            EXPECT_EQ(s.undo_depth(), st["undo"].get<std::size_t>()) << where;
            EXPECT_EQ(s.redo_depth(), st["redo"].get<std::size_t>()) << where;
            if (st["wipe"].is_null()) {
                EXPECT_FALSE(s.wipe.has_value()) << where;
            } else {
                ASSERT_TRUE(s.wipe.has_value()) << where;
                EXPECT_EQ(*s.wipe, st["wipe"].get<double>()) << where;
            }
            // What the panels show.
            const auto& p = st["panel"];
            auto* regions = find<app::RegionEditor>(w, "regions");
            ASSERT_EQ(std::size_t(regions->rows()), p["regions"].size()) << where;
            for (int r = 0; r < regions->rows(); ++r) {
                const auto& g = p["regions"][std::size_t(r)];
                EXPECT_EQ(regions->range(r)->text().toStdString(), g["q"].get<std::string>()) << where;
                EXPECT_EQ(regions->value(r)->text().toStdString(), g["ev"].get<std::string>()) << where;
                EXPECT_EQ(regions->value(r)->property("live").toBool(), g["live"].get<bool>()) << where << " row " << r;
                EXPECT_EQ(regions->row(r)->property("sel").toBool(), g["sel"].get<bool>()) << where << " row " << r;
            }
            EXPECT_EQ(w.findChild<QLabel*>("regionCount")->text().toStdString(), p["regionCount"].get<std::string>()) << where;
            EXPECT_EQ(w.findChild<QLabel*>("strengthVal")->text().toStdString(), p["strengthVal"].get<std::string>()) << where;
            EXPECT_EQ(w.findChild<QLabel*>("peakVal")->text().toStdString(), p["peakVal"].get<std::string>()) << where;
            EXPECT_EQ(w.findChild<QLabel*>("preserveHint")->text().toStdString(), p["preserveHint"].get<std::string>()) << where;
            EXPECT_EQ(find<app::CheckRow>(w, "preserve")->on(), p["preserveOn"].get<bool>()) << where;
            EXPECT_EQ(find<app::Seg>(w, "mode")->on().toStdString(), p["mode"].get<std::string>()) << where;
        }
    }
}

// ---- steps 7 and 8: the probe, the Frame panel and the bars, wired --------

namespace {

// A small frame with clipped highlights, the masks on them, and a residual
// that lifts them: enough for every read-out to have something to say.
void synthetic_frame(app::MainWindow& w) {
    const int W = 96, H = 64;
    PlanarBuffer sdr(3, H, W), residual(3, H, W), hi(1, H, W), sh(1, H, W);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const std::size_t i = std::size_t(y) * W + std::size_t(x);
            const float v = std::min(1.0f, float(x) / float(W - 16));
            for (int c = 0; c < 3; ++c) {
                sdr.plane(c)[i] = std::clamp(v * (c == 0 ? 1.0f : c == 1 ? 0.9f : 0.75f), 0.0f, 1.0f);
                residual.plane(c)[i] = v >= 0.99f ? 0.35f : 0.0f;
            }
            hi.plane(0)[i] = v >= 0.99f ? 1.0f : 0.0f;
            sh.plane(0)[i] = x < 4 ? 1.0f : 0.0f;
        }
    FrameScalars scalars;
    scalars.shadow_weight = 0.87f;
    FrameHeader header;
    header.source_resolution = header.resolution = "96x64";
    header.elapsed_s = 0.25;
    w.present_frame(SdrImage(std::move(sdr)), Fields{std::move(residual), std::move(hi), std::move(sh)}, scalars,
                    ModelConstants{16.0f, 4.0f, -1.0f}, header);
}

std::vector<std::vector<std::string>> rows_of(QWidget* ms) {
    std::vector<std::vector<std::string>> out;
    for (QWidget* r : ms->findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly)) {
        if (r->property("role").toString() != "ro") continue;
        std::vector<std::string> row;
        for (QLabel* l : r->findChildren<QLabel*>()) row.push_back(l->text().toStdString());
        row.push_back(r->findChildren<QLabel*>()[1]->property("state").toString().toStdString());
        out.push_back(row);
    }
    return out;
}

}  // namespace

TEST(AppMeasure, TheFramePanelAndBarsShowTheMeasurement) {
    app::MainWindow w(false);
    w.resize(1600, 1000);
    w.show();
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&w));
    EXPECT_FALSE(w.action("remeasure")->isEnabled());
    synthetic_frame(w);
    w.measure_now();
    const FrameMeasure* m = w.measurement();
    ASSERT_NE(m, nullptr);
    const MetricsText want = metrics_text(m->frame_metrics(), [] {
        FrameHeader h;
        h.source_resolution = h.resolution = "96x64";
        h.elapsed_s = 0.25;
        return h;
    }(), m->coverage.clipped_pct);
    auto expect_rows = [](const std::vector<std::vector<std::string>>& got, const std::vector<MetricRow>& rows) {
        ASSERT_EQ(got.size(), rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            EXPECT_EQ(got[i][0], rows[i].k);
            EXPECT_EQ(got[i][1], rows[i].v) << rows[i].k;
            EXPECT_EQ(got[i][2], rows[i].u);
            EXPECT_EQ(got[i][3], rows[i].warn ? "warn" : "") << rows[i].k;
        }
    };
    expect_rows(rows_of(w.findChild<QWidget*>("measA")), want.a);
    expect_rows(rows_of(w.findChild<QWidget*>("measB")), want.b);
    EXPECT_EQ(w.findChild<QLabel*>("statusMask")->text().toStdString(), want.status_mask);
    EXPECT_EQ(w.findChild<QLabel*>("statusTime")->text().toStdString(), want.status_time);
    EXPECT_EQ(w.findChild<QLabel*>("srcInfo")->text().toStdString(), want.src_info);
    EXPECT_GT(m->coverage.clipped_pct, 0.0);
    EXPECT_GT(m->measured.metrics.maxcll, 203.0);
    // The clip bar: red for the clipped share, amber after it for the mask.
    auto* bar = find<app::ClipBar>(w, "clipBar");
    ASSERT_TRUE(bar->isVisible());
    EXPECT_NEAR(bar->lost()->width(), bar->width() * std::min(100.0, m->coverage.clipped_pct) / 100.0, 1.0);
    EXPECT_EQ(bar->acted()->x(), bar->lost()->width());
    // The scopes are drawn from it.
    EXPECT_TRUE(find<app::ScopePlot>(w, "wave")->has_data());
    EXPECT_TRUE(find<app::ScopePlot>(w, "hist")->has_data());
    // The pipeline bar: MaxCLL over the view peak warns, a peak above it does not.
    EXPECT_TRUE(w.findChild<QLabel*>("pipeWarn")->isVisible());
    EXPECT_EQ(w.findChild<QLabel*>("pipeWarn")->text().toStdString(),
              pipe_text("aces", 203.0, m->measured.metrics.maxcll, std::nullopt).warn);
    w.session().peak_input(5.0);
    EXPECT_EQ(w.findChild<QLabel*>("viewTransform")->text().toStdString(), "exposure + clip \u00b7 6,496 nits");
    EXPECT_FALSE(w.findChild<QLabel*>("pipeWarn")->isVisible());
    w.run("container-linear");
    EXPECT_EQ(w.findChild<QLabel*>("pipeMaster")->text().toStdString(), "linear Rec.2020 EXR, float32");
    // A grade change measures again, on its own thread, after the grade settles.
    const double before = m->measured.metrics.peak_nits;
    w.run("mode-off");
    // (By value: the new measurement may well sit where the old one was freed.)
    for (int i = 0; i < 500 && !(w.measurement()->measured.metrics.peak_nits < before - 1.0); ++i) QTest::qWait(10);
    ASSERT_LT(w.measurement()->measured.metrics.peak_nits, before - 1.0) << "no new measurement within 5 s";
    EXPECT_EQ(rows_of(w.findChild<QWidget*>("measA"))[2][1],
              metrics_text(w.measurement()->frame_metrics(), std::nullopt, 0).a[2].v);   // Peak, redrawn
}

// The compare line (beta 3): what the model changed against the baseline, how
// much of it shows at this view peak, and a button that raises the peak far
// enough to see it. Changes tints the picture (core/view.cpp, display.frag).
TEST(AppMeasure, TheCompareLineSaysWhatChangedAndRaisesThePeak) {
    app::MainWindow w(false);
    w.resize(1600, 1000);
    w.show();
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&w));
    auto* line = w.findChild<QLabel*>("compareLine");
    auto* fit = w.findChild<QPushButton*>("compareFit");
    ASSERT_NE(line, nullptr);
    ASSERT_NE(fit, nullptr);
    EXPECT_FALSE(fit->isVisible());
    synthetic_frame(w);
    w.measure_now();
    const FrameMeasure* m = w.measurement();
    ASSERT_NE(m, nullptr);
    const double inf = std::numeric_limits<double>::infinity();
    const CompareText want = compare_text(m->compare, 203.0, inf);
    EXPECT_EQ(line->text().toStdString(), want.line);
    EXPECT_EQ(line->property("state").toString().toStdString(), want.warn ? "warn" : "");
    EXPECT_EQ(fit->isVisible(), want.fit_shown);
    if (want.fit_shown) {
        EXPECT_EQ(fit->text().toStdString(), want.fit_label);
        fit->click();
        EXPECT_DOUBLE_EQ(w.session().peak_ev, want.fit_peak_ev);
        const CompareText after = compare_text(m->compare, w.session().display_nits(), inf);
        EXPECT_EQ(line->text().toStdString(), after.line);
        EXPECT_FALSE(fit->isVisible());
    }
    // Changes: a view setting, not part of the grade or its undo.
    auto* changes = w.findChild<QPushButton*>("changesBtn");
    ASSERT_NE(changes, nullptr);
    const std::string params = w.session().params_json();
    const auto undo = w.session().undo_depth();
    changes->click();
    EXPECT_TRUE(w.session().show_changes);
    EXPECT_TRUE(changes->isChecked());
    EXPECT_EQ(w.session().params_json(), params);
    EXPECT_EQ(w.session().undo_depth(), undo);
    changes->click();
    EXPECT_FALSE(w.session().show_changes);
    // Invented (roadmap 3.5): a layer of its own, and the line becomes its legend.
    auto* invented = w.findChild<QPushButton*>("inventedBtn");
    ASSERT_NE(invented, nullptr);
    invented->click();
    EXPECT_EQ(w.session().view_layer, 3);
    EXPECT_TRUE(invented->isChecked());
    EXPECT_EQ(line->text().toStdString(), support_text(m->support));
    EXPECT_FALSE(fit->isVisible());
    find<app::Seg>(w, "viewLayer")->button("0")->click();   // Image ends it
    EXPECT_FALSE(invented->isChecked());
    invented->click();
    invented->click();                                      // and so does a second click
    EXPECT_EQ(w.session().view_layer, 0);
    EXPECT_EQ(line->text().toStdString(), compare_text(m->compare, w.session().display_nits(), inf).line);
}

TEST(AppMeasure, TheProbeReadsThePixelUnderThePointer) {
    app::MainWindow w(false);
    w.resize(1600, 1000);
    w.show();
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&w));
    synthetic_frame(w);
    w.measure_now();
    const FrameMeasure* m = w.measurement();
    for (const auto& at : std::vector<std::pair<double, double>>{{90.4, 10.9}, {2.0, 60.0}, {40.5, 30.5}}) {
        const auto probe = m->probe_at(at.first, at.second);
        ASSERT_TRUE(probe.has_value());
        w.probe_pixel(at, w.mapToGlobal(QPoint(500, 400)));
        const ProbePanelText t = probe_panel(probe);
        EXPECT_EQ(w.findChild<QLabel*>("probeXY")->text().toStdString(), t.xy);
        EXPECT_EQ(w.findChild<QLabel*>("probeNits")->text().toStdString(), t.nits);
        EXPECT_EQ(w.findChild<QLabel*>("probeNits")->property("state").toString(), "");
        EXPECT_TRUE(w.findChild<QLabel*>("probeNitsUnit")->isVisibleTo(&w));
        EXPECT_EQ(w.findChild<QLabel*>("probeDelta")->text().toStdString(), t.delta);
        EXPECT_EQ(w.findChild<QLabel*>("probeSrc")->text().toStdString(), t.src);
        EXPECT_EQ(w.findChild<QLabel*>("probeSrc")->property("state").toString().toStdString(), t.src_class);
        EXPECT_EQ(w.findChild<QLabel*>("probeBase")->text().toStdString(), t.base);
        EXPECT_EQ(w.findChild<QLabel*>("probeModel")->text().toStdString(), t.model);
        EXPECT_EQ(w.findChild<QLabel*>("probeModel")->property("state").toString().toStdString(), t.model_class);
        EXPECT_EQ(w.findChild<QLabel*>("probeMask")->text().toStdString(), t.mask);
        // The floating box: the page's rows, beside the pointer.
        ASSERT_TRUE(w.probe_box()->isVisible());
        const auto rows = probe_box(*probe);
        const auto labels = w.probe_box()->findChildren<QLabel*>();
        ASSERT_EQ(labels.size(), qsizetype(rows.size() * 2));
        for (std::size_t r = 0; r < rows.size(); ++r) {
            EXPECT_EQ(labels[qsizetype(r * 2)]->text().toStdString(), rows[r].k);
            EXPECT_EQ(labels[qsizetype(r * 2 + 1)]->text().toStdString(), rows[r].v);
        }
    }
    // The clipped corner reads as clipped.
    w.probe_pixel(std::pair{95.0, 0.0}, w.mapToGlobal(QPoint(500, 400)));
    EXPECT_EQ(w.findChild<QLabel*>("probeSrc")->property("state").toString(), "clip");
    // Off the frame, or the probe put away: idle, no box.
    w.probe_pixel(std::nullopt);
    EXPECT_EQ(w.findChild<QLabel*>("probeNits")->property("state").toString(), "idle");
    EXPECT_EQ(w.findChild<QLabel*>("probeDelta")->text().toStdString(), "pick a pixel with Probe, or hold Alt");
    EXPECT_FALSE(w.probe_box()->isVisible());
}

// ---- step 9: the Deliver tab's master, as a background job ----------------

namespace {

MasterFrame small_frame() {
    MasterFrame f;
    f.sdr = SdrImage(PlanarBuffer(3, 12, 20, 0.6f));
    f.fields = Fields{PlanarBuffer(3, 12, 20, 0.1f), PlanarBuffer(1, 12, 20, 0.5f), PlanarBuffer(1, 12, 20, 0.0f)};
    return f;
}

std::filesystem::path fresh_dir(const std::string& name) {
    // The temporary folder by its real path: on macOS /var is /private/var, and
    // the app reports the paths it writes as they are.
    const auto root = std::filesystem::canonical(std::filesystem::temp_directory_path()) / ("rudra-app-master-" + name);
    std::filesystem::remove_all(root);
    return root;
}

QString status_of(app::MainWindow& w) { return w.findChild<QLabel*>("renderStatus")->text(); }

bool wait_for(const std::function<bool()>& done, int ms = 10000) {
    for (int i = 0; i < ms / 10 && !done(); ++i) QTest::qWait(10);
    return done();
}

}  // namespace

TEST(AppDeliver, MasterRendersThePlanAndSaysSo) {
    app::MainWindow w(false);
    w.show();
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&w));
    auto prepare = [](std::size_t) -> Result<MasterFrame> { return small_frame(); };
    // No folder: the page's message, nothing rendered.
    w.master(prepare, 3);
    EXPECT_EQ(status_of(w), "Choose a render folder first.");
    EXPECT_FALSE(w.mastering());
    const auto root = fresh_dir("ok");
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString(root.string()));
    w.findChild<QLineEdit*>("renderName")->setText("shot_010");
    w.findChild<QComboBox*>("renderMode")->setCurrentIndex(1);   // All loaded frames, sequence
    w.findChild<QSpinBox*>("renderStart")->setValue(1);
    w.master(prepare, 3);
    ASSERT_TRUE(wait_for([&] { return !w.mastering() && status_of(w).startsWith("Rendered"); }));
    EXPECT_EQ(status_of(w).toStdString(), "Rendered 3 frame(s) to " + root.string());
    EXPECT_EQ(w.findChild<QPushButton*>("btnMaster")->text(), "Master EXR");
    const QString log = w.findChild<QPlainTextEdit*>("log")->toPlainText();
    const auto first = root / "shot_010.000001.exr", last = root / "shot_010.000003.exr";
    EXPECT_TRUE(log.contains(QString::fromStdString("Render destination: " + first.string() + " \u2026 " + last.string())));
    for (int i = 1; i <= 3; ++i) {
        const auto exr = root / ("shot_010.00000" + std::to_string(i) + ".exr");
        EXPECT_TRUE(std::filesystem::exists(exr)) << exr;
        EXPECT_TRUE(log.contains(QString::fromStdString("Saved " + exr.string() + " \u00b7 20x12"))) << i;
    }
    // Again: refused, nothing replaced.
    w.master(prepare, 3);
    EXPECT_EQ(status_of(w).toStdString(), "Stopped after 0 / 3: Refusing to overwrite existing render: " + first.string());
    // A bad name and a relative folder, as the plan refuses them.
    w.findChild<QLineEdit*>("renderName")->setText("my shot");
    w.master(prepare, 3);
    EXPECT_EQ(status_of(w), "Stopped after 0 / 3: Render name must contain only letters, numbers, dots, underscores or hyphens");
    w.findChild<QLineEdit*>("renderName")->setText("shot");
    w.findChild<QLineEdit*>("renderDir")->setText("renders/out");
    w.findChild<QComboBox*>("renderMode")->setCurrentIndex(0);   // Current image
    w.master(prepare, 3);
    EXPECT_EQ(status_of(w), "Stopped after 0 / 1: Choose an absolute render folder on the Studio computer");
    std::filesystem::remove_all(root);
}

TEST(AppDeliver, MasterEXRStopsARenderAfterTheFrameInHand) {
    app::MainWindow w(false);
    w.show();
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&w));
    const auto root = fresh_dir("cancel");
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString(root.string()));
    w.findChild<QComboBox*>("renderMode")->setCurrentIndex(1);
    std::atomic<int> started{0};
    w.master([&](std::size_t) -> Result<MasterFrame> {
        ++started;
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        return small_frame();
    }, 20);
    ASSERT_TRUE(w.mastering());
    ASSERT_TRUE(wait_for([&] { return started >= 1; }));
    w.findChild<QPushButton*>("btnMaster")->click();   // the button stops it
    ASSERT_TRUE(wait_for([&] { return !w.mastering() && status_of(w).startsWith("Stopped"); }));
    EXPECT_TRUE(status_of(w).endsWith(": cancelled")) << status_of(w).toStdString();
    EXPECT_LT(started.load(), 20);
    EXPECT_EQ(w.findChild<QPushButton*>("btnMaster")->text(), "Master EXR");
    std::filesystem::remove_all(root);
}

// ---- step 5: the scope widgets against the page's own rasters -------------

namespace {

const json& scope_golden() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/scopes/drawings.json");
        return json::parse(f);
    }();
    return g;
}

// How far two rasters of the same drawing are apart: the mean difference in
// codes, and the share of pixels whose channels all lie within `tol`.
struct Agreement {
    double mean = 1e9, within = 0.0;
    int worst = 0;
};
Agreement agreement(const QImage& a, const QImage& b, int tol) {
    Agreement r;
    const QImage x = a.convertToFormat(QImage::Format_RGB32), y = b.convertToFormat(QImage::Format_RGB32);
    if (x.size() != y.size()) return r;
    double sum = 0.0;
    long good = 0, all = 0;
    for (int row = 0; row < x.height(); ++row)
        for (int col = 0; col < x.width(); ++col) {
            const QRgb p = x.pixel(col, row), q = y.pixel(col, row);
            const int dr = std::abs(qRed(p) - qRed(q)), dg = std::abs(qGreen(p) - qGreen(q)),
                      db = std::abs(qBlue(p) - qBlue(q));
            sum += (dr + dg + db) / 3.0;
            const int d = std::max({dr, dg, db});
            r.worst = std::max(r.worst, d);
            good += d <= tol;
            ++all;
        }
    r.mean = sum / double(all);
    r.within = double(good) / double(all);
    return r;
}

// The pixels a drawing's labels change: where `with` and `without` differ.
std::vector<bool> changed(const QImage& with, const QImage& without) {
    const QImage x = with.convertToFormat(QImage::Format_RGB32), y = without.convertToFormat(QImage::Format_RGB32);
    std::vector<bool> m(std::size_t(x.width()) * std::size_t(x.height()), false);
    for (int row = 0; row < x.height(); ++row)
        for (int col = 0; col < x.width(); ++col) {
            const QRgb p = x.pixel(col, row), q = y.pixel(col, row);
            m[std::size_t(row) * std::size_t(x.width()) + std::size_t(col)] =
                std::max({std::abs(qRed(p) - qRed(q)), std::abs(qGreen(p) - qGreen(q)),
                          std::abs(qBlue(p) - qBlue(q))}) > 16;
        }
    return m;
}

// The share of `a`'s pixels with a pixel of `b` within one pixel of them.
double covered(const std::vector<bool>& a, const std::vector<bool>& b, int w, int h) {
    long n = 0, hit = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if (!a[std::size_t(y) * std::size_t(w) + std::size_t(x)]) continue;
            ++n;
            bool near = false;
            for (int dy = -1; dy <= 1 && !near; ++dy)
                for (int dx = -1; dx <= 1 && !near; ++dx) {
                    const int u = x + dx, v = y + dy;
                    near = u >= 0 && v >= 0 && u < w && v < h && b[std::size_t(v) * std::size_t(w) + std::size_t(u)];
                }
            hit += near;
        }
    return n ? double(hit) / double(n) : 1.0;
}

// Where each <text> of a drawing lands in a widget of `target`: the
// transform paint_svg uses (viewBox, xMidYMid meet), a monospaced width.
struct LabelBox {
    std::string text;
    QRectF box;
    bool middle = false;
};
std::vector<LabelBox> label_boxes(const SvgDrawing& d, const QRectF& target) {
    std::vector<LabelBox> out;
    double vx = 0, vy = 0, vw = 1, vh = 1;
    std::sscanf(d.view_box.c_str(), "%lf %lf %lf %lf", &vx, &vy, &vw, &vh);
    const double k = std::min(target.width() / vw, target.height() / vh);
    const double ox = target.x() + (target.width() - vw * k) / 2.0 - vx * k;
    const double oy = target.y() + (target.height() - vh * k) / 2.0 - vy * k;
    std::function<void(const SvgElement&)> walk = [&](const SvgElement& e) {
        for (const auto& c : e.children) walk(c);
        if (e.tag != "text") return;
        const double size = std::stod(*e.attr("font-size")) * k;
        const double wd = 0.6 * size * double(e.text.size());
        double x = ox + std::stod(*e.attr("x")) * k;
        const double y = oy + std::stod(*e.attr("y")) * k;
        const std::string* a = e.attr("text-anchor");
        const bool middle = a && *a == "middle";
        if (middle) x -= wd / 2.0;
        out.push_back({e.text, QRectF(x - 2, y - size - 2, wd + 4, size * 1.3 + 4), middle});
    };
    for (const auto& e : d.elements) walk(e);
    return out;
}

// A mask inside a box: its left and right extent, its vertical centre of
// mass, and how many pixels it has there.
struct Extent {
    double left = 0, right = 0, cy = 0;
    int n = 0;
};
Extent extent(const std::vector<bool>& m, int w, int h, const QRectF& box) {
    Extent e;
    double sy = 0;
    e.left = 1e9;
    e.right = -1e9;
    for (int y = std::max(0, int(box.top())); y < std::min(h, int(std::ceil(box.bottom()))); ++y)
        for (int x = std::max(0, int(box.left())); x < std::min(w, int(std::ceil(box.right()))); ++x)
            if (m[std::size_t(y) * std::size_t(w) + std::size_t(x)]) {
                e.left = std::min(e.left, double(x));
                e.right = std::max(e.right, double(x));
                sy += y;
                ++e.n;
            }
    if (e.n) e.cy = sy / e.n;
    return e;
}

QImage grab_at(QWidget& w, int width, int height) {
    w.setFixedSize(width, height);
    w.setAttribute(Qt::WA_DontShowOnScreen, true);
    w.show();
    QApplication::processEvents();
    return w.grab().toImage();
}

}  // namespace

TEST(AppScopes, WaveformAndHistogramLookAsThePageDrawsThem) {
    for (const auto& c : scope_golden()["cases"]) {
        const std::string name = c["name"];
        ScopeData s;
        for (const char* k : {"lo", "q1", "mid", "q3", "hi"})
            (k == std::string("lo") ? s.lo : k == std::string("q1") ? s.q1 : k == std::string("mid") ? s.mid
             : k == std::string("q3") ? s.q3 : s.hi) = c["scopes"][k].get<std::vector<double>>();
        s.histogram = c["scopes"]["histogram"].get<std::vector<double>>();
        std::optional<double> maxcll;
        if (!c["metrics"].is_null()) {
            maxcll = c["metrics"]["maxcll"].is_null() ? std::nan("") : c["metrics"]["maxcll"].get<double>();
        }
        for (const char* part : {"wave", "hist"}) {
            const auto& box = c["drawn"][part]["box"];
            app::ScopePlot plot(part, part == std::string("wave") ? app::ScopePlot::Kind::Waveform
                                                                  : app::ScopePlot::Kind::Histogram,
                                box[1].get<int>());
            plot.set_data(s, maxcll);
            plot.set_text_shown(false);
            const QImage got = grab_at(plot, box[0].get<int>(), box[1].get<int>());
            const std::string base = std::string(RUDRA_GOLDEN_DIR) + "/scopes/" + name + "_" + part;
            const QImage want(QString::fromStdString(base + "_notext.png"));
            ASSERT_FALSE(want.isNull()) << name << " " << part;
            // The geometry and colour, labels hidden: equal but for how two
            // rasterisers cover an edge (a missing or misplaced band, bar or
            // line moves the mean past a code and the share under 97 %).
            const Agreement a = agreement(got, want, 8);
            if (a.mean > 1.0 || a.within < 0.97) got.save(QString::fromStdString("/tmp/native_scope_" + name + "_" + part + ".png"));
            EXPECT_LE(a.mean, 1.0) << name << " " << part;
            EXPECT_GE(a.within, 0.97) << name << " " << part << ": " << a.within * 100 << " % within 8 codes";
            // The labels: drawn where the page draws them. Glyphs differ
            // between rasterisers, so this compares where, not how.
            plot.set_text_shown(true);
            const QImage got_text = plot.grab().toImage();
            const QImage want_text(QString::fromStdString(base + ".png"));
            const int w = got.width(), h = got.height();
            const auto ours = changed(got_text, got), page = changed(want_text, want);
            EXPECT_GE(covered(ours, page, w, h), 0.8) << name << " " << part << ": our labels where the page has none";
            EXPECT_GE(covered(page, ours, w, h), 0.8) << name << " " << part << ": the page's labels missing";
            // And each label in its place: around every <text> of the display
            // list, the page's glyphs and ours start (or, centred, centre)
            // within a pixel and sit on the same line. How long a string runs
            // is the platform's (Chromium on Linux rounds each advance to a
            // whole pixel at these sizes, on Windows and macOS it does not),
            // so the far end is not compared.
            for (const auto& l : label_boxes(plot.drawing(), QRectF(1, 1, w - 2, h - 2))) {
                const Extent pe = extent(page, w, h, l.box), oe = extent(ours, w, h, l.box);
                ASSERT_GT(pe.n, 0) << name << " " << part << ": the page drew no \"" << l.text << "\"";
                ASSERT_GT(oe.n, 0) << name << " " << part << ": no \"" << l.text << "\"";
                const double px = l.middle ? (pe.left + pe.right) / 2.0 : pe.left;
                const double ox = l.middle ? (oe.left + oe.right) / 2.0 : oe.left;
                EXPECT_LE(std::abs(px - ox), 1.0) << name << " " << part << " \"" << l.text << "\" x";
                EXPECT_LE(std::abs(pe.cy - oe.cy), 1.0) << name << " " << part << " \"" << l.text << "\" y";
            }        }
    }
}

TEST(AppScopes, VectorscopeLooksAsThePageDrawsIt) {
    const auto& v = scope_golden()["vector"];
    const int w = v["sample"]["width"], h = v["sample"]["height"];
    const auto rgba = v["sample"]["rgba"].get<std::vector<double>>();
    PlanarBuffer sample(3, h, w);
    for (int i = 0; i < w * h; ++i)
        for (int ch = 0; ch < 3; ++ch) sample.plane(ch)[i] = float(rgba[std::size_t(i) * 4 + std::size_t(ch)]);
    app::VectorscopeView view;
    view.set_image(vectorscope(sample));
    const QImage got = grab_at(view, v["plot_box"][0].get<int>(), v["plot_box"][1].get<int>());
    const QImage want(QString::fromStdString(std::string(RUDRA_GOLDEN_DIR) + "/scopes/vector.png"));
    ASSERT_FALSE(want.isNull());
    // The picture itself is Phase 2's, equal to the page's canvas; what is
    // compared here is its placing, scaling and frame (labels included).
    const Agreement a = agreement(got, want, 8);
    if (a.mean > 1.0 || a.within < 0.985) got.save("/tmp/native_scope_vector.png");
    EXPECT_LE(a.mean, 1.0);
    EXPECT_GE(a.within, 0.985) << a.within * 100 << " % within 8 codes (worst " << a.worst << ")";
}


// ---------------------------------------------------------------------------
// Phase 3 step 10: the checkpoint manager and the first run
// ---------------------------------------------------------------------------

namespace {

// A backend that answers at once with fields of its own level, so a frame
// says which model made it.
class FakeBackend : public InferenceBackend {
public:
    FakeBackend(float level, std::shared_ptr<std::atomic<int>> calls, std::shared_ptr<std::atomic<bool>> pump = {})
        : level_(level), calls_(std::move(calls)), pump_(std::move(pump)) {}
    BackendInfo info() const override { return {Runtime::OnnxRuntime, Device::Cpu, "fake", "CPUExecutionProvider"}; }
    // Pumping: the shadow weight jumps with a marker in the frame's corner (1
    // where it is bright, 0.25 where it is dark), as a gate can on grain.
    Result<FrameScalars> frame_pass(const SdrImage& s) override {
        FrameScalars f;
        if (pump_ && pump_->load()) f.shadow_weight = s.buffer().plane(0)[0] > 0.5f ? 1.0f : 0.25f;
        return f;
    }
    Result<Fields> tile_pass(const SdrImage& tile, const FrameScalars&) override {
        ++*calls_;
        const int h = tile.height(), w = tile.width();
        return Fields{PlanarBuffer(3, h, w, level_), PlanarBuffer(1, h, w, 0.5f), PlanarBuffer(1, h, w, 0.0f)};
    }

private:
    float level_;
    std::shared_ptr<std::atomic<int>> calls_;
    std::shared_ptr<std::atomic<bool>> pump_;
};

void write_manifest(const std::filesystem::path& dir, const std::string& name) {
    std::filesystem::create_directories(dir);
    const json tol{{"atol", 1e-5}, {"rtol", 1e-5}};
    const std::string z(64, '0');
    const json m{{"contract", "1.0"}, {"name", name}, {"source", {{"file", name + ".pt"}, {"sha256", z}}},
                 {"exported", "2026-09-24T00:00:00Z"},
                 {"network", {{"base_channels", 32}, {"corpus_ev", -1.0}, {"log_scale", 16.0}, {"max_hdr", 4.0},
                              {"heads", {{"residual_gate", false}, {"shadow_gate", true}, {"curve", false}}},
                              {"curve_params", 1}}},
                 {"tiling", {{"tile_size", 512}, {"overlap", 64}}},
                 {"files", {{"torchscript", "model.ts"}, {"onnx_frame", "f.onnx"}, {"onnx_tile", "t.onnx"},
                            {"golden", "golden/golden.json"}, {"torchscript_sha256", z}, {"onnx_frame_sha256", z},
                            {"onnx_tile_sha256", name + z.substr(name.size())}}},
                 {"onnx_inputs", {{"frame", {"sdr"}}, {"tile", {"sdr"}}}},
                 {"tolerance", {{"torchscript", tol}, {"onnx", tol}}}};
    std::ofstream(dir / "manifest.json") << m.dump(1);
}

struct Models {
    std::filesystem::path root;
    std::map<std::string, std::shared_ptr<std::atomic<int>>> calls;   // tile passes, by package name
    std::map<std::string, int> tests;                                 // self-tests run, by package name
    std::set<std::string> drifting;                                   // these fail their self-test
    std::shared_ptr<std::atomic<bool>> pump = std::make_shared<std::atomic<bool>>(false);   // FakeBackend
};

// Three packages (alpha, beta, gamma) in a models.json root; the window opens
// them through fakes and runs a fake self-test.
std::shared_ptr<Models> fake_models(app::MainWindow& w, const std::string& tag) {
    auto m = std::make_shared<Models>();
    m->root = fresh_dir("models-" + tag);
    const std::map<std::string, float> levels{{"alpha", 0.1f}, {"beta", 0.2f}, {"gamma", 0.3f}};
    for (const auto& [name, level] : levels) {
        write_manifest(m->root / name, name);
        m->calls[name] = std::make_shared<std::atomic<int>>(0);
    }
    std::ofstream(m->root / "models.json") << json{{"default", "beta.pt"},
                                                   {"models", {{{"file", "alpha.pt"}, {"kind", "sdr2hdr"}, {"title", "Alpha"}},
                                                               {{"file", "beta.pt"}, {"kind", "sdr2hdr"}, {"title", "Beta"}},
                                                               {{"file", "gamma.pt"}, {"kind", "sdr2hdr"}, {"title", "Gamma"}}}}}
                                                  .dump();
    app::MainWindow::ModelHooks h;
    h.verify_files = false;
    h.open = [m, levels](const ModelManifest& man, std::optional<BackendChoice> c) -> Result<OpenedBackend> {
        return OpenedBackend{std::make_unique<FakeBackend>(levels.at(man.name), m->calls.at(man.name), m->pump),
                             c.value_or(BackendChoice{Runtime::OnnxRuntime, Device::Cpu})};
    };
    h.test = [m](const ModelManifest& man, InferenceBackend& b) -> Result<SelfTestReport> {
        SelfTestReport r;
        r.backend = b.info();
        r.frames = 16;
        r.stitched = true;
        r.outputs["residual"].max_abs = m->drifting.count(man.name) ? 0.5 : 1e-6;
        r.outputs["residual"].excess = m->drifting.count(man.name) ? 0.4 : -1.0;
        ++m->tests[man.name];
        return r;
    };
    w.set_model_hooks(h);
    w.set_model_roots({m->root});
    return m;
}

bool use(app::MainWindow& w, const std::filesystem::path& p, QString* why = nullptr,
         std::optional<BackendChoice> c = std::nullopt) {
    std::optional<bool> result;
    w.use_model(p, c, [&](bool ok, const QString& y) {
        result = ok;
        if (why) *why = y;
    });
    EXPECT_TRUE(wait_for([&] { return result.has_value(); }));
    return result.value_or(false);
}

QString text_of(QWidget& w, const char* id) {
    auto* l = w.findChild<QLabel*>(id);
    return l ? l->text() : QString("<no %1>").arg(id);
}

}  // namespace

TEST(AppModels, TheCatalogPicksTheRegistrysDefaultAndThePillsSayIt) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "pick");
    ASSERT_EQ(w.catalog().entries.size(), 3u);
    ASSERT_TRUE(w.catalog().chosen);
    EXPECT_EQ(w.catalog().entries[*w.catalog().chosen].package.filename(), "beta");
    EXPECT_EQ(text_of(w, "ckpt"), "no model");
    ASSERT_TRUE(use(w, m->root / "beta"));
    EXPECT_EQ(text_of(w, "ckpt"), "beta");
    EXPECT_EQ(text_of(w, "device"), "CPU");   // the page's info.gpu for a CPU run
    EXPECT_EQ(w.findChild<QLabel*>("lamp")->property("state").toString(), "on");
    const QString log = w.findChild<QPlainTextEdit*>("log")->toPlainText();
    EXPECT_TRUE(log.contains("self-test passed, 16 golden frames + stitch")) << log.toStdString();
    EXPECT_TRUE(log.contains("device CPU"));
    EXPECT_TRUE(log.contains("drop frames — the network runs once each, then the grade is local"));
    EXPECT_EQ(QSettings().value("model/package").toString().toStdString(), (m->root / "beta").string());
}

#ifdef RUDRA_HAVE_STILL_DECODE
TEST(AppModels, SwitchingModelsMidSessionKeepsTheSession) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "switch");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.set_use_model(true);   // the model's fields, not the analytic reconstruction's
    w.open_source(QString::fromStdString((std::filesystem::path(RUDRA_GOLDEN_DIR) / "decode").string()));
    ASSERT_GE(w.frame_count(), 10u);
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    w.run("next");
    w.run("next");
    ASSERT_TRUE(wait_for([&] { return w.current_index() == 2 && w.frame_fields() &&
                                      w.frame_fields()->residual.at(0, 0, 0) == 0.1f && *m->calls["alpha"] >= 3; }));
    // A grade with undo behind it.
    w.run("strength-up");
    w.run("mode-shadows");
    w.run("preserve");
    const std::string params = w.session().params_json();
    const auto undo = w.session().undo_depth();
    ASSERT_GE(undo, 3u);

    ASSERT_TRUE(use(w, m->root / "gamma"));
    EXPECT_EQ(text_of(w, "ckpt"), "gamma");
    // The same frames, the same frame, the same grade and undo; the new
    // model's fields for that frame.
    EXPECT_GE(w.frame_count(), 10u);
    EXPECT_EQ(w.current_index(), 2);
    EXPECT_EQ(w.session().params_json(), params);
    EXPECT_EQ(w.session().undo_depth(), undo);
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() && w.frame_fields()->residual.at(0, 0, 0) == 0.3f; }));
    EXPECT_GE(*m->calls["gamma"], 1);
    // And back: alpha's fields again, from alpha, not a cache of gamma's.
    ASSERT_TRUE(use(w, m->root / "alpha"));
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() && w.frame_fields()->residual.at(0, 0, 0) == 0.1f; }));
    EXPECT_EQ(w.session().params_json(), params);
}
#endif

TEST(AppModels, AModelThatDriftsIsRefusedAndTheOneInUseStays) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "drift");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    m->drifting.insert("gamma");
    QString why;
    EXPECT_FALSE(use(w, m->root / "gamma", &why));
    EXPECT_TRUE(why.startsWith("model gamma refused on ONNX Runtime on CPU: self-test FAILED (residual)")) << why.toStdString();
    EXPECT_EQ(text_of(w, "ckpt"), "alpha");
    EXPECT_EQ(w.model_package(), m->root / "alpha");
    // A folder that is not a package says so and changes nothing.
    EXPECT_FALSE(use(w, m->root, &why));
    EXPECT_TRUE(why.startsWith("This folder is not a RUDRA model package.")) << why.toStdString();
    EXPECT_EQ(text_of(w, "ckpt"), "alpha");
}

TEST(AppModels, TheGoldensRunOncePerPackageAndBackend) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "once");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    ASSERT_TRUE(use(w, m->root / "beta"));
    ASSERT_TRUE(use(w, m->root / "alpha"));
    EXPECT_EQ(m->tests["alpha"], 1);
    EXPECT_EQ(m->tests["beta"], 1);
    // Another backend is another check.
    ASSERT_TRUE(use(w, m->root / "alpha", nullptr, BackendChoice{Runtime::LibTorch, Device::Cpu}));
    EXPECT_EQ(m->tests["alpha"], 2);
    EXPECT_EQ(w.model_backend(), (std::optional<BackendChoice>{BackendChoice{Runtime::LibTorch, Device::Cpu}}));
}

TEST(AppModels, ABareStartOpensTheLastPackageOnItsBackend) {
    QSettings().clear();
    QSettings().setValue("firstRun/done", true);
    {
        app::MainWindow w(false);
        auto m = fake_models(w, "boot");
        w.boot();   // nothing remembered: the registry's default
        ASSERT_TRUE(wait_for([&] { return !w.loading_model() && !w.model_package().empty(); }));
        EXPECT_EQ(w.model_package(), m->root / "beta");
        ASSERT_TRUE(use(w, m->root / "gamma", nullptr, BackendChoice{Runtime::LibTorch, Device::Cpu}));
    }
    app::MainWindow w(false);
    auto m = fake_models(w, "boot2");
    // The remembered one is from the first window's folder (still there).
    w.boot();
    ASSERT_TRUE(wait_for([&] { return !w.loading_model() && !w.model_package().empty(); }));
    EXPECT_EQ(w.model_package().filename(), "gamma");
    EXPECT_EQ(w.model_backend()->key(), "libtorch/cpu");
    EXPECT_EQ(w.findChild<QDialog*>("firstRun"), nullptr);
}

namespace {
// Point the fake registry's default at another package, as a release that
// replaces its default model does (shadow_v1 -> v8, 9 Oct 2026).
void set_default(const Models& m, const std::string& file) {
    std::ofstream(m.root / "models.json") << json{{"default", file},
                                                  {"models", {{{"file", "alpha.pt"}, {"kind", "sdr2hdr"}, {"title", "Alpha"}},
                                                              {{"file", "beta.pt"}, {"kind", "sdr2hdr"}, {"title", "Beta"}},
                                                              {{"file", "gamma.pt"}, {"kind", "sdr2hdr"}, {"title", "Gamma"}}}}}
                                                 .dump();
}
QString log_text(app::MainWindow& w) { return w.findChild<QPlainTextEdit*>("log")->toPlainText(); }
bool booted(app::MainWindow& w) {
    w.boot();
    return wait_for([&] { return !w.loading_model() && !w.model_package().empty(); });
}
}  // namespace

TEST(AppModels, ABareStartFollowsTheDefaultWhenItMoves) {
    QSettings().clear();
    QSettings().setValue("firstRun/done", true);
    app::MainWindow w(false);
    auto m = fake_models(w, "follow");
    ASSERT_TRUE(booted(w));
    EXPECT_EQ(w.model_package(), m->root / "beta");   // the default, never picked
    set_default(*m, "alpha.pt");
    ASSERT_TRUE(booted(w));
    EXPECT_EQ(w.model_package(), m->root / "alpha");
    EXPECT_TRUE(log_text(w).contains("the default model is now Alpha; opening it instead of beta")) << log_text(w).toStdString();
}

TEST(AppModels, APickedPackageStaysPickedWhenTheDefaultMoves) {
    QSettings().clear();
    QSettings().setValue("firstRun/done", true);
    app::MainWindow w(false);
    auto m = fake_models(w, "picked");
    ASSERT_TRUE(booted(w));
    ASSERT_TRUE(use(w, m->root / "gamma"));
    set_default(*m, "alpha.pt");
    ASSERT_TRUE(booted(w));
    EXPECT_EQ(w.model_package(), m->root / "gamma");
    EXPECT_FALSE(log_text(w).contains("the default model is now"));
}

TEST(AppModels, OldSettingsGiveWayToTheDefaultOnlyForAPackageOutsideTheCatalog) {
    // Settings written before model/followsDefault existed: an older beta's
    // own package (outside every folder this install searches) gives way...
    QSettings().clear();
    QSettings().setValue("firstRun/done", true);
    app::MainWindow w(false);
    auto old = fake_models(w, "old-install");
    auto m = fake_models(w, "new-install");   // the roots are now only this one
    QSettings().setValue("model/package", QString::fromStdString((old->root / "alpha").string()));
    ASSERT_TRUE(booted(w));
    EXPECT_EQ(w.model_package(), m->root / "beta");
    EXPECT_TRUE(log_text(w).contains("the default model is now Beta")) << log_text(w).toStdString();
    // ...while one inside them is taken as picked.
    QSettings().clear();
    QSettings().setValue("firstRun/done", true);
    QSettings().setValue("model/package", QString::fromStdString((m->root / "gamma").string()));
    ASSERT_TRUE(booted(w));
    EXPECT_EQ(w.model_package(), m->root / "gamma");
}

// Startup (9 Oct 2026): a packaged RUDRA ignores Qt settings other software
// left in the environment, and keeps a log file.
namespace {
struct SavedEnv {
    std::vector<std::pair<const char*, std::optional<QByteArray>>> saved;
    explicit SavedEnv(std::initializer_list<const char*> names) {
        for (const char* n : names)
            saved.emplace_back(n, qEnvironmentVariableIsSet(n) ? std::optional<QByteArray>(qgetenv(n)) : std::nullopt);
    }
    ~SavedEnv() {
        for (const auto& [n, v] : saved) {
            if (v) qputenv(n, *v);
            else qunsetenv(n);
        }
    }
};
}  // namespace

TEST(AppStartup, ForeignQtSettingsGiveWayToTheShippedPlugins) {
    SavedEnv keep{"QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_PLATFORM"};
    const auto exe = std::filesystem::temp_directory_path() / "rudra-app-startup-exe";
    std::filesystem::remove_all(exe);
    // A development build on the system's Qt ships no plugins: nothing changes.
    qputenv("QT_QPA_PLATFORM", "offscreen");
    EXPECT_TRUE(app::shipped_platforms_dir(exe).empty());
    EXPECT_TRUE(app::ignore_foreign_qt_environment({}).empty());
    EXPECT_EQ(qgetenv("QT_QPA_PLATFORM"), "offscreen");
    // A package: the plugins beside the exe (Windows), or in the bundle (macOS).
    std::filesystem::create_directories(exe / "platforms");
    std::ofstream(exe / "platforms" / "qwindows.dll") << "";
    const auto dir = app::shipped_platforms_dir(exe);
    ASSERT_EQ(dir, exe / "platforms");
    qputenv("QT_PLUGIN_PATH", "C:/conda/Library/plugins");
    qputenv("QT_QPA_PLATFORM_PLUGIN_PATH", "C:/conda/Library/plugins/platforms");
    qputenv("QT_QPA_PLATFORM", "offscreen");   // the package has no offscreen plugin
    const auto ignored = app::ignore_foreign_qt_environment(dir);
    ASSERT_EQ(ignored.size(), 3u);
    EXPECT_NE(ignored[0].find("QT_PLUGIN_PATH=C:/conda/Library/plugins"), std::string::npos) << ignored[0];
    EXPECT_NE(ignored[2].find("no 'offscreen' platform plugin"), std::string::npos) << ignored[2];
    EXPECT_FALSE(qEnvironmentVariableIsSet("QT_PLUGIN_PATH"));
    EXPECT_FALSE(qEnvironmentVariableIsSet("QT_QPA_PLATFORM_PLUGIN_PATH"));
    EXPECT_FALSE(qEnvironmentVariableIsSet("QT_QPA_PLATFORM"));
    // A plugin the package does ship stays chosen, options and all.
    qputenv("QT_QPA_PLATFORM", "windows:darkmode=2");
    EXPECT_TRUE(app::ignore_foreign_qt_environment(dir).empty());
    EXPECT_EQ(qgetenv("QT_QPA_PLATFORM"), "windows:darkmode=2");
    // macOS: RUDRA.app/Contents/MacOS/RUDRA beside Contents/PlugIns/platforms.
    const auto bundle = exe / "RUDRA.app" / "Contents";
    std::filesystem::create_directories(bundle / "MacOS");
    std::filesystem::create_directories(bundle / "PlugIns" / "platforms");
    EXPECT_EQ(app::shipped_platforms_dir(bundle / "MacOS"), bundle / "PlugIns" / "platforms");
    std::filesystem::remove_all(exe);
}

TEST(AppStartup, TheLogFileRotatesAndKeepsTheWindowsLog) {
    const auto dir = std::filesystem::temp_directory_path() / "rudra-app-startup-log";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "rudra.log") << std::string(64, 'x');   // past the limit below
    std::ofstream(dir / "rudra.1.log") << "older";
    const QString path = app::install_app_log(QString::fromStdString(dir.string()), 32, 2);
    ASSERT_FALSE(path.isEmpty());
    EXPECT_EQ(app::app_log_path(), path);
    EXPECT_TRUE(std::filesystem::exists(dir / "rudra.1.log"));   // the big one, moved
    EXPECT_TRUE(std::filesystem::exists(dir / "rudra.2.log"));   // the older one, moved on
    app::MainWindow w(false);
    w.log("a line in the window");
    qWarning("a Qt warning");
    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::ReadOnly));
    const QString text = QString::fromUtf8(f.readAll());
    EXPECT_TRUE(text.contains("app   a line in the window")) << text.toStdString();
    EXPECT_TRUE(text.contains("warn  a Qt warning")) << text.toStdString();
}

TEST(AppModels, NoPackageAnywhereSaysWhereItLooked) {
    QSettings().clear();
    QSettings().setValue("firstRun/done", true);
    app::MainWindow w(false);
    w.set_model_roots({fresh_dir("empty-a"), fresh_dir("empty-b")});
    w.boot();
    EXPECT_EQ(text_of(w, "ckpt"), "no model package found");
    EXPECT_TRUE(w.findChild<QPlainTextEdit*>("log")->toPlainText().contains(
        "no model loaded: no model package found in 2 folder(s)"));
}

TEST(AppModels, TheManagerListsSwitchesAndAddsFolders) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "manager");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.run("models");
    auto* dlg = find<app::ModelManager>(w, "modelManager");
    ASSERT_NE(dlg, nullptr);
    auto* list = dlg->list();
    ASSERT_EQ(list->topLevelItemCount(), 3);
    EXPECT_EQ(list->topLevelItem(0)->text(0), "Alpha");
    EXPECT_EQ(list->topLevelItem(0)->text(3), "in use");
    EXPECT_EQ(list->topLevelItem(1)->text(3), "opens on a bare start");
    EXPECT_EQ(list->currentItem(), list->topLevelItem(0));
    dlg->select(2);
    EXPECT_EQ(list->currentItem(), list->topLevelItem(2));
    dlg->use_selected();
    ASSERT_TRUE(wait_for([&] { return !w.loading_model() && w.model_package().filename() == "gamma"; }));
    EXPECT_EQ(list->topLevelItem(2)->text(3), "in use");
    EXPECT_TRUE(dlg->findChild<QLabel*>("ckptStatus")->text().startsWith("In use: gamma on ONNX Runtime on CPU"));
    // A second folder joins the list.
    const auto more = fresh_dir("models-more");
    write_manifest(more / "delta", "delta");
    w.add_model_root(more);
    dlg->refresh();
    EXPECT_EQ(list->topLevelItemCount(), 4);
    EXPECT_EQ(list->topLevelItem(3)->text(0), "delta");
    // The backend list is every choice this build has, after Automatic.
    auto* combo = dlg->findChild<QComboBox*>("ckptBackend");
    EXPECT_EQ(combo->count(), int(backend_choices().size()) + 1);
}

TEST(AppModels, TheFirstRunReportsTheDisplaysRealPeak) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "first");
    w.boot();   // first start: the check, nothing loaded behind it
    auto* fr = find<app::FirstRun>(w, "firstRun");
    ASSERT_NE(fr, nullptr);
    EXPECT_TRUE(fr->isVisible());
    EXPECT_TRUE(w.model_package().empty());
    EXPECT_EQ(fr->findChild<QLabel*>("frDisplay")->text(), "Display not checked");   // no viewer in the tests
    // What the card's viewer reports on the PA279CRV through DXGI.
    fr->set_display(DisplayTarget::scrgb(418.0), "scRGB", "DXGI");
    EXPECT_EQ(fr->findChild<QLabel*>("frDisplay")->text(), "HDR display, 418 nits");
    EXPECT_TRUE(fr->findChild<QLabel*>("frDisplayDetail")->text().contains("the 600, 1,000 and 2,000 nit patches clip"));
    fr->set_display(DisplayTarget::sdr(), "SDR", "swapchain");
    EXPECT_EQ(fr->findChild<QLabel*>("frDisplay")->text(), "SDR display");
    EXPECT_TRUE(fr->findChild<QLabel*>("frModel")->text().startsWith("Beta"));
    fr->test_model();
    ASSERT_TRUE(wait_for([&] { return !w.loading_model() && !w.model_package().empty(); }));
    EXPECT_EQ(fr->findChild<QLabel*>("frModelStatus")->text(), "Ready: beta on ONNX Runtime on CPU");
    fr->start();
    EXPECT_TRUE(QSettings().value("firstRun/done").toBool());
    EXPECT_FALSE(fr->isVisible());
}


// ---------------------------------------------------------------------------
// Phase 3 step 11: the sheets, the copies, drop to open, recent shots, settings
// ---------------------------------------------------------------------------

#ifdef RUDRA_HAVE_STILL_DECODE
namespace {

std::filesystem::path decode_dir() { return std::filesystem::path(RUDRA_GOLDEN_DIR) / "decode"; }

void drop(QWidget& target, const QStringList& paths) {
    auto* mime = new QMimeData;
    QList<QUrl> urls;
    for (const auto& p : paths) urls << QUrl::fromLocalFile(p);
    mime->setUrls(urls);
    QDragEnterEvent enter(QPoint(10, 10), Qt::CopyAction, mime, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(&target, &enter);
    QDropEvent d(QPointF(10, 10), Qt::CopyAction, mime, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(&target, &d);
    delete mime;
}

QString png(const char* name) { return QString::fromStdString((decode_dir() / name).string()); }

QString last_log(app::MainWindow& w) {
    const QStringList lines = w.findChild<QPlainTextEdit*>("log")->toPlainText().split('\n', Qt::SkipEmptyParts);
    return lines.isEmpty() ? QString() : lines.last();
}

}  // namespace

TEST(AppCopy, TheCopiesAreThePagesTextsOfTheFrameOnScreen) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "copy");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.refresh_enabled();
    EXPECT_FALSE(w.action("copy-metrics")->isEnabled());   // !!state.metrics
    EXPECT_FALSE(w.action("copy-delivery")->isEnabled());  // no frame
    w.add_files({decode_dir() / "png8_rgb.png"});
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    w.measure_now();
    w.refresh_enabled();
    ASSERT_NE(w.measurement(), nullptr);
    EXPECT_TRUE(w.action("copy-metrics")->isEnabled());
    EXPECT_TRUE(w.action("copy-scopes")->isEnabled());
    EXPECT_TRUE(w.action("copy-delivery")->isEnabled());

    w.run("copy-metrics");
    const auto* me = w.measurement();
    EXPECT_EQ(QGuiApplication::clipboard()->text().toStdString(),
              metrics_value(me->measured.metrics, me->compose_ms).stringify());
    EXPECT_EQ(last_log(w), "copied measurements to the clipboard");
    w.run("copy-scopes");
    const json scopes = json::parse(QGuiApplication::clipboard()->text().toStdString());
    EXPECT_EQ(scopes["lo"].size(), 230u);
    EXPECT_EQ(scopes["histogram"].size(), 76u);
    EXPECT_EQ(last_log(w), "copied scope data to the clipboard");

    // Ungraded: region_ev is null; graded, it is the three regions.
    w.run("copy-delivery");
    json d = json::parse(QGuiApplication::clipboard()->text().toStdString());
    std::vector<std::string> keys;
    for (auto it = d.begin(); it != d.end(); ++it) keys.push_back(it.key());
    EXPECT_EQ(d["checkpoint"], "alpha");
    EXPECT_TRUE(d["step"].is_null());
    EXPECT_EQ(d["frame"], "png8_rgb.png");
    EXPECT_EQ(d["container"], "ACES 2065-1 (AP0)");
    EXPECT_EQ(d["diffuse_white_nits"], 203);
    EXPECT_TRUE(d["region_ev"].is_null());
    EXPECT_EQ(d["measurements"]["maxcll"], me->measured.metrics.maxcll);
    EXPECT_EQ(last_log(w), "copied delivery metadata to the clipboard");
    w.session().region_press(1, 100.0);
    w.session().region_move(130.0, false);
    w.session().region_release();
    w.run("container-linear");
    w.run("copy-delivery");
    d = json::parse(QGuiApplication::clipboard()->text().toStdString());
    ASSERT_TRUE(d["region_ev"].is_array());
    EXPECT_EQ(d["region_ev"][1]["label"], "speculars");
    EXPECT_NE(d["region_ev"][1]["ev"], 0.0);
    EXPECT_EQ(d["container"], "linear Rec.2020");
    // The text is the page's bytes for this state (core/copy_texts, held to the page).
    EXPECT_EQ(QGuiApplication::clipboard()->text().toStdString(), w.delivery_text());
}

TEST(AppCopy, SheetsAreThePagesAndAClickClosesThem) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "sheets");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.run("shortcuts");
    ASSERT_NE(w.sheet(), nullptr);
    EXPECT_TRUE(w.sheet()->isVisible());
    EXPECT_EQ(w.sheet()->findChild<QLabel*>("sheetTitle")->text(), "Keyboard");
    EXPECT_EQ(w.sheet()->findChild<QLabel*>("sheetClose")->text(), "Esc, or click anywhere, to close");
    // The title, the rows (key and what), the close line.
    EXPECT_EQ(w.sheet()->findChildren<QLabel*>().size(), qsizetype(2 + 2 * (shortcut_sheet().size() + 1))); // native F row
    QPointer<QDialog> first = w.sheet();
    QTest::mouseClick(first, Qt::LeftButton);
    QApplication::processEvents();
    QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    EXPECT_TRUE(first.isNull() || !first->isVisible());
    w.run("about");
    ASSERT_NE(w.sheet(), nullptr);
    bool device = false;
    const auto labels = w.sheet()->findChildren<QLabel*>();
    for (qsizetype i = 0; i + 1 < labels.size(); ++i)
        if (labels[i]->text() == "Running on") device = labels[i + 1]->text() == "CPU";   // $("device").textContent
    EXPECT_TRUE(device);
    QTest::keyClick(w.sheet(), Qt::Key_Escape);
    QApplication::processEvents();
    EXPECT_TRUE(w.sheet() == nullptr || !w.sheet()->isVisible());
}

TEST(AppOpen, ADropAddsFramesOpensAShotOrUsesAModel) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "drop");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.show();
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&w));
    // Enter lights the drop zone; the drop puts it out.
    auto* dz = w.findChild<QWidget*>("dropzone");
    ASSERT_NE(dz, nullptr);
    {
        QMimeData mime;
        mime.setUrls({QUrl::fromLocalFile(png("png8_rgb.png"))});
        QDragEnterEvent enter(QPoint(10, 10), Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(&w, &enter);
        EXPECT_TRUE(enter.isAccepted());
        EXPECT_EQ(dz->property("state").toString(), "hot");
        QDragLeaveEvent leave;
        QApplication::sendEvent(&w, &leave);
        EXPECT_EQ(dz->property("state").toString(), "");
    }
    // Two stills: the frames, the first shown (addFiles).
    drop(w, {png("png8_rgb.png"), png("png16_rgb.png")});
    EXPECT_EQ(w.frame_count(), 2u);
    EXPECT_EQ(w.current_index(), 0);
    EXPECT_EQ(last_log(w), "added 2 frames");
    // One more: appended, and it is the one shown.
    drop(w, {png("bmp24.bmp")});
    EXPECT_EQ(w.frame_count(), 3u);
    EXPECT_EQ(w.current_index(), 2);
    EXPECT_EQ(last_log(w), "added 1 frame");
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() && w.frame_fields()->residual.width() > 0 &&
                                      w.findChild<QLabel*>("shotCount")->text() == "3"; }));
    // A model package folder: used, the frames kept.
    drop(w, {QString::fromStdString((m->root / "beta").string())});
    ASSERT_TRUE(wait_for([&] { return !w.loading_model() && w.model_package().filename() == "beta"; }));
    EXPECT_EQ(w.frame_count(), 3u);
    // A folder of frames: that shot instead.
    drop(w, {QString::fromStdString(decode_dir().string())});
    EXPECT_GE(w.frame_count(), 10u);
    EXPECT_EQ(w.current_index(), 0);
}

TEST(AppOpen, RecentShotsNewestFirstAndClearable) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "recent");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.open_source(QString::fromStdString(decode_dir().string()));
    w.add_files({decode_dir() / "png8_rgb.png"});
    w.open_source(QString::fromStdString(decode_dir().string()));   // again: moves to the top, not twice
    const QStringList r = w.recent_sources();
    ASSERT_EQ(r.size(), 2);
    // Qt stores portable separators, including on Windows.
    EXPECT_EQ(r[0].toStdString(), decode_dir().lexically_normal().generic_string());
    EXPECT_TRUE(r[1].endsWith("png8_rgb.png"));
    // The menu: File > Open recent, the shots then Clear recent.
    QMenu* file = w.menuBar()->actions()[0]->menu();
    QMenu* recent = nullptr;
    for (QAction* a : file->actions())
        if (a->menu() && a->text() == "Open recent") recent = a->menu();
    ASSERT_NE(recent, nullptr);
    w.fill_recent();
    std::vector<QAction*> items;
    for (QAction* a : recent->actions())
        if (a->objectName().startsWith("recent:")) items.push_back(a);
    ASSERT_EQ(items.size(), 2u);
    EXPECT_EQ(recent->actions().last(), w.action("recent-clear"));
    EXPECT_TRUE(items[1]->text().startsWith("png8_rgb.png"));
    items[1]->trigger();   // a still: added to the frames
    EXPECT_EQ(last_log(w), "added 1 frame");
    w.run("recent-clear");
    EXPECT_TRUE(w.recent_sources().isEmpty());
    EXPECT_FALSE(w.action("recent-clear")->isEnabled());
}
#endif

TEST(AppSettings, TheWindowComesBackAsItWasLeft) {
    QSettings().clear();
    QSize left_at;
    {
        app::MainWindow w(false);
        w.set_workspace("simple");
        w.show_tab("deliver");
        w.run("rail-left");
        w.run("scopes");
        w.run("container-linear");
        w.findChild<QLineEdit*>("renderDir")->setText("/renders/show");
        w.findChild<QLineEdit*>("renderName")->setText("sh010");
        w.findChild<QComboBox*>("renderMode")->setCurrentIndex(1);
        w.findChild<QSpinBox*>("renderStart")->setValue(1001);
        w.show();
        ASSERT_TRUE(QTest::qWaitForWindowExposed(&w));
        w.resize(720, 560);   // inside the offscreen screen (800 x 800), so nothing clamps it
        QApplication::processEvents();
        left_at = w.size();   // what the window made of it (its minimum may be wider)
        w.close();   // closeEvent saves
    }
    app::MainWindow w(false);
    EXPECT_EQ(w.workspace(), "full");   // nothing restored until asked (the tests' windows start clean)
    w.restore_settings();
    EXPECT_EQ(w.workspace(), "simple");
    EXPECT_EQ(w.tab(), "deliver");
    EXPECT_FALSE(w.session().rail_left);
    EXPECT_TRUE(w.session().rail_right);
    EXPECT_FALSE(w.session().scopes_open);
    EXPECT_FALSE(w.action("rail-left")->isChecked());
    EXPECT_EQ(w.container(), "linear");
    EXPECT_EQ(w.findChild<QLineEdit*>("renderDir")->text(), "/renders/show");
    EXPECT_EQ(w.findChild<QLineEdit*>("renderName")->text(), "sh010");
    EXPECT_EQ(w.findChild<QComboBox*>("renderMode")->currentIndex(), 1);
    EXPECT_EQ(w.findChild<QSpinBox*>("renderStart")->value(), 1001);
    // The size (restoreGeometry keeps a window on its screen: the offscreen
    // one is 800 wide, narrower than the window's minimum, so the height is
    // the part that shows the round trip here).
    EXPECT_EQ(w.size().height(), left_at.height());
    QSettings().clear();
}


#ifdef RUDRA_HAVE_STILL_DECODE
// Phase 3 step 12: the scripted workflow runs end to end (with the fakes; the
// real model and the CLI comparison run in scripts/NATIVE_PHASE3_EXIT.ps1).
TEST(AppWorkflow, TheScriptedWorkflowPassesAndReportsEveryStep) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "workflow");
    const auto shot = fresh_dir("workflow-shot");
    std::filesystem::create_directories(shot);
    int k = 0;
    for (const char* f : {"png8_rgb.png", "png16_rgb.png", "bmp24.bmp", "jpeg_q92_420.jpg", "png8_grey.png"})
        std::filesystem::copy_file(decode_dir() / f, shot / ("f" + std::to_string(k++) + "_" + f));
    app::WorkflowArgs a;
    a.report = QString::fromStdString((shot / "report.json").string());
    a.package = QString::fromStdString((m->root / "beta").string());
    a.frames = QString::fromStdString(shot.string());
    a.out = QString::fromStdString((fresh_dir("workflow-out")).string());
    a.check_every = 1;
    EXPECT_EQ(app::run_workflow_check(w, a), 0);
    std::ifstream in(shot / "report.json");
    const json r = json::parse(in);
    EXPECT_EQ(r["verdict"], "PASS");
    for (const char* s : {"model", "open", "scrub", "grade", "compare", "measure", "master"})
        EXPECT_TRUE(r[s]["ok"].get<bool>()) << s;
    EXPECT_EQ(r["scrub"]["delivered"], 5);
    EXPECT_EQ(r["scrub"]["checked_against_decode"], 5);
    EXPECT_EQ(r["master"]["masters"].size(), 3u);
    // Each master names the frame, the files and the parameters the CLI takes.
    const auto& first = r["master"]["masters"][0];
    EXPECT_TRUE(std::filesystem::exists(first["exr"].get<std::string>()));
    auto q = master_request_from_json(first["params"].get<std::string>());
    ASSERT_TRUE(q);
    EXPECT_EQ(q->checkpoint, "beta");
    EXPECT_EQ(q->recovery_mode, "highlights");
    EXPECT_NEAR(q->strength, 1.2, 1e-12);
}

TEST(AppWorkflow, AnalyticWorkflowExportsWithoutLoadingAnyModel) {
    QSettings().clear();
    app::MainWindow w(false);
    const auto shot = fresh_dir("analytic-workflow-shot");
    std::filesystem::create_directories(shot);
    for (const char* file : {"png8_rgb.png", "png16_rgb.png", "bmp24.bmp"})
        std::filesystem::copy_file(decode_dir() / file, shot / file);
    app::WorkflowArgs a;
    a.analytic = true; a.check_every = 1;
    a.report = QString::fromStdString((shot / "report.json").string());
    a.frames = QString::fromStdString(shot.string());
    a.out = QString::fromStdString(fresh_dir("analytic-workflow-out").string());
    if (find_executable("ffmpeg") && find_executable("ffprobe"))
        a.movie = QString::fromStdString((std::filesystem::path(RUDRA_GOLDEN_DIR) / "video/clips/bars_709.mp4").string());
    EXPECT_EQ(app::run_workflow_check(w, a), 0);
    std::ifstream in(shot / "report.json"); const json r = json::parse(in);
    EXPECT_EQ(r["verdict"], "PASS"); EXPECT_EQ(r["mode"], "analytic");
    EXPECT_TRUE(r["model"]["skipped"].get<bool>());
    EXPECT_TRUE(r["unexpected_dialogs"].empty());
    EXPECT_TRUE(w.model_package().empty()); EXPECT_FALSE(w.using_model());
    for (const auto& master : r["master"]["masters"]) {
        std::ifstream side(master["sidecar"].get<std::string>());
        EXPECT_EQ(json::parse(side)["reconstruction"], "analytic (no model)");
    }
    if (!a.movie.isEmpty()) {
        EXPECT_TRUE(r["movie"]["ok"].get<bool>());
    }
}

TEST(AppWorkflow, StillInspectionFollowsTheDecodedFrameAndSourceReplacement) {
    QSettings().clear(); app::MainWindow w(false);
    const auto shot = fresh_dir("profile-inspection"); std::filesystem::create_directories(shot);
    QImage image(24, 16, QImage::Format_RGB32); image.fill(QColor(110, 140, 180));
    image.setColorSpace(QColorSpace::DisplayP3);
    const auto p3 = QString::fromStdString((shot / "a_p3.png").string());
    ASSERT_TRUE(image.save(p3));
    std::filesystem::copy_file(decode_dir() / "bmp24.bmp", shot / "b_untagged.bmp");
    w.open_source(QString::fromStdString(shot.string()), true);
    ASSERT_TRUE(wait_for([&] { return w.presented_index() == 0; }));
    EXPECT_TRUE(text_of(w, "sourceInspection").contains("Display P3"));
    w.run("next");
    ASSERT_TRUE(wait_for([&] { return w.presented_index() == 1; }));
    EXPECT_TRUE(text_of(w, "sourceInspection").contains("no colour profile"));
    EXPECT_FALSE(text_of(w, "sourceInspection").contains("Display P3"));
    w.run("first");
    ASSERT_TRUE(wait_for([&] { return w.presented_index() == 0; }));
    EXPECT_TRUE(text_of(w, "sourceInspection").contains("Display P3"));
    w.open_source(QString::fromStdString((decode_dir() / "bmp24.bmp").string()));
    ASSERT_TRUE(wait_for([&] { return w.presented_index() == 0; }));
    EXPECT_FALSE(text_of(w, "sourceInspection").contains("Display P3"));
}

TEST(AppWorkflow, UnexpectedDialogFailsAndReturnsInsteadOfWaitingForAnOperator) {
    QSettings().clear(); app::MainWindow w(false);
    const auto shot = fresh_dir("workflow-dialog"); std::filesystem::create_directories(shot);
    std::filesystem::copy_file(decode_dir() / "png8_rgb.png", shot / "frame.png");
    app::WorkflowArgs a; a.analytic = true; a.timeout_ms = 3000;
    a.frames = QString::fromStdString(shot.string());
    a.report = QString::fromStdString((shot / "report.json").string());
    a.out = QString::fromStdString(fresh_dir("workflow-dialog-out").string());
    QTimer::singleShot(0, &w, [&] { QMessageBox::warning(&w, "Test export failure", "Output unavailable"); });
    const auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(app::run_workflow_check(w, a), 1);
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(3));
    std::ifstream in(shot / "report.json"); const json r = json::parse(in);
    EXPECT_EQ(r["verdict"], "FAIL"); ASSERT_EQ(r["unexpected_dialogs"].size(), 1u);
    EXPECT_EQ(r["unexpected_dialogs"][0]["text"], "Output unavailable");
    EXPECT_FALSE(r.contains("master"));
}

TEST(AppWorkflow, ExistingOutputsAndAmbiguousModesAreRefused) {
    QSettings().clear(); app::MainWindow w(false);
    const auto root = fresh_dir("workflow-refusal"); std::filesystem::create_directories(root);
    app::WorkflowArgs a; a.analytic = true; a.frames = QString::fromStdString(decode_dir().string());
    a.out = QString::fromStdString(root.string());
    a.report = QString::fromStdString((root / "report.json").string());
    std::ofstream(root / "keep.txt") << "keep";
    EXPECT_EQ(app::run_workflow_check(w, a), 1);
    std::ifstream in(root / "report.json"); const json r = json::parse(in);
    EXPECT_FALSE(r["arguments"]["ok"].get<bool>());
    EXPECT_EQ(app::run_workflow_check(w, a), 1); // existing report remains intact
    std::ifstream unchanged(root / "report.json"); EXPECT_EQ(json::parse(unchanged), r);
    a.report = QString::fromStdString((root / "ambiguous.json").string());
    a.out = QString::fromStdString((root / "fresh").string()); a.package = "research-package";
    EXPECT_EQ(app::run_workflow_check(w, a), 1);
    EXPECT_FALSE(std::filesystem::exists(root / "fresh"));
    EXPECT_TRUE(std::filesystem::exists(root / "keep.txt"));
}
#endif

#ifdef RUDRA_HAVE_STILL_DECODE
// Phase 4 step 11: a movie is a shot, read a frame at a time; its exports go to
// the queue as batch.py queues, shown in the queue window to their end.
namespace {
std::filesystem::path movie(const char* clip) { return std::filesystem::path(RUDRA_GOLDEN_DIR) / "video" / "clips" / clip; }
bool have_ffmpeg() { return rudra::find_executable("ffmpeg") && rudra::find_executable("ffprobe"); }
}  // namespace

TEST(AppVideo, AMovieOpensAsAShotAndScrubs) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "movie");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    // A drop of a movie opens it as its own shot.
    drop(w, {QString::fromStdString(movie("bars_709.mp4").string())});
    ASSERT_TRUE(w.shot_is_video());
    EXPECT_EQ(w.frame_count(), 6u);
    EXPECT_EQ(w.shot_title(), "bars_709.mp4");
    ASSERT_TRUE(w.shot_video()->stream.has_value());
    EXPECT_TRUE(text_of(w, "sourceInspection").contains("Reported source:"));
    EXPECT_TRUE(text_of(w, "sourceInspection").contains("bt709"));
    EXPECT_EQ(w.frame_name(0), "bars_709_000001");
    EXPECT_EQ(w.frame_name(5), "bars_709_000006");
    EXPECT_TRUE(last_log(w).startsWith("opened bars_709.mp4: 6 frames at 23.976"));
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() && w.frame_fields()->residual.width() == 64; }));
    EXPECT_EQ(text_of(w, "shotTitle"), "bars_709.mp4");
    EXPECT_TRUE(text_of(w, "shotSub").contains("6 frames") && text_of(w, "shotSub").contains("23.976 fps")) << text_of(w, "shotSub").toStdString();
    // Scrub to the last frame: extracted on demand, delivered as itself.
    w.run("last");
    ASSERT_TRUE(wait_for([&] { return w.current_index() == 5 && std::filesystem::is_regular_file(w.frames()[5]); }));
    EXPECT_EQ(w.frames()[5].filename(), "000005.png");   // the Studio's cache name
    // A folder after it: no longer a movie.
    w.open_source(QString::fromStdString(decode_dir().string()));
    EXPECT_FALSE(w.shot_is_video());
    EXPECT_TRUE(text_of(w, "sourceInspection").contains("Still / frame sequence"));
    EXPECT_EQ(w.frame_name(0), w.frames()[0].filename().string());
}

TEST(AppVideo, TheExportTilesQueueAMovieAndTheQueueWindowFollowsIt) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "export-movie");
    ASSERT_TRUE(use(w, m->root / "beta"));
    w.set_use_model(true);   // the model's export: the fakes cannot run it, so it fails below
    const auto out = fresh_dir("export-movie-out");
    std::filesystem::create_directories(out);
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString(out.string()));
    w.findChild<QLineEdit*>("renderName")->setText("sh010");
    // Stills: the video tiles are shown, not offered.
    w.add_files({decode_dir() / "png8_rgb.png"});
    w.open_export_sheet();
    auto* sheet = static_cast<app::ExportSheet*>(w.findChild<QDialog*>("exportSheet"));
    ASSERT_NE(sheet, nullptr);
    EXPECT_FALSE(sheet->findChild<QWidget*>("hdr10")->isEnabled());
    sheet->pick("hdr10");
    EXPECT_EQ(sheet->picked(), "exr-aces");
    sheet->close();
    // A movie: the tiles are live, the sheet says what the file will be.
    w.open_source(QString::fromStdString(movie("h264_709_audio.mp4").string()));
    ASSERT_TRUE(w.shot_is_video());
    w.open_export_sheet();
    EXPECT_TRUE(sheet->findChild<QWidget*>("hdr10")->isEnabled());
    sheet->pick("hdr10");
    EXPECT_EQ(sheet->picked(), "hdr10");
    EXPECT_EQ(text_of(*sheet, "exportFrames"), "All 6, as a movie with its audio");
    EXPECT_TRUE(text_of(*sheet, "exportDest").endsWith("sh010.mp4"));
    EXPECT_EQ(text_of(*sheet, "exportSignal"), "PQ · Rec.2020 · HEVC 10-bit · HDR10 metadata");
    // ProRes 4444: the finishing master, a .mov; PQ, ACEScct or LogC4.
    EXPECT_FALSE(sheet->findChild<QWidget*>("exportEncodingRow")->isVisibleTo(sheet));   // HDR10: no encoding row
    EXPECT_TRUE(sheet->findChild<QWidget*>("prores4444")->isEnabled());
    sheet->pick("prores4444");
    EXPECT_EQ(sheet->picked(), "prores4444");
    EXPECT_TRUE(sheet->findChild<QWidget*>("exportEncodingRow")->isVisibleTo(sheet));
    EXPECT_EQ(sheet->video_format(), "prores4444");
    EXPECT_TRUE(text_of(*sheet, "exportDest").endsWith("sh010.mov"));
    EXPECT_EQ(text_of(*sheet, "exportSignal"), "PQ · Rec.2020 · ProRes 4444");
    sheet->set_encoding("logc4");
    EXPECT_EQ(sheet->video_format(), "prores4444_logc4");
    EXPECT_TRUE(text_of(*sheet, "exportDest").endsWith("sh010_logc4.mov"));
    EXPECT_EQ(text_of(*sheet, "exportSignal"), "LogC4 · ARRI Wide Gamut 4 · ProRes 4444 · scene-referred");
    EXPECT_TRUE(sheet->findChild<QLabel*>("exportHint")->isVisibleTo(sheet));
    sheet->pick("prores");
    sheet->set_encoding("acescct");
    EXPECT_EQ(sheet->video_format(), "prores422hq_acescct");
    EXPECT_EQ(text_of(*sheet, "exportSignal"), "ACEScct · AP1 · ProRes 422 HQ 10-bit · scene-referred");
    sheet->set_encoding("pq");
    EXPECT_EQ(sheet->video_format(), "prores422hq");
    sheet->pick("ffv1rgb16");
    EXPECT_EQ(sheet->video_format(), "ffv1rgb16");
    EXPECT_TRUE(text_of(*sheet, "exportDest").endsWith("sh010.mkv"));
    EXPECT_TRUE(text_of(*sheet, "exportSignal").contains("lossless 16-bit RGB"));
    EXPECT_FALSE(sheet->findChild<QWidget*>("exportEncodingRow")->isVisibleTo(sheet));
    sheet->pick("hdr10");
    EXPECT_EQ(sheet->video_format(), "hdr10");
    sheet->export_now();
    // The queue file, in batch.py's format, beside the master to be.
    const auto queue = out / "sh010.hdr10.queue.json";
    ASSERT_TRUE(std::filesystem::exists(queue)) << status_of(w).toStdString();
    {
        std::ifstream in(queue);
        const json q = json::parse(in);
        EXPECT_EQ(q["version"], 1);
        EXPECT_EQ(std::filesystem::path(q["defaults"]["checkpoint"].get<std::string>()).filename(), "manifest.json");
        EXPECT_EQ(q["jobs"][0]["options"]["format"], "hdr10");
        // The app's settings travel with the export.
        const auto settings = std::filesystem::path(q["jobs"][0]["options"]["grade"].get<std::string>());
        EXPECT_EQ(settings.filename(), "sh010.mp4.settings.json");
        auto g = master_request_from_json([&] {
            std::ifstream f(settings);
            std::stringstream ss;
            ss << f.rdbuf();
            return ss.str();
        }());
        ASSERT_TRUE(g);
        EXPECT_FALSE(g->analytic);
        EXPECT_EQ(g->checkpoint, "beta");
        EXPECT_EQ(std::filesystem::path(q["jobs"][0]["output"].get<std::string>()), out / "sh010.mp4");
        EXPECT_TRUE(rudra::load_queue(queue).ok());   // what the Python and the native runner both read
    }
    auto* qw = w.findChild<QDialog*>("queueWindow");
    ASSERT_NE(qw, nullptr);
    EXPECT_TRUE(qw->isVisible());
    // The fakes cannot run a package, so the job fails; the window says why and offers Resume.
    ASSERT_TRUE(wait_for([&] { return !w.video_queues().running() && app::read_queue_state(queue).status == "failed"; }));
    static_cast<app::QueueWindow*>(qw)->refresh();
    auto* row = qw->findChild<QWidget*>("queueRow");
    ASSERT_NE(row, nullptr);
    EXPECT_TRUE(row->findChild<QLabel*>("queueStatus")->text().startsWith("Failed: ")) << row->findChild<QLabel*>("queueStatus")->text().toStdString();
    EXPECT_TRUE(row->findChild<QPushButton*>("queueResume")->isVisibleTo(qw));
    EXPECT_FALSE(row->findChild<QPushButton*>("queueStop")->isVisibleTo(qw));
    // The same name again: refused, nothing written (the first queue stays the only one).
    EXPECT_TRUE(w.queue_video_export("hdr10"));   // no master yet, so a second queue is allowed ...
    EXPECT_TRUE(std::filesystem::exists(out / "sh010.hdr10 2.queue.json"));   // ... under its own name
    ASSERT_TRUE(wait_for([&] { return !w.video_queues().running(); }));
    { std::ofstream(out / "sh010.mp4") << "x"; }
    EXPECT_FALSE(w.queue_video_export("hdr10"));
    EXPECT_TRUE(last_log(w).contains("already there"));
    // Kept between runs.
    w.save_settings();
    app::MainWindow w2(false);
    w2.restore_settings();
    EXPECT_EQ(w2.video_queues().entries().size(), 2u);
    // The words a row gives each state.
    app::QueueJobState st;
    st.status = "running", st.phase = "inference", st.frames_done = 12, st.frames_total = 240;
    EXPECT_EQ(app::QueueWindow::describe(st, false, true), "Reconstructing, frame 12 of 240");
    st.status = "interrupted";
    EXPECT_EQ(app::QueueWindow::describe(st, false, false), "Stopped; Resume starts this clip again");
    st.status = "complete";
    EXPECT_EQ(app::QueueWindow::describe(st, false, false), "Complete, QC passed");
    QSettings().clear();
}
#endif

// ---- Projects, autosave, updates, help (product items 3 to 5) --------------

// The Log's last line (last_log above is only in builds with still decode).
QString log_tail(app::MainWindow& w) {
    const QStringList lines = w.findChild<QPlainTextEdit*>("log")->toPlainText().split('\n', Qt::SkipEmptyParts);
    return lines.isEmpty() ? QString() : lines.last();
}

TEST(AppProject, SaveAndOpenBringTheGradeAndTheModelBack) {
    QSettings().clear();
    const auto dir = fresh_dir("project");
    std::filesystem::path saved;
    std::string params;
    {
        app::MainWindow w(false);
        auto m = fake_models(w, "project-a");
        ASSERT_TRUE(use(w, m->root / "gamma"));
        w.set_use_model(true);   // the project runs the model: opening it brings that model back
        EXPECT_EQ(w.windowTitle(), "RUDRA");
        w.run("strength-up");
        w.run("mode-shadows");
        w.run("preserve");
        w.run("container-linear");
        w.session().peak_input(1.25);
        params = w.session().params_json();
        ASSERT_TRUE(w.save_project_to(dir / "my shot"));
        saved = w.project_path();
        EXPECT_EQ(saved, dir / "my shot.rudra");
        EXPECT_FALSE(w.project_dirty());
        EXPECT_EQ(w.windowTitle(), QString::fromUtf8("my shot.rudra[*] — RUDRA"));
        EXPECT_FALSE(w.isWindowModified());
        EXPECT_EQ(w.recent_sources().value(0).toStdString(), saved.generic_string());   // File > Open recent has it
        w.run("strength-up");
        EXPECT_TRUE(w.project_dirty());
        EXPECT_TRUE(w.isWindowModified());
    }
    app::MainWindow w(false);
    auto m = fake_models(w, "project-b");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    ASSERT_TRUE(w.open_project(saved));
    EXPECT_EQ(w.session().params_json(), params);
    EXPECT_EQ(w.session().container, "linear");
    EXPECT_DOUBLE_EQ(w.session().peak_ev, 1.25);
    // The project's model, not the one in use.
    ASSERT_TRUE(wait_for([&] { return !w.loading_model() && w.model_package().filename() == "gamma"; }));
    EXPECT_FALSE(w.project_dirty());
    EXPECT_EQ(w.project_path(), saved);
    EXPECT_TRUE(log_tail(w).contains("opened project") || w.findChild<QPlainTextEdit*>("log")->toPlainText().contains(
                                                              "opened project"));
    // Undo goes back to the grade before the project.
    w.run("undo");
    EXPECT_NE(w.session().params_json(), params);
    // A project from a newer RUDRA is refused, and says why; the session stays.
    const std::string now = w.session().params_json();
    std::ofstream(dir / "newer.rudra") << R"({"rudra_project": 99})";
    EXPECT_FALSE(w.open_project(dir / "newer.rudra"));
    EXPECT_TRUE(log_tail(w).contains("newer RUDRA")) << log_tail(w).toStdString();
    EXPECT_EQ(w.session().params_json(), now);
    EXPECT_EQ(w.project_path(), saved);
    // A headroom project is refused before changing the current shot/grade.
    Project headroom;
    headroom.grade.headroom_policy = HeadroomPolicy::Bt1886V1;
    headroom.source_kind = "movie"; headroom.sources = {dir / "headroom.mkv"};
    headroom.grade.source = "bt1886";
    headroom.anchor = headroom.carry_chroma = headroom.settle_grain = headroom.grade.clip_fill = false;
    ASSERT_TRUE(save_project(dir / "headroom.rudra", headroom));
    const bool was_dirty = w.project_dirty();
    EXPECT_FALSE(w.open_project(dir / "headroom.rudra"));
    EXPECT_TRUE(log_tail(w).contains("headroom projects")) << log_tail(w).toStdString();
    EXPECT_EQ(w.session().params_json(), now);
    EXPECT_EQ(w.project_path(), saved);
    EXPECT_EQ(w.project_dirty(), was_dirty);
    // A dropped project opens like File > Open project.
    w.open_paths({QString::fromStdString(saved.string())});
    EXPECT_EQ(w.session().params_json(), params);
}

TEST(AppProject, TheSessionIsAutosavedAndTheLastOneReopens) {
    QSettings().clear();
    const auto dir = fresh_dir("autosave");
    const auto file = dir / "autosave" / "last-session.rudra";
    std::string params;
    {
        app::MainWindow w(false);
        auto m = fake_models(w, "autosave-a");
        ASSERT_TRUE(use(w, m->root / "alpha"));
        w.set_use_model(true);
        w.set_autosave_path(file, 30);
        EXPECT_TRUE(w.reopen_path().empty());
        EXPECT_FALSE(w.action("reopen-session")->isEnabled());
        w.run("mode-highlights");
        ASSERT_TRUE(wait_for([&] { return std::filesystem::exists(file); }));
        EXPECT_TRUE(w.project_path().empty());   // the autosave is not the project
        EXPECT_EQ(w.windowTitle(), "RUDRA");
        // A change just before the window closes is saved by the close.
        w.run("strength-down");
        params = w.session().params_json();
        w.show();
        w.close();
    }
    auto q = load_project(file);
    ASSERT_TRUE(q) << q.error().message;
    EXPECT_EQ(q->grade.mode, "highlights");
    // The next run keeps that session aside before writing its own.
    app::MainWindow w(false);
    auto m = fake_models(w, "autosave-b");
    ASSERT_TRUE(use(w, m->root / "beta"));
    w.set_autosave_path(file, 30);
    EXPECT_EQ(w.reopen_path(), dir / "autosave" / "previous-session.rudra");
    EXPECT_FALSE(std::filesystem::exists(file));
    w.refresh_enabled();
    EXPECT_TRUE(w.action("reopen-session")->isEnabled());
    w.run("reopen-session");
    EXPECT_EQ(w.session().params_json(), params);
    EXPECT_TRUE(w.project_path().empty());
    EXPECT_TRUE(w.project_dirty());   // unsaved work until it is saved as a project
    ASSERT_TRUE(wait_for([&] { return !w.loading_model() && w.model_package().filename() == "alpha"; }));
    ASSERT_TRUE(wait_for([&] { return std::filesystem::exists(file); }));   // and autosaved again
}

TEST(AppProject, AProjectOpenedWhileAModelLoadsGetsItsOwnModel) {
    // A .rudra dropped or double-clicked while the start-up model still loads:
    // the project's model follows that load instead of being refused, and a
    // save meanwhile still names the project's model.
    QSettings().clear();
    const auto dir = fresh_dir("project-queued");
    std::filesystem::path saved;
    {
        app::MainWindow w(false);
        auto m = fake_models(w, "queued-a");
        ASSERT_TRUE(use(w, m->root / "gamma"));
        w.set_use_model(true);
        ASSERT_TRUE(w.save_project_to(dir / "g.rudra"));
        saved = w.project_path();
    }
    app::MainWindow w(false);
    auto m = fake_models(w, "queued-b");
    w.use_model(m->root / "alpha");   // not waited for
    ASSERT_TRUE(w.loading_model());
    ASSERT_TRUE(w.open_project(saved));
    EXPECT_EQ(w.current_project().package.filename(), "gamma");
    ASSERT_TRUE(wait_for([&] { return !w.loading_model() && w.model_package().filename() == "gamma"; }));
    EXPECT_EQ(w.current_project().package.filename(), "gamma");
}

#ifdef RUDRA_HAVE_STILL_DECODE
TEST(AppProject, TheShotAndTheFrameComeBack) {
    QSettings().clear();
    const auto dir = fresh_dir("project-shot");
    std::filesystem::create_directories(dir);
    std::filesystem::path saved;
    std::size_t count = 0;
    {
        app::MainWindow w(false);
        auto m = fake_models(w, "shot-a");
        ASSERT_TRUE(use(w, m->root / "alpha"));
        w.open_source(QString::fromStdString(decode_dir().string()), true);
        count = w.frame_count();
        ASSERT_GE(count, 3u);
        w.run("next");
        w.run("next");
        ASSERT_TRUE(w.save_project_to(dir / "folder.rudra"));
        auto p = load_project(dir / "folder.rudra");
        ASSERT_TRUE(p);
        EXPECT_EQ(p->source_kind, "folder");
        EXPECT_EQ(p->frame, 2);
    }
    app::MainWindow w(false);
    auto m = fake_models(w, "shot-b");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    ASSERT_TRUE(w.open_project(dir / "folder.rudra"));
    EXPECT_EQ(w.frame_count(), count);
    EXPECT_EQ(w.current_index(), 2);
    EXPECT_FALSE(w.project_dirty());
    // Stills added to the frames are saved by name, in order.
    w.run("close");
    w.add_files({decode_dir() / "png8_rgb.png", decode_dir() / "png8_rgb.png"});
    ASSERT_TRUE(w.save_project_to(dir / "stills.rudra"));
    auto p = load_project(dir / "stills.rudra");
    ASSERT_TRUE(p);
    EXPECT_EQ(p->source_kind, "files");
    EXPECT_EQ(p->sources.size(), 2u);
    // Footage that has gone is named, and the grade still opens.
    std::ofstream(dir / "gone.rudra") << R"({"rudra_project": 1, "shot": {"kind": "folder", "sources": [{"path": "/nowhere/plates"}]}, "grade": {"mode": "off"}})";
    ASSERT_TRUE(w.open_project(dir / "gone.rudra"));
    EXPECT_EQ(w.frame_count(), 0u);
    EXPECT_EQ(w.session().grade.mode, "off");
    EXPECT_TRUE(w.findChild<QPlainTextEdit*>("log")->toPlainText().contains("the project's footage is missing: /nowhere/plates"));
}
#endif

TEST(AppUpdates, VersionsCompareAsSemverSays) {
    using app::compare_versions;
    EXPECT_EQ(compare_versions("0.9.0", "0.9.0"), 0);
    EXPECT_EQ(compare_versions("v0.9.0", "0.9.0"), 0);
    EXPECT_GT(compare_versions("0.9.0", "0.9.0-beta.5"), 0);   // a release is newer than its betas
    EXPECT_GT(compare_versions("0.9.0-beta.10", "0.9.0-beta.9"), 0);
    EXPECT_GT(compare_versions("0.9.0-beta.6", "0.9.0-beta.5"), 0);
    EXPECT_GT(compare_versions("0.9.0-rc.1", "0.9.0-beta.12"), 0);
    EXPECT_GT(compare_versions("0.10.0", "0.9.9"), 0);
    EXPECT_GT(compare_versions("1.0.0-alpha.1", "1.0.0-alpha"), 0);
    EXPECT_LT(compare_versions("1.0.0-alpha.1", "1.0.0-alpha.beta"), 0);   // numeric before alphanumeric
    EXPECT_EQ(compare_versions("1.0.0+build.7", "1.0.0"), 0);
    EXPECT_LT(compare_versions("0.9.0-beta.5", "0.9.1-beta.1"), 0);
}

TEST(AppUpdates, TheNewestReleaseForThisChannel) {
    const std::string feed = R"([
        {"tag_name": "v0.9.0-beta.7", "html_url": "u7", "prerelease": true, "draft": true},
        {"tag_name": "v0.9.0-beta.6", "html_url": "u6", "prerelease": true},
        {"tag_name": "v0.8.2", "html_url": "u82", "prerelease": false},
        {"tag_name": "nightly", "html_url": "un", "prerelease": true},
        {"tag_name": "v0.9.0-beta.10", "html_url": "u10", "prerelease": true}])";
    const auto beta = app::newest_release(feed, true);
    ASSERT_TRUE(beta);
    EXPECT_EQ(beta->version, "0.9.0-beta.10");   // the draft and the non-version tag are skipped
    EXPECT_EQ(beta->url, "u10");
    const auto release = app::newest_release(feed, false);
    ASSERT_TRUE(release);
    EXPECT_EQ(release->version, "0.8.2");
    EXPECT_FALSE(app::newest_release("{\"message\": \"rate limited\"}", true));
    EXPECT_FALSE(app::newest_release("not json", true));
    EXPECT_TRUE(app::is_prerelease("0.9.0-beta.5"));
    EXPECT_FALSE(app::is_prerelease("0.9.0"));
}

TEST(AppUpdates, TheCheckSaysWhatIsNewer) {
    const QString keep = QCoreApplication::applicationVersion();
    QCoreApplication::setApplicationVersion("0.9.0-beta.5");
    const auto dir = fresh_dir("updates");
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "feed.json") << R"([{"tag_name": "v0.9.0-beta.6", "html_url": "https://example.test/b6", "prerelease": true}])";
    app::MainWindow w(false);
    w.set_updates_feed(QUrl::fromLocalFile(QString::fromStdString((dir / "feed.json").string())).toString());
    std::optional<QString> line;
    w.check_for_updates(false, [&](const QString& l) { line = l; });
    ASSERT_TRUE(wait_for([&] { return line.has_value(); }));
    EXPECT_EQ(*line, "RUDRA 0.9.0-beta.6 is available (this is 0.9.0-beta.5): https://example.test/b6");
    EXPECT_EQ(w.update_url(), "https://example.test/b6");
    EXPECT_EQ(w.statusBar()->currentMessage(), *line);
    // Up to date.
    QCoreApplication::setApplicationVersion("0.9.0-beta.6");
    line.reset();
    w.check_for_updates(false, [&](const QString& l) { line = l; });
    ASSERT_TRUE(wait_for([&] { return line.has_value(); }));
    EXPECT_EQ(*line, "RUDRA is up to date (0.9.0-beta.6)");
    EXPECT_TRUE(w.update_url().isEmpty());
    // No network: said, not thrown.
    w.set_updates_feed(QUrl::fromLocalFile(QString::fromStdString((dir / "absent.json").string())).toString());
    line.reset();
    w.check_for_updates(false, [&](const QString& l) { line = l; });
    ASSERT_TRUE(wait_for([&] { return line.has_value(); }));
    EXPECT_TRUE(line->startsWith("update check failed")) << line->toStdString();
    // Once a day at most, and never when turned off.
    QSettings().setValue("updates/auto", false);
    QSettings().remove("updates/last");
    w.maybe_check_for_updates();
    EXPECT_FALSE(QSettings().contains("updates/last"));
    QSettings().setValue("updates/auto", true);
    w.maybe_check_for_updates();
    EXPECT_TRUE(QSettings().contains("updates/last"));
    QCoreApplication::setApplicationVersion(keep);
}

TEST(AppHelp, GettingStartedTheGuideAndTips) {
    QSettings().clear();
    app::MainWindow w(false);
    w.run("getting-started");
    ASSERT_NE(w.sheet(), nullptr);
    EXPECT_EQ(w.sheet()->windowTitle(), "Getting started");
    auto* pages = w.sheet()->findChild<QStackedWidget*>("guidePages");
    auto* next = w.sheet()->findChild<QPushButton*>("guideNext");
    auto* back = w.sheet()->findChild<QPushButton*>("guideBack");
    ASSERT_NE(pages, nullptr);
    ASSERT_NE(next, nullptr);
    ASSERT_NE(back, nullptr);
    EXPECT_EQ(pages->count(), 5);
    EXPECT_EQ(pages->currentIndex(), 0);
    EXPECT_FALSE(back->isEnabled());
    if (const auto capture = qEnvironmentVariable("RUDRA_GUIDE_CAPTURE"); !capture.isEmpty()) {
        w.sheet()->resize(980, 640);
        QCoreApplication::processEvents();
        ASSERT_TRUE(w.sheet()->grab().save(capture));
    }
    next->click();
    EXPECT_EQ(pages->currentIndex(), 1);
    back->click();
    EXPECT_EQ(pages->currentIndex(), 0);
    auto* deliver = w.sheet()->findChild<QPushButton*>("guideStep4");
    ASSERT_NE(deliver, nullptr);
    deliver->click();
    EXPECT_EQ(pages->currentIndex(), 4);
    EXPECT_EQ(next->text(), "Start working");
    EXPECT_TRUE(deliver->isChecked());
    // Clicking instructional content no longer dismisses the guide.
    QMouseEvent click(QEvent::MouseButtonPress, QPointF(20, 20), QPointF(20, 20),
                      Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(w.sheet(), &click);
    EXPECT_TRUE(w.sheet()->isVisible());
    next->click();
    EXPECT_FALSE(w.sheet()->isVisible());
    w.run("getting-started");
    EXPECT_EQ(w.sheet()->findChild<QStackedWidget*>("guidePages")->currentIndex(), 0);
    w.sheet()->findChild<QPushButton*>("guideClose")->click();
    EXPECT_FALSE(w.sheet()->isVisible());
    w.run("getting-started");
    QTest::keyClick(w.sheet(), Qt::Key_Escape);
    EXPECT_FALSE(w.sheet()->isVisible());
    // Once per edition of the steps: a beta user (who saw the old ones) sees RUDRA 1.0's once.
    QSettings().setValue("help/gettingStartedShown", true);
    w.show_getting_started_once();
    EXPECT_EQ(QSettings().value("help/gettingStarted").toInt(), 2);
    EXPECT_TRUE(app::MainWindow::user_guide_url().endsWith("docs/USER_GUIDE.md"));
    EXPECT_EQ(w.action("user-guide")->shortcut(), QKeySequence(Qt::Key_F1));
    EXPECT_EQ(w.action("save-project")->shortcut(), QKeySequence(QKeySequence::Save));
    // Every control a new user asks about says what it does.
    for (const char* id : {"mode", "renderDir", "renderName", "renderMode", "btnMaster", "btnReprocess", "calibrate"}) {
        QWidget* c = w.findChild<QWidget*>(id);
        ASSERT_NE(c, nullptr) << id;
        EXPECT_FALSE(c->toolTip().isEmpty()) << id;
    }
    for (auto* s : w.findChildren<QSlider*>()) EXPECT_FALSE(s->toolTip().isEmpty()) << s->objectName().toStdString();
}

#ifdef RUDRA_HAVE_STILL_DECODE
// The 10 Oct 2026 review: "masters default to the analytic inverse" must be
// true of the app. By default nothing runs a model: no package is needed to
// open, view and master, and a loaded one is not called until it is turned on.
TEST(AppAnalytic, TheDefaultRunsNoModelAndSaysSo) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "analytic");
    EXPECT_FALSE(w.using_model());
    EXPECT_FALSE(w.action("use-model")->isChecked());
    EXPECT_EQ(w.findChild<QLabel*>("reconstructionMode")->text(), "Analytic reconstruction");
    // No package at all: the shot opens and its fields are the analytic ones
    // (with the clip fill off: since 12 Oct 2026 it fills clipped pixels
    // through the fields, and these test frames clip).
    ASSERT_TRUE(w.model_package().empty());
    w.session().grade.clip_fill = false;
    w.open_source(QString::fromStdString(decode_dir().string()), true);
    ASSERT_GE(w.frame_count(), 3u);
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    const Fields* f = w.frame_fields();
    for (float v : f->residual.span()) ASSERT_EQ(v, 0.0f);
    for (float v : f->highlight.span()) ASSERT_EQ(v, 0.0f);
    // A master with no package: written, and its sidecar and EXR say how.
    const auto root = fresh_dir("analytic-master");
    std::filesystem::create_directories(root);
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString(root.string()));
    w.findChild<QLineEdit*>("renderName")->setText("plain");
    w.findChild<QComboBox*>("renderMode")->setCurrentIndex(0);   // the frame on screen
    EXPECT_TRUE(w.action("master")->isEnabled());
    w.run("master");
    ASSERT_TRUE(wait_for([&] { return !w.mastering() && status_of(w).startsWith("Rendered"); })) << status_of(w).toStdString();
    std::ifstream sc(root / "plain.json");
    const json sidecar = json::parse(sc);
    EXPECT_EQ(sidecar["reconstruction"], "analytic (no model)");
    EXPECT_EQ(sidecar["checkpoint"], "");
    // A package loaded in the background is not called while analytic.
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.run("next");
    ASSERT_TRUE(wait_for([&] { return w.current_index() == 1 && w.frame_fields() != nullptr; }));
    QTest::qWait(50);
    EXPECT_EQ(*m->calls["alpha"], 0);
    EXPECT_EQ(w.frame_fields()->residual.at(0, 0, 0), 0.0f);
    // Turned on: the model runs, the frame on screen is composed again, and it is remembered.
    w.run("use-model");
    EXPECT_TRUE(w.using_model());
    EXPECT_TRUE(w.action("use-model")->isChecked());
    EXPECT_EQ(w.findChild<QLabel*>("reconstructionMode")->text(), "AI-assisted: alpha");
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() && w.frame_fields()->residual.at(0, 0, 0) == 0.1f; }));
    EXPECT_GE(*m->calls["alpha"], 1);
    EXPECT_TRUE(QSettings().value("model/use").toBool());
    EXPECT_TRUE(w.current_project().use_model);
    w.run("use-model");
    EXPECT_FALSE(w.using_model());
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() && w.frame_fields()->residual.at(0, 0, 0) == 0.0f; }));
}
#endif

#ifdef RUDRA_HAVE_STILL_DECODE
// Playback with the model (10 Oct 2026): inference ahead of the playhead, a
// smaller copy while playing, the stopped frame inferred again at full size;
// a movie decoded in order by one ffmpeg.
TEST(AppPlayback, PlaysWithTheModelAndRefinesWhereItStops) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "playback");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.set_use_model(true);
    w.set_playback_side(8);   // every decode frame is larger: all proxies while playing
    EXPECT_TRUE(w.action("playback-side-960")->isChecked() == false);
    w.open_source(QString::fromStdString(decode_dir().string()), true);
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    w.run("play");
    ASSERT_TRUE(w.playing());
    ASSERT_TRUE(wait_for([&] { return w.frame_proxy(); }));   // what plays is the proxy
    QTest::qWait(400);
    w.run("play");   // stop
    EXPECT_FALSE(w.playing());
    const auto st = w.playback_stats();
    EXPECT_GE(st.shown, 3) << "frames shown while playing";
    EXPECT_TRUE(w.findChild<QPlainTextEdit*>("log")->toPlainText().contains("the model at 8 px while playing"));
    EXPECT_TRUE(log_tail(w).startsWith("played ")) << log_tail(w).toStdString();
    // Stopped: the frame on screen is inferred again at full size.
    ASSERT_TRUE(wait_for([&] { return !w.frame_proxy(); }));
    // The settings are kept.
    EXPECT_EQ(QSettings().value("playback/side").toInt(), 8);
    w.run("playback-side-960");
    EXPECT_EQ(w.playback_side(), 960);
    EXPECT_TRUE(w.action("playback-side-960")->isChecked());
}

TEST(AppPlayback, TheMovieDecoderSkipsAheadWithoutStalling) {
    if (!rudra::find_executable("ffmpeg") || !rudra::find_executable("ffprobe")) GTEST_SKIP() << "no ffmpeg";
    const auto clip = std::filesystem::path(RUDRA_GOLDEN_DIR) / "video" / "clips" / "bars_709.mp4";
    auto seq = open_sequence(clip.string());
    ASSERT_TRUE(seq);
    const int n = seq->count();
    ASSERT_GE(n, 6);
    // Two frames of room: a skip of four would wait on a full buffer if the
    // frames behind it were kept.
    auto f = app::MovieFeeder::start(clip, *seq->fps, 0, n, 0, 2);
    ASSERT_TRUE(f) << f.error().message;
    ASSERT_TRUE((*f)->take(0));
    QElapsedTimer t;
    t.start();
    auto img = (*f)->take(5, 3000);
    ASSERT_TRUE(img);
    EXPECT_LT(t.elapsed(), 2500);
    EXPECT_FALSE((*f)->take(2, 10));   // behind: not from here
    // Stopped: a waiter is released at once.
    (*f)->stop();
    EXPECT_FALSE((*f)->take(n - 1, 3000));
}

TEST(AppPlayback, AMovieIsDecodedInOrderWhilePlaying) {
    if (!rudra::find_executable("ffmpeg") || !rudra::find_executable("ffprobe")) GTEST_SKIP() << "no ffmpeg";
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "playback-movie");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    const auto clip = std::filesystem::path(RUDRA_GOLDEN_DIR) / "video" / "clips" / "bars_709.mp4";
    w.open_source(QString::fromStdString(clip.string()));
    ASSERT_GE(w.frame_count(), 3u);
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    w.set_playback_realtime(false);   // every frame, for the count below
    w.run("play");
    ASSERT_TRUE(wait_for([&] { return w.playback_stats().shown >= int(w.frame_count()) - 1; }, 8000));
    w.run("play");
    const QString log = w.findChild<QPlainTextEdit*>("log")->toPlainText();
    EXPECT_FALSE(log.contains("playback decodes frame by frame")) << log.toStdString();
    EXPECT_TRUE(log.contains("; analytic"));   // the default reconstruction plays too
}
#endif

#ifdef RUDRA_HAVE_STILL_DECODE
// The default export (10 Oct 2026): analytic, with the grade, no package needed.
TEST(AppVideo, AnAnalyticMovieExportRunsWithoutAModelAndCarriesTheGrade) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    QSettings().clear();
    app::MainWindow w(false);
    ASSERT_FALSE(w.using_model());
    const auto out = fresh_dir("export-analytic");
    std::filesystem::create_directories(out);
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString(out.string()));
    w.findChild<QLineEdit*>("renderName")->setText("plain");
    w.open_source(QString::fromStdString(movie("h264_709.mp4").string()));
    ASSERT_TRUE(w.shot_is_video());
    w.run("mode-highlights");
    ASSERT_TRUE(w.queue_video_export("hlg")) << last_log(w).toStdString();
    const auto queue = out / "plain.hlg.queue.json";
    ASSERT_TRUE(wait_for([&] { return !w.video_queues().running() && app::read_queue_state(queue).status == "complete"; }, 60000))
        << app::read_queue_state(queue).status;
    std::ifstream in(out / "plain.mp4.json");
    const json rep = json::parse(in);
    EXPECT_EQ(rep["reconstruction"], "analytic (no model)");
    EXPECT_EQ(rep["recovery_mode"], "highlights");
    EXPECT_TRUE(rep["checkpoint"].is_null());
    // Held steady within a shot (the default), as playback holds it.
    std::ifstream qf(queue);
    EXPECT_EQ(json::parse(qf)["jobs"][0]["options"]["temporal-smoothing"], 0.25);
    EXPECT_DOUBLE_EQ(rep["temporal_smoothing"].get<double>(), 0.25);
    EXPECT_TRUE(rep.contains("temporal"));
}

TEST(AppVideo, EveryMovieDeliveryEncodingCompletesWithoutAModel) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    QSettings().clear();
    app::MainWindow w(false);
    const auto out = fresh_dir("all-movie-encodings");
    std::filesystem::create_directories(out);
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString(out.string()));
    w.open_source(QString::fromStdString(movie("h264_709.mp4").string()));
    ASSERT_TRUE(w.shot_is_video());
    for (const char* format : {"hdr10", "hlg", "ffv1rgb16", "rgb16mov", "prores422hq", "prores4444",
                               "prores422hq_acescct", "prores422hq_logc4",
                               "prores4444_acescct", "prores4444_logc4"}) {
        SCOPED_TRACE(format);
        const QString name = QString::fromLatin1(format);
        w.findChild<QLineEdit*>("renderName")->setText(name);
        ASSERT_TRUE(w.queue_video_export(name)) << last_log(w).toStdString();
        QString outputName = name;
        if (name.endsWith("_acescct")) outputName += "_acescct";
        else if (name.endsWith("_logc4")) outputName += "_logc4";
        const auto queue = out / (outputName.toStdString() + "." + format + ".queue.json");
        ASSERT_TRUE(wait_for([&] { return !w.video_queues().running() && app::read_queue_state(queue).status == "complete"; }, 60000))
            << app::read_queue_state(queue).status << " " << last_log(w).toStdString();
    }
}
#endif

#ifdef RUDRA_HAVE_STILL_DECODE
// Temporal consistency while playing (10 Oct 2026): a shot whose frames hardly
// change but whose shadow weight jumps frame to frame plays with the weight
// held, as its export holds it; stopped, the frame shows its own; off, every
// frame plays with its own.
TEST(AppPlayback, PlayingHoldsTheShadowWeightAcrossTheShot) {
    QSettings().clear();
    const auto dir = fresh_dir("steady-frames");
    std::filesystem::create_directories(dir);
    for (int i = 0; i < 12; ++i) {
        QImage img(64, 48, QImage::Format_RGB888);
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 64; ++x) {
                const int v = (x < 10 && y < 10) ? (i % 2 ? 250 : 5) : 60 + 2 * x;   // the marker, then a ramp
                img.setPixel(x, y, qRgb(v, v, v));
            }
        ASSERT_TRUE(img.save(QString::fromStdString((dir / QStringLiteral("f%1.png").arg(i, 3, 10, QChar('0')).toStdString()).string())));
    }
    app::MainWindow w(false);
    auto m = fake_models(w, "steady");
    m->pump->store(true);
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.set_use_model(true);
    w.set_playback_realtime(false);
    EXPECT_TRUE(w.steady_shots());
    EXPECT_TRUE(w.action("steady-shots")->isChecked());
    w.open_source(QString::fromStdString(dir.string()), true);
    ASSERT_EQ(w.frame_count(), 12u);
    ASSERT_TRUE(wait_for([&] { return w.frame_scalars() != nullptr; }));
    auto play = [&] {
        std::map<int, float> shown;   // frame -> the weight it was shown with
        w.run("first");
        EXPECT_TRUE(wait_for([&] { return w.current_index() == 0 && w.frame_scalars(); }));
        w.run("play");
        EXPECT_TRUE(wait_for([&] {
            if (w.frame_scalars()) shown[int(w.current_index())] = w.frame_scalars()->shadow_weight;
            return w.playback_stats().shown >= 8;
        }, 15000));
        w.run("play");   // stop
        return shown;
    };
    const std::map<int, float> held = play();
    int between = 0;
    for (const auto& [i, v] : held)
        if (i > 1 && v > 0.3f && v < 0.95f) ++between;
    EXPECT_GE(between, 3) << "held weights while playing";
    // Stopped: the frame on screen with its own weight.
    ASSERT_TRUE(wait_for([&] {
        const float v = w.frame_scalars()->shadow_weight;
        return v == 1.0f || v == 0.25f;
    }));
    // Off: each frame plays with its own, and the setting is kept.
    w.run("steady-shots");
    EXPECT_FALSE(w.steady_shots());
    EXPECT_FALSE(QSettings().value("video/steady").toBool());
    const std::map<int, float> own = play();
    for (const auto& [i, v] : own) EXPECT_TRUE(v == 1.0f || v == 0.25f) << i << ": " << v;
}
#endif

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    // Settings of their own, never the user's.
    QCoreApplication::setOrganizationName("FXTD Studios (tests)");
    QCoreApplication::setApplicationName("rudra_app_tests");
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       QString::fromStdString((std::filesystem::temp_directory_path() /
                                                 ("rudra-app-tests-settings-" + std::to_string(QCoreApplication::applicationPid())))
                                                    .string()));
    app::ThemeReport theme = app::apply_theme(app);
    g_theme = &theme;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// 16 Oct 2026: "in export HDR10, HLG and ProRes are not enabled" for a folder
// of frames. They are now: the frames are mastered as a linear EXR sequence
// beside the movie, then encoded as `rudra-native deliver` encodes them.
#ifdef RUDRA_HAVE_STILL_DECODE
TEST(AppVideo, AFolderOfFramesExportsToAMovie) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    QSettings().clear();
    const auto dir = fresh_dir("folder-movie-frames");
    std::filesystem::create_directories(dir);
    for (int i = 0; i < 4; ++i) {
        QImage img(64, 48, QImage::Format_RGB888);
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 64; ++x) img.setPixel(x, y, qRgb(20 + 3 * x, 40 + 2 * y, 30 + 10 * i));
        ASSERT_TRUE(img.save(QString::fromStdString((dir / QStringLiteral("f%1.png").arg(i, 3, 10, QChar('0')).toStdString()).string())));
    }
    app::MainWindow w(false);
    const auto out = fresh_dir("folder-movie-out");
    std::filesystem::create_directories(out);
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString(out.string()));
    w.findChild<QLineEdit*>("renderName")->setText("sh020");
    w.open_source(QString::fromStdString(dir.string()), true);
    ASSERT_EQ(w.frame_count(), 4u);
    ASSERT_TRUE(w.shot_is_frame_folder());
    EXPECT_FALSE(w.shot_is_video());
    w.open_export_sheet();
    auto* sheet = static_cast<app::ExportSheet*>(w.findChild<QDialog*>("exportSheet"));
    ASSERT_NE(sheet, nullptr);
    for (const char* id : {"hdr10", "hlg", "prores", "prores4444"}) EXPECT_TRUE(sheet->findChild<QWidget*>(id)->isEnabled()) << id;
    sheet->pick("prores");
    EXPECT_EQ(sheet->video_format(), "prores422hq");
    EXPECT_FALSE(sheet->findChild<QWidget*>("exportEncodingRow")->isVisibleTo(sheet));   // PQ only from frames
    EXPECT_TRUE(sheet->findChild<QWidget*>("exportFpsRow")->isVisibleTo(sheet));
    sheet->pick("hdr10");
    EXPECT_EQ(text_of(*sheet, "exportFrames"), "All 4, mastered as EXR, then as a movie (no audio)");
    EXPECT_TRUE(text_of(*sheet, "exportDest").endsWith("sh020_hdr10.mp4")) << text_of(*sheet, "exportDest").toStdString();
    auto* fps = sheet->findChild<QComboBox*>("exportFps");
    fps->setCurrentIndex(fps->findText("25"));
    EXPECT_DOUBLE_EQ(sheet->fps(), 25.0);
    sheet->export_now();
    ASSERT_TRUE(wait_for([&] { return !w.mastering() && !w.encoding_movie() &&
                                      std::filesystem::exists(out / "sh020_hdr10.mp4"); }, 60000))
        << status_of(w).toStdString();
    EXPECT_TRUE(status_of(w).startsWith("Movie written")) << status_of(w).toStdString();
    int exrs = 0;
    for (const auto& e : std::filesystem::directory_iterator(out / "sh020_hdr10_frames")) exrs += e.path().extension() == ".exr";
    EXPECT_EQ(exrs, 4);
    const auto movie = rudra::probe_video(out / "sh020_hdr10.mp4");
    ASSERT_TRUE(movie) << movie.error().message;
    ASSERT_EQ(movie->video_streams.size(), 1u);
    const auto& stream = movie->video_streams.front();
    EXPECT_EQ(stream.avg_frame_rate, "25/1");
    EXPECT_EQ(stream.color_primaries, "bt2020");
    EXPECT_EQ(stream.color_transfer, "smpte2084");
    EXPECT_EQ(stream.color_space, "bt2020nc");
    EXPECT_EQ(movie->frames.size(), 4u);
    // Nothing is replaced: a second export under the same name is refused.
    EXPECT_FALSE(w.export_frames_to_movie("hdr10", 25.0));
    EXPECT_TRUE(status_of(w).contains("already there")) << status_of(w).toStdString();
    // Exercise every remaining folder-delivery format, not only enabled tiles.
    for (const auto& format : {"hlg", "ffv1rgb16", "rgb16mov", "prores422hq", "prores4444"}) {
        SCOPED_TRACE(format);
        w.findChild<QLineEdit*>("renderName")->setText(format);
        ASSERT_TRUE(w.export_frames_to_movie(format, 24.0));
        const auto output = out / (std::string(format) + "_" + format +
                                  (std::string(format) == "ffv1rgb16" ? ".mkv" : std::string(format) == "hlg" ? ".mp4" : ".mov"));
        ASSERT_TRUE(wait_for([&] { return !w.mastering() && !w.encoding_movie() && std::filesystem::exists(output); }, 60000))
            << status_of(w).toStdString();
        ASSERT_TRUE(std::filesystem::exists(output)) << status_of(w).toStdString();
        auto probe = rudra::probe_video(output);
        ASSERT_TRUE(probe) << probe.error().message;
        ASSERT_EQ(probe->video_streams.size(), 1u);
        EXPECT_EQ(probe->frames.size(), 4u);
        EXPECT_EQ(probe->video_streams.front().avg_frame_rate, "24/1");
        EXPECT_EQ(probe->video_streams.front().codec_name, std::string(format) == "ffv1rgb16" ? "ffv1"
                                                           : std::string(format) == "hlg"      ? "hevc"
                                                           : std::string(format) == "rgb16mov" ? "rawvideo"
                                                                                               : "prores");
    }
    // A single still is EXR only.
    w.add_files({decode_dir() / "png8_rgb.png"});
    EXPECT_FALSE(w.shot_is_frame_folder());
    w.open_export_sheet();
    EXPECT_FALSE(sheet->findChild<QWidget*>("hdr10")->isEnabled());
}
#endif

// ---- Headroom preview in the window (E1, 17 Oct 2026) ------------------------
// The routing, refusal, undo/restore and playback lifecycle, over a fake
// renderer (the real decoder needs Windows source leases; the Windows workflow
// check runs it end to end).
#ifdef RUDRA_HAVE_STILL_DECODE
namespace {
class FakeHeadroomPort final : public rudra::HeadroomPreviewPort {
public:
    struct Shared {
        std::mutex mu;
        int opens = 0, seeks = 0, configures = 0;
        std::optional<rudra::Error> fail_render;
    };
    explicit FakeHeadroomPort(std::shared_ptr<Shared> sh) : sh_(std::move(sh)) {}
    struct Token final : Pending {
        int revision = 0;
        rudra::HeadroomShown shown;
    };
    rudra::Result<void> configure(const rudra::Session& s, bool, bool model, bool steady) override {
        std::lock_guard lock(mu_);
        ++revision_;
        { std::lock_guard l(sh_->mu); ++sh_->configures; }
        ok_ = false;
        if (s.grade.headroom_policy != rudra::HeadroomPolicy::Bt1886V1)
            return rudra::make_error(rudra::ErrorCode::Unsupported, "Headroom preview requires bt1886-headroom-v1");
        if (model || steady)
            return rudra::make_error(rudra::ErrorCode::Unsupported, "Headroom preview requires model and temporal hold off");
        if (s.anchor || s.carry_chroma || s.grade.clip_fill)
            return rudra::make_error(rudra::ErrorCode::Unsupported, "Headroom preview requires anchor, carry chroma and clip fill off");
        ok_ = true;
        exposure_ = s.grade.exposure_ev;
        return {};
    }
    rudra::Result<void> open(const std::filesystem::path&, rudra::FloatFrameRange r) override {
        std::lock_guard lock(mu_);
        if (!ok_) return rudra::make_error(rudra::ErrorCode::Unsupported, "no grade");
        { std::lock_guard l(sh_->mu); ++sh_->opens; }
        set(r);
        return {};
    }
    rudra::Result<void> seek(rudra::FloatFrameRange r) override {
        std::lock_guard lock(mu_);
        { std::lock_guard l(sh_->mu); ++sh_->seeks; }
        set(r);
        return {};
    }
    rudra::Result<std::shared_ptr<Pending>> render_next(std::chrono::milliseconds wait) override {
        std::unique_lock lock(mu_);
        if (!ok_) return rudra::make_error(rudra::ErrorCode::Unsupported, "no grade");
        {
            std::lock_guard l(sh_->mu);
            if (sh_->fail_render) {
                auto e = *sh_->fail_render;
                sh_->fail_render.reset();
                return e;
            }
        }
        if (next_ < 0) {
            cv_.wait_for(lock, wait);
            return rudra::make_error(rudra::ErrorCode::Busy, "nothing yet");
        }
        if (next_ >= end_) return rudra::make_error(rudra::ErrorCode::NotFound, "range done");
        auto t = std::make_shared<Token>();
        t->index = next_++;
        t->revision = revision_;
        auto img = std::make_shared<rudra::NetworkLinearImage>(2, 3, float(0.02 * std::exp2(exposure_)));
        t->shown.index = t->index;
        t->shown.width = 3;
        t->shown.height = 2;
        t->shown.model = img;
        t->shown.baseline = img;
        t->shown.rendered_bytes = 2 * 2 * 3 * 3 * sizeof(float);
        t->rendered_bytes = t->shown.rendered_bytes;
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        return std::shared_ptr<Pending>(t);
    }
    rudra::Result<void> present(const Pending& p, const std::function<void(const rudra::HeadroomShown&)>& d) override {
        std::lock_guard lock(mu_);
        const auto& t = static_cast<const Token&>(p);
        if (t.revision != revision_) return rudra::make_error(rudra::ErrorCode::Cancelled, "stale");
        d(t.shown);
        return {};
    }
    void cancel() override {
        std::lock_guard lock(mu_);
        ++revision_;
        next_ = -1;
        cv_.notify_all();
    }

private:
    void set(rudra::FloatFrameRange r) {
        ++revision_;
        next_ = r.first;
        end_ = r.first + r.count;
        cv_.notify_all();
    }
    std::shared_ptr<Shared> sh_;
    std::mutex mu_;
    std::condition_variable cv_;
    int revision_ = 0, next_ = -1, end_ = -1;
    bool ok_ = false;
    double exposure_ = 0.0;
};

std::shared_ptr<FakeHeadroomPort::Shared> fake_headroom(app::MainWindow& w) {
    auto sh = std::make_shared<FakeHeadroomPort::Shared>();
    w.set_headroom_port_factory([sh]() -> rudra::Result<std::unique_ptr<rudra::HeadroomPreviewPort>> {
        return std::unique_ptr<rudra::HeadroomPreviewPort>(new FakeHeadroomPort(sh));
    });
    return sh;
}

std::filesystem::path headroom_clip() {
    return std::filesystem::path(RUDRA_GOLDEN_DIR) / "video" / "clips" / "bars_709.mp4";
}

// The controls the v1 preview refuses, turned off as an artist would.
void headroom_ready_controls(app::MainWindow& w) {
    w.set_steady_shots(false);
    if (w.session().anchor) w.session().toggle_anchor();
    if (w.session().carry_chroma) w.session().toggle_carry_chroma();
    if (w.session().grade.clip_fill) w.session().toggle_clip_fill();
}
}  // namespace

TEST(AppHeadroom, ClosedUnlessTheWindowIsCapable) {
    QSettings().clear();
    app::MainWindow w(false);
    EXPECT_FALSE(w.headroom_capable());
    EXPECT_FALSE(w.set_headroom_preview(true));
    EXPECT_EQ(w.session().grade.headroom_policy, HeadroomPolicy::Disabled);
    EXPECT_FALSE(w.headroom_routed());
}

TEST(AppSuperWhites, TheMenuSaysWhyWithoutAMovie) {
    QSettings().clear();
    app::MainWindow w(false);
    w.run("keep-super-whites");
    EXPECT_FALSE(w.headroom_capable());
    EXPECT_EQ(w.session().grade.headroom_policy, HeadroomPolicy::Disabled);
    EXPECT_TRUE(w.problem().contains("Open a movie first")) << w.problem().toStdString();
    EXPECT_FALSE(w.action("keep-super-whites")->isChecked());
    EXPECT_FALSE(QSettings().value("experimental/superWhites", false).toBool());
}

TEST(AppSuperWhites, TheMenuOpensSupportTurnsOffWhatV1RefusesAndRoutes) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    QSettings().clear();
    app::MainWindow w(false);   // a window started without the choice
    auto sh = fake_headroom(w);
    w.open_source(QString::fromStdString(headroom_clip().string()));
    ASSERT_GE(w.frame_count(), 3u);
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    // Another Source is refused, and nothing changes.
    w.session().set_source("aces");
    w.run("keep-super-whites");
    EXPECT_TRUE(w.problem().contains("BT.1886")) << w.problem().toStdString();
    EXPECT_EQ(w.session().grade.headroom_policy, HeadroomPolicy::Disabled);
    EXPECT_FALSE(w.headroom_capable());
    w.session().set_source("bt1886");
    ASSERT_TRUE(w.session().grade.clip_fill);   // the default: turned off by the menu
    w.run("keep-super-whites");
    EXPECT_TRUE(w.headroom_capable());
    EXPECT_TRUE(QSettings().value("experimental/superWhites", false).toBool());
    EXPECT_EQ(w.session().grade.headroom_policy, HeadroomPolicy::Bt1886V1);
    EXPECT_FALSE(w.session().grade.clip_fill);
    EXPECT_FALSE(w.session().anchor);
    EXPECT_FALSE(w.session().carry_chroma);
    EXPECT_TRUE(w.action("keep-super-whites")->isChecked());
    EXPECT_TRUE(w.headroom_routed());
    ASSERT_TRUE(wait_for([&] { return w.headroom_frame() != nullptr; })) << w.headroom_problem().toStdString();
    // Off again: the usual reading, and the support stays for this window.
    w.run("keep-super-whites");
    EXPECT_EQ(w.session().grade.headroom_policy, HeadroomPolicy::Disabled);
    EXPECT_FALSE(w.headroom_routed());
    EXPECT_FALSE(w.action("keep-super-whites")->isChecked());
    EXPECT_TRUE(w.headroom_capable());
}

TEST(AppHeadroom, PausedFramesRouteThroughTheDriverAndUndoReturnsToLegacy) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    QSettings().clear();
    app::MainWindow w(false, true);
    auto sh = fake_headroom(w);
    w.open_source(QString::fromStdString(headroom_clip().string()));
    ASSERT_GE(w.frame_count(), 3u);
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    headroom_ready_controls(w);
    ASSERT_TRUE(w.set_headroom_preview(true));
    EXPECT_TRUE(w.headroom_routed());
    ASSERT_TRUE(wait_for([&] { return w.headroom_frame() != nullptr; })) << w.headroom_problem().toStdString();
    EXPECT_EQ(w.headroom_frame()->index, 0);
    EXPECT_EQ(w.presented_index(), 0);
    EXPECT_EQ(w.frame_fields(), nullptr) << "no legacy frame is kept on screen beside the preview";
    // E2: the readouts and scopes are the headroom frame's own (2 x 3 at 0.02 = 200 nits).
    ASSERT_TRUE(wait_for([&] { return w.measurement() != nullptr; }));
    EXPECT_EQ(w.measurement()->width, 3);
    EXPECT_NEAR(w.measurement()->measured.metrics.peak_nits, 200.0, 0.5);
    EXPECT_TRUE(w.measurement()->sdr8.empty()) << "nothing borrowed from the legacy path";
    // Stepping seeks the same prepared source.
    w.run("next");
    ASSERT_TRUE(wait_for([&] { return w.presented_index() == 1; }));
    { std::lock_guard l(sh->mu); EXPECT_EQ(sh->opens, 1); EXPECT_GE(sh->seeks, 1); }
    // A grade change renders the paused frame again with the new grade.
    const auto before = w.headroom_stats().presented;
    w.session().nudge_exposure(1);
    ASSERT_TRUE(wait_for([&] { return w.headroom_stats().presented > before; }));
    EXPECT_EQ(w.presented_index(), 1);
    EXPECT_NEAR(w.headroom_frame()->model->buffer().at(0, 0, 0), 0.04f, 1e-6f);
    // Undo the exposure, then the policy: the legacy engine shows the frame again.
    w.run("undo");
    ASSERT_TRUE(wait_for([&] { return w.headroom_frame() && w.headroom_frame()->model->buffer().at(0, 0, 0) < 0.03f; }));
    w.run("undo");
    EXPECT_EQ(w.session().grade.headroom_policy, HeadroomPolicy::Disabled);
    EXPECT_FALSE(w.headroom_routed());
    EXPECT_EQ(w.headroom_frame(), nullptr);
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr && w.presented_index() == 1; }));
    // Redo: back to the preview, on the same frame.
    w.run("redo");
    ASSERT_TRUE(wait_for([&] { return w.headroom_frame() != nullptr && w.presented_index() == 1; }));
    EXPECT_EQ(w.frame_fields(), nullptr);
}

TEST(AppHeadroom, ARefusedGradeShowsWhyAndNoLegacyPicture) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    QSettings().clear();
    app::MainWindow w(false, true);
    fake_headroom(w);
    w.open_source(QString::fromStdString(headroom_clip().string()));
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    // Steady shots, anchor, chroma and the clip fill are on by default: refused,
    // and nothing is changed to make it pass.
    ASSERT_TRUE(w.set_headroom_preview(true));
    ASSERT_TRUE(wait_for([&] { return !w.headroom_problem().isEmpty(); }));
    EXPECT_TRUE(w.headroom_problem().contains("temporal hold")) << w.headroom_problem().toStdString();
    EXPECT_TRUE(w.steady_shots());
    EXPECT_TRUE(w.session().anchor);
    EXPECT_EQ(w.headroom_frame(), nullptr);
    EXPECT_EQ(w.frame_fields(), nullptr);
    EXPECT_EQ(w.presented_index(), -1);
    QTest::qWait(100);   // a late legacy frame must not land either
    EXPECT_EQ(w.frame_fields(), nullptr);
    // Fixing the controls brings the preview without another request.
    headroom_ready_controls(w);
    ASSERT_TRUE(wait_for([&] { return w.headroom_frame() != nullptr; })) << w.headroom_problem().toStdString();
    EXPECT_TRUE(w.headroom_problem().isEmpty());
}

TEST(AppHeadroom, PlaysThroughTheDriverAndStopsOnTheFrameReached) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    QSettings().clear();
    app::MainWindow w(false, true);
    fake_headroom(w);
    w.open_source(QString::fromStdString(headroom_clip().string()));
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    headroom_ready_controls(w);
    ASSERT_TRUE(w.set_headroom_preview(true));
    ASSERT_TRUE(wait_for([&] { return w.headroom_frame() != nullptr; }));
    w.run("play");
    EXPECT_TRUE(w.playing());
    ASSERT_TRUE(wait_for([&] { return w.presented_index() >= 2; }));
    w.run("play");   // stop
    EXPECT_FALSE(w.playing());
    const int at = w.current_index();
    ASSERT_TRUE(wait_for([&] { return w.presented_index() == at && !w.headroom_stats().active; }));
    QTest::qWait(60);
    EXPECT_EQ(w.presented_index(), at) << "no frame of the stopped range lands afterwards";
    // To the end: playback stops by itself.
    w.run("play");
    ASSERT_TRUE(wait_for([&] { return !w.playing(); }, 15000));
    EXPECT_EQ(w.presented_index(), int(w.frame_count()) - 1);
    EXPECT_TRUE(log_tail(w).contains("end of the shot")) << log_tail(w).toStdString();
}

TEST(AppHeadroom, AFailureIsReportedWithoutFallingBack) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    QSettings().clear();
    app::MainWindow w(false, true);
    auto sh = fake_headroom(w);
    w.open_source(QString::fromStdString(headroom_clip().string()));
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    headroom_ready_controls(w);
    {
        std::lock_guard l(sh->mu);
        sh->fail_render = rudra::make_error(rudra::ErrorCode::IntegrityError, "The source changed while previewing");
    }
    ASSERT_TRUE(w.set_headroom_preview(true));
    ASSERT_TRUE(wait_for([&] { return !w.headroom_problem().isEmpty(); }));
    EXPECT_TRUE(w.headroom_problem().contains("source changed"));
    EXPECT_EQ(w.headroom_frame(), nullptr);
    EXPECT_EQ(w.frame_fields(), nullptr);
    // A new request (another frame) tries again.
    w.run("next");
    ASSERT_TRUE(wait_for([&] { return w.presented_index() == 1; }));
    EXPECT_TRUE(w.headroom_problem().isEmpty());
}

TEST(AppHeadroom, MovieExportsRunTheTransactionalHeadroomPath) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    QSettings().clear();
    const auto out = fresh_dir("headroom-export");
    std::filesystem::create_directories(out);
    // The real float decoder accepts YUV444; the generic playback fixture is
    // YUV420 and only worked with the fake preview port, not a real export.
    const auto source444 = out.parent_path() / (out.filename().string() + "-source444.mkv");
    auto prepared444 = rudra::run_tool({"ffmpeg", "-v", "error", "-y", "-i", headroom_clip().string(),
        "-c:v", "ffv1", "-pix_fmt", "yuv444p10le", "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709", source444.string()});
    ASSERT_TRUE(prepared444) << prepared444.error().message;
    app::MainWindow w(false, true);
    fake_headroom(w);
    w.open_source(QString::fromStdString(source444.string()));
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    headroom_ready_controls(w);
    ASSERT_TRUE(w.set_headroom_preview(true));
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString(out.string()));
    w.findChild<QLineEdit*>("renderName")->setText("graded");
    // Log masters and stills are not supported yet: refused, nothing written.
    EXPECT_FALSE(w.queue_video_export("prores4444_logc4"));
    EXPECT_TRUE(log_tail(w).contains("does not support the headroom policy")) << log_tail(w).toStdString();
    w.master();
    EXPECT_FALSE(w.mastering());
    EXPECT_TRUE(log_tail(w).contains("Master EXR does not support")) << log_tail(w).toStdString();
    // HDR10 runs the headroom export, never the legacy queue.
    ASSERT_TRUE(w.queue_video_export("hdr10"));
    EXPECT_TRUE(w.exporting_headroom());
    EXPECT_FALSE(w.video_queues().running());
    ASSERT_TRUE(wait_for([&] { return !w.exporting_headroom(); }, 60000));
#ifdef _WIN32
    EXPECT_TRUE(std::filesystem::exists(out / "graded.mp4")) << w.headroom_export_result().toStdString();
    EXPECT_TRUE(std::filesystem::exists(out / "graded.mp4.json"));
    w.findChild<QLineEdit*>("renderName")->setText("graded16");
    ASSERT_TRUE(w.queue_video_export("ffv1rgb16"));
    ASSERT_TRUE(wait_for([&] { return !w.exporting_headroom(); }, 60000));
    ASSERT_TRUE(std::filesystem::exists(out / "graded16.mkv")) << w.headroom_export_result().toStdString();
    EXPECT_TRUE(std::filesystem::exists(out / "graded16.mkv.json"));
    auto rgb16 = rudra::probe_video(out / "graded16.mkv");
    ASSERT_TRUE(rgb16);
    ASSERT_EQ(rgb16->video_streams.size(), 1u);
    EXPECT_EQ(rgb16->video_streams.front().pix_fmt, "gbrp16le");
    EXPECT_EQ(rgb16->video_streams.front().color_space, "gbr");
#else
    // Source leases are Windows-only: refused at preparation, nothing published or staged.
    EXPECT_TRUE(log_tail(w).contains("did not publish anything")) << log_tail(w).toStdString();
    EXPECT_TRUE(std::filesystem::is_empty(out));
#endif
}

TEST(AppHeadroom, AProjectRestoresThePreviewOnlyInACapableWindow) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    QSettings().clear();
    const auto dir = fresh_dir("headroom-project");
    {
        app::MainWindow w(false, true);
        fake_headroom(w);
        w.open_source(QString::fromStdString(headroom_clip().string()));
        ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
        headroom_ready_controls(w);
        ASSERT_TRUE(w.set_headroom_preview(true));
        w.run("next");
        w.run("next");
        ASSERT_TRUE(wait_for([&] { return w.presented_index() == 2; }));
        ASSERT_TRUE(w.save_project_to(dir / "hr.rudra")) << log_tail(w).toStdString();
    }
    {
        app::MainWindow w(false, true);
        fake_headroom(w);
        w.set_steady_shots(false);   // a window setting, not in the project
        ASSERT_TRUE(w.open_project(dir / "hr.rudra")) << log_tail(w).toStdString();
        EXPECT_EQ(w.session().grade.headroom_policy, HeadroomPolicy::Bt1886V1);
        ASSERT_TRUE(wait_for([&] { return w.presented_index() == 2 && w.headroom_frame(); }))
            << w.headroom_problem().toStdString();
        EXPECT_EQ(w.frame_fields(), nullptr);
        w.run("close");
        EXPECT_FALSE(w.headroom_routed());
        EXPECT_EQ(w.headroom_frame(), nullptr);
        EXPECT_FALSE(w.headroom_stats().active);
    }
    {
        app::MainWindow w(false);   // the default window still refuses it
        EXPECT_FALSE(w.open_project(dir / "hr.rudra"));
    }
}
TEST(AppHeadroom, ReplacingTheShotReopensOrLeavesTheRoute) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    QSettings().clear();
    app::MainWindow w(false, true);
    auto sh = fake_headroom(w);
    w.open_source(QString::fromStdString(headroom_clip().string()));
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    headroom_ready_controls(w);
    ASSERT_TRUE(w.set_headroom_preview(true));
    ASSERT_TRUE(wait_for([&] { return w.headroom_frame() != nullptr; }));
    // A folder of stills is not a movie: the legacy engine takes it, the policy stays in the grade.
    w.open_source(QString::fromStdString(decode_dir().string()), true);
    EXPECT_FALSE(w.headroom_routed());
    EXPECT_EQ(w.headroom_frame(), nullptr);
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    EXPECT_FALSE(w.headroom_stats().active);
    // A movie again: a fresh driver opens it.
    w.open_source(QString::fromStdString(headroom_clip().string()));
    ASSERT_TRUE(wait_for([&] { return w.headroom_frame() != nullptr; })) << w.headroom_problem().toStdString();
    EXPECT_EQ(w.frame_fields(), nullptr);
    std::lock_guard l(sh->mu);
    EXPECT_EQ(sh->opens, 2);
}
#endif


// ---- Deliver > Learned ACES 2.0 master… (aces_master.hpp, 26 Oct 2026) ------
#include <QCheckBox>
#include <QComboBox>
#include <QFileInfo>
#include <QProgressBar>
#include <QPushButton>
#include "aces_master.hpp"
#include "batch_queue.hpp"
#include "batch_window.hpp"
#include "edl.hpp"
#include <QJsonArray>
#include "aces_engine.hpp"
#include <QJsonObject>

namespace {

std::filesystem::path aces_tmp(const std::string& name) {
    const auto d = std::filesystem::temp_directory_path() / ("rudra-app-aces-" + name);
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

void write_text(const std::filesystem::path& p, const std::string& s) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream(p) << s;
}

json aces_manifest(bool parity, bool guard) {
    return json{{"contract", "rudra_native_aces_v2_staged"}, {"contract_major", 2},
                {"model_contract", {{"id", "aces2_test"}, {"full_context", "independent_pixel_rms"}, {"ocio_version", "2.5.2"},
                                    {"config", "cg-config-v4.0.0_aces-v2.0_ocio-v2.5"}, {"view", "ACES 2.0 - SDR 100 nits (Rec.709)"},
                                    {"display", "Rec.1886 Rec.709 - Display"}}},
                {"checkpoint_sha256", "f73f14c7e74d5bb54d0cbf949b94365f"}, {"g1_sha256", "dd10d83287efa1ab"}, {"g1_file", "rudra_g1.ts"},
                {"onnx_file", nullptr},
                {"clipped_smoothing", {{"name", "hybrid64"}, {"sigma", 64.0}, {"ramp", 1.0}, {"start", 0.0}, {"mode", "hybrid"}}},
                {"appearance_constraint", {{"max_display_error_codes", 1}, {"include_fully_clipped", true}, {"forbid_new_negative_y", guard}}},
                {"parity_pass", parity}, {"release_authorized", false}};
}

}  // namespace

TEST(AcesMaster, ParsesTheCommandsProgressLines) {
    auto p = app::parse_aces_progress("frame 001.png -> frame 001.exr  361 ms  remaining_display_errors=0");
    ASSERT_TRUE(p);
    EXPECT_EQ(p->source, "frame 001.png");
    EXPECT_EQ(p->exr, "frame 001.exr");
    EXPECT_DOUBLE_EQ(p->milliseconds, 361.0);
    EXPECT_EQ(p->remaining_display_errors, 0u);
    p = app::parse_aces_progress("a.tif -> a.exr  12.5 ms  remaining_display_errors=7\r");
    ASSERT_TRUE(p);
    EXPECT_EQ(p->remaining_display_errors, 7u);
    EXPECT_FALSE(app::parse_aces_progress("error [NotFound]: No stills found."));
    EXPECT_FALSE(app::parse_aces_progress(""));
}

TEST(AcesMaster, CollectsStillsAndRefusesSameNameExrs) {
    const auto d = aces_tmp("stills");
    for (const char* f : {"b.png", "a.jpg", "notes.txt", "c.TIF"}) write_text(d / f, "x");
    const auto stills = app::collect_aces_stills(d);
    ASSERT_EQ(stills.size(), 3u);
    EXPECT_EQ(stills[0].filename(), "a.jpg");
    EXPECT_FALSE(app::aces_name_collision(stills));
    write_text(d / "A.png", "x");   // a.jpg and A.png both want a.exr
    EXPECT_TRUE(app::aces_name_collision(app::collect_aces_stills(d)));
    EXPECT_EQ(app::collect_aces_stills(d / "b.png").size(), 1u);
    EXPECT_TRUE(app::collect_aces_stills(d / "notes.txt").empty());
    EXPECT_TRUE(app::collect_aces_stills(d / "missing").empty());
}

TEST(AcesMaster, CollectsAVideoAsItsOwnInput) {
    const auto d = aces_tmp("video_inputs");
    for (const char* f : {"a.png", "clip.MP4", "take.mov"}) write_text(d / f, "x");
    EXPECT_TRUE(app::is_aces_video(d / "clip.MP4"));
    EXPECT_TRUE(app::is_aces_video(d / "take.mov"));
    EXPECT_FALSE(app::is_aces_video(d / "a.png"));
    // A folder gives its stills only; a video is chosen on its own.
    ASSERT_EQ(app::collect_aces_stills(d).size(), 1u);
    ASSERT_EQ(app::collect_aces_stills(d / "clip.MP4").size(), 1u);
    EXPECT_EQ(app::collect_aces_stills(d / "clip.MP4")[0].filename(), "clip.MP4");
}

TEST(AcesMaster, ReadsAVideoAndOffersItsFramesForPreview) {
    if (QStandardPaths::findExecutable("ffmpeg").isEmpty() || QStandardPaths::findExecutable("ffprobe").isEmpty())
        GTEST_SKIP() << "ffmpeg and ffprobe are not on the PATH";
    QSettings().clear();
    const auto root = aces_tmp("video_dialog");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    const auto clip = root / "shot.mp4";
    QProcess ff;
    ff.start("ffmpeg", {"-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x32:rate=24", "-frames:v", "30",
                        "-pix_fmt", "yuv420p", "-c:v", "libx264", QString::fromStdString(clip.string())});
    ASSERT_TRUE(ff.waitForFinished(60000));
    ASSERT_EQ(ff.exitCode(), 0) << ff.readAllStandardError().toStdString();
    app::AcesMasterDialog d(nullptr, {root}, clip);
    const QString note = d.findChild<QLabel*>("acesFramesNote")->text();
    EXPECT_TRUE(note.startsWith("Video: 30 frames, 64 × 32, 24 fps, 8-bit")) << note.toStdString();
    auto* pick = d.findChild<QComboBox*>("acesPreviewFrame");
    ASSERT_EQ(pick->count(), 12);   // spread over the clip, first and last included
    EXPECT_EQ(pick->itemData(0, Qt::UserRole + 1).toInt(), 0);
    EXPECT_EQ(pick->itemData(11, Qt::UserRole + 1).toInt(), 29);
    EXPECT_EQ(pick->itemText(11), "shot.mp4 · frame 29");
    EXPECT_EQ(pick->itemData(5).toString(), QString::fromStdString(clip.string()));
    // An unreadable video says so, and offers nothing to preview or master.
    write_text(root / "broken.mp4", "not a video");
    d.set_input(root / "broken.mp4");
    EXPECT_TRUE(d.findChild<QLabel*>("acesFramesNote")->text().startsWith("This video cannot be read"));
    EXPECT_EQ(pick->count(), 0);
    EXPECT_FALSE(d.ready());
    QSettings().clear();
}

TEST(AcesMaster, FindsOnlyPackagesTheContractAccepts) {
    const auto root = aces_tmp("packages");
    write_text(root / "good" / "manifest.json", aces_manifest(true, true).dump());
    write_text(root / "failed_parity" / "manifest.json", aces_manifest(false, true).dump());
    write_text(root / "v1" / "manifest.json", json{{"contract", "rudra_native_v1"}}.dump());
    std::filesystem::create_directories(root / "empty");
    const auto found = app::find_aces_packages({root, root / "good", root / "missing"});
    ASSERT_EQ(found.size(), 1u);   // the good one, once, though two roots reach it
    EXPECT_EQ(found[0].root.filename(), "good");
    EXPECT_TRUE(found[0].luminance_guard);
    EXPECT_FALSE(found[0].has_onnx);
    EXPECT_FALSE(found[0].release_authorized);
}

TEST(AcesMaster, CommandArgumentsCarryTheChoices) {
    app::AcesRunSettings s;
    s.package = "pkg";
    s.input = "in";
    s.out = "out";
    s.runtime = "onnxruntime";
    s.device = "directml";
    s.luminance_guard = false;
    const QStringList a = app::aces_cli_arguments(s);
    EXPECT_EQ(a.first(), "aces-v2");
    EXPECT_TRUE(a.contains("--no-luminance-guard"));
    EXPECT_EQ(a.at(a.indexOf("--runtime") + 1), "onnxruntime");
    EXPECT_EQ(a.at(a.indexOf("--device") + 1), "directml");
    s.luminance_guard = true;
    EXPECT_TRUE(app::aces_cli_arguments(s).contains("--luminance-guard"));
    ASSERT_EQ(app::aces_run_targets().size(), 4u);
    EXPECT_EQ(app::aces_run_targets()[0].device, "cuda");
    // No log copy: no --log. A LogC4 choice carries the curve and the formats.
    EXPECT_FALSE(app::aces_cli_arguments(s).contains("--log"));
    s.log_curve = "logc4";
    s.log_formats = "dpx10,prores422hq";
    const QStringList l = app::aces_cli_arguments(s);
    EXPECT_EQ(l.at(l.indexOf("--log") + 1), "logc4");
    EXPECT_EQ(l.at(l.indexOf("--log-format") + 1), "dpx10,prores422hq");
    EXPECT_EQ(app::aces_log_choices().front().key, "");
    bool has_dpx = false;
    for (const auto& c : app::aces_log_choices()) has_dpx = has_dpx || c.key == "dpx10";
    EXPECT_TRUE(has_dpx);
}

TEST(AcesMaster, SumsUpTheRunsProvenance) {
    const auto d = aces_tmp("provenance");
    EXPECT_FALSE(app::read_aces_provenance(d));
    write_text(d / "provenance.json",
               json{{"device", "cuda"}, {"runtime", "libtorch"}, {"appearance_constraint_pass", true}, {"luminance_guard", true},
                    {"frames", json::array({{{"ms", 300.0}, {"reconstructed_fraction", 0.02}, {"remaining_display_errors", 0},
                                             {"new_negative_luminance", 0}},
                                            {{"ms", 400.0}, {"reconstructed_fraction", 0.04}, {"remaining_display_errors", 0},
                                             {"new_negative_luminance", 0}}})}}
                   .dump());
    auto s = app::read_aces_provenance(d);
    ASSERT_TRUE(s);
    EXPECT_EQ(s->frames, 2u);
    EXPECT_DOUBLE_EQ(s->mean_ms, 350.0);
    EXPECT_NEAR(s->mean_reconstructed_fraction, 0.03, 1e-12);
    EXPECT_TRUE(s->appearance_pass);
    const std::string t = app::aces_summary_text(*s);
    EXPECT_NE(t.find("2 masters written on cuda (libtorch), 350 ms a frame"), std::string::npos) << t;
    EXPECT_NE(t.find("within 1 code"), std::string::npos) << t;
    EXPECT_EQ(t.find("LogC4"), std::string::npos) << t;
    // Camera-log copies are named in the summary.
    auto j = json::parse(std::ifstream(d / "provenance.json"));
    j["log_delivery"] = {{"curve", "ARRI LogC4"},
                         {"files", json::array({{{"kind", "10-bit DPX per frame (RGB)"}}, {{"kind", "ProRes 422 HQ, 10-bit 4:2:2 (lossy)"}}})}};
    write_text(d / "provenance.json", j.dump());
    const std::string t2 = app::aces_summary_text(*app::read_aces_provenance(d));
    EXPECT_NE(t2.find("ARRI LogC4 copies: 10-bit DPX per frame (RGB), ProRes 422 HQ, 10-bit 4:2:2 (lossy)."), std::string::npos) << t2;
}

TEST(AcesMaster, OpensFromTheDeliverMenuAndWaitsForWhatItNeeds) {
    QSettings().clear();
    const auto root = aces_tmp("dialog");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    const auto frames = root / "frames";
    write_text(frames / "a.png", "x");
    app::MainWindow w(false);
    w.set_model_roots({root});
    ASSERT_NE(w.action("aces-master"), nullptr);
    EXPECT_EQ(w.action("aces-master")->text(), "Learned ACES 2.0 master…");
    w.run("aces-master");
    auto* d = static_cast<app::AcesMasterDialog*>(w.findChild<QDialog*>("acesMaster"));   // no Q_OBJECT: found as a QDialog
    ASSERT_NE(d, nullptr);
    auto* pkg = d->findChild<QComboBox*>("acesPackage");
    ASSERT_NE(pkg, nullptr);
    EXPECT_GE(pkg->findData(QString::fromStdString((root / "pkg").string())), 0);
    EXPECT_TRUE(d->findChild<QCheckBox*>("acesGuard")->isChecked());   // the package's default
    d->set_input(frames);
    // No ACES-capable rudra-native here: Start stays off and says why.
    d->set_cli(root / "missing-rudra-native");
    QString why;
    EXPECT_FALSE(d->ready(&why));
    EXPECT_EQ(why, "Choose a rudra-native built with ACES v2.");
    EXPECT_FALSE(d->findChild<QPushButton*>("acesStart")->isEnabled());
    EXPECT_FALSE(d->running());
    EXPECT_FALSE(app::cli_has_aces_v2(root / "missing-rudra-native"));
    // The suggested output folder sits beside the frames and does not exist yet.
    const QString out = d->findChild<QLineEdit*>("acesOut")->text();
    EXPECT_TRUE(out.endsWith("frames_aces_master")) << out.toStdString();
    EXPECT_FALSE(QFileInfo::exists(out));
}

TEST(AcesMaster, PutsTheConfiguredRuntimeFoldersFirstOnThePath) {
    QSettings().clear();
    const QString before = app::aces_child_environment().value("PATH");
    QSettings().setValue("aces/path_prepend", QStringList{"/opt/torch/lib", "/opt/ort/lib"});
    const QString path = app::aces_child_environment().value("PATH");
#ifdef _WIN32
    EXPECT_TRUE(path.startsWith("/opt/torch/lib;/opt/ort/lib;")) << path.toStdString();
#else
    EXPECT_TRUE(path.startsWith("/opt/torch/lib:/opt/ort/lib:")) << path.toStdString();
#endif
    EXPECT_TRUE(path.endsWith(before));
    QSettings().clear();
}

#ifndef _WIN32
// The whole run against a stand-in rudra-native (a shell script): progress from
// its per-frame lines, the summary from its provenance.json.
TEST(AcesMaster, RunsTheCommandAndReportsItsMasters) {
    QSettings().clear();
    const auto root = aces_tmp("run");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    for (const char* f : {"a.png", "b.png"}) write_text(root / "frames" / f, "x");
    const auto cli = root / "rudra-native";
    write_text(cli, R"(#!/bin/sh
[ "$1" = "aces-v2-serve" ] && exit 3
if [ "$#" -lt 3 ]; then echo "usage: rudra-native aces-v2 <package> <image|folder> --out <dir>" >&2; exit 64; fi
out=""; prev=""
for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done
mkdir -p "$out"
echo "a.png -> a.exr  100 ms  remaining_display_errors=0"
echo "b.png -> b.exr  300 ms  remaining_display_errors=0"
printf '%s' '{"device":"cpu","runtime":"libtorch","appearance_constraint_pass":true,"luminance_guard":true,"frames":[{"ms":100,"reconstructed_fraction":0.1,"remaining_display_errors":0,"new_negative_luminance":0},{"ms":300,"reconstructed_fraction":0.3,"remaining_display_errors":0,"new_negative_luminance":0}]}' > "$out/provenance.json"
exit 0
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    EXPECT_TRUE(app::cli_has_aces_v2(cli));
    app::AcesMasterDialog d(nullptr, {root}, root / "frames");
    d.set_cli(cli);
    QString why;
    ASSERT_TRUE(d.ready(&why)) << why.toStdString();
    EXPECT_TRUE(d.findChild<QPushButton*>("acesStart")->isEnabled());
    d.start();
    ASSERT_TRUE(wait_for([&] { return !d.running(); }));
    const QString status = d.findChild<QLabel*>("acesStatus")->text();
    EXPECT_TRUE(status.startsWith("2 masters written on cpu (libtorch), 200 ms a frame")) << status.toStdString();
    EXPECT_EQ(d.findChild<QProgressBar*>("acesProgress")->value(), 2);
    EXPECT_TRUE(d.findChild<QPlainTextEdit*>("acesLog")->toPlainText().contains("b.png -> b.exr"));
    // The summary stays; the folder now exists, so Start waits for a new one.
    EXPECT_FALSE(d.findChild<QPushButton*>("acesStart")->isEnabled());
    EXPECT_TRUE(d.findChild<QPushButton*>("acesOpenOut")->isEnabled());
    QSettings().clear();
}
#endif

#ifndef _WIN32
TEST(AcesMaster, CancelStopsTheCommandAndSaysWhatIsLeft) {
    QSettings().clear();
    const auto root = aces_tmp("cancel");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    for (const char* f : {"a.png", "b.png"}) write_text(root / "frames" / f, "x");
    const auto cli = root / "rudra-native";
    write_text(cli, R"(#!/bin/sh
[ "$1" = "aces-v2-serve" ] && exit 3
if [ "$#" -lt 3 ]; then echo "usage: rudra-native aces-v2 <package> <image|folder> --out <dir>" >&2; exit 64; fi
echo "a.png -> a.exr  100 ms  remaining_display_errors=0"
exec sleep 30
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::AcesMasterDialog d(nullptr, {root}, root / "frames");
    d.set_cli(cli);
    d.start();
    ASSERT_TRUE(wait_for([&] { return d.findChild<QProgressBar*>("acesProgress")->value() == 1; }));
    EXPECT_TRUE(d.findChild<QPushButton*>("acesCancel")->isEnabled());
    d.cancel();
    ASSERT_TRUE(wait_for([&] { return !d.running(); }, 5000));
    EXPECT_TRUE(d.findChild<QLabel*>("acesStatus")->text().startsWith("Cancelled after 1 of 2 frames"));
    QSettings().clear();
}
#endif

#ifndef _WIN32
// The model kept loaded: a stand-in `aces-v2-serve` (Python) answers preview,
// view and master over JSON lines; the dialog shows the frame and masters
// through it without starting another process.
TEST(AcesMaster, PreviewsAndMastersThroughTheLoadedModel) {
    QSettings().clear();
    const auto root = aces_tmp("engine");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    for (const char* f : {"a.png", "b.png"}) write_text(root / "frames" / f, "x");
    const auto cli = root / "rudra-native";
    write_text(cli, R"(#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
if a[:2] == ["aces-v2", "--help"]:
    sys.stderr.write("usage: rudra-native aces-v2 <package> <image|folder> --out <dir>\n"); sys.exit(64)
if not a or a[0] != "aces-v2-serve": sys.exit(2)
def say(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
say({"event": "ready", "device": "cpu", "runtime": "libtorch", "load_ms": 1234.0})
for line in sys.stdin:
    r = json.loads(line); op = r.get("op")
    if op == "quit": break
    if op == "preview":
        say({"id": r["id"], "ok": True, "width": 4, "height": 2, "bits": 8, "ms": 12.0, "reconstructed_fraction": 0.25,
             "peak": 9.5, "peak_plain": 3.0, "remaining_display_errors": 0, "new_negative_luminance": 0})
    elif op == "view":
        with open(r["out"], "wb") as f: f.write(b"P6\n4 2\n255\n" + bytes([200, 100, 50]) * 8)
        say({"id": r["id"], "ok": True, "path": r["out"], "width": 4, "height": 2, "ms": 3.0, "ev": r["ev"], "mode": r["mode"]})
    elif op == "master":
        os.makedirs(r["out"])
        for i, s in enumerate(r["sources"]):
            n = os.path.splitext(os.path.basename(s))[0]
            say({"id": r["id"], "progress": {"index": i + 1, "total": len(r["sources"]), "source": os.path.basename(s),
                                             "exr": n + ".exr", "ms": 50.0, "remaining_display_errors": 0}})
        json.dump({"device": "cpu", "runtime": "libtorch", "appearance_constraint_pass": True, "luminance_guard": True,
                   "frames": [{"ms": 50.0, "reconstructed_fraction": 0.25, "remaining_display_errors": 0,
                               "new_negative_luminance": 0}] * len(r["sources"])}, open(os.path.join(r["out"], "provenance.json"), "w"))
        say({"id": r["id"], "ok": True, "done": True, "cancelled": False, "frames": len(r["sources"])})
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::AcesMasterDialog d(nullptr, {root}, root / "frames");
    d.set_cli(cli);
    ASSERT_TRUE(wait_for([&] { return d.engine()->state() == app::AcesEngine::State::Ready; }))
        << d.engine()->status_text().toStdString();
    EXPECT_EQ(d.engine()->status_text(), "ACES model ready on cpu (libtorch), loaded in 1.2 s");
    // The first frame is previewed as soon as the model is ready.
    auto* image = d.findChild<QLabel*>("acesPreviewImage");
    ASSERT_TRUE(wait_for([&] { return !image->pixmap().isNull() && !d.preview_busy(); }));
    EXPECT_TRUE(d.findChild<QLabel*>("acesPreviewStats")->text().contains("25.00 % reconstructed"));
    // Exposure, mode and frame changes ask again.
    d.set_preview_ev(-4.0);
    d.set_preview_mode("split");
    d.preview_frame(1);
    ASSERT_TRUE(wait_for([&] { return !d.preview_busy(); }));
    EXPECT_EQ(d.findChild<QLabel*>("acesEvValue")->text(), "-4.0 EV");
    // Start masters through the loaded model: no second process, progress per frame.
    ASSERT_TRUE(d.ready());
    d.start();
    EXPECT_TRUE(d.running());
    ASSERT_TRUE(wait_for([&] { return !d.running(); }));
    EXPECT_TRUE(d.findChild<QLabel*>("acesStatus")->text().startsWith("2 masters written on cpu (libtorch), 50 ms a frame"))
        << d.findChild<QLabel*>("acesStatus")->text().toStdString();
    EXPECT_TRUE(d.findChild<QPlainTextEdit*>("acesLog")->toPlainText().contains("loaded model"));
    EXPECT_EQ(d.findChild<QProgressBar*>("acesProgress")->value(), 2);
    EXPECT_EQ(d.engine()->state(), app::AcesEngine::State::Ready);   // still loaded afterwards
    QSettings().clear();
}

TEST(AcesMaster, TheMainViewerShowsFramesThroughTheLoadedModel) {
#ifndef RUDRA_HAVE_STILL_DECODE
    GTEST_SKIP() << "no still decode in this build";
#else
    QSettings().clear();
    const auto root = aces_tmp("viewer_route");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    std::filesystem::create_directories(root / "frames");
    for (const char* f : {"a.png", "b.png"}) {
        QImage img(4, 2, QImage::Format_RGB888);
        img.fill(QColor(200, 120, 60));
        ASSERT_TRUE(img.save(QString::fromStdString((root / "frames" / f).string())));
    }
    const auto log = root / "requests.jsonl";
    const auto cli = root / "rudra-native";
    write_text(cli, std::string(R"(#!/usr/bin/env python3
import json, struct, sys
LOG = ")") + log.string() + R"("
def say(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
if sys.argv[1:2] != ["aces-v2-serve"]: sys.exit(64)
say({"event": "ready", "device": "cpu", "runtime": "libtorch", "load_ms": 10.0})
for line in sys.stdin:
    r = json.loads(line)
    open(LOG, "a").write(line)
    if r.get("op") == "quit": break
    if r.get("op") == "preview":
        for k, v in (("model_out", 2.0), ("baseline_out", 1.0)):
            if k in r: open(r[k], "wb").write(struct.pack("<24f", *([v] * 24)))
        say({"id": r["id"], "ok": True, "width": 4, "height": 2, "bits": 8, "ms": 5.0, "reconstructed_fraction": 0.5,
             "peak": 2.0, "peak_plain": 1.0, "remaining_display_errors": 0, "new_negative_luminance": 0})
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::MainWindow w(false);
    w.set_model_roots({root});
    app::AcesEngine::Config c;
    c.cli = cli;
    c.package = root / "pkg";
    c.device = "cpu";
    w.aces_engine()->start(c);
    ASSERT_TRUE(wait_for([&] { return w.aces_engine()->state() == app::AcesEngine::State::Ready; }));
    w.open_source(QString::fromStdString((root / "frames").string()), true);
    ASSERT_TRUE(wait_for([&] { return w.aces_shown() == 0; })) << w.problem().toStdString();
    EXPECT_TRUE(w.frame_info().contains("learned ACES 2.0, 4x2")) << w.frame_info().toStdString();
    w.run("next");
    ASSERT_TRUE(wait_for([&] { return w.aces_shown() == 1; }));
    // The viewer's requests leave the ACES window's kept frame alone and ask for the raw master.
    std::ifstream in(log);
    std::string line, last;
    while (std::getline(in, line))
        if (line.find("\"preview\"") != std::string::npos) last = line;
    EXPECT_NE(last.find("\"keep\":false"), std::string::npos) << last;
    EXPECT_NE(last.find("model_out"), std::string::npos);
    EXPECT_NE(last.find("b.png"), std::string::npos);
    // Off: the legacy path shows frames again, nothing more goes to the model.
    w.set_aces_viewer(false);
    EXPECT_FALSE(w.aces_routed());
    QSettings().clear();
#endif
}

TEST(AcesMaster, EngineThatCannotLoadSaysWhy) {
    QSettings().clear();
    const auto root = aces_tmp("engine-fail");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    const auto cli = root / "rudra-native";
    write_text(cli, R"(#!/bin/sh
[ "$1" = "aces-v2" ] && { echo "usage: rudra-native aces-v2 <package> <image|folder> --out <dir>" >&2; exit 64; }
echo '{"event":"error","error":"This build has no LibTorch.","detail":""}'
exit 2
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::AcesEngine e(nullptr);
    int changes = 0;
    QObject owner;
    e.listen(&owner, [&] { ++changes; });
    e.start({cli, root / "pkg", "libtorch", "cpu", 2});
    ASSERT_TRUE(wait_for([&] { return e.state() == app::AcesEngine::State::Failed; }));
    EXPECT_EQ(e.status_text(), "ACES model failed: This build has no LibTorch.");
    EXPECT_GE(changes, 2);   // loading, then failed
    bool answered = false;
    e.request(QJsonObject{{"op", "preview"}}, [&](const QJsonObject& o) { answered = !o.value("ok").toBool(); });
    EXPECT_TRUE(answered);   // not ready: answered at once, with the reason
    QSettings().clear();
}
#endif

// ---------------------------------------------------------------------------
// The Deliver tab (27 Oct 2026, deliver_choice.hpp): learned ACES 2.0 or
// analytic; still, sequence or MOV; EXR 32/16-bit or DPX 10-bit; a 10- or
// 16-bit movie.
// ---------------------------------------------------------------------------

TEST(Deliver, ChoicesMapToTheEnginesFields) {
    app::DeliverChoice c;   // a still, 32-bit float EXR in ACES 2065-1
    auto f = app::aces_deliver_fields(c);
    EXPECT_EQ(f.value("exr").toString(), "float");
    EXPECT_EQ(f.value("exr_gamut").toString(), "ap0");
    EXPECT_FALSE(f.contains("log"));
    c.image = "exr16";
    c.gamut = "acescg";
    f = app::aces_deliver_fields(c);
    EXPECT_EQ(f.value("exr").toString(), "half");
    EXPECT_EQ(f.value("exr_gamut").toString(), "acescg");
    c.image = "dpx10";
    c.encoding = "pq";
    f = app::aces_deliver_fields(c);
    EXPECT_EQ(f.value("exr").toString(), "none");
    EXPECT_EQ(f.value("log").toString(), "pq1000");
    EXPECT_EQ(f.value("log_formats").toString(), "dpx10");
    c.output = "movie";
    c.bits = 16;
    c.encoding = "logc4";
    f = app::aces_deliver_fields(c);
    EXPECT_EQ(f.value("log_formats").toString(), "mov16");
    EXPECT_EQ(f.value("exr").toString(), "none");
    c.keep_exr = true;
    c.bits = 10;
    c.codec = "prores4444";
    f = app::aces_deliver_fields(c);
    EXPECT_EQ(f.value("log_formats").toString(), "prores4444");
    EXPECT_EQ(f.value("exr").toString(), "float");
    // The analytic movie formats the export queue takes.
    EXPECT_EQ(app::analytic_movie_format(c), "prores4444_logc4");
    c.codec = "prores422hq";
    c.encoding = "pq";
    EXPECT_EQ(app::analytic_movie_format(c), "prores422hq");
    c.bits = 16;
    EXPECT_EQ(app::analytic_movie_format(c), "rgb16mov");
}

TEST(Deliver, TheAnalyticSourceKeepsWhatItCanWrite) {
    app::DeliverChoice c;
    c.image = "dpx10";
    c.gamut = "acescg";
    c.output = "movie";
    c.bits = 16;
    c.encoding = "logc4";
    // A single still has no movie.
    EXPECT_EQ(app::deliver_fit(c, true, false, false).output, "still");
    // The learned source takes everything.
    EXPECT_EQ(app::deliver_fit(c, true, true, false), c);
    // The analytic master: EXR 32-bit, ACES 2065-1 or linear Rec.2020, a 16-bit movie in PQ.
    const auto a = app::deliver_fit(c, false, true, false);
    EXPECT_EQ(a.image, "exr32");
    EXPECT_EQ(a.gamut, "linear");
    EXPECT_EQ(a.encoding, "pq");
    c.bits = 10;
    EXPECT_EQ(app::deliver_fit(c, false, true, false).encoding, "logc4");   // a movie's ProRes takes a log
    EXPECT_EQ(app::deliver_fit(c, false, false, true).encoding, "pq");      // a folder's movie is PQ
    // The analytic EXR keeps its words; the button too.
    app::DeliverChoice d;
    const auto sig = app::deliver_signal(app::deliver_fit(d, false, false, false), false);
    EXPECT_EQ(sig.container, "OpenEXR — ACES 2065-1");
    EXPECT_EQ(sig.primaries, "AP0 (ST 2065-4)");
    EXPECT_EQ(app::deliver_button_text(d, false), "Master EXR");
    EXPECT_EQ(app::deliver_button_text(d, true), "Render EXR");
    d.output = "movie";
    d.bits = 16;
    EXPECT_EQ(app::deliver_signal(d, true).container, "MOV — RGB 16-bit");
    EXPECT_EQ(app::deliver_button_text(d, true), "Render MOV");
}

#ifdef RUDRA_HAVE_STILL_DECODE
TEST(Deliver, RendersStillsSequencesAndMoviesThroughTheLoadedModel) {
    QSettings().clear();
    const auto root = aces_tmp("deliver_route");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    std::filesystem::create_directories(root / "frames");
    std::filesystem::create_directories(root / "renders");
    for (const char* f : {"a.png", "b.png", "c.png"}) {
        QImage img(4, 2, QImage::Format_RGB888);
        img.fill(QColor(200, 120, 60));
        ASSERT_TRUE(img.save(QString::fromStdString((root / "frames" / f).string())));
    }
    const auto log = root / "requests.jsonl";
    const auto cli = root / "rudra-native";
    write_text(cli, std::string(R"(#!/usr/bin/env python3
import json, os, struct, sys
LOG = ")") + log.string() + R"("
def say(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
if sys.argv[1:2] != ["aces-v2-serve"]: sys.exit(64)
say({"event": "ready", "device": "cpu", "runtime": "libtorch", "load_ms": 10.0})
for line in sys.stdin:
    r = json.loads(line)
    open(LOG, "a").write(line)
    op = r.get("op")
    if op == "quit": break
    if op == "preview":
        for k, v in (("model_out", 2.0), ("baseline_out", 1.0)):
            if k in r: open(r[k], "wb").write(struct.pack("<24f", *([v] * 24)))
        say({"id": r["id"], "ok": True, "width": 4, "height": 2, "bits": 8, "ms": 5.0, "reconstructed_fraction": 0.5,
             "peak": 2.0, "peak_plain": 1.0, "remaining_display_errors": 0, "new_negative_luminance": 0})
    elif op == "master":
        os.makedirs(r["out"], exist_ok=True)
        n = len(r.get("frames") or r["sources"])
        for i in range(n):
            say({"id": r["id"], "progress": {"index": i + 1, "total": n, "source": "x", "exr": "f%d.exr" % i, "ms": 40.0,
                                             "remaining_display_errors": 0}})
        json.dump({"device": "cpu", "runtime": "libtorch", "appearance_constraint_pass": True, "luminance_guard": True,
                   "frames": [{"ms": 40.0, "reconstructed_fraction": 0.5, "remaining_display_errors": 0,
                               "new_negative_luminance": 0}] * n}, open(os.path.join(r["out"], "provenance.json"), "w"))
        say({"id": r["id"], "ok": True, "done": True, "cancelled": False, "frames": n})
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::MainWindow w(false);
    w.set_model_roots({root});
    // Before the model is ready, the Deliver tab is the analytic master's.
    EXPECT_FALSE(w.deliver_aces());
    EXPECT_EQ(w.findChild<QPushButton*>("btnMaster")->text(), "Master EXR");
    EXPECT_FALSE(find<app::Seg>(w, "deliverSource")->button("aces")->isEnabled());
    app::AcesEngine::Config c;
    c.cli = cli;
    c.package = root / "pkg";
    c.device = "cpu";
    w.aces_engine()->start(c);
    ASSERT_TRUE(wait_for([&] { return w.aces_engine()->state() == app::AcesEngine::State::Ready; }));
    w.open_source(QString::fromStdString((root / "frames").string()), true);
    ASSERT_TRUE(wait_for([&] { return w.aces_shown() == 0; })) << w.problem().toStdString();
    EXPECT_TRUE(w.deliver_aces());
    EXPECT_EQ(w.findChild<QPushButton*>("btnMaster")->text(), "Render EXR");
    EXPECT_TRUE(w.findChild<QWidget*>("deliverAnalyticOptions")->isHidden());
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString((root / "renders").string()));
    w.findChild<QLineEdit*>("renderName")->setText("SH010");
    w.findChild<QSpinBox*>("renderStart")->setValue(1001);
    auto last_master = [&] {
        std::ifstream in(log);
        std::string line, last;
        while (std::getline(in, line))
            if (line.find("\"master\"") != std::string::npos) last = line;
        return last.empty() ? json() : json::parse(last);
    };
    // A still: the frame on screen, ACES 2065-1 as the ST 2065-4 container (half), numbered from the start frame.
    w.run("master");
    ASSERT_TRUE(wait_for([&] { return !w.aces_delivering(); }));
    auto r = last_master();
    ASSERT_FALSE(r.is_null());
    EXPECT_EQ(r["exr"], "half");
    EXPECT_EQ(r["exr_gamut"], "ap0");
    EXPECT_EQ(r["name"], "SH010");
    EXPECT_EQ(r["start"], 1001);
    ASSERT_EQ(r["sources"].size(), 1u);
    EXPECT_NE(r["sources"][0].get<std::string>().find("a.png"), std::string::npos);
    // Into the File's folder, as Nuke's Write: SH010.1001.exr (#### by default), made when missing.
    EXPECT_EQ(r["out"].get<std::string>(), (root / "renders").string());
    EXPECT_EQ(r["into_existing"], true);
    EXPECT_EQ(r["create_dirs"], true);
    EXPECT_EQ(r["padding"], 4);
    EXPECT_EQ(r["separator"], ".");
    EXPECT_EQ(w.findChild<QLineEdit*>("renderFile")->text().toStdString(),
              QDir::toNativeSeparators(QString::fromStdString((root / "renders" / "SH010.####.exr").string())).toStdString());
    EXPECT_TRUE(status_of(w).startsWith("Rendered 1 frame(s)")) << status_of(w).toStdString();
    // A sequence of 10-bit DPX in ACEScct: every frame, a new folder (nothing replaced).
    w.set_deliver("output", "sequence");
    w.set_deliver("image", "dpx10");
    w.set_deliver("encoding", "acescct");
    EXPECT_EQ(w.findChild<QComboBox*>("renderMode")->currentIndex(), 1);
    EXPECT_FALSE(w.findChild<QWidget*>("deliverEncodingRow")->isHidden());
    EXPECT_TRUE(w.findChild<QWidget*>("deliverGamutRow")->isHidden());
    EXPECT_EQ(w.findChild<QPushButton*>("btnMaster")->text(), "Render DPX");
    w.run("master");
    ASSERT_TRUE(wait_for([&] { return !w.aces_delivering(); }));
    r = last_master();
    EXPECT_EQ(r["exr"], "none");
    EXPECT_EQ(r["log"], "acescct");
    EXPECT_EQ(r["log_formats"], "dpx10");
    EXPECT_EQ(r["sources"].size(), 3u);
    EXPECT_EQ(r["out"].get<std::string>(), (root / "renders").string());
    EXPECT_TRUE(status_of(w).startsWith("Rendered 3 frame(s)")) << status_of(w).toStdString();
    // A 16-bit MOV in PQ (the ACES 2.0 HDR output) at 25 fps.
    w.set_deliver("output", "movie");
    w.set_deliver("bits", "16");
    w.set_deliver("encoding", "pq");
    w.set_deliver("fps", "25");
    EXPECT_FALSE(w.findChild<QWidget*>("deliverBitsRow")->isHidden());
    EXPECT_TRUE(w.findChild<QWidget*>("deliverCodecRow")->isHidden());
    EXPECT_FALSE(w.findChild<QWidget*>("deliverFpsRow")->isHidden());
    EXPECT_EQ(w.findChild<QLabel*>("containerField")->text(), "MOV — RGB 16-bit");
    w.run("master");
    ASSERT_TRUE(wait_for([&] { return !w.aces_delivering(); }));
    r = last_master();
    EXPECT_EQ(r["log"], "pq1000");
    EXPECT_EQ(r["log_formats"], "mov16");
    EXPECT_EQ(r["exr"], "none");
    EXPECT_DOUBLE_EQ(r["fps"].get<double>(), 25.0);
    // 10-bit: ProRes 422 HQ.
    w.set_deliver("bits", "10");
    EXPECT_EQ(app::aces_deliver_fields(w.deliver_effective()).value("log_formats").toString(), "prores422hq");
    // The analytic source again: its EXR master and words.
    w.set_deliver("source", "analytic");
    w.set_deliver("output", "still");
    EXPECT_FALSE(w.deliver_aces());
    EXPECT_EQ(w.findChild<QPushButton*>("btnMaster")->text(), "Master EXR");
    EXPECT_EQ(w.findChild<QLabel*>("containerField")->text(), "OpenEXR — ACES 2065-1");
    EXPECT_FALSE(w.findChild<QWidget*>("deliverAnalyticOptions")->isHidden());
    QSettings().clear();
}
#endif

// ---------------------------------------------------------------------------
// The AI fill (RUDRA layer 3, 27 Oct 2026, genfill_job.hpp): the learned EXR
// render's copy filled by the Wan2.1-VACE LoRA (rudra.genfill.run, a Python
// process speaking JSON lines). The fakes below stand in for the Python.
// ---------------------------------------------------------------------------
#include "genfill_job.hpp"

TEST(Deliver, TheAiFillIsForTheLearnedExrOnly) {
    app::DeliverChoice c;
    c.ai_fill = true;
    EXPECT_TRUE(app::ai_fill_possible(c, true));
    EXPECT_TRUE(app::deliver_fit(c, true, false, false).ai_fill);
    EXPECT_FALSE(app::deliver_fit(c, false, false, false).ai_fill);   // the analytic master
    c.image = "exr16";
    EXPECT_TRUE(app::deliver_fit(c, true, false, true).ai_fill);
    c.image = "dpx10";
    EXPECT_FALSE(app::deliver_fit(c, true, false, true).ai_fill);
    c.image = "exr32";
    c.output = "movie";
    EXPECT_FALSE(app::deliver_fit(c, true, true, false).ai_fill);
    // Thresholds stay in range, the shadow one below the highlight one.
    app::DeliverChoice d;
    d.ai.tau_high = 999;
    d.ai.tau_low = 300;
    d.ai.steps = 0;
    d.ai.seed = -4;
    const auto f = app::deliver_fit(d, true, false, false);
    EXPECT_EQ(f.ai.tau_high, 256);
    EXPECT_EQ(f.ai.tau_low, 64);
    EXPECT_EQ(f.ai.steps, 1);
    EXPECT_EQ(f.ai.seed, 0);
}

TEST(GenFill, ArgumentsCarryTheChoicesAndTheFolderIsNew) {
    app::GenFillSetup s{"/py", "/repo", "/models/wan", "/runs/lora.safetensors"};
    app::GenFillOptions o;
    o.tau_high = 245;
    o.tau_low = -1;
    o.seed = 7;
    o.steps = 12;
    o.prompt_highlight = "  warm sun  ";
    const QStringList a = app::genfill_arguments(s, o, "/r/SH010", "/r/SH010_aifill");
    const QStringList want{"-m", "rudra.genfill.run", "--master", "/r/SH010", "--out", "/r/SH010_aifill", "--base", "/models/wan",
                           "--lora", "/runs/lora.safetensors", "--tau-high", "245", "--tau-low", "-1", "--seed", "7",
                           "--steps", "12", "--json", "--prompt-highlight", "warm sun"};
    EXPECT_EQ(a, want);
    const auto root = aces_tmp("genfill-out");
    const QString m = QString::fromStdString((root / "SH010").string());
    std::filesystem::create_directories(root / "SH010");
    EXPECT_EQ(app::genfill_out_dir(m), m + "_aifill");
    std::filesystem::create_directories(root / "SH010_aifill");
    EXPECT_EQ(app::genfill_out_dir(m), m + "_aifill_2");
    write_text(root / "still.001001.exr", "x");
    EXPECT_EQ(app::genfill_out_dir(QString::fromStdString((root / "still.001001.exr").string())),
              QString::fromStdString((root / "still.001001").string()) + "_aifill");
}

#ifndef _WIN32
namespace {
// A stand-in for the Python: `<python> -m rudra.genfill.run ... --out X ... --json`.
std::filesystem::path fake_genfill(const std::filesystem::path& root, const std::string& body) {
    write_text(root / "repo" / "rudra" / "genfill" / "run.py", "# fake");
    write_text(root / "models" / "wan" / "transformer" / "config.json", "{}");
    write_text(root / "runs" / "rudra_genfill.safetensors", "x");
    const auto py = root / "python";
    write_text(py, "#!/bin/sh\nout=\"\"\nwhile [ $# -gt 0 ]; do [ \"$1\" = \"--out\" ] && out=\"$2\"; shift; done\n" + body);
    std::filesystem::permissions(py, std::filesystem::perms::owner_all);
    QSettings st;
    st.setValue("genfill/python", QString::fromStdString(py.string()));
    st.setValue("genfill/repo", QString::fromStdString((root / "repo").string()));
    st.setValue("genfill/base", QString::fromStdString((root / "models" / "wan").string()));
    st.setValue("genfill/lora", QString::fromStdString((root / "runs" / "rudra_genfill.safetensors").string()));
    return py;
}
const char* kFillOk = R"(mkdir -p "$out"
echo '{"event":"ready","device":"cuda","load_seconds":3.0}'
echo 'a library line that is not JSON'
echo '{"event":"progress","window":1,"windows":1,"step":15,"steps":30}'
echo '{"event":"frame","index":1,"of":2,"file":"'"$out"'/a.exr","generated_fraction":0.01}'
echo '{"event":"frame","index":2,"of":2,"file":"'"$out"'/b.exr","generated_fraction":0.02}'
echo '{"event":"done","ok":true,"frames":2,"generated_fraction":0.015}'
)";
}  // namespace

TEST(GenFill, SetupSaysWhatIsMissing) {
    QSettings().clear();
    const auto root = aces_tmp("genfill-setup");
    fake_genfill(root, kFillOk);
    QString why;
    auto s = app::genfill_setup(&why);
    ASSERT_TRUE(s) << why.toStdString();
    EXPECT_TRUE(why.isEmpty());
    std::filesystem::remove(root / "runs" / "rudra_genfill.safetensors");
    EXPECT_FALSE(app::genfill_setup(&why));
    EXPECT_TRUE(why.contains("no trained LoRA")) << why.toStdString();
    std::filesystem::remove_all(root / "models");
    EXPECT_FALSE(app::genfill_setup(&why));
    EXPECT_TRUE(why.contains("Wan2.1-VACE")) << why.toStdString();
    QSettings().setValue("genfill/repo", QString::fromStdString((root / "nowhere").string()));
    EXPECT_FALSE(app::genfill_setup(&why));
    EXPECT_TRUE(why.contains("repository")) << why.toStdString();
    QSettings().clear();
}

TEST(GenFill, TheJobReadsTheFillersLines) {
    QSettings().clear();
    const auto root = aces_tmp("genfill-job");
    fake_genfill(root, kFillOk);
    auto s = app::genfill_setup();
    ASSERT_TRUE(s);
    app::GenFillJob job(nullptr);
    std::vector<QString> events;
    bool finished = false, ok = false;
    QJsonObject last;
    const QString out = QString::fromStdString((root / "SH010_aifill").string());
    ASSERT_TRUE(job.start(*s, {}, QString::fromStdString((root / "SH010").string()), out,
                          [&](const QJsonObject& o) { events.push_back(o.value("event").toString()); },
                          [&](bool k, const QJsonObject& o) { finished = true; ok = k; last = o; }));
    ASSERT_TRUE(wait_for([&] { return finished; }));
    EXPECT_TRUE(ok);
    EXPECT_EQ(events, (std::vector<QString>{"ready", "progress", "frame", "frame", "done"}));
    EXPECT_EQ(last.value("frames").toInt(), 2);
    EXPECT_TRUE(std::filesystem::exists(root / "SH010_aifill"));
    // A failing fill says why (its own error line).
    fake_genfill(root, "echo '{\"event\":\"error\",\"ok\":false,\"error\":\"FileExistsError\",\"detail\":\"exists\"}'\nexit 2\n");
    finished = false;
    ASSERT_TRUE(job.start(*app::genfill_setup(), {}, "m", "o", {}, [&](bool k, const QJsonObject& o) { finished = true; ok = k; last = o; }));
    ASSERT_TRUE(wait_for([&] { return finished; }));
    EXPECT_FALSE(ok);
    EXPECT_EQ(last.value("error").toString(), "FileExistsError");
    // A crash without a line: the exit code and stderr.
    fake_genfill(root, "echo 'Traceback: boom' >&2\nexit 1\n");
    finished = false;
    ASSERT_TRUE(job.start(*app::genfill_setup(), {}, "m", "o", {}, [&](bool k, const QJsonObject& o) { finished = true; ok = k; last = o; }));
    ASSERT_TRUE(wait_for([&] { return finished; }));
    EXPECT_FALSE(ok);
    EXPECT_TRUE(last.value("error").toString().contains("exit 1"));
    EXPECT_TRUE(last.value("detail").toString().contains("boom"));
    // Cancel asks it (a "cancel" line on stdin); it stops at its next step.
    fake_genfill(root, "echo '{\"event\":\"ready\"}'\nread line\n[ \"$line\" = cancel ] && exit 0\nsleep 30\n");
    finished = false;
    ASSERT_TRUE(job.start(*app::genfill_setup(), {}, "m", "o", {}, [&](bool k, const QJsonObject& o) { finished = true; ok = k; last = o; }));
    EXPECT_TRUE(job.running());
    QString why;
    EXPECT_FALSE(job.start(*app::genfill_setup(), {}, "m", "o", {}, {}, &why));   // one at a time
    EXPECT_TRUE(why.contains("already running"));
    job.cancel();
    ASSERT_TRUE(wait_for([&] { return finished; }));
    EXPECT_TRUE(last.value("cancelled").toBool());
    // One that does not listen: a second Stop kills it.
    fake_genfill(root, "echo '{\"event\":\"ready\"}'\nexec sleep 30\n");
    finished = false;
    ASSERT_TRUE(job.start(*app::genfill_setup(), {}, "m", "o", {}, [&](bool k, const QJsonObject& o) { finished = true; ok = k; last = o; }));
    job.cancel();
    QCoreApplication::processEvents();
    EXPECT_TRUE(job.running());
    job.cancel();
    ASSERT_TRUE(wait_for([&] { return finished; }));
    EXPECT_TRUE(last.value("cancelled").toBool());
    QSettings().clear();
}

TEST(GenFill, TheDeliverTabOffersItAndRunsIt) {
    QSettings().clear();
    const auto root = aces_tmp("genfill-tab");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    const auto cli = root / "rudra-native";
    write_text(cli, R"(#!/usr/bin/env python3
import json, sys
def say(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
if sys.argv[1:2] != ["aces-v2-serve"]: sys.exit(64)
say({"event": "ready", "device": "cpu", "runtime": "libtorch", "load_ms": 10.0})
for line in sys.stdin:
    r = json.loads(line)
    if r.get("op") == "quit": break
    say({"id": r.get("id"), "ok": False, "error": "fake"})
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    fake_genfill(root, kFillOk);
    app::MainWindow w(false);
    w.set_model_roots({root});
    // Not offered for the analytic master.
    w.set_deliver("source", "analytic");
    EXPECT_TRUE(w.findChild<QWidget*>("deliverAiFill")->isHidden());
    app::AcesEngine::Config c;
    c.cli = cli;
    c.package = root / "pkg";
    c.device = "cpu";
    w.aces_engine()->start(c);
    ASSERT_TRUE(wait_for([&] { return w.aces_engine()->state() == app::AcesEngine::State::Ready; }));
    w.set_deliver("source", "aces");
    ASSERT_TRUE(w.deliver_aces());
    EXPECT_FALSE(w.findChild<QWidget*>("deliverAiFill")->isHidden());
    EXPECT_TRUE(w.findChild<QWidget*>("deliverAiFillOptions")->isHidden());
    w.set_deliver("ai_fill", "1");
    EXPECT_FALSE(w.findChild<QWidget*>("deliverAiFillOptions")->isHidden());
    EXPECT_TRUE(w.findChild<QLabel*>("deliverAiSetup")->text().startsWith("Ready")) << w.findChild<QLabel*>("deliverAiSetup")->text().toStdString();
    w.findChild<QSpinBox*>("deliverAiTauHigh")->setValue(245);
    w.set_deliver("ai_prompt_highlight", "warm sun");
    EXPECT_EQ(w.deliver_effective().ai.tau_high, 245);
    EXPECT_EQ(QSettings().value("deliver/aiTauHigh").toInt(), 245);
    EXPECT_EQ(QSettings().value("deliver/aiPromptHighlight").toString(), "warm sun");
    // DPX and movies have no AI fill.
    w.set_deliver("image", "dpx10");
    EXPECT_TRUE(w.findChild<QWidget*>("deliverAiFill")->isHidden());
    EXPECT_FALSE(w.deliver_effective().ai_fill);
    w.set_deliver("image", "exr32");
    EXPECT_TRUE(w.deliver_effective().ai_fill);
    // Run on a master folder: the status says what it wrote.
    std::filesystem::create_directories(root / "renders" / "SH010");
    ASSERT_TRUE(w.start_genfill(QString::fromStdString((root / "renders" / "SH010").string())));
    EXPECT_TRUE(w.genfill_running() || !w.genfill_last_out().isEmpty());
    ASSERT_TRUE(wait_for([&] { return !w.genfill_running() && status_of(w).startsWith("AI fill: 2 frame(s)"); }))
        << status_of(w).toStdString();
    EXPECT_EQ(w.genfill_last_out(), QString::fromStdString((root / "renders" / "SH010_aifill").string()));
    EXPECT_TRUE(status_of(w).contains("1.50 % of pixels generated")) << status_of(w).toStdString();
    // A missing model: refused with the reason, nothing started.
    std::filesystem::remove(root / "runs" / "rudra_genfill.safetensors");
    QString why;
    EXPECT_FALSE(w.start_genfill(QString::fromStdString((root / "renders" / "SH010").string()), &why));
    EXPECT_TRUE(why.contains("no trained LoRA"));
    QSettings().clear();
}
#endif

// ---------------------------------------------------------------------------
// The batch queue (28 Oct 2026, batch_queue.hpp): shots rendered one after
// another through the learned ACES 2.0 model.
// ---------------------------------------------------------------------------

namespace {
const app::BatchShot* shot_named(const std::vector<app::BatchShot>& v, const QString& title) {
    for (const auto& s : v)
        if (s.title == title) return &s;
    return nullptr;
}
}  // namespace

TEST(Batch, GroupsMoviesSequencesAndStills) {
    const auto d = aces_tmp("batch_group");
    for (const char* f : {"clip.mov", "SH010.9.png", "SH010.10.png", "SH010.11.png", "SH020_0001.png", "plate.png",
                          "notes.txt", "SH030/a_01.tif", "SH030/a_02.tif"})
        write_text(d / f, "x");
    const auto shots = app::batch_shots_from({d});
    ASSERT_EQ(shots.size(), 5u);
    const auto* movie = shot_named(shots, "clip.mov");
    ASSERT_NE(movie, nullptr);
    EXPECT_EQ(movie->kind, "movie");
    const auto* seq = shot_named(shots, "SH010");
    ASSERT_NE(seq, nullptr);
    EXPECT_EQ(seq->kind, "sequence");
    ASSERT_EQ(seq->sources.size(), 3u);
    EXPECT_EQ(seq->sources[0].filename(), "SH010.9.png");    // by frame number, not by name
    EXPECT_EQ(seq->sources[2].filename(), "SH010.11.png");
    ASSERT_NE(shot_named(shots, "SH020_0001.png"), nullptr);  // a lone numbered still is a still
    EXPECT_EQ(shot_named(shots, "plate.png")->kind, "still");
    const auto* sub = shot_named(shots, "SH030");             // a sub-folder's sequence takes its name
    ASSERT_NE(sub, nullptr);
    EXPECT_EQ(sub->sources.size(), 2u);
    // Loose files are grouped the same way.
    const auto loose = app::batch_shots_from({d / "SH010.10.png", d / "SH010.9.png", d / "clip.mov"});
    ASSERT_EQ(loose.size(), 2u);
    EXPECT_EQ(loose[0].title, "SH010");
    EXPECT_EQ(loose[1].kind, "movie");
}

TEST(Batch, NonAsciiFoldersSurviveTheQueueAndTheRequest) {
    // Review 28 Oct: paths go Qt <-> std::filesystem through UTF-16 (paths.hpp).
    const QString root = QString::fromStdString(aces_tmp("batch_utf").string()) + QStringLiteral("/Été 撮影");
    const auto in = app::to_path(root + "/in");
    std::filesystem::create_directories(in);
    for (const char* f : {"Ünï.1001.png", "Ünï.1002.png"}) write_text(in / app::to_path(QString::fromUtf8(f)), "x");
    const auto file = app::to_path(root + "/queue.json");
    app::DeliverChoice c;
    c.output = "sequence";
    std::vector<QString> ids;
    {
        app::BatchQueue q(file);
        ids = q.add(app::batch_shots_from({in}), c, root + "/out", 1001);
        ASSERT_EQ(ids.size(), 1u);
    }
    app::BatchQueue q(file);
    ASSERT_EQ(q.jobs().size(), 1u);
    EXPECT_EQ(q.jobs()[0].shot.title, QString::fromUtf8("Ünï"));
    const auto r = app::batch_request(*q.find(ids[0]));
    // The folder keeps its letters; the render's own name is ASCII (what a conform relinks).
    EXPECT_EQ(r["out"].toString().toStdString(), (root + "/out/_n_").toStdString());
    EXPECT_TRUE(r["sources"].toArray()[0].toString().endsWith(QString::fromUtf8("Ünï.1001.png")));
}

TEST(Batch, TheQueueKeepsItsJobsAndBuildsTheirRequests) {
    const auto d = aces_tmp("batch_queue");
    for (const char* f : {"SH010.1001.png", "SH010.1002.png", "clip.mov"}) write_text(d / "in" / f, "x");
    const auto file = d / "queue.json";
    app::DeliverChoice c;
    c.output = "movie";
    c.bits = 16;
    c.encoding = "pq";
    c.fps = 25;
    c.source = "analytic";   // the batch is the learned model's
    std::vector<QString> ids;
    {
        app::BatchQueue q(file);
        ids = q.add(app::batch_shots_from({d / "in"}), c, QString::fromStdString((d / "out").string()), 1001);
        ASSERT_EQ(ids.size(), 2u);
        EXPECT_EQ(q.jobs()[0].choice.source, "aces");
        q.find(ids[0])->status = "running";   // as if the app stopped mid-render
        q.save();
    }
    app::BatchQueue q(file);
    ASSERT_EQ(q.jobs().size(), 2u);
    EXPECT_EQ(q.jobs()[0].status, "pending");
    EXPECT_EQ(q.jobs()[0].note, "interrupted");
    const app::BatchJob& seq = *q.find(ids[0]);
    EXPECT_EQ(seq.shot.title, "SH010");
    EXPECT_EQ(seq.name, "SH010");
    // A sequence to a 16-bit PQ MOV at its frame rate, into a new folder per run.
    QString why;
    auto r = app::batch_request(seq, &why);
    ASSERT_FALSE(r.isEmpty()) << why.toStdString();
    EXPECT_EQ(r["log_formats"].toString(), "mov16");
    EXPECT_EQ(r["log"].toString(), "pq1000");
    EXPECT_EQ(r["exr"].toString(), "none");
    EXPECT_DOUBLE_EQ(r["fps"].toDouble(), 25.0);
    EXPECT_EQ(r["sources"].toArray().size(), 2);
    EXPECT_EQ(r["start"].toInt(), 1001);
    EXPECT_EQ(r["out"].toString().toStdString(), (d / "out" / "SH010").string());
    std::filesystem::create_directories(d / "out" / "SH010");
    EXPECT_EQ(app::batch_request(seq).value("out").toString().toStdString(), (d / "out" / "SH010_2").string());
    // A movie keeps its own clock; a still of it is its first frame.
    const app::BatchJob& movie = *q.find(ids[1]);
    EXPECT_FALSE(app::batch_request(movie).contains("fps"));
    app::DeliverChoice still;
    still.image = "exr16";
    still.gamut = "acescg";
    q.apply({ids[1]}, still, {}, 1);
    r = app::batch_request(*q.find(ids[1]));
    EXPECT_EQ(r["exr"].toString(), "half");
    EXPECT_EQ(r["frames"].toArray(), QJsonArray{0});
    EXPECT_EQ(q.find(ids[1])->start, 1);
    EXPECT_EQ(app::batch_output_text(*q.find(ids[1])), "ACEScg EXR, half · first frame");
    // Order, names, retry, removal.
    q.move(ids[1], -1);
    EXPECT_EQ(q.jobs()[0].id, ids[1]);
    q.rename(ids[1], "plate v2/final");
    EXPECT_EQ(q.find(ids[1])->name, "plate v2_final");
    q.find(ids[1])->status = "failed";
    q.retry({});
    EXPECT_EQ(q.find(ids[1])->status, "pending");
    q.remove({ids[0]});
    EXPECT_EQ(q.jobs().size(), 1u);
    // A missing source is refused before anything runs.
    std::filesystem::remove(d / "in" / "clip.mov");
    EXPECT_TRUE(app::batch_request(*q.find(ids[1]), &why).isEmpty());
    EXPECT_TRUE(why.startsWith("Missing")) << why.toStdString();
}

TEST(Batch, RendersEveryShotThroughTheLoadedModel) {
    QSettings().clear();
    const auto root = aces_tmp("batch_run");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    for (const char* f : {"A.1001.png", "A.1002.png", "A.1003.png", "B.png"}) write_text(root / "in" / f, "x");
    QSettings().setValue("batch/file", QString::fromStdString((root / "queue.json").string()));
    const auto log = root / "requests.jsonl";
    const auto cli = root / "rudra-native";
    write_text(cli, std::string(R"(#!/usr/bin/env python3
import json, os, sys
LOG = ")") + log.string() + R"("
def say(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
if sys.argv[1:2] != ["aces-v2-serve"]: sys.exit(64)
say({"event": "ready", "device": "cpu", "runtime": "libtorch", "load_ms": 10.0})
for line in sys.stdin:
    r = json.loads(line)
    open(LOG, "a").write(line)
    if r.get("op") == "quit": break
    if r.get("op") == "master":
        if any("missing" in s for s in r["sources"]):
            say({"id": r["id"], "ok": False, "error": "Not a supported still or video.", "detail": r["sources"][0]}); continue
        os.makedirs(r["out"])
        n = len(r.get("frames") or r["sources"])
        for i in range(n):
            say({"id": r["id"], "progress": {"index": i + 1, "total": n, "source": "x", "exr": "f.exr", "ms": 10.0,
                                             "remaining_display_errors": 0}})
        say({"id": r["id"], "ok": True, "done": True, "cancelled": False, "frames": n, "maxcll": 1000.0, "maxfall": 120.0})
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::MainWindow w(false);
    // Nothing loaded: the batch waits for the model.
    QString why;
    EXPECT_FALSE(w.batch_start(&why));
    app::AcesEngine::Config c;
    c.cli = cli;
    c.package = root / "pkg";
    c.device = "cpu";
    w.aces_engine()->start(c);
    ASSERT_TRUE(wait_for([&] { return w.aces_engine()->state() == app::AcesEngine::State::Ready; }));
    w.open_batch();
    auto* win = w.findChild<QDialog*>("batchWindow");
    ASSERT_NE(win, nullptr);
    w.findChild<QLineEdit*>("batchDir")->setText(QString::fromStdString((root / "out").string()));
    static_cast<app::BatchWindow*>(win)->add_paths({QString::fromStdString((root / "in").string())});
    ASSERT_EQ(w.batch_queue().jobs().size(), 2u);
    // A shot whose file has gone fails on its own; the rest still render.
    w.batch_queue().add({app::BatchShot{"gone", "still", {root / "in" / "B.png"}}}, w.deliver_choice(),
                        QString::fromStdString((root / "out").string()), 1);
    auto& q = w.batch_queue();
    const_cast<app::BatchJob&>(q.jobs().back()).shot.sources = {root / "in" / "missing.png"};
    EXPECT_EQ(w.findChild<QTreeWidget*>("batchList")->topLevelItemCount(), 3);
    ASSERT_TRUE(w.findChild<QPushButton*>("batchStart")->isEnabled());
    w.findChild<QPushButton*>("batchStart")->click();
    EXPECT_TRUE(w.batch_running());
    ASSERT_TRUE(wait_for([&] { return !w.batch_running(); }));
    EXPECT_EQ(q.jobs()[0].status, "done");
    EXPECT_EQ(q.jobs()[1].status, "done");
    EXPECT_DOUBLE_EQ(q.jobs()[1].maxcll, 1000.0);   // the shot's light level, from the render
    EXPECT_EQ(w.findChild<QTreeWidget*>("batchList")->topLevelItem(1)->text(6), "1000 / 120");
    EXPECT_EQ(q.jobs()[2].status, "failed");
    EXPECT_TRUE(w.findChild<QLabel*>("batchStatus")->text().startsWith("Batch ended: 2 done, 0 waiting, 1 failed"))
        << w.findChild<QLabel*>("batchStatus")->text().toStdString();
    EXPECT_TRUE(q.jobs()[2].error.startsWith("Missing")) << q.jobs()[2].error.toStdString();
    EXPECT_TRUE(std::filesystem::is_directory(root / "out" / "A"));
    EXPECT_TRUE(std::filesystem::is_directory(root / "out" / "B"));
    EXPECT_TRUE(w.findChild<QLabel*>("batchSub")->text().startsWith("3 shot(s) · 0 waiting · 2 done"))
        << w.findChild<QLabel*>("batchSub")->text().toStdString();
    // Two master requests reached the engine: the sequence (a still: its first frame) and the still.
    std::ifstream in(log);
    std::string line;
    int masters = 0;
    while (std::getline(in, line)) masters += line.find("\"master\"") != std::string::npos;
    EXPECT_EQ(masters, 2);
    // The queue is on disk: a new window finds it.
    app::BatchQueue again(root / "queue.json");
    EXPECT_EQ(again.jobs().size(), 3u);
    EXPECT_EQ(again.jobs()[0].status, "done");
    QSettings().clear();
}

// ---------------------------------------------------------------------------
// The colorist's route (28 Oct 2026): scopes and readouts of the learned
// ACES 2.0 frame, the trims in the viewer, Deliver and the batch, and the
// timecode and reel a render carries.
// ---------------------------------------------------------------------------

#ifdef RUDRA_HAVE_STILL_DECODE
TEST(Colorist, TheLearnedRouteIsMeasuredTrimmedAndStamped) {
    QSettings().clear();
    const auto root = aces_tmp("colorist");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    std::filesystem::create_directories(root / "frames");
    for (const char* f : {"SH010.1001.png", "SH010.1002.png"}) {
        QImage img(4, 2, QImage::Format_RGB888);
        img.fill(QColor(250, 240, 230));
        ASSERT_TRUE(img.save(QString::fromStdString((root / "frames" / f).string())));
    }
    QSettings().setValue("batch/file", QString::fromStdString((root / "queue.json").string()));
    const auto log = root / "requests.jsonl";
    const auto cli = root / "rudra-native";
    write_text(cli, std::string(R"(#!/usr/bin/env python3
import json, os, struct, sys
LOG = ")") + log.string() + R"("
def say(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
if sys.argv[1:2] != ["aces-v2-serve"]: sys.exit(64)
say({"event": "ready", "device": "cpu", "runtime": "libtorch", "load_ms": 10.0})
for line in sys.stdin:
    r = json.loads(line)
    open(LOG, "a").write(line)
    if r.get("op") == "quit": break
    if r.get("op") == "preview":
        for k, v in (("model_out", 6.0), ("baseline_out", 1.0)):
            if k in r: open(r[k], "wb").write(struct.pack("<24f", *([v] * 24)))
        say({"id": r["id"], "ok": True, "width": 4, "height": 2, "bits": 8, "ms": 5.0, "reconstructed_fraction": 0.5,
             "peak": 6.0, "peak_plain": 1.0, "remaining_display_errors": 0, "new_negative_luminance": 0})
    elif r.get("op") == "master":
        os.makedirs(r["out"])
        say({"id": r["id"], "ok": True, "done": True, "cancelled": False, "frames": 1})
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::MainWindow w(false);
    EXPECT_TRUE(w.findChild<QWidget*>("acesTrimsCard")->isHidden());   // nothing to trim yet
    app::AcesEngine::Config c;
    c.cli = cli;
    c.package = root / "pkg";
    c.device = "cpu";
    w.aces_engine()->start(c);
    ASSERT_TRUE(wait_for([&] { return w.aces_engine()->state() == app::AcesEngine::State::Ready; }));
    w.open_source(QString::fromStdString((root / "frames").string()), true);
    ASSERT_TRUE(wait_for([&] { return w.aces_shown() == 0; })) << w.problem().toStdString();
    // The readouts and scopes are the learned frame's, in nits.
    ASSERT_TRUE(wait_for([&] { return w.measurement() != nullptr; }));
    EXPECT_GT(w.measurement()->measured.metrics.maxcll, 1000);   // 6 x diffuse white in ACEScg ~ 1200 nits
    EXPECT_FALSE(w.findChild<QWidget*>("acesTrimsCard")->isHidden());
    auto lines = [&](const char* op) {
        std::vector<json> out;
        std::ifstream in(log);
        std::string line;
        while (std::getline(in, line))
            if (line.find(std::string("\"") + op + "\"") != std::string::npos) out.push_back(json::parse(line));
        return out;
    };
    EXPECT_FALSE(lines("preview").back().contains("trims"));   // the model's own master by default
    // Trims: the frame is asked for again with them; the sliders show them.
    rudra::aces_v2::AcesTrims t;
    t.strength = 0.8;
    t.saturation = 1.2;
    t.peak_nits = 1000;
    w.set_aces_trims(t);
    ASSERT_TRUE(wait_for([&] { return lines("preview").back().contains("trims"); }));
    EXPECT_DOUBLE_EQ(lines("preview").back()["trims"]["peak_nits"].get<double>(), 1000.0);
    EXPECT_EQ(w.findChild<QLabel*>("acesPeakVal")->text(), "1000");
    EXPECT_EQ(w.findChild<QSlider*>("acesStrength")->value(), 80);
    w.findChild<QSlider*>("acesPeak")->setValue(w.findChild<QSlider*>("acesPeak")->maximum());
    EXPECT_EQ(w.aces_trims().peak_nits, 0.0);   // far right: no limit
    w.set_aces_trims(t);
    EXPECT_DOUBLE_EQ(QSettings().value("aces/trimStrength").toDouble(), 0.8);
    // Deliver: the trims, a typed start timecode (a still of frame 2 of the sequence is start + 1) and the reel.
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString((root / "renders").string()));
    w.findChild<QLineEdit*>("renderTimecode")->setText("01:00:00:00");
    w.findChild<QLineEdit*>("renderReel")->setText("A001");
    w.run("next");
    ASSERT_TRUE(wait_for([&] { return w.aces_shown() == 1; }));
    QString why;
    auto req = w.aces_deliver_request(&why);
    ASSERT_FALSE(req.isEmpty()) << why.toStdString();
    EXPECT_DOUBLE_EQ(req["trims"].toObject()["strength"].toDouble(), 0.8);
    EXPECT_EQ(req["timecode"].toString(), "01:00:00:01");
    EXPECT_EQ(req["reel"].toString(), "A001");
    w.findChild<QLineEdit*>("renderTimecode")->setText("1:00");
    EXPECT_TRUE(w.aces_deliver_request(&why).isEmpty());
    EXPECT_TRUE(why.contains("HH:MM:SS:FF")) << why.toStdString();
    w.findChild<QLineEdit*>("renderTimecode")->setText("01:00:00:00");
    // The batch takes the trims, timecode and reel with the shot, and keeps them on disk.
    ASSERT_TRUE(w.batch_add_current(&why)) << why.toStdString();
    const auto& job = w.batch_queue().jobs().back();
    EXPECT_DOUBLE_EQ(job.peak_nits, 1000.0);
    EXPECT_EQ(job.timecode, "01:00:00:00");
    EXPECT_TRUE(app::batch_output_text(job).endsWith("trimmed")) << app::batch_output_text(job).toStdString();
    const auto breq = app::batch_request(job);
    EXPECT_DOUBLE_EQ(breq["trims"].toObject()["saturation"].toDouble(), 1.2);
    EXPECT_EQ(breq["reel"].toString(), "A001");
    app::BatchQueue again(root / "queue.json");
    EXPECT_DOUBLE_EQ(again.jobs().back().strength, 0.8);
    EXPECT_EQ(again.jobs().back().reel, "A001");
    // Reset: the model's own master again.
    w.findChild<QPushButton*>("acesTrimsReset")->click();
    EXPECT_TRUE(w.aces_trims().identity());
    QSettings().clear();
}
#endif

// ---------------------------------------------------------------------------
// The EDL into the batch (28 Oct 2026, edl.hpp): only the frames each edit
// uses, with handles, matched by clip name, reel and timecode.
// ---------------------------------------------------------------------------

namespace {
const char* kEdl = R"(TITLE: REEL_ONE
FCM: NON-DROP FRAME

001  A001C003 V     C        01:00:01:00 01:00:02:00 00:00:00:00 00:00:01:00
* FROM CLIP NAME: A001C003.mov
002  AX       AA    C        01:00:00:00 01:00:01:00 00:00:01:00 00:00:02:00
003  BL       V     C        00:00:00:00 00:00:00:12 00:00:02:00 00:00:02:12
004  B002     V     D    012 10:00:00:06 10:00:00:18 00:00:02:12 00:00:03:00
005  ZZZ      V     C        01:00:00:00 01:00:00:10 00:00:03:00 00:00:03:10
006  A001C003 V     C        02:00:00:00 02:00:00:10 00:00:03:10 00:00:03:20
* FROM CLIP NAME: A001C003.mov
)";
}  // namespace

TEST(Edl, ParsesEventsAndMatchesThemToMovies) {
    QString title;
    const auto ev = app::parse_edl(kEdl, &title);
    EXPECT_EQ(title, "REEL_ONE");
    ASSERT_EQ(ev.size(), 6u);
    EXPECT_EQ(ev[0].clip_name, "A001C003.mov");
    EXPECT_FALSE(ev[1].video());                 // audio only
    EXPECT_EQ(ev[3].transition, "D 012");
    EXPECT_EQ(ev[3].src_in, "10:00:00:06");
    std::vector<app::EdlMedia> media(2);
    media[0].path = "/m/A001C003.mov";
    media[0].start_tc = "01:00:00:00";
    media[0].fps = 24;
    media[0].frames = 24 * 5;
    media[1].path = "/m/day2_B002.mov";
    media[1].reel = "B002";                      // its reel tag
    media[1].start_tc = "10:00:00:00";
    media[1].fps = 25;
    media[1].frames = 50;
    const auto m = app::edl_match(ev, media, 4);
    ASSERT_EQ(m.shots.size(), 2u);
    EXPECT_EQ(m.shots[0].name, "001_A001C003");
    EXPECT_EQ(m.shots[0].first, 24 - 4);         // 01:00:01:00 is frame 24; 4 handles
    EXPECT_EQ(m.shots[0].last, 47 + 4);          // the out point is exclusive
    EXPECT_EQ(m.shots[1].shot.title, "day2_B002.mov");
    EXPECT_EQ(m.shots[1].first, 2);              // frame 6 minus 4 handles
    EXPECT_EQ(m.shots[1].last, 21);
    ASSERT_EQ(m.unmatched.size(), 2);            // ZZZ (no movie) and 006 (outside the movie's timecode)
    EXPECT_TRUE(m.unmatched[0].startsWith("005 ZZZ: no movie"));
    EXPECT_TRUE(m.unmatched[1].contains("does not hold this timecode"));
    // Handles stop at the movie's ends.
    EXPECT_EQ(app::edl_match(ev, media, 100).shots[1].first, 0);
    EXPECT_EQ(app::edl_match(ev, media, 100).shots[1].last, 49);
}

TEST(Edl, AnEdlBecomesBatchShotsOfOnlyTheirFrames) {
    if (QStandardPaths::findExecutable("ffmpeg").isEmpty() || QStandardPaths::findExecutable("ffprobe").isEmpty())
        GTEST_SKIP() << "ffmpeg and ffprobe are not on the PATH";
    QSettings().clear();
    const auto root = aces_tmp("edl_import");
    QSettings().setValue("batch/file", QString::fromStdString((root / "queue.json").string()));
    std::filesystem::create_directories(root / "media" / "day1");
    const auto movie = root / "media" / "day1" / "A001C003.mov";
    const std::string cmd = "ffmpeg -v error -y -f lavfi -i testsrc2=s=64x36:r=24:d=5 -pix_fmt yuv420p -c:v libx264 "
                            "-timecode 01:00:00:00 \"" + movie.string() + "\"";
    ASSERT_EQ(std::system(cmd.c_str()), 0);
    write_text(root / "cut.edl", kEdl);
    app::MainWindow w(false);
    w.set_deliver("output", "still");   // a still choice renders a range as frames
    QString report;
    const int n = w.batch_import_edl(QString::fromStdString((root / "cut.edl").string()),
                                     QString::fromStdString((root / "media").string()), 12,
                                     QString::fromStdString((root / "out").string()), &report);
    EXPECT_EQ(n, 1) << report.toStdString();   // 001 only: B002 and ZZZ have no movie here, 006 is outside its timecode
    const auto& jobs = w.batch_queue().jobs();
    ASSERT_GE(jobs.size(), 1u);
    const auto& j = jobs.front();
    EXPECT_EQ(j.name, "001_A001C003");
    EXPECT_EQ(j.first, 12);
    EXPECT_EQ(j.last, 59);
    EXPECT_EQ(j.choice.output, "sequence");
    const auto req = app::batch_request(j);
    EXPECT_EQ(req["range"].toArray(), (QJsonArray{12, 59}));
    EXPECT_FALSE(req.contains("timecode"));   // the movie's own timecode follows each frame
    EXPECT_TRUE(report.contains("not matched")) << report.toStdString();
    // The range is editable, and only for movies.
    EXPECT_TRUE(w.batch_queue().set_range(j.id, 0, 10));
    EXPECT_FALSE(w.batch_queue().set_range(j.id, 10, 0));
    EXPECT_TRUE(w.batch_queue().set_range(j.id, -1, -1));
    EXPECT_FALSE(app::batch_request(w.batch_queue().jobs().front()).contains("range"));
    QSettings().clear();
}

TEST(Edl, RetimesAndShortHandlesAreSaidAndTheRecordTimecodeKept) {
    // Review 28 Oct: an M2 speed change renders the source frames it uses; handles
    // the movie cannot give are reported, not silently cut; the record timecode is kept.
    const char* edl = R"(TITLE: RETIME
007  A001C003 V     C        01:00:00:05 01:00:00:15 00:01:00:00 00:01:01:00
M2   A001C003       048.0    01:00:00:05
* FROM CLIP NAME: A001C003.mov
)";
    const auto ev = app::parse_edl(edl);
    ASSERT_EQ(ev.size(), 1u);
    EXPECT_DOUBLE_EQ(ev[0].speed, 48.0);
    std::vector<app::EdlMedia> media(1);
    media[0].path = "/m/A001C003.mov";
    media[0].start_tc = "01:00:00:00";
    media[0].fps = 24;
    media[0].frames = 120;
    const auto m = app::edl_match(ev, media, 12);
    ASSERT_EQ(m.shots.size(), 1u);
    const auto& s = m.shots[0];
    // One record second at 200 %: 48 source frames from frame 5, handles 12 (only 5 at the head).
    EXPECT_EQ(s.first, 0);
    EXPECT_EQ(s.last, 5 + 48 - 1 + 12);
    EXPECT_EQ(s.rec_in, "00:01:00:00");
    ASSERT_EQ(s.warnings.size(), 2);
    EXPECT_TRUE(s.warnings[0].contains("speed 200%")) << s.warnings[0].toStdString();
    EXPECT_TRUE(s.warnings[1].contains("5 head, 12 tail")) << s.warnings[1].toStdString();
    EXPECT_EQ(m.warnings.size(), 2);
}

#ifdef RUDRA_HAVE_STILL_DECODE
// Monitoring, reference stills and the playback proxy on the learned route (28 Oct 2026).
TEST(Colorist, MonitorStillsAndProxy) {
    QSettings().clear();
    const auto root = aces_tmp("monitor");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    std::filesystem::create_directories(root / "frames");
    for (const char* f : {"a.png", "b.png", "c.png"}) {
        QImage img(4, 2, QImage::Format_RGB888);
        img.fill(QColor(250, 240, 230));
        ASSERT_TRUE(img.save(QString::fromStdString((root / "frames" / f).string())));
    }
    const auto log = root / "requests.jsonl";
    const auto cli = root / "rudra-native";
    write_text(cli, std::string(R"(#!/usr/bin/env python3
import json, os, struct, sys
LOG = ")") + log.string() + R"("
def say(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
if sys.argv[1:2] != ["aces-v2-serve"]: sys.exit(64)
say({"event": "ready", "device": "cpu", "runtime": "libtorch", "load_ms": 10.0})
for line in sys.stdin:
    r = json.loads(line)
    open(LOG, "a").write(line)
    if r.get("op") == "quit": break
    if r.get("op") == "preview":
        v = 3.0 if r.get("display") == "hdr1000" else 2.0
        for k, x in (("model_out", v), ("baseline_out", 1.0)):
            if k in r: open(r[k], "wb").write(struct.pack("<24f", *([x] * 24)))
        if "codes_out" in r: open(r["codes_out"], "wb").write(struct.pack("<8f", *([1.0] * 4 + [0.5] * 4)))
        say({"id": r["id"], "ok": True, "width": 4, "height": 2, "bits": 8, "ms": 5.0, "reconstructed_fraction": 0.5,
             "peak": v, "peak_plain": 1.0, "remaining_display_errors": 0, "new_negative_luminance": 0})
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::MainWindow w(false);
    app::AcesEngine::Config c;
    c.cli = cli;
    c.package = root / "pkg";
    c.device = "cpu";
    w.aces_engine()->start(c);
    ASSERT_TRUE(wait_for([&] { return w.aces_engine()->state() == app::AcesEngine::State::Ready; }));
    w.open_source(QString::fromStdString((root / "frames").string()), true);
    ASSERT_TRUE(wait_for([&] { return w.aces_shown() == 0; })) << w.problem().toStdString();
    auto previews = [&] {
        std::vector<json> out;
        std::ifstream in(log);
        std::string line;
        while (std::getline(in, line))
            if (line.find("\"preview\"") != std::string::npos) out.push_back(json::parse(line));
        return out;
    };
    // The Invented layer's codes come with every frame.
    EXPECT_TRUE(previews().back().contains("codes_out"));
    ASSERT_EQ(w.aces_codes().size(), 8u);
    EXPECT_FLOAT_EQ(w.aces_codes()[0], 1.0f);
    // Review 28 Oct: on the learned route the analytic controls are locked and say so;
    // the input is read as BT.1886, or re-encoded from sRGB when the master was made for it.
    EXPECT_FALSE(w.findChild<QSlider*>("exposure")->isEnabled());
    EXPECT_FALSE(w.findChild<QSlider*>("strength")->isEnabled());
    EXPECT_TRUE(w.findChild<QSlider*>("peak")->isEnabled());   // the scene monitor's view peak
    EXPECT_FALSE(w.findChild<QLabel*>("routeNoteRec")->isHidden());
    EXPECT_TRUE(w.findChild<QLabel*>("pipeIn")->text().contains("BT.1886")) << w.findChild<QLabel*>("pipeIn")->text().toStdString();
    EXPECT_FALSE(previews().back().contains("source_display"));
    w.set_aces_source_display("srgb");
    ASSERT_TRUE(wait_for([&] { return previews().back().value("source_display", "") == "srgb" && w.aces_shown() == 0; }));
    EXPECT_TRUE(w.findChild<QLabel*>("pipeIn")->text().contains("sRGB display"));
    EXPECT_EQ(QSettings().value("aces/sourceDisplay").toString(), "srgb");
    w.set_aces_source_display("bt1886");
    // Monitor: the ACES 2.0 HDR output, the view peak at 1000 nits; kept between runs.
    EXPECT_EQ(w.aces_monitor(), "scene");
    w.run("monitor-hdr1000");
    ASSERT_TRUE(wait_for([&] { return previews().back().value("display", "") == "hdr1000" && w.aces_shown() == 0; }));
    EXPECT_NEAR(w.session().display_nits(), 1000.0, 60.0);
    EXPECT_TRUE(w.action("monitor-hdr1000")->isChecked());
    EXPECT_EQ(QSettings().value("aces/monitor").toString(), "hdr1000");
    EXPECT_EQ(find<app::Seg>(w, "acesMonitor")->on(), "hdr1000");
    ASSERT_TRUE(wait_for([&] { return w.measurement() && w.measurement()->measured.metrics.maxcll > 500; }));
    EXPECT_FALSE(w.findChild<QSlider*>("peak")->isEnabled());   // locked to the monitor's 1000 nits
    w.set_aces_monitor("sdr100");
    EXPECT_NEAR(w.session().display_nits(), 100.0, 10.0);
    w.set_aces_monitor("scene");
    ASSERT_TRUE(wait_for([&] { return !previews().back().contains("display"); }));
    // A reference still: grabbed, then compared in the baseline's place on the next frame.
    EXPECT_FALSE(w.findChild<QPushButton*>("refCompare")->isEnabled());
    w.run("still-grab");
    EXPECT_EQ(w.reference_count(), 1);
    EXPECT_TRUE(w.findChild<QComboBox*>("refStills")->currentText().contains("a.png")) << w.findChild<QComboBox*>("refStills")->currentText().toStdString();
    w.run("still-compare");
    EXPECT_TRUE(w.reference_compare());
    EXPECT_TRUE(w.action("still-compare")->isChecked());
    w.run("next");
    ASSERT_TRUE(wait_for([&] { return w.aces_shown() == 1; }));
    EXPECT_TRUE(w.reference_shown());
    EXPECT_FALSE(w.findChild<QLabel*>("refNote")->isHidden());
    w.run("still-compare");
    EXPECT_FALSE(w.reference_compare());
    // Playing at a proxy size: the model runs on the smaller frame while playing.
    w.run("playback-side-640");
    w.run("first");
    ASSERT_TRUE(wait_for([&] { return w.aces_shown() == 0; }));
    w.run("play");
    ASSERT_TRUE(wait_for([&] {
        for (const auto& p : previews())
            if (p.value("max_side", 0) == 640) return true;
        return false;
    }));
    // Review 28 Oct: the plays fill a cache; with every frame in, play is real time from it.
    ASSERT_TRUE(wait_for([&] { return !w.aces_playing(); }));
    if (w.aces_cached_frames() < 3) {
        w.run("play");
        ASSERT_TRUE(wait_for([&] { return !w.aces_playing(); }));
    }
    EXPECT_EQ(w.aces_cached_frames(), 3);
    const auto asked = previews().size();
    w.run("play");
    EXPECT_TRUE(w.aces_realtime_playing());
    QTest::qWait(150);
    EXPECT_EQ(previews().size(), asked);   // nothing asked of the model while it plays from the cache
    w.run("play");
    EXPECT_FALSE(w.aces_realtime_playing());
    // A trim changes what the screen shows: the cache no longer counts.
    aces_v2::AcesTrims t;
    t.strength = 0.7;
    w.set_aces_trims(t);
    EXPECT_EQ(w.aces_cached_frames(), 0);
    QSettings().clear();
}
#endif

// The Render card's File path, as Nuke's Write takes it (29 Oct 2026).
TEST(Deliver, TheFilePathIsNukeStyle) {
    auto p = app::parse_render_path("D:/Renders/SH010/SH010_hdr.####.exr");
    EXPECT_EQ(p.dir, "D:/Renders/SH010");
    EXPECT_EQ(p.name, "SH010_hdr");
    EXPECT_EQ(p.padding, 4);
    EXPECT_EQ(p.separator, ".");
    EXPECT_EQ(p.ext, "exr");
    p = app::parse_render_path("/r/plate_%06d.dpx");
    EXPECT_EQ(p.name, "plate");
    EXPECT_EQ(p.separator, "_");
    EXPECT_EQ(p.padding, 6);
    EXPECT_EQ(app::parse_render_path("/r/SH010_hdr.mov").padding, 0);
    EXPECT_EQ(app::parse_render_path("/r/SH010_hdr.mov").name, "SH010_hdr");
    app::RenderPath q{"/r", "SH020", "exr", 4, "."};
    EXPECT_EQ(app::compose_render_path(q), QDir::toNativeSeparators("/r/SH020.####.exr"));
    EXPECT_EQ(app::parse_render_path(app::compose_render_path(q)), q);
    QSettings().clear();
    app::MainWindow w(false);
    const auto root = aces_tmp("render_path");
    auto* file = w.findChild<QLineEdit*>("renderFile");
    ASSERT_NE(file, nullptr);
    EXPECT_TRUE(w.findChild<QLineEdit*>("renderDir")->isHidden());   // the path replaces the folder and name fields
    file->setText(QString::fromStdString((root / "out" / "SH030_v2_####.exr").string()));
    emit file->editingFinished();
    EXPECT_EQ(w.render_dir().toStdString(), QDir::toNativeSeparators(QString::fromStdString((root / "out").string())).toStdString());
    EXPECT_EQ(w.findChild<QLineEdit*>("renderName")->text(), "SH030_v2");
    // The type follows the extension: .dpx picks DPX (on the learned source), .mov a movie.
    w.findChild<QLineEdit*>("renderName")->setText("SH030_v3");
    EXPECT_TRUE(file->text().contains("SH030_v3_####.exr")) << file->text().toStdString();
    EXPECT_TRUE(w.findChild<QCheckBox*>("renderCreateDirs")->isChecked());
    QSettings().clear();
}

// The batch's AI fill (29 Oct 2026): after a shot's master, for EXR stills and sequences.
TEST(Batch, EndsWhenTheEngineDiesMidShot) {
    // Review 28 Oct: the replies ran while the engine still read Ready, so the
    // batch sent its next job to the dead process and hung "running" for good.
    QSettings().clear();
    const auto root = aces_tmp("batch_dies");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    for (const char* f : {"A.1001.png", "A.1002.png", "clip.mov"}) write_text(root / "in" / f, "x");
    QSettings().setValue("batch/file", QString::fromStdString((root / "queue.json").string()));
    QSettings().setValue("deliver/renderPadding", 5);
    QSettings().setValue("deliver/renderSeparator", "_");
    const auto cli = root / "rudra-native";
    write_text(cli, R"(#!/usr/bin/env python3
import json, os, sys
def say(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
if sys.argv[1:2] != ["aces-v2-serve"]: sys.exit(64)
say({"event": "ready", "device": "cpu", "runtime": "libtorch", "load_ms": 1.0})
for line in sys.stdin:
    r = json.loads(line)
    if r.get("op") == "master":
        open(os.path.join(os.path.dirname(r["out"]), "req.json"), "w").write(line)
        sys.exit(3)   # dies mid-render
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::MainWindow w(false);
    app::AcesEngine::Config c;
    c.cli = cli;
    c.package = root / "pkg";
    c.device = "cpu";
    w.aces_engine()->start(c);
    ASSERT_TRUE(wait_for([&] { return w.aces_engine()->state() == app::AcesEngine::State::Ready; }));
    auto& q = w.batch_queue();
    app::DeliverChoice d;
    d.output = "sequence";
    std::filesystem::create_directories(root / "out");
    const auto ids = q.add(app::batch_shots_from({root / "in"}), d, QString::fromStdString((root / "out").string()), 1001);
    ASSERT_EQ(ids.size(), 2u);
    ASSERT_TRUE(w.batch_start());
    ASSERT_TRUE(wait_for([&] { return !w.batch_running(); }));
    EXPECT_EQ(q.find(ids[0])->status, "failed");
    EXPECT_EQ(q.find(ids[1])->status, "pending");   // not failed for an engine it never reached
    EXPECT_EQ(w.aces_engine()->state(), app::AcesEngine::State::Failed);
    // The batch numbers frames as the Deliver tab does.
    std::ifstream rf(root / "out" / "req.json");
    const auto req = json::parse(rf);
    EXPECT_EQ(req.value("padding", 0), 5);
    EXPECT_EQ(req.value("separator", std::string("?")), "_");
    QSettings().clear();
}

TEST(Batch, OneMasterAtATimeCancelByIdAndQuittingAsks) {
    // Review 28 Oct: the shared engine runs one master at a time, is not reloaded
    // under one, cancels the one asked for, and quitting asks before stopping it.
    QSettings().clear();
    const auto root = aces_tmp("batch_one");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    for (const char* f : {"A.1001.png", "A.1002.png"}) write_text(root / "in" / f, "x");
    QSettings().setValue("batch/file", QString::fromStdString((root / "queue.json").string()));
    const auto cli = root / "rudra-native";
    write_text(cli, R"(#!/usr/bin/env python3
import json, os, sys
def say(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
if sys.argv[1:2] != ["aces-v2-serve"]: sys.exit(64)
say({"event": "ready", "device": "cpu", "runtime": "libtorch", "load_ms": 1.0})
held = None
log = open(os.environ["RUDRA_TEST_LOG"], "a")
for line in sys.stdin:
    r = json.loads(line)
    log.write(line); log.flush()
    if r.get("op") == "quit": break
    if r.get("op") == "master":
        os.makedirs(r["out"], exist_ok=True)
        held = r["id"]   # holds it until cancelled
    if r.get("op") == "cancel" and r.get("target") == held:
        say({"id": held, "ok": True, "done": True, "cancelled": True, "frames": 0})
        held = None
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    const auto logf = root / "engine.log";
    qputenv("RUDRA_TEST_LOG", QByteArray::fromStdString(logf.string()));
    auto w = std::make_unique<app::MainWindow>(false);
    app::AcesEngine::Config c;
    c.cli = cli;
    c.package = root / "pkg";
    c.device = "cpu";
    auto* e = w->aces_engine();
    e->start(c);
    ASSERT_TRUE(wait_for([&] { return e->state() == app::AcesEngine::State::Ready; }));
    auto& q = w->batch_queue();
    app::DeliverChoice d;
    d.output = "sequence";
    const auto ids = q.add(app::batch_shots_from({root / "in"}), d, QString::fromStdString((root / "out").string()), 1001);
    ASSERT_EQ(ids.size(), 1u);
    ASSERT_TRUE(w->batch_start());
    ASSERT_TRUE(wait_for([&] { return e->busy(); }));
    // A second master is refused at once; the engine is not stopped or swapped under the render.
    QString err;
    e->request(QJsonObject{{"op", "master"}}, [&](const QJsonObject& o) { err = o.value("error").toString(); });
    EXPECT_EQ(err, app::AcesEngine::busy_text());
    EXPECT_FALSE(e->stop());
    app::AcesEngine::Config other = c;
    other.device = "cuda";
    EXPECT_FALSE(e->start(other));
    EXPECT_EQ(e->state(), app::AcesEngine::State::Ready);
    // Quitting asks; "Keep working" leaves it running.
    QString asked;
    w->quit_question = [&](const QString& t) { asked = t; return false; };
    EXPECT_FALSE(w->close());
    EXPECT_TRUE(asked.contains("the batch"));
    EXPECT_TRUE(w->batch_running());
    // "Stop and quit": the batch's own master is cancelled by its id.
    w->quit_question = [](const QString&) { return true; };
    EXPECT_TRUE(w->close());
    ASSERT_TRUE(wait_for([&] { return !w->batch_running(); }));
    EXPECT_FALSE(e->busy());
    std::ifstream lf(logf);
    std::string line, cancel;
    int master_id = -1;
    while (std::getline(lf, line)) {
        const auto j = json::parse(line);
        if (j.value("op", "") == "master") master_id = j.value("id", -1);
        if (j.value("op", "") == "cancel") cancel = line;
    }
    ASSERT_FALSE(cancel.empty());
    EXPECT_EQ(json::parse(cancel).value("target", -2), master_id);
    w.reset();
    qunsetenv("RUDRA_TEST_LOG");
    QSettings().clear();
}

TEST(Batch, RunsTheAiFillAfterAShotsMaster) {
    QSettings().clear();
    const auto root = aces_tmp("batch_ai");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    for (const char* f : {"A.1001.png", "A.1002.png", "clip.mov"}) write_text(root / "in" / f, "x");
    QSettings().setValue("batch/file", QString::fromStdString((root / "queue.json").string()));
    const auto cli = root / "rudra-native";
    write_text(cli, R"(#!/usr/bin/env python3
import json, os, sys
def say(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
if sys.argv[1:2] != ["aces-v2-serve"]: sys.exit(64)
say({"event": "ready", "device": "cpu", "runtime": "libtorch", "load_ms": 1.0})
for line in sys.stdin:
    r = json.loads(line)
    if r.get("op") == "quit": break
    if r.get("op") == "master":
        os.makedirs(r["out"], exist_ok=True)
        say({"id": r["id"], "ok": True, "done": True, "cancelled": False, "frames": 2})
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::MainWindow w(false);
    app::AcesEngine::Config c;
    c.cli = cli;
    c.package = root / "pkg";
    c.device = "cpu";
    w.aces_engine()->start(c);
    ASSERT_TRUE(wait_for([&] { return w.aces_engine()->state() == app::AcesEngine::State::Ready; }));
    auto& q = w.batch_queue();
    app::DeliverChoice d;
    d.output = "sequence";
    const auto ids = q.add(app::batch_shots_from({root / "in"}), d, QString::fromStdString((root / "out").string()), 1001);
    ASSERT_EQ(ids.size(), 2u);   // the sequence and the movie
    q.set_ai_fill({}, true);       // every waiting shot
    EXPECT_TRUE(app::batch_effective(*q.find(ids[0])).ai_fill);
    EXPECT_TRUE(app::batch_output_text(*q.find(ids[0])).endsWith("+ AI fill"));
    q.set_ai_fill({ids[1]}, true);
    q.find(ids[1])->choice.output = "movie";
    EXPECT_FALSE(app::batch_effective(*q.find(ids[1])).ai_fill);   // a movie has none
    // The fill runs through the window's job; here a stand-in that finishes at once.
    std::vector<QString> filled;
    w.batch_runner().fill = [&](const app::BatchJob& j, QString*, std::function<void(bool, const QString&)> done) {
        filled.push_back(j.out);
        QTimer::singleShot(10, [done] { done(true, "AI fill: 2 frame(s)"); });
        return true;
    };
    ASSERT_TRUE(w.batch_start());
    ASSERT_TRUE(wait_for([&] { return !w.batch_running(); }));
    ASSERT_EQ(filled.size(), 1u);
    EXPECT_EQ(filled[0].toStdString(), (root / "out" / "A").string());
    EXPECT_EQ(q.find(ids[0])->status, "done");
    EXPECT_EQ(q.find(ids[0])->note, "AI fill done");
    EXPECT_EQ(q.find(ids[1])->status, "done");
    // Kept on disk with the job.
    app::BatchQueue again(root / "queue.json");
    EXPECT_TRUE(again.find(ids[0])->choice.ai_fill);
    QSettings().clear();
}

TEST(Deliver, Hdr10HevcAndProResXqAreTheLearnedEnginesAndSayP3) {
    // Review 28 Oct: HDR10 metadata in a learned PQ movie; XQ in the menu; P3 D65 limited said.
    app::DeliverChoice c;
    c.output = "movie";
    c.codec = "hevc10";
    c.encoding = "logc4";
    c.bits = 16;
    const auto fit = app::deliver_fit(c, true, true, false);
    EXPECT_EQ(fit.codec, "hevc10");
    EXPECT_EQ(fit.encoding, "pq");   // HEVC HDR10 is PQ, 10-bit
    EXPECT_EQ(fit.bits, 10);
    EXPECT_EQ(app::aces_deliver_fields(fit)["log_formats"].toString(), "hevc10");
    EXPECT_EQ(app::render_ext(fit), "mp4");
    EXPECT_EQ(app::deliver_signal(fit, true).container, QString::fromUtf8("MP4 — HEVC HDR10 (Main 10)"));
    EXPECT_EQ(app::deliver_signal(fit, true).primaries, "Rec.2020 container, P3 D65 limited");
    EXPECT_EQ(app::deliver_fit(c, false, true, false).codec, "prores422hq");   // the analytic route has no HEVC here
    c.codec = "prores4444xq";
    c.encoding = "pq";
    c.bits = 10;
    EXPECT_EQ(app::aces_deliver_fields(app::deliver_fit(c, true, true, false))["log_formats"].toString(), "prores4444xq");
    EXPECT_EQ(app::deliver_fit(c, false, true, false).codec, "prores4444");
}

TEST(GenFill, AFillIsReviewedBeforeItGoesOut) {
    // Review 28 Oct: A master / B fill / the invented pixels per frame, flicker flagged.
    QSettings().clear();
    const auto root = aces_tmp("fill_review");
    const auto dir = root / "SH010_aifill";
    std::filesystem::create_directories(dir / "review");
    QString why;
    EXPECT_FALSE(app::read_genfill_review(QString::fromStdString(dir.string()), &why));
    EXPECT_TRUE(why.contains("genfill.json"));
    write_text(dir / "genfill.json", R"({"model": "wan", "review": {"folder": "review", "exposure_stops": -3, "temporal_flag_stops": 0.25},
      "frames": [{"name": "SH010.1001.exr", "generated_fraction": 0.01, "temporal_excess_stops": null, "flagged": false},
                 {"name": "SH010.1002.exr", "generated_fraction": 0.02, "temporal_excess_stops": 0.6, "flagged": true}]})");
    for (const char* n : {"SH010.1001", "SH010.1002"})
        for (const char* s : {"a", "b", "mask"}) {
            QImage img(8, 4, QImage::Format_RGB888);
            img.fill(QColor(std::string(s) == "a" ? 10 : 200, 100, 50));
            ASSERT_TRUE(img.save(QString::fromStdString((dir / "review" / (std::string(n) + "." + s + ".png")).string())));
        }
    auto r = app::read_genfill_review(QString::fromStdString(dir.string()), &why);
    ASSERT_TRUE(r) << why.toStdString();
    ASSERT_EQ(r->frames.size(), 2u);
    EXPECT_FALSE(r->frames[0].temporal);
    EXPECT_EQ(r->flagged(), 1);
    EXPECT_FALSE(r->frames[1].mask.isEmpty());
    app::GenFillReviewDialog d(*r);
    EXPECT_EQ(d.current(), 1);   // the flagged frame first
    EXPECT_EQ(d.showing(), "b");
    QTest::keyClick(&d, Qt::Key_Space);
    EXPECT_EQ(d.showing(), "a");
    QTest::keyClick(&d, Qt::Key_M);
    EXPECT_EQ(d.showing(), "mask");
    QTest::keyClick(&d, Qt::Key_Left);
    EXPECT_EQ(d.current(), 0);
    EXPECT_TRUE(d.findChild<QLabel*>("fillReviewCaption")->text().contains("SH010.1001.exr"));
    // The window opens it from the Deliver tab (here without a window: read only).
    app::MainWindow w(false);
    EXPECT_TRUE(w.review_genfill(QString::fromStdString(dir.string())));
    // The model's parts are found under RUDRA_GENFILL_DIR, not a workstation's drive.
    write_text(root / "gf" / "Wan2.1-VACE-1.3B-diffusers" / "transformer" / "config.json", "{}");
    write_text(root / "gf" / "rudra_genfill.safetensors", "x");
    qputenv("RUDRA_GENFILL_DIR", QByteArray::fromStdString((root / "gf").string()));
    QSettings().setValue("genfill/repo", QString::fromStdString(root.string()));
    write_text(root / "rudra" / "genfill" / "run.py", "");
    QSettings().setValue("genfill/python", "python3");
    auto s = app::genfill_setup(&why);
    EXPECT_TRUE(s) << why.toStdString();
    if (s) {
        EXPECT_EQ(s->lora, root / "gf" / "rudra_genfill.safetensors");
    }
    qunsetenv("RUDRA_GENFILL_DIR");
    QSettings().clear();
}

#ifdef RUDRA_HAVE_STILL_DECODE
TEST(Deliver, ALearnedRenderIsNeverTheAnalyticLookInItsPlace) {
    // Review 28 Oct: the model chosen but failed or loading: the render is refused, not
    // silently swapped for the analytic reconstruction (a different look).
    QSettings().clear();
    const auto root = aces_tmp("no_fallback");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    std::filesystem::create_directories(root / "frames");
    for (const char* f : {"a.png", "b.png"}) {
        QImage img(4, 2, QImage::Format_RGB888);
        img.fill(QColor(250, 240, 230));
        ASSERT_TRUE(img.save(QString::fromStdString((root / "frames" / f).string())));
    }
    const auto cli = root / "rudra-native";
    write_text(cli, "#!/bin/sh\necho '{\"event\":\"error\",\"error\":\"no GPU\"}'\nexit 2\n");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::MainWindow w(false);
    app::AcesEngine::Config c;
    c.cli = cli;
    c.package = root / "pkg";
    c.device = "cuda";
    w.aces_engine()->start(c);
    ASSERT_TRUE(wait_for([&] { return w.aces_engine()->state() == app::AcesEngine::State::Failed; }));
    w.open_source(QString::fromStdString((root / "frames").string()), true);
    ASSERT_TRUE(wait_for([&] { return w.frame_count() == 2u; }));
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString((root / "out").string()));
    EXPECT_TRUE(w.deliver_learned_wanted());
    EXPECT_EQ(find<app::Seg>(w, "deliverSource")->on(), "aces");
    w.run("master");
    QCoreApplication::processEvents();
    EXPECT_FALSE(w.mastering());
    EXPECT_FALSE(w.aces_delivering());
    EXPECT_TRUE(w.findChild<QLabel*>("renderStatus")->text().startsWith("Not rendered")) << w.findChild<QLabel*>("renderStatus")->text().toStdString();
    EXPECT_FALSE(std::filesystem::exists(root / "out"));
    // Choosing Analytic is the colorist's call: then it renders the analytic master.
    w.set_deliver("source", "analytic");
    EXPECT_FALSE(w.deliver_learned_wanted());
    QSettings().clear();
}
#endif

#ifdef RUDRA_HAVE_STILL_DECODE
TEST(Colorist, TrimsAreRememberedPerShotCopiedAndCarryExposureAndContrast) {
    // Review 28 Oct: per-shot trims, copy / paste, and exposure and contrast trims reach the render.
    QSettings().clear();
    const auto root = aces_tmp("shot_trims");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    for (const char* shot : {"A", "B"}) {
        std::filesystem::create_directories(root / shot);
        for (const char* f : {"1.png", "2.png"}) {
            QImage img(4, 2, QImage::Format_RGB888);
            img.fill(QColor(250, 240, 230));
            ASSERT_TRUE(img.save(QString::fromStdString((root / shot / f).string())));
        }
    }
    const auto log = root / "requests.jsonl";
    const auto cli = root / "rudra-native";
    write_text(cli, std::string(R"(#!/usr/bin/env python3
import json, struct, sys
LOG = ")") + log.string() + R"("
def say(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
say({"event": "ready", "device": "cpu", "runtime": "libtorch", "load_ms": 10.0})
for line in sys.stdin:
    r = json.loads(line)
    open(LOG, "a").write(line)
    if r.get("op") == "quit": break
    if r.get("op") == "preview":
        for k in ("model_out", "baseline_out"):
            if k in r: open(r[k], "wb").write(struct.pack("<24f", *([1.0] * 24)))
        if "codes_out" in r: open(r["codes_out"], "wb").write(struct.pack("<8f", *([0.5] * 8)))
        say({"id": r["id"], "ok": True, "width": 4, "height": 2, "bits": 8, "ms": 5.0, "reconstructed_fraction": 0.0,
             "peak": 1.0, "peak_plain": 1.0, "remaining_display_errors": 0, "new_negative_luminance": 0})
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::MainWindow w(false);
    app::AcesEngine::Config c;
    c.cli = cli;
    c.package = root / "pkg";
    c.device = "cpu";
    w.aces_engine()->start(c);
    ASSERT_TRUE(wait_for([&] { return w.aces_engine()->state() == app::AcesEngine::State::Ready; }));
    auto last_preview = [&] {
        std::ifstream in(log);
        std::string line, last;
        while (std::getline(in, line))
            if (line.find("\"preview\"") != std::string::npos) last = line;
        return last.empty() ? json() : json::parse(last);
    };
    w.open_source(QString::fromStdString((root / "A").string()), true);
    ASSERT_TRUE(wait_for([&] { return w.aces_shown() == 0; }));
    aces_v2::AcesTrims a;
    a.exposure_ev = 0.5;
    a.contrast = 1.2;
    w.set_aces_trims(a);
    auto trim = [&](const char* k) {
        const json p = last_preview();
        return p.contains("trims") && p["trims"].is_object() ? p["trims"].value(k, 0.0) : 0.0;
    };
    ASSERT_TRUE(wait_for([&] { return trim("contrast") == 1.2; }));
    EXPECT_DOUBLE_EQ(trim("exposure_ev"), 0.5);
    EXPECT_EQ(w.findChild<QSlider*>("acesExposure")->value(), 5);
    w.copy_aces_trims();
    EXPECT_TRUE(w.findChild<QPushButton*>("acesTrimsPaste")->isEnabled());
    // Shot B: keeps the trims in hand until it gets its own.
    w.open_source(QString::fromStdString((root / "B").string()), true);
    ASSERT_TRUE(wait_for([&] { return w.aces_shown() == 0 && w.frame_count() == 2u && w.shot_trim_key().endsWith("B"); }));
    aces_v2::AcesTrims b;
    b.strength = 0.5;
    w.set_aces_trims(b);
    // Back to A: its own trims return; then B's.
    w.open_source(QString::fromStdString((root / "A").string()), true);
    ASSERT_TRUE(wait_for([&] { return w.shot_trim_key().endsWith("A"); }));
    EXPECT_EQ(w.aces_trims(), a);
    w.open_source(QString::fromStdString((root / "B").string()), true);
    ASSERT_TRUE(wait_for([&] { return w.shot_trim_key().endsWith("B"); }));
    EXPECT_EQ(w.aces_trims(), b);
    w.paste_aces_trims();
    EXPECT_EQ(w.aces_trims(), a);
    EXPECT_EQ(*w.remembered_trims(w.shot_trim_key()), a);   // pasted onto B, and kept with it
    QSettings().clear();
}
#endif

TEST(Batch, AQueueFileThatCannotBeReadIsKeptAsideNotWrittenOver) {
    // Review 28 Oct: a corrupt queue was silently replaced by an empty one.
    const auto d = aces_tmp("batch_corrupt");
    const auto file = d / "queue.json";
    write_text(file, "{\"jobs\": [ {\"id\": \"x\", ");   // cut short
    app::BatchQueue q(file);
    EXPECT_TRUE(q.jobs().empty());
    EXPECT_TRUE(q.load_warning().contains("kept as")) << q.load_warning().toStdString();
    EXPECT_FALSE(std::filesystem::exists(file));
    int aside = 0;
    for (const auto& e : std::filesystem::directory_iterator(d))
        if (e.path().filename().string().rfind("queue.json.corrupt-", 0) == 0) {
            ++aside;
            std::ifstream in(e.path());
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            EXPECT_EQ(text, "{\"jobs\": [ {\"id\": \"x\", ");
        }
    EXPECT_EQ(aside, 1);
}

TEST(Deliver, LearnedTargetsReachTheEngineAndTheUiSetsThem) {
    // Review 28 Oct: PQ 2000 / 4000, HLG, the SDR pass; XQ and HEVC settable from the menu.
    app::DeliverChoice c;
    c.output = "movie";
    c.encoding = "pq";
    c.pq_peak = 4000;
    EXPECT_EQ(app::aces_deliver_fields(app::deliver_fit(c, true, true, false))["log"].toString(), "pq4000");
    EXPECT_EQ(app::deliver_signal(app::deliver_fit(c, true, true, false), true).transfer, QString::fromUtf8("PQ — ACES 2.0 HDR 4000 nits"));
    EXPECT_EQ(app::deliver_fit(c, false, true, false).pq_peak, 1000);   // the analytic PQ is graded on its tab
    c.encoding = "hlg";
    EXPECT_EQ(app::aces_deliver_fields(app::deliver_fit(c, true, true, false))["log"].toString(), "hlg1000");
    EXPECT_EQ(app::deliver_fit(c, false, true, false).encoding, "pq");
    c.encoding = "sdr";
    EXPECT_EQ(app::aces_deliver_fields(app::deliver_fit(c, true, true, false))["log"].toString(), "sdr709");
    EXPECT_EQ(app::deliver_signal(app::deliver_fit(c, true, true, false), true).primaries, "Rec.709");
    c.codec = "hevc10";
    EXPECT_EQ(app::deliver_fit(c, true, true, false).encoding, "pq");   // HDR10 is PQ
    QSettings().clear();
    app::MainWindow w(false);
    w.set_deliver("codec", "prores4444xq");
    EXPECT_EQ(w.deliver_choice().codec, "prores4444xq");
    w.set_deliver("codec", "hevc10");
    EXPECT_EQ(w.deliver_choice().codec, "hevc10");
    w.set_deliver("encoding", "hlg");
    EXPECT_EQ(w.deliver_choice().encoding, "hlg");
    w.set_deliver("pq_peak", "2000");
    EXPECT_EQ(w.deliver_choice().pq_peak, 2000);
    EXPECT_EQ(QSettings().value("deliver/pqPeak").toInt(), 2000);
    QSettings().clear();
}

TEST(Scopes, TheParadeIsOnThePqScaleAndCountsWhatIsOutOfGamut) {
    // Review 28 Oct: RGB parade, PQ code scale, out-of-P3 / out-of-Rec.2020 share.
    PlanarBuffer img(3, 10, 10, 0.0203f / 1.0f);   // diffuse white grey: 203 nits
    // One pixel far outside P3 (a Rec.709 value with a negative blue).
    img.plane(0)[0] = 0.05f;
    img.plane(1)[0] = 0.0f;
    img.plane(2)[0] = -0.01f;
    const auto q = app::qc_scope(img, 300, 120);
    ASSERT_FALSE(q.empty());
    EXPECT_EQ(q.parade.width(), 300);
    EXPECT_NEAR(q.out_p3, 0.01, 1e-9);
    EXPECT_GT(q.out_2020, 0.0);
    EXPECT_GT(q.peak_nits, 203.0);
    // The 203-nit trace sits at PQ(203) in every panel.
    const int row = int(std::lround((1.0 - rudra::pq_oetf(203.0f)) * 119));
    EXPECT_GT(q.parade.pixelColor(199, row).alpha(), 0);   // the G panel, the last column
    app::ParadeView v;
    v.set_scope(q);
    EXPECT_FALSE(v.scope().empty());
    EXPECT_TRUE(app::qc_scope(PlanarBuffer{}, 300, 120).empty());
}

TEST(Batch, TheListUpdatesInPlaceSoAnEditSurvivesARun) {
    // Review 28 Oct: the list was rebuilt on every progress frame, closing any inline editor.
    QSettings().clear();
    const auto root = aces_tmp("batch_inplace");
    QSettings().setValue("batch/file", QString::fromStdString((root / "queue.json").string()));
    for (const char* f : {"A.1001.png", "A.1002.png", "clip.mov"}) write_text(root / "in" / f, "x");
    app::MainWindow w(false);
    auto& q = w.batch_queue();
    const auto ids = q.add(app::batch_shots_from({root / "in"}), w.deliver_choice(), QString::fromStdString(root.string()), 1001);
    ASSERT_EQ(ids.size(), 2u);
    w.open_batch();
    auto* list = w.findChild<QTreeWidget*>("batchList");
    ASSERT_NE(list, nullptr);
    QTreeWidgetItem* first = list->topLevelItem(0);
    q.find(ids[0])->done = 5;   // progress: the same shots
    q.touch();
    EXPECT_EQ(list->topLevelItem(0), first);   // the same row, updated
    q.remove({ids[1]});                       // the shots changed: rebuilt
    EXPECT_EQ(list->topLevelItemCount(), 1);
    QSettings().clear();
}

TEST(JsonLines, WholeLinesInAnyPiecesAndNothingElse) {
    app::JsonLines j;
    std::vector<QJsonObject> got;
    auto take = [&](const QJsonObject& o) { got.push_back(o); };
    j.feed("{\"a\":1}\r\n{\"b\"", take);
    EXPECT_EQ(got.size(), 1u);
    j.feed(":2}\nWarning: not json\n[1,2]\n", take);
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[1].value("b").toInt(), 2);
}

#ifdef RUDRA_HAVE_STILL_DECODE
TEST(Deliver, ATypedTimecodeIsCheckedAtTheMoviesOwnRate) {
    // Review 28 Oct: a 29.97 movie's ;29 frames were refused (checked at 24); frames send their rate.
    if (QStandardPaths::findExecutable("ffmpeg").isEmpty()) GTEST_SKIP() << "ffmpeg is not on the PATH";
    QSettings().clear();
    const auto root = aces_tmp("tc2997");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    const auto movie = root / "clip.mov";
    const std::string cmd = "ffmpeg -v error -y -f lavfi -i testsrc2=s=64x36:r=30000/1001:d=1 -pix_fmt yuv420p -c:v libx264 \"" +
                            movie.string() + "\"";
    ASSERT_EQ(std::system(cmd.c_str()), 0);
    const auto cli = root / "rudra-native";
    write_text(cli, R"(#!/usr/bin/env python3
import json, sys
print(json.dumps({"event": "ready", "device": "cpu", "runtime": "libtorch", "load_ms": 1.0}), flush=True)
for line in sys.stdin:
    if json.loads(line).get("op") == "quit": break
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::MainWindow w(false);
    app::AcesEngine::Config c;
    c.cli = cli;
    c.package = root / "pkg";
    c.device = "cpu";
    w.aces_engine()->start(c);
    ASSERT_TRUE(wait_for([&] { return w.aces_engine()->state() == app::AcesEngine::State::Ready; }));
    w.open_source(QString::fromStdString(movie.string()), true);
    ASSERT_TRUE(wait_for([&] { return w.frame_count() > 0u; }));
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString((root / "out").string()));
    w.findChild<QLineEdit*>("renderName")->setText("SH010");
    w.findChild<QLineEdit*>("renderTimecode")->setText("01:00:00;29");
    w.set_deliver("output", "movie");
    QString why;
    const QJsonObject req = w.aces_deliver_request(&why);
    EXPECT_FALSE(req.isEmpty()) << why.toStdString();
    EXPECT_EQ(req.value("timecode").toString(), "01:00:00;29");
    QSettings().clear();
}
#endif

TEST(AppLayout, TheScopesSitBesideTheViewerAndTheirButtonHidesThem) {
    // 29 Oct 2026: the scopes in their own column beside the viewer, a Scopes button on the viewer bar.
    QSettings().clear();
    app::MainWindow w(false);
    w.resize(1800, 1000);
    w.show();
    auto* dock = w.findChild<QWidget*>("scopesDock");
    auto* btn = w.findChild<QPushButton*>("scopesBtn");
    ASSERT_NE(dock, nullptr);
    ASSERT_NE(btn, nullptr);
    EXPECT_TRUE(dock->isAncestorOf(w.findChild<QWidget*>("scopes")));            // not in the right panel any more
    EXPECT_FALSE(w.findChild<QWidget*>("railRight")->isAncestorOf(w.findChild<QWidget*>("scopes")));
    const bool on = dock->isVisible();
    EXPECT_EQ(btn->isChecked(), on);
    btn->click();
    EXPECT_EQ(dock->isVisible(), !on);
    EXPECT_EQ(btn->isChecked(), !on);
    w.run("scopes");   // the toolbar icon and the menu do the same
    EXPECT_EQ(dock->isVisible(), on);
    EXPECT_EQ(btn->isChecked(), on);
    // Simple: the scopes show on Check only, and the button still shows or hides them.
    w.set_workspace("simple");
    EXPECT_FALSE(dock->isVisible());
    EXPECT_FALSE(btn->isHidden());
    btn->click();
    EXPECT_TRUE(dock->isVisible());
    w.show_tab("check");
    EXPECT_TRUE(dock->isVisible());
    btn->click();
    EXPECT_FALSE(dock->isVisible());
    QSettings().clear();
}

// ---------------------------------------------------------------------------
// The start screen (29 Oct 2026, splash.hpp).
// ---------------------------------------------------------------------------
#include "splash.hpp"

TEST(Splash, PaintsTheMarkTheWordAndTheProgress) {
    app::Splash s;
    EXPECT_EQ(s.size(), QSize(app::Splash::kWidth, app::Splash::kHeight));
    s.set_status("Loading the model");
    s.set_progress(1.7);                       // clamped
    EXPECT_EQ(s.status(), "Loading the model");
    EXPECT_DOUBLE_EQ(s.progress(), 1.0);
    s.settle();
    QImage img(s.size(), QImage::Format_ARGB32_Premultiplied);
    img.fill(Qt::transparent);
    s.render(&img);
    // The Pro ground (#1c1c1c), the accent dot of the lockup, the lit mark at 40 %
    // height, and the 2 px rule filled with the accent (here: full).
    EXPECT_EQ(img.pixelColor(5, 5), QColor(0x1c, 0x1c, 0x1c));
    EXPECT_EQ(img.pixelColor(28, 30), QColor(0x0a, 0x84, 0xff));
    int lit = 0;
    const int cy = int(app::Splash::kHeight * 0.40);
    for (int y = cy - 30; y < cy + 30; ++y)
        for (int x = app::Splash::kWidth / 2 - 4; x <= app::Splash::kWidth / 2 + 4; ++x)
            if (img.pixelColor(x, y).lightness() > 160) ++lit;
    EXPECT_GT(lit, 10);
    const QColor rule = img.pixelColor(app::Splash::kWidth / 2 - 90, cy + 150);
    EXPECT_GT(rule.blue(), 200);
    EXPECT_LT(rule.red(), 60);
}

TEST(Splash, FinishFadesOutOnceTheWindowIsUp) {
    app::Splash s;
    s.show();
    QWidget w;
    w.show();
    s.finish(&w, 50);
    ASSERT_TRUE(wait_for([&] { return !s.isVisible(); }, 3000));
    EXPECT_DOUBLE_EQ(s.progress(), 1.0);
}

// ---------------------------------------------------------------------------
// Simple (29 Oct 2026): Look, Check, Deliver, and the first-run tips.
// ---------------------------------------------------------------------------
#include "coach_marks.hpp"

TEST(AppSimple, LookCheckDeliverAndAdvancedGoesToFull) {
    QSettings().clear();
    Window win("simple");
    auto& w = win.w;
    // Three numbered tabs; Grade is Full's.
    EXPECT_EQ(w.findChild<QPushButton*>("tabRec")->text(), "1  Look");
    EXPECT_EQ(w.findChild<QPushButton*>("tabCheck")->text(), "2  Check");
    EXPECT_EQ(w.findChild<QPushButton*>("tabDeliver")->text(), "3  Deliver");
    EXPECT_TRUE(win.shown(win.get("tabCheck")));
    EXPECT_FALSE(win.shown(win.get("tabGrade")));
    // The screen keeps to the picture: no compare line, pipeline, probe, collections or model card.
    for (const char* id : {"compareBar", "pipe", "probeBtn", "probePanelWrap", "collections", "modelCard", "viewMode",
                           "viewLayer", "changesBtn", "inventedBtn", "guideBtn", "frameStats", "logCard", "recSourceCard",
                           "mode", "preserve", "scopesDock"})
        EXPECT_FALSE(win.shown(win.get(id))) << id;
    for (const char* id : {"holdHint", "viewMenuBtn", "zoomSeg", "scopesBtn", "emptySimple", "lookAdvanced", "lookWaiting"})
        EXPECT_TRUE(win.shown(win.get(id))) << id;
    // No learned model: the analytic look (1.0's guaranteed path) with Exposure, Strength and Peak.
    EXPECT_TRUE(win.shown(win.get("recAnalytic")));
    EXPECT_TRUE(win.shown(win.get("recReconCard")));
    EXPECT_EQ(w.findChild<QLabel*>("recReconTitle")->text(), "Look");
    EXPECT_FALSE(win.shown(win.get("emptyFull")));
    EXPECT_EQ(w.findChild<QLabel*>("holdHint")->text(), "Hold B for the original SDR");
    // The viewer fills what the compare line and the pipeline left.
    const auto tools = win.box(win.get("viewerTools")), viewer = win.box(win.get("viewer")), time = win.box(win.get("transport"));
    EXPECT_EQ(viewer[1], tools[1] + tools[3]);
    EXPECT_EQ(time[1], viewer[1] + viewer[3]);
    // Check: the summary, the stills, the scopes beside the picture.
    w.findChild<QPushButton*>("tabCheck")->click();
    EXPECT_EQ(w.shown_tab(), "check");
    EXPECT_TRUE(win.shown(win.get("checkSummary")));
    EXPECT_TRUE(win.shown(win.get("scopesDock")));
    EXPECT_EQ(w.findChild<QLabel*>("checkPeak")->text(), "—");
    // Deliver: four choices; Custom… opens the full card.
    w.findChild<QPushButton*>("tabDeliver")->click();
    EXPECT_FALSE(win.shown(win.get("scopesDock")));
    EXPECT_TRUE(win.shown(win.get("deliverPresets")));
    EXPECT_EQ(w.deliver_preset(), "master");
    EXPECT_FALSE(win.shown(win.get("deliverFullCards")));
    for (const char* id : {"renderTimecode", "renderReel", "renderStart", "btnReprocess"}) EXPECT_FALSE(win.shown(win.get(id))) << id;
    EXPECT_TRUE(win.shown(win.get("renderFile")));
    EXPECT_TRUE(win.shown(win.get("btnMaster")));
    auto* presets = dynamic_cast<app::Seg*>(win.get("deliverPreset"));
    ASSERT_NE(presets, nullptr);
    EXPECT_FALSE(presets->button("hdr10")->isEnabled());   // no shot: no movie
    presets->button("custom")->click();
    EXPECT_EQ(w.deliver_preset(), "custom");
    EXPECT_TRUE(win.shown(win.get("deliverFullCards")));
    presets->button("master")->click();
    EXPECT_EQ(w.deliver_preset(), "master");
    EXPECT_FALSE(win.shown(win.get("deliverFullCards")));
    // Grade asked for in Simple shows Look; Check asked for in Full shows Reconstruct.
    w.show_tab("grade");
    EXPECT_EQ(w.shown_tab(), "rec");
    // Advanced: Full, as it was.
    w.findChild<QPushButton*>("lookAdvanced")->click();
    EXPECT_EQ(w.workspace(), "full");
    w.show_tab("check");
    EXPECT_EQ(w.shown_tab(), "rec");
    EXPECT_EQ(w.findChild<QPushButton*>("tabRec")->text(), "Reconstruct");
    for (const char* id : {"compareBar", "pipe", "recAnalytic", "frameStats", "collections", "emptyFull"})
        EXPECT_TRUE(win.shown(win.get(id))) << id;
    for (const char* id : {"holdHint", "viewMenuBtn", "tabCheck", "deliverPresets", "emptySimple"})
        EXPECT_FALSE(win.shown(win.get(id))) << id;
    QSettings().clear();
}

#ifdef RUDRA_HAVE_STILL_DECODE
TEST(AppSimple, PresetsSetTheDeliverChoice) {
    QSettings().clear();
    app::MainWindow w(false);
    w.set_workspace("simple");
    w.add_files({decode_dir() / "png8_rgb.png", decode_dir() / "png8_rgb.png"});   // a folder's worth: movies allowed
    // No learned model: the grading master is the analytic EXR; the movies need the model.
    w.apply_deliver_preset("master");
    EXPECT_EQ(w.deliver_preset(), "master");
    EXPECT_EQ(w.deliver_choice().source, "analytic");
    EXPECT_EQ(w.deliver_choice().output, "sequence");
    EXPECT_EQ(w.deliver_choice().gamut, "ap0");
    w.apply_deliver_preset("hdr10");
    EXPECT_EQ(w.deliver_preset(), "master");   // refused, nothing changed
    auto* presets = dynamic_cast<app::Seg*>(w.findChild<QWidget*>("deliverPreset"));
    ASSERT_NE(presets, nullptr);
    EXPECT_FALSE(presets->button("hdr10")->isEnabled());
    EXPECT_TRUE(presets->button("hdr10")->toolTip().contains("learned ACES 2.0 model"));
#ifdef RUDRA_HAVE_STILL_DECODE
    // The learned model running (a stand-in engine): the movies, through it.
    const auto root = aces_tmp("simple-presets");
    write_text(root / "pkg" / "manifest.json", aces_manifest(true, true).dump());
    const auto cli = root / "rudra-native";
    write_text(cli, R"(#!/usr/bin/env python3
import json, sys
def say(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
if sys.argv[1:2] != ["aces-v2-serve"]: sys.exit(64)
say({"event": "ready", "device": "cpu", "runtime": "libtorch", "load_ms": 1.0})
for line in sys.stdin:
    r = json.loads(line)
    if r.get("op") == "quit": break
    say({"id": r.get("id"), "ok": False, "error": "not in this test"})
)");
    std::filesystem::permissions(cli, std::filesystem::perms::owner_all);
    app::AcesEngine::Config c;
    c.cli = cli;
    c.package = root / "pkg";
    c.device = "cpu";
    w.aces_engine()->start(c);
    ASSERT_TRUE(wait_for([&] { return w.aces_engine()->state() == app::AcesEngine::State::Ready; }));
    w.apply_deliver_preset("hdr10");
    EXPECT_EQ(w.deliver_preset(), "hdr10");
    EXPECT_EQ(w.deliver_choice().source, "aces");
    EXPECT_EQ(w.deliver_choice().output, "movie");
    EXPECT_EQ(w.deliver_choice().codec, "hevc10");
    EXPECT_EQ(w.deliver_choice().encoding, "pq");
    EXPECT_EQ(w.deliver_choice().pq_peak, 1000);
    w.apply_deliver_preset("hlg");
    EXPECT_EQ(w.deliver_preset(), "hlg");
    EXPECT_EQ(w.deliver_choice().codec, "prores422hq");
    EXPECT_EQ(w.deliver_choice().encoding, "hlg");
    w.apply_deliver_preset("master");
    EXPECT_EQ(w.deliver_preset(), "master");
    EXPECT_EQ(w.deliver_choice().source, "aces");
#endif
    QSettings().clear();
}
#endif  // RUDRA_HAVE_STILL_DECODE

TEST(AppSimple, TheViewMenuDoesWhatTheCompareControlsDid) {
    QSettings().clear();
    app::MainWindow w(false);
    w.set_workspace("simple");
    auto* menu = w.findChild<QMenu*>("viewMenu");
    ASSERT_NE(menu, nullptr);
    auto act = [&](const char* id) {
        auto* a = menu->findChild<QAction*>(id);
        EXPECT_NE(a, nullptr) << id;
        if (a) a->trigger();
    };
    act("viewSdr");
    EXPECT_EQ(w.session().show, "baseline");
    EXPECT_TRUE(w.findChild<QPushButton*>("viewMenuBtn")->text().contains("Original SDR"));
    act("viewWipe");
    EXPECT_TRUE(w.session().wipe.has_value());
    act("viewFalseColour");
    EXPECT_EQ(w.session().view_layer, 1);
    act("viewChanged");
    EXPECT_TRUE(w.session().show_changes);
    act("viewHdr");
    EXPECT_EQ(w.session().show, "model");
    EXPECT_FALSE(w.session().wipe.has_value());
    EXPECT_EQ(w.session().view_layer, 0);
    EXPECT_FALSE(w.session().show_changes);
    EXPECT_TRUE(w.findChild<QPushButton*>("viewMenuBtn")->text().contains("HDR"));
    QSettings().clear();
}

#ifdef RUDRA_HAVE_STILL_DECODE
TEST(AppSimple, TheWelcomeThenFourTipsOnceAndHelpBringsThemBack) {
    QSettings().clear();
    Window win("full");
    auto& w = win.w;
    auto* coach = w.coach_marks();
    ASSERT_NE(coach, nullptr);
    // A new user: the welcome card (show_getting_started_once, after the first-run check).
    w.show_getting_started_once();
    EXPECT_EQ(QSettings().value("help/gettingStarted").toInt(), 2);
    EXPECT_EQ(w.sheet(), nullptr);   // not the five pages any more
    QTest::qWait(700);
    ASSERT_TRUE(coach->isVisible());
    EXPECT_TRUE(coach->welcoming());
    EXPECT_EQ(w.findChild<QPushButton*>("coachSkip")->text(), "I know RUDRA — no tips");
    w.findChild<QPushButton*>("coachNext")->click();   // Got it
    EXPECT_FALSE(coach->isVisible());
    EXPECT_EQ(w.workspace(), "simple");
    EXPECT_TRUE(w.tips_pending());   // the tips wait for a shot
    // A shot opens: the four tips.
    w.add_files({decode_dir() / "png8_rgb.png"});
    QTest::qWait(1100);
    ASSERT_TRUE(coach->isVisible());
    EXPECT_FALSE(coach->welcoming());
    EXPECT_EQ(coach->count(), 4);
    EXPECT_EQ(coach->step(), 0);
    EXPECT_FALSE(coach->hole().isEmpty());   // the viewer, lit
    EXPECT_EQ(w.findChild<QLabel*>("coachCount")->text(), "Tip 1 of 4");
    EXPECT_EQ(w.findChild<QLabel*>("coachTitle")->text(), "This is your HDR");
    // The viewer has no room beside it: the card sits over its lower part, inside the window.
    EXPECT_TRUE(coach->rect().contains(QRect(coach->card()->pos(), coach->card()->size())));
    w.findChild<QPushButton*>("coachNext")->click();
    EXPECT_EQ(w.findChild<QLabel*>("coachTitle")->text(), "Shape it");
    // The trims: the card beside them, not over them.
    EXPECT_FALSE(QRect(coach->card()->pos(), coach->card()->size()).intersects(coach->hole()));
    w.findChild<QPushButton*>("coachNext")->click();
    EXPECT_EQ(w.shown_tab(), "check");           // each tip brings its tab forward
    EXPECT_TRUE(win.shown(win.get("scopesDock")));
    EXPECT_FALSE(coach->hole().isEmpty());
    w.findChild<QPushButton*>("coachNext")->click();
    EXPECT_EQ(w.shown_tab(), "deliver");
    EXPECT_EQ(w.findChild<QPushButton*>("coachNext")->text(), "Done");
    EXPECT_FALSE(coach->hole().isEmpty());
    w.findChild<QPushButton*>("coachNext")->click();
    EXPECT_FALSE(coach->isVisible());
    EXPECT_EQ(w.shown_tab(), "rec");
    EXPECT_TRUE(QSettings().value("help/tipsDone").toBool());
    // Once: not again.
    w.show_tips_once();
    QTest::qWait(700);
    EXPECT_FALSE(coach->isVisible());
    // Help > Show tips: again, from Full too; Esc skips.
    w.set_workspace("full");
    QAction* tips = w.findChild<QAction*>("showTips");
    ASSERT_NE(tips, nullptr);
    tips->trigger();
    EXPECT_TRUE(coach->isVisible());
    EXPECT_EQ(w.workspace(), "simple");
    QTest::keyClick(coach, Qt::Key_Escape);
    EXPECT_FALSE(coach->isVisible());
    QSettings().clear();
}

TEST(AppSimple, NoTipsMeansNone) {
    QSettings().clear();
    app::MainWindow w(false);
    w.show();
    (void)QTest::qWaitForWindowExposed(&w);
    w.show_welcome_tips();
    ASSERT_TRUE(w.coach_marks()->isVisible());
    w.findChild<QPushButton*>("coachSkip")->click();   // I know RUDRA
    EXPECT_FALSE(w.coach_marks()->isVisible());
    EXPECT_TRUE(QSettings().value("help/tipsDone").toBool());
    EXPECT_FALSE(w.tips_pending());
    w.add_files({decode_dir() / "png8_rgb.png"});
    QTest::qWait(1100);
    EXPECT_FALSE(w.coach_marks()->isVisible());
    QSettings().clear();
}
#endif  // RUDRA_HAVE_STILL_DECODE

// ---------------------------------------------------------------------------
// Model downloads and About (29 Oct 2026).
// ---------------------------------------------------------------------------
#include <QCryptographicHash>
#include <QProgressBar>
#include <QTemporaryDir>
#include "model_dialogs.hpp"
#include "model_downloads.hpp"

namespace {

QString sha256_of(const QByteArray& b) { return QString::fromLatin1(QCryptographicHash::hash(b, QCryptographicHash::Sha256).toHex()); }

// A served model: two files under `src`, an index naming them by file:// URL.
QByteArray served_index(const QString& src, const QByteArray& a, const QByteArray& b, const QString& sha_b = {}) {
    QDir().mkpath(src);
    for (const auto& [n, data] : std::vector<std::pair<QString, QByteArray>>{{"manifest.json", a}, {"weights.bin", b}}) {
        QFile f(src + "/" + n);
        f.open(QIODevice::WriteOnly);
        f.write(data);
    }
    QJsonArray files;
    files.append(QJsonObject{{"path", "manifest.json"}, {"url", QUrl::fromLocalFile(src + "/manifest.json").toString()},
                             {"size", a.size()}, {"sha256", sha256_of(a)}});
    files.append(QJsonObject{{"path", "sub/weights.bin"}, {"url", QUrl::fromLocalFile(src + "/weights.bin").toString()},
                             {"size", b.size()}, {"sha256", sha_b.isEmpty() ? sha256_of(b) : sha_b}});
    QJsonArray models;
    models.append(QJsonObject{{"id", "demo"}, {"title", "Demo model"}, {"version", "1.0"}, {"channel", "preview"},
                              {"kind", "package"}, {"summary", "a test"}, {"licence", "Apache-2.0"},
                              {"folder", "demo_pkg"}, {"files", files}});
    return QJsonDocument(QJsonObject{{"version", 1}, {"models", models}}).toJson();
}

}  // namespace

TEST(AppModels, TheIndexIsReadSafely) {
    QTemporaryDir tmp;
    const QByteArray good = served_index(tmp.path() + "/src", "{}", QByteArray(5000, 'x'));
    QString why;
    auto m = app::parse_model_index(good, &why);
    ASSERT_TRUE(m) << why.toStdString();
    ASSERT_EQ(m->size(), 1u);
    EXPECT_EQ((*m)[0].size(), 5002);
    EXPECT_EQ((*m)[0].files[1].path, "sub/weights.bin");
    EXPECT_FALSE(app::parse_model_index("not json", &why));
    EXPECT_FALSE(app::parse_model_index(R"({"version": 2, "models": []})", &why));
    EXPECT_TRUE(why.contains("update RUDRA"));
    // An entry with a path out of its folder, or a bad checksum, is left out.
    for (const char* bad : {R"("path": "../evil")", R"("path": "/abs")", R"("path": "a\\b")"}) {
        const QString j = QStringLiteral(R"({"version":1,"models":[{"id":"x","files":[{%1,"url":"https://a/b","size":3,"sha256":"%2"}]}]})")
                              .arg(QString::fromLatin1(bad), QString(64, 'a'));
        auto r = app::parse_model_index(j.toUtf8());
        ASSERT_TRUE(r);
        EXPECT_TRUE(r->empty()) << bad;
    }
    auto r = app::parse_model_index(R"({"version":1,"models":[{"id":"x","files":[{"path":"a","url":"http://a/b","size":3,"sha256":"00"}]}]})");
    ASSERT_TRUE(r);
    EXPECT_TRUE(r->empty());   // plain http, short checksum
    EXPECT_EQ(app::human_size(2100000000), "2.1 GB");
    EXPECT_EQ(app::human_size(1500000), "1.5 MB");
}

TEST(AppModels, ADownloadIsCheckedThenInstalledAndAPartIsNeverUsed) {
    QSettings().clear();
    QTemporaryDir tmp;
    const QByteArray weights(300000, 'w');
    const QByteArray index = served_index(tmp.path() + "/src", R"({"name":"demo"})", weights);
    QFile idx(tmp.path() + "/models.json");
    ASSERT_TRUE(idx.open(QIODevice::WriteOnly));
    idx.write(index);
    idx.close();
    QNetworkAccessManager net;
    QObject owner;
    app::ModelDownloads md(&net, &owner);
    md.set_root(tmp.path() + "/models");
    int changes = 0;
    md.listen(&owner, [&] { ++changes; });
    md.fetch_index(QUrl::fromLocalFile(tmp.path() + "/models.json").toString());
    ASSERT_TRUE(wait_for([&] { return !md.fetching(); }));
    ASSERT_EQ(md.models().size(), 1u) << md.index_status().toStdString();
    EXPECT_EQ(md.state("demo"), app::InstallState::NotInstalled);
    // A stale .part from an earlier try: replaced, never trusted.
    QDir().mkpath(tmp.path() + "/models/demo_pkg/sub");
    {
        QFile part(tmp.path() + "/models/demo_pkg/sub/weights.bin.part");
        part.open(QIODevice::WriteOnly);
        part.write(QByteArray(1000, 'z'));
    }
    EXPECT_EQ(md.state("demo"), app::InstallState::Partial);
    QString folder;
    md.installed = [&](const app::RemoteModel&, const QString& f) { folder = f; };
    ASSERT_TRUE(md.download("demo"));
    QString why;
    EXPECT_FALSE(md.download("demo", &why));   // one at a time
    ASSERT_TRUE(wait_for([&] { return md.active().isEmpty(); }));
    EXPECT_EQ(md.error("demo"), "");
    EXPECT_EQ(md.state("demo"), app::InstallState::Installed);
    EXPECT_EQ(QDir(folder).absolutePath(), QDir(tmp.path() + "/models/demo_pkg").absolutePath());
    QFile got(tmp.path() + "/models/demo_pkg/sub/weights.bin");
    ASSERT_TRUE(got.open(QIODevice::ReadOnly));
    EXPECT_EQ(got.readAll(), weights);
    EXPECT_FALSE(QFileInfo::exists(tmp.path() + "/models/demo_pkg/sub/weights.bin.part"));
    EXPECT_TRUE(QFileInfo::exists(tmp.path() + "/models/demo_pkg/.rudra-download.json"));
    EXPECT_GT(changes, 2);
    QSettings().clear();
}

TEST(AppModels, AFileThatFailsItsChecksumIsDeletedAndSaysSo) {
    QSettings().clear();
    QTemporaryDir tmp;
    const QByteArray index = served_index(tmp.path() + "/src", "{}", QByteArray(4000, 'w'), QString(64, '0'));
    QNetworkAccessManager net;
    QObject owner;
    app::ModelDownloads md(&net, &owner);
    md.set_root(tmp.path() + "/models");
    md.set_models(*app::parse_model_index(index));
    ASSERT_TRUE(md.download("demo"));
    ASSERT_TRUE(wait_for([&] { return md.active().isEmpty(); }));
    EXPECT_TRUE(md.error("demo").contains("checksum")) << md.error("demo").toStdString();
    EXPECT_NE(md.state("demo"), app::InstallState::Installed);
    EXPECT_FALSE(QFileInfo::exists(tmp.path() + "/models/demo_pkg/sub/weights.bin"));
    EXPECT_FALSE(QFileInfo::exists(tmp.path() + "/models/demo_pkg/sub/weights.bin.part"));
    // Discard: the parts that are left go.
    md.cancel("demo");
    EXPECT_EQ(md.error("demo"), "");
    QSettings().clear();
}

TEST(AppModels, TheModelWindowListsDownloadsAndSimpleOffersOne) {
    QSettings().clear();
    QTemporaryDir tmp;
    QSettings().setValue("models/indexUrl", QUrl::fromLocalFile(tmp.path() + "/none.json").toString());
    app::MainWindow w(false);
    w.set_model_roots({std::filesystem::path(tmp.path().toStdString()) / "roots"});
    auto* md = w.model_downloads();
    md->set_root(tmp.path() + "/models");
    md->set_models(*app::parse_model_index(served_index(tmp.path() + "/src", "{}", QByteArray(100000, 'w'))));
    w.open_model_manager();
    auto* dlg = find<app::ModelManager>(w, "modelManager");
    ASSERT_NE(dlg, nullptr);
    ASSERT_TRUE(wait_for([&] { return !md->fetching(); }));
    md->set_models(*app::parse_model_index(served_index(tmp.path() + "/src", "{}", QByteArray(100000, 'w'))));
    auto* get = dlg->findChild<QPushButton*>("dlGet:demo");
    ASSERT_NE(get, nullptr);
    EXPECT_EQ(get->text(), "Download");
    EXPECT_TRUE(dlg->findChild<QLabel*>("dlTitle:demo")->text().contains("PREVIEW"));
    EXPECT_TRUE(dlg->findChild<QLabel*>("dlSub:demo")->text().contains("Apache-2.0"));
    get->click();
    ASSERT_TRUE(wait_for([&] { return md->state("demo") == app::InstallState::Installed && md->active().isEmpty(); }));
    QApplication::processEvents();
    EXPECT_NE(dlg->findChild<QPushButton*>("dlShow:demo"), nullptr);
    EXPECT_NE(dlg->findChild<QPushButton*>("dlUse:demo"), nullptr);
    // The download's folder is a model folder now.
    const auto roots = w.model_roots();
    EXPECT_NE(std::find(roots.begin(), roots.end(), std::filesystem::path((tmp.path() + "/models").toStdU16String())), roots.end());
    dlg->close();
    // Simple, no learned model here: Look offers the download.
    w.resize(1600, 1000);
    w.show();
    w.set_workspace("simple");
    QApplication::processEvents();
    auto* offer = w.findChild<QPushButton*>("lookDownload");
    ASSERT_NE(offer, nullptr);
    EXPECT_TRUE(offer->isVisible());
    EXPECT_TRUE(w.findChild<QLabel*>("lookWaiting")->text().contains("optional download"));
    QSettings().clear();
}

#ifdef RUDRA_HAVE_STILL_DECODE
TEST(AppAbout, TheVersionTheMachineAndTheLinks) {
    QSettings().clear();
    app::MainWindow w(false);
    w.run("about");
    ASSERT_NE(w.sheet(), nullptr);
    EXPECT_EQ(w.sheet()->objectName(), "aboutDialog");
    EXPECT_EQ(w.sheet()->windowTitle(), "About RUDRA");
    auto* site = w.sheet()->findChild<QPushButton*>("aboutWebsite");
    ASSERT_NE(site, nullptr);
    EXPECT_EQ(site->toolTip(), "https://www.fxtdstudios.com");
    EXPECT_TRUE(site->text().startsWith("www.fxtdstudios.com"));
    for (const char* id : {"aboutGuide", "aboutNotes", "aboutLicences", "aboutIssue", "aboutUpdates", "aboutCopy", "aboutClose"})
        EXPECT_NE(w.sheet()->findChild<QPushButton*>(id), nullptr) << id;
    EXPECT_TRUE(w.sheet()->findChild<QLabel*>("aboutCopyright")->text().contains("Apache License 2.0"));
    w.sheet()->findChild<QPushButton*>("aboutCopy")->click();
    EXPECT_EQ(last_log(w), "About details copied");
    w.sheet()->findChild<QPushButton*>("aboutClose")->click();
    QApplication::processEvents();
    EXPECT_TRUE(w.sheet() == nullptr || !w.sheet()->isVisible());
    QSettings().clear();
}

#endif  // RUDRA_HAVE_STILL_DECODE (uses decode_dir / last_log)
// Open… takes movies too (29 Oct 2026): every movie the reader opens is in the filter.
TEST(AppOpen, TheOpenDialogOffersMoviesAndStills) {
    const QString f = app::open_file_filter();
    for (const char* s : {"*.mov", "*.mp4", "*.mxf", "*.mkv", "*.png", "*.jpg", "*.tif", "*.exr", "*.rudra"}) {
        if (QString(s) == "*.exr") continue;   // EXR is an output, not an SDR source
        EXPECT_TRUE(f.contains(s)) << s;
    }
    EXPECT_TRUE(f.startsWith("Stills and movies ("));
    EXPECT_TRUE(f.contains(";;Movies ("));
    EXPECT_TRUE(f.contains(";;All files (*)"));
}

// ---------------------------------------------------------------------------
// The AI fill v3 (30 Oct 2026): the artist's highlight brightness offset.
// ---------------------------------------------------------------------------
TEST(GenFill, TheArtistsBrightnessOffsetReachesTheFill) {
    app::GenFillSetup s{"/py", "/repo", "/models/wan", "/runs/lora.safetensors"};
    app::GenFillOptions o;
    EXPECT_FALSE(app::genfill_arguments(s, o, "m", "o").contains("--brightness-offset"));   // 0 = the estimate
    o.brightness = -0.75;
    const QStringList a = app::genfill_arguments(s, o, "m", "o");
    const qsizetype i = a.indexOf("--brightness-offset");
    ASSERT_GE(i, 0);
    EXPECT_EQ(a[i + 1], "-0.75");
    app::DeliverChoice c;
    c.ai.brightness = 9.0;
    EXPECT_DOUBLE_EQ(app::deliver_fit(c, true, false, false).ai.brightness, 3.0);   // clamped to +-3 stops
    QSettings().clear();
    app::MainWindow w(false);
    w.set_deliver("ai_brightness", "1.25");
    EXPECT_DOUBLE_EQ(w.deliver_choice().ai.brightness, 1.25);
    EXPECT_DOUBLE_EQ(QSettings().value("deliver/aiBrightness").toDouble(), 1.25);
    w.set_deliver("ai_brightness", "7");                                               // refused: out of range
    EXPECT_DOUBLE_EQ(w.deliver_choice().ai.brightness, 1.25);
    auto* bs = w.findChild<QDoubleSpinBox*>("deliverAiBrightness");
    ASSERT_NE(bs, nullptr);
    bs->setValue(-0.5);
    EXPECT_DOUBLE_EQ(w.deliver_choice().ai.brightness, -0.5);
    QSettings().clear();
}

#ifdef RUDRA_HAVE_STILL_DECODE
// A downloaded AI fill LoRA is the one Deliver > AI fill uses (31 Oct 2026).
TEST(AppModels, ADownloadedLoraBecomesTheAiFills) {
    QSettings().clear();
    QTemporaryDir tmp;
    QSettings().setValue("models/indexUrl", QUrl::fromLocalFile(tmp.path() + "/none.json").toString());
    QDir().mkpath(tmp.path() + "/src");
    const QByteArray lora(2048, 'l'), bright("{\"w\":1}");
    QJsonArray files;
    for (const auto& [name, data] : std::vector<std::pair<QString, QByteArray>>{{"rudra_genfill.safetensors", lora}, {"brightness.json", bright}}) {
        QFile f(tmp.path() + "/src/" + name);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly));
        f.write(data);
        f.close();
        files.append(QJsonObject{{"path", name}, {"url", QUrl::fromLocalFile(f.fileName()).toString()}, {"size", data.size()},
                                 {"sha256", QString::fromLatin1(QCryptographicHash::hash(data, QCryptographicHash::Sha256).toHex())}});
    }
    QJsonArray models;
    models.append(QJsonObject{{"id", "fill"}, {"title", "AI fill"}, {"version", "3"}, {"kind", "genfill"},
                              {"folder", "rudra_genfill_v3"}, {"files", files}});
    app::MainWindow w(false);
    auto* md = w.model_downloads();
    md->set_root(tmp.path() + "/models");
    md->set_models(*app::parse_model_index(QJsonDocument(QJsonObject{{"version", 1}, {"models", models}}).toJson()));
    ASSERT_TRUE(md->download("fill"));
    ASSERT_TRUE(wait_for([&] { return md->active().isEmpty(); }));
    EXPECT_EQ(md->state("fill"), app::InstallState::Installed);
    EXPECT_EQ(QDir::cleanPath(QSettings().value("genfill/lora").toString()),
              QDir::cleanPath(tmp.path() + "/models/rudra_genfill_v3/rudra_genfill.safetensors"));
    QSettings().clear();
}
#endif
