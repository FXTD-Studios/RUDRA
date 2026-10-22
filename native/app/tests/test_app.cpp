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
#include <QSettings>
#include <QClipboard>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QPointer>
#include <QUrl>
#include <QTest>
#include <QTreeWidget>

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
        } else out.push_back(id_of(a));
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
};

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
            for (auto* b : x->findChildren<QPushButton*>(QString(), Qt::FindDirectChildrenOnly)) got.push_back(norm(b->text()).toStdString());
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
    for (const auto& f : snap["fields"]) fwant.push_back(f.get<std::vector<std::string>>());
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
    for (const std::string state : {"full", "simple", "rails_hidden"}) {
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
        EXPECT_EQ(win.shown(win.get("railLeft")), rails) << state;
        EXPECT_EQ(win.shown(win.get("railRight")), rails) << state;
        if (rails) {
            EXPECT_EQ(side[2], 240) << state;
            EXPECT_EQ(side[0], 0) << state;
            EXPECT_EQ(insp[2], 320) << state;
            EXPECT_EQ(insp[0] + insp[2], W) << state;
            EXPECT_EQ(centre[0], 240) << state;
            EXPECT_EQ(centre[2], W - 240 - 320) << state;
        } else {
            EXPECT_EQ(centre[0], 0) << state;
            EXPECT_EQ(centre[2], W) << state;
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
    EXPECT_EQ(w.findChild<QLabel*>("pipeMaster")->text().toStdString(), "linear Rec.2020 EXR, half");
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
        if (labels[i]->text() == "Device") device = labels[i + 1]->text() == "CPU";   // $("device").textContent
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
    if (!a.movie.isEmpty()) EXPECT_TRUE(r["movie"]["ok"].get<bool>());
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
    for (const char* format : {"hdr10", "hlg", "prores422hq", "prores4444",
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
    for (const auto& format : {"hlg", "prores422hq", "prores4444"}) {
        SCOPED_TRACE(format);
        w.findChild<QLineEdit*>("renderName")->setText(format);
        ASSERT_TRUE(w.export_frames_to_movie(format, 24.0));
        const auto output = out / (std::string(format) + "_" + format +
                                  (std::string(format) == "hlg" ? ".mp4" : ".mov"));
        ASSERT_TRUE(wait_for([&] { return !w.mastering() && !w.encoding_movie() && std::filesystem::exists(output); }, 60000))
            << status_of(w).toStdString();
        ASSERT_TRUE(std::filesystem::exists(output)) << status_of(w).toStdString();
        auto probe = rudra::probe_video(output);
        ASSERT_TRUE(probe) << probe.error().message;
        ASSERT_EQ(probe->video_streams.size(), 1u);
        EXPECT_EQ(probe->frames.size(), 4u);
        EXPECT_EQ(probe->video_streams.front().avg_frame_rate, "24/1");
        EXPECT_EQ(probe->video_streams.front().codec_name, std::string(format) == "hlg" ? "hevc" : "prores");
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
    app::MainWindow w(false, true);
    fake_headroom(w);
    w.open_source(QString::fromStdString(headroom_clip().string()));
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

