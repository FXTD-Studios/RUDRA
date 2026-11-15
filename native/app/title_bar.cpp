// The merged title bar (11 Nov 2026): on Windows the system's white caption and the menu row go,
// and the toolbar becomes the window's top edge, in the app's own colours. The menus move behind
// one button at the toolbar's left; minimise, maximise and close are the toolbar's last three.
//
// The window stays a real Windows window: it keeps WS_CAPTION and WS_THICKFRAME, so dragging the
// bar, double-click to maximise, Aero snap, the Win+arrow keys and the system menu all behave as
// in any other program. Only the non-client area is taken over (WM_NCCALCSIZE: the client is the
// whole window) and the hit test says which part of the bar is the caption (WM_NCHITTEST).
// Not there: the snap-layout flyout on hovering the maximise button (Win+Z opens it).
//
// Off: Settings key window/mergedTitleBar = false, or RUDRA_NATIVE_TITLE_BAR set. The app's tests
// and its --check modes never turn it on (main.cpp asks for it), so they see the plain window.
// macOS and Linux keep the system's title bar.

#include "main_window.hpp"

#include <algorithm>

#include <QAbstractButton>
#include <QCursor>
#include <QEvent>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QSettings>
#include <QShowEvent>
#include <QToolButton>

#ifdef _WIN32
#include <windows.h>
#include <dwmapi.h>
#endif

namespace rudra::app {

#ifdef _WIN32
namespace {

// Segoe MDL2 Assets is on every Windows 10 and 11; these are the system's own caption glyphs.
constexpr char16_t kGlyphMenu = 0xE700, kGlyphMin = 0xE921, kGlyphMax = 0xE922, kGlyphRestore = 0xE923, kGlyphClose = 0xE8BB;

QToolButton* bar_button(QWidget* bar, const char* id, char16_t glyph, const QString& tip, int width) {
    auto* b = new QToolButton(bar);
    b->setObjectName(id);
    b->setText(QString(QChar(glyph)));
    b->setToolTip(tip);
    b->setFocusPolicy(Qt::NoFocus);
    b->setFixedSize(width, 28);
    QFont f(QStringLiteral("Segoe MDL2 Assets"));
    f.setPixelSize(10);
    b->setFont(f);
    return b;
}

// Keeps the maximise button's glyph in step with the window (it is maximised by snap and by the
// keyboard too, not only by the button).
class StateWatch : public QObject {
public:
    StateWatch(QWidget* window, QToolButton* button) : QObject(window), window_(window), max_(button) { window->installEventFilter(this); }
    bool eventFilter(QObject* o, QEvent* e) override {
        if (o == window_ && e->type() == QEvent::WindowStateChange) {
            const bool on = window_->isMaximized();
            max_->setText(QString(QChar(on ? kGlyphRestore : kGlyphMax)));
            max_->setToolTip(on ? QStringLiteral("Restore") : QStringLiteral("Maximise"));
        }
        return false;
    }

private:
    QWidget* window_;
    QToolButton* max_;
};

}  // namespace
#endif

void MainWindow::use_merged_title_bar() {
#ifdef _WIN32
    if (merged_title_ || qEnvironmentVariableIsSet("RUDRA_NATIVE_TITLE_BAR") ||
        !QSettings().value("window/mergedTitleBar", true).toBool())
        return;
    auto* bar = findChild<QWidget*>("toolbar");
    auto* row = bar ? qobject_cast<QHBoxLayout*>(bar->layout()) : nullptr;
    if (!row) return;
    merged_title_ = true;
    title_bar_ = bar;
    // Qt then counts no frame of its own, so the window's and the client's sizes agree; the
    // styles Windows needs for snap and resize are put back when the window is first shown.
    setWindowFlag(Qt::FramelessWindowHint, true);

    // The menus, behind one button. The same QActions: what Simple hides stays hidden here.
    auto* menu_btn = bar_button(bar, "iMenu", kGlyphMenu, QStringLiteral("Menu"), 32);
    QFont mf = menu_btn->font();
    mf.setPixelSize(14);
    menu_btn->setFont(mf);
    auto* popup = new QMenu(menu_btn);
    for (QAction* a : menuBar()->actions()) popup->addAction(a);
    menu_btn->setMenu(popup);
    menu_btn->setPopupMode(QToolButton::InstantPopup);
    menu_btn->setStyleSheet(QStringLiteral("QToolButton::menu-indicator { image: none; }"));
    row->insertWidget(0, menu_btn);
    menuBar()->hide();

    auto* btn_min = bar_button(bar, "winMin", kGlyphMin, QStringLiteral("Minimise"), 40);
    auto* btn_max = bar_button(bar, "winMax", kGlyphMax, QStringLiteral("Maximise"), 40);
    auto* btn_close = bar_button(bar, "winClose", kGlyphClose, QStringLiteral("Close"), 40);
    // Its hover colour is the theme's (QToolButton#winClose in theme/studio.qss.in): no colour in the sources.
    connect(btn_min, &QToolButton::clicked, this, [this] { showMinimized(); });
    connect(btn_max, &QToolButton::clicked, this, [this] {
        if (isMaximized()) showNormal();
        else showMaximized();
    });
    connect(btn_close, &QToolButton::clicked, this, [this] { close(); });
    row->addSpacing(6);
    row->addWidget(btn_min);
    row->addWidget(btn_max);
    row->addWidget(btn_close);
    new StateWatch(this, btn_max);
#else
    title_bar_ = nullptr;   // macOS and Linux keep the system's title bar
    merged_title_styled_ = false;
#endif
}

void MainWindow::showEvent(QShowEvent* e) {
    QMainWindow::showEvent(e);
#ifdef _WIN32
    if (!merged_title_ || merged_title_styled_) return;
    merged_title_styled_ = true;
    const HWND hwnd = reinterpret_cast<HWND>(winId());
    const LONG_PTR style = GetWindowLongPtrW(hwnd, GWL_STYLE);
    SetWindowLongPtrW(hwnd, GWL_STYLE, style | WS_CAPTION | WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_SYSMENU);
    const BOOL dark = TRUE;   // the frame's thin outline and shadow, dark like the window
    DwmSetWindowAttribute(hwnd, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &dark, sizeof dark);
    SetWindowPos(hwnd, nullptr, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
#endif
}

bool MainWindow::nativeEvent(const QByteArray& type, void* message, qintptr* result) {
#ifdef _WIN32
    if (merged_title_ && merged_title_styled_) {
        const MSG* msg = static_cast<const MSG*>(message);
        const HWND hwnd = msg->hwnd;
        if (msg->message == WM_NCCALCSIZE && msg->wParam == TRUE) {
            // The client area is the whole window: no caption, no borders drawn by the system.
            // Maximised, Windows may place the window a frame's width past the screen on every
            // side; the client is then held to the monitor's work area so nothing is cut off.
            auto* p = reinterpret_cast<NCCALCSIZE_PARAMS*>(msg->lParam);
            if (IsZoomed(hwnd)) {
                MONITORINFO mi{};
                mi.cbSize = sizeof mi;
                if (GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mi)) {
                    RECT& r = p->rgrc[0];
                    r.left = (std::max)(r.left, mi.rcWork.left);
                    r.top = (std::max)(r.top, mi.rcWork.top);
                    r.right = (std::min)(r.right, mi.rcWork.right);
                    r.bottom = (std::min)(r.bottom, mi.rcWork.bottom);
                }
            }
            *result = 0;
            return true;
        }
        if (msg->message == WM_NCHITTEST) {
            const POINT pt{static_cast<int>(static_cast<short>(LOWORD(msg->lParam))),
                           static_cast<int>(static_cast<short>(HIWORD(msg->lParam)))};
            RECT wr{};
            GetWindowRect(hwnd, &wr);
            if (!IsZoomed(hwnd)) {   // the edges resize, as the system's frame did
                const int b = MulDiv(6, static_cast<int>(GetDpiForWindow(hwnd)), 96);
                const bool l = pt.x < wr.left + b, r = pt.x >= wr.right - b, t = pt.y < wr.top + b, bt = pt.y >= wr.bottom - b;
                const LRESULT edge = t && l ? HTTOPLEFT : t && r ? HTTOPRIGHT : bt && l ? HTBOTTOMLEFT : bt && r ? HTBOTTOMRIGHT
                                     : l ? HTLEFT : r ? HTRIGHT : t ? HTTOP : bt ? HTBOTTOM : HTNOWHERE;
                if (edge != HTNOWHERE) {
                    *result = edge;
                    return true;
                }
            }
            // The bar itself and its labels are the caption (drag, double-click, snap); its
            // buttons and switches stay the app's.
            if (title_bar_) {
                const QPoint local = mapFromGlobal(QCursor::pos());
                if (title_bar_->geometry().contains(title_bar_->parentWidget()->mapFrom(this, local))) {
                    QWidget* under = childAt(local);
                    bool caption = under == title_bar_ || qobject_cast<QLabel*>(under) != nullptr;
                    // A plain container inside the bar that holds no button (the shot title's
                    // block, the device pill) is the bar too; anything with a control in it is not.
                    if (!caption && under && title_bar_->isAncestorOf(under) && !under->inherits("QAbstractButton") &&
                        !under->inherits("QComboBox") && !under->inherits("QLineEdit") && !under->children().isEmpty() &&
                        under->findChild<QAbstractButton*>() == nullptr)
                        caption = true;
                    if (caption) {
                        *result = HTCAPTION;
                        return true;
                    }
                }
            }
        }
    }
#endif
    return QMainWindow::nativeEvent(type, message, result);
}

}  // namespace rudra::app
