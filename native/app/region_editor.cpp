#include "region_editor.hpp"

#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QStyle>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

#include "rudra/core/js_format.hpp"
#include "widgets.hpp"

namespace rudra::app {
namespace {

void repolish(QWidget* w) {
    w->style()->unpolish(w);
    w->style()->polish(w);
    w->update();
}

// The page's .sw: an 11 px square, accent-dim when the row is selected.
class Swatch : public QWidget {
public:
    explicit Swatch(QWidget* parent) : QWidget(parent) { setFixedSize(11, 11); }
    bool on = false;

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setPen(QPen(palette().color(on ? QPalette::Link : QPalette::Midlight), 1));
        p.setBrush(on ? palette().color(QPalette::Highlight) : Qt::transparent);
        p.drawRoundedRect(QRectF(0.5, 0.5, 10, 10), 2, 2);
    }
};

}  // namespace

RegionEditor::RegionEditor(Session& session, QWidget* parent) : QWidget(parent), session_(session) {
    setObjectName("regions");
    setProperty("role", "regions");
    setAttribute(Qt::WA_StyledBackground, true);
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(4, 0, 4, 8);
    v->setSpacing(0);
    sync();
}

void RegionEditor::build(int n) {
    auto* v = static_cast<QVBoxLayout*>(layout());
    for (auto& r : rows_) delete r.row;
    rows_.clear();
    for (int i = 0; i < n; ++i) {
        Row r;
        r.row = new QWidget(this);
        r.row->setProperty("role", "region");
        r.row->setAttribute(Qt::WA_StyledBackground, true);
        r.row->setCursor(Qt::PointingHandCursor);
        auto* h = new QHBoxLayout(r.row);
        h->setContentsMargins(8, 5, 8, 5);
        h->setSpacing(8);
        h->addWidget(new Swatch(r.row));
        r.q = new QLabel(r.row);
        r.q->setProperty("role", "region-q");
        h->addWidget(r.q, 1);
        r.ev = new QLabel(r.row);
        r.ev->setProperty("role", "region-ev");
        r.ev->setCursor(Qt::SizeHorCursor);
        h->addWidget(r.ev);
        auto* u = new QLabel("EV", r.row);
        u->setProperty("role", "region-u");
        u->setFixedWidth(20);
        h->addWidget(u);
        if (i < kMaxMaskBands) {
            r.mask = new QPushButton("\u2205", r.row);
            r.mask->setObjectName(QString("maskBtn%1").arg(i));
            r.mask->setProperty("role", "mask");
            r.mask->setCheckable(true);
            r.mask->setFixedSize(30, 20);
            r.mask->setToolTip("Paint a mask for this band");
            r.mask->setCursor(Qt::ArrowCursor);
            connect(r.mask, &QPushButton::clicked, this, [this, i] {
                if (mask_pressed) mask_pressed(i);
            });
            h->addWidget(r.mask);
        } else {
            auto* spacer = new QWidget(r.row);
            spacer->setFixedSize(30, 20);
            h->addWidget(spacer);
        }
        r.row->installEventFilter(this);
        r.ev->installEventFilter(this);
        v->addWidget(r.row);
        rows_.push_back(r);
    }
}

void RegionEditor::sync() {
    const auto& regions = session_.grade.regions;
    if (int(rows_.size()) != int(regions.size())) build(int(regions.size()));
    for (std::size_t i = 0; i < regions.size(); ++i) {
        const auto& g = regions[i];
        auto& r = rows_[i];
        r.q->setText(QString::fromStdString(nits_label(g.low_nits) + " – " + nits_label(g.high_nits) + " nits"));
        r.ev->setText(QString::fromStdString(js_signed(g.ev, 2)));
        const bool live = std::abs(g.ev) > 1e-9, sel = int(i) == session_.region_sel;
        if (r.mask) {
            const bool has = session_.grade.masks && session_.grade.masks->has(int(i));
            r.mask->setText(has ? QString::number(std::lround(session_.grade.masks->coverage(int(i)) * 100.0)) + "%"
                                : QStringLiteral("\u2205"));
            r.mask->setChecked(session_.painted_band() == int(i));
            if (r.mask->property("has").toBool() != has) {
                r.mask->setProperty("has", has);
                repolish(r.mask);
            }
        }
        if (r.ev->property("live").toBool() != live) {
            r.ev->setProperty("live", live);
            repolish(r.ev);
        }
        if (r.row->property("sel").toBool() != sel) {
            r.row->setProperty("sel", sel);
            repolish(r.row);
            if (auto* sw = r.row->findChild<QWidget*>()) {
                static_cast<Swatch*>(sw)->on = sel;
                sw->update();
            }
        }
    }
}

bool RegionEditor::eventFilter(QObject* o, QEvent* e) {
    int i = -1;
    bool on_value = false;
    for (std::size_t k = 0; k < rows_.size(); ++k) {
        if (o == rows_[k].ev) {
            i = int(k);
            on_value = true;
        } else if (o == rows_[k].row) {
            i = int(k);
        }
    }
    if (i < 0) return QWidget::eventFilter(o, e);
    auto* me = static_cast<QMouseEvent*>(e);
    switch (e->type()) {
        case QEvent::MouseButtonPress:
            if (me->button() != Qt::LeftButton) break;
            if (on_value) {
                // pointerdown on .ev: the row is selected and the drag begins.
                dragging_ = i;
                // x in the value label's own coordinates: it does not move while
                // it is dragged, and they keep a high-DPI mouse's fractions.
                session_.region_press(i, me->position().x());
            } else {
                session_.select_region(i);
            }
            return true;
        case QEvent::MouseMove:
            if (dragging_ >= 0) {
                session_.region_move(me->position().x(), me->modifiers() & Qt::ShiftModifier);
                return true;
            }
            break;
        case QEvent::MouseButtonRelease:
            if (dragging_ >= 0) {
                dragging_ = -1;
                session_.region_release();
                return true;
            }
            break;
        case QEvent::MouseButtonDblClick:
            // dblclick on .ev: zero it (the second press of the pair is this).
            if (on_value && me->button() == Qt::LeftButton) {
                session_.region_zero(i);
                return true;
            }
            break;
        default: break;
    }
    return QWidget::eventFilter(o, e);
}

// ---- the brush card (3.3) -------------------------------------------------

MaskPanel::MaskPanel(Session& session, QWidget* parent) : QWidget(parent), session_(session) {
    setObjectName("maskPanel");
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(8, 8, 8, 4);
    v->setSpacing(6);
    auto* head = new QWidget(this);
    auto* hh = new QHBoxLayout(head);
    hh->setContentsMargins(0, 0, 0, 0);
    title_ = new QLabel(head);
    title_->setProperty("role", "card-head");
    hh->addWidget(title_);
    hh->addStretch(1);
    coverage_ = new QLabel(head);
    coverage_->setObjectName("maskCoverage");
    coverage_->setProperty("role", "note");
    hh->addWidget(coverage_);
    v->addWidget(head);
    auto* mode = new QWidget(this);
    mode->setProperty("role", "seg");
    auto* mh = new QHBoxLayout(mode);
    mh->setContentsMargins(2, 2, 2, 2);
    mh->setSpacing(2);
    add_ = new QPushButton("Add", mode);
    add_->setObjectName("maskAdd");
    erase_ = new QPushButton("Erase", mode);
    erase_->setObjectName("maskErase");
    for (QPushButton* b : {add_, erase_}) {
        b->setCheckable(true);
        b->setProperty("role", "seg-btn");
        mh->addWidget(b, 1);
    }
    connect(add_, &QPushButton::clicked, this, [this] { Brush b = session_.brush(); b.erase = false; session_.set_brush(b); });
    connect(erase_, &QPushButton::clicked, this, [this] { Brush b = session_.brush(); b.erase = true; session_.set_brush(b); });
    v->addWidget(mode);
    auto slider = [&](const char* name, const QString& label, int lo, int hi, QSlider*& out, QLabel*& value) {
        auto* row = new QWidget(this);
        auto* rh = new QHBoxLayout(row);
        rh->setContentsMargins(0, 0, 0, 0);
        auto* k = new QLabel(label, row);
        k->setProperty("role", "key");
        rh->addWidget(k);
        rh->addStretch(1);
        value = new QLabel(row);
        value->setProperty("role", "value");
        rh->addWidget(value);
        v->addWidget(row);
        out = new QSlider(Qt::Horizontal, this);
        out->setObjectName(name);
        out->setRange(lo, hi);
        v->addWidget(out);
    };
    slider("maskSize", "Size", 2, 600, size_, size_v_);
    slider("maskSoft", "Softness", 0, 100, soft_, soft_v_);
    slider("maskFlow", "Flow", 1, 100, flow_, flow_v_);
    connect(size_, &QSlider::valueChanged, this, [this](int v) { if (syncing_) return; Brush b = session_.brush(); b.size_px = v; session_.set_brush(b); });
    connect(soft_, &QSlider::valueChanged, this, [this](int v) { if (syncing_) return; Brush b = session_.brush(); b.softness = v / 100.0; session_.set_brush(b); });
    connect(flow_, &QSlider::valueChanged, this, [this](int v) { if (syncing_) return; Brush b = session_.brush(); b.flow = v / 100.0; session_.set_brush(b); });
    show_ = new CheckRow("maskShow", "Show mask on the viewer", "maskShowHint", "tinted in the band's colour", this);
    show_->clicked = [this] {
        session_.show_masks = !session_.show_masks;
        session_.set_brush(session_.brush());   // a view change: the tint follows
    };
    v->addWidget(show_);
    auto* btns = new QWidget(this);
    auto* bh = new QHBoxLayout(btns);
    bh->setContentsMargins(0, 0, 0, 0);
    bh->addStretch(1);
    invert_ = new QPushButton("Invert", btns);
    invert_->setObjectName("maskInvert");
    connect(invert_, &QPushButton::clicked, this, [this] { session_.invert_mask(); });
    clear_ = new QPushButton("Clear", btns);
    clear_->setObjectName("maskClear");
    connect(clear_, &QPushButton::clicked, this, [this] { session_.clear_mask(session_.painted_band()); });
    done_ = new QPushButton("Done", btns);
    done_->setObjectName("maskDone");
    done_->setProperty("role", "primary");
    connect(done_, &QPushButton::clicked, this, [this] {
        if (done) done();
    });
    for (QPushButton* b : {invert_, clear_, done_}) bh->addWidget(b);
    v->addWidget(btns);
    sync();
}

void MaskPanel::sync() {
    const int band = session_.painted_band();
    setVisible(band >= 0);
    if (band < 0) return;
    syncing_ = true;
    static const char* kNames[] = {"Highlights", "Speculars", "Shadows", "Band 4"};
    const auto& regions = session_.grade.regions;
    const QString name = band < int(regions.size()) && !regions[std::size_t(band)].label.empty()
                             ? QString::fromStdString(regions[std::size_t(band)].label)
                             : QString(kNames[std::min(band, 3)]);
    title_->setText("Mask \u00b7 " + name);
    const bool has = session_.grade.masks && session_.grade.masks->has(band);
    coverage_->setText(has ? QString::number(std::lround(session_.grade.masks->coverage(band) * 100.0)) + "% of the frame"
                           : QStringLiteral("nothing painted yet"));
    const Brush& b = session_.brush();
    add_->setChecked(!b.erase);
    erase_->setChecked(b.erase);
    size_->setValue(int(std::lround(b.size_px)));
    soft_->setValue(int(std::lround(b.softness * 100.0)));
    flow_->setValue(int(std::lround(b.flow * 100.0)));
    size_v_->setText(QString::number(std::lround(b.size_px)) + " px");
    soft_v_->setText(QString::number(std::lround(b.softness * 100.0)) + "%");
    flow_v_->setText(QString::number(std::lround(b.flow * 100.0)) + "%");
    show_->set_on(session_.show_masks);
    invert_->setEnabled(has);
    clear_->setEnabled(has);
    syncing_ = false;
}

}  // namespace rudra::app
