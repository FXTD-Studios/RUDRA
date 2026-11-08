#include "flow_layout.hpp"

#include <QSpacerItem>
#include <QWidget>
#include <algorithm>

namespace rudra::app {

FlowLayout::FlowLayout(QWidget* parent, int hspacing, int vspacing)
    : QLayout(parent), hspace_(hspacing), vspace_(vspacing) {
    setContentsMargins(0, 0, 0, 0);
}

FlowLayout::~FlowLayout() {
    while (QLayoutItem* item = takeAt(0)) delete item;
}

void FlowLayout::add_gap(int width) { addItem(new QSpacerItem(width, 0, QSizePolicy::Fixed, QSizePolicy::Minimum)); }

void FlowLayout::addItem(QLayoutItem* item) { items_.append(item); }

QLayoutItem* FlowLayout::itemAt(int index) const {
    return index >= 0 && index < items_.size() ? items_.at(index) : nullptr;
}

QLayoutItem* FlowLayout::takeAt(int index) {
    return index >= 0 && index < items_.size() ? items_.takeAt(index) : nullptr;
}

QSize FlowLayout::minimumSize() const {
    QSize size;
    for (const QLayoutItem* item : items_) size = size.expandedTo(item->minimumSize());
    const QMargins m = contentsMargins();
    return size + QSize(m.left() + m.right(), m.top() + m.bottom());
}

QSize FlowLayout::sizeHint() const {
    // One line, as wide as every item: what the row asks for when there is room.
    int w = 0, h = 0;
    for (const QLayoutItem* item : items_) {
        if (item->isEmpty()) continue;
        const QSize s = item->sizeHint();
        w += (w ? hspace_ : 0) + s.width();
        h = std::max(h, s.height());
    }
    const QMargins m = contentsMargins();
    return {w + m.left() + m.right(), h + m.top() + m.bottom()};
}

void FlowLayout::setGeometry(const QRect& rect) {
    QLayout::setGeometry(rect);
    arrange(rect, false);
}

int FlowLayout::arrange(const QRect& rect, bool test_only) const {
    const QMargins m = contentsMargins();
    const QRect r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom());
    int x = r.x(), y = r.y(), line_h = 0;
    // Items of one line share its height and are centred in it, as in a QHBoxLayout row.
    QList<QPair<QLayoutItem*, QPoint>> line;
    auto flush = [&] {
        if (!test_only)
            for (const auto& [item, at] : line) {
                const QSize s = item->sizeHint();
                item->setGeometry(QRect(QPoint(at.x(), at.y() + (line_h - s.height()) / 2), s));
            }
        line.clear();
    };
    for (QLayoutItem* item : items_) {
        if (item->isEmpty()) continue;
        const QSize s = item->sizeHint();
        const int gap = x > r.x() ? hspace_ : 0;
        if (x > r.x() && x + gap + s.width() > r.right() + 1) {   // does not fit: a new line
            flush();
            x = r.x();
            y += line_h + vspace_;
            line_h = 0;
        } else {
            x += gap;
        }
        if (item->spacerItem() && x == r.x()) continue;   // no gap at the start of a line
        line.append({item, QPoint(x, y)});
        x += s.width();
        line_h = std::max(line_h, s.height());
    }
    flush();
    return y + line_h - rect.y() + m.bottom();
}

}  // namespace rudra::app
