#pragma once
// A layout that places its items left to right and starts a new line when the next one
// does not fit (8 Nov 2026), after Qt's Flow Layout example. The viewer's toolbar uses it so
// a narrow window wraps the row instead of growing past the screen and pushing the right
// rail (Grade, Deliver) off it. Its minimum width is the widest single item, not the row.

#include <QLayout>
#include <QList>
#include <QStyle>

namespace rudra::app {

class FlowLayout final : public QLayout {
public:
    explicit FlowLayout(QWidget* parent = nullptr, int hspacing = 6, int vspacing = 4);
    ~FlowLayout() override;

    // A fixed gap of `width` pixels at this point of the row (QHBoxLayout::addSpacing).
    void add_gap(int width);

    void addItem(QLayoutItem* item) override;
    int count() const override { return int(items_.size()); }
    QLayoutItem* itemAt(int index) const override;
    QLayoutItem* takeAt(int index) override;
    Qt::Orientations expandingDirections() const override { return {}; }
    bool hasHeightForWidth() const override { return true; }
    int heightForWidth(int width) const override { return arrange(QRect(0, 0, width, 0), true); }
    QSize minimumSize() const override;
    QSize sizeHint() const override;
    void setGeometry(const QRect& rect) override;

private:
    // Lays the items out in `rect` (or only measures, test_only) and returns the height used.
    int arrange(const QRect& rect, bool test_only) const;

    QList<QLayoutItem*> items_;
    int hspace_, vspace_;
};

}  // namespace rudra::app
