#include "getting_started.hpp"
#include "main_window.hpp"
#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QButtonGroup>
#include <QStackedWidget>
#include <QScrollArea>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDesktopServices>
#include <QUrl>
#include <QScreen>
#include <QGuiApplication>
#include <QKeySequence>
#include <QStyle>
#include <array>
#include <algorithm>

namespace rudra::app {
namespace {
using Step = std::array<QString, 10>;
const std::array<Step, 5> steps = {{
    {QStringLiteral("Source"), QStringLiteral("Open your footage"), QStringLiteral("Start with your source"), QStringLiteral("Bring in a still, a frame sequence or a movie. Choose the source curve so RUDRA can build the right HDR starting point."), QStringLiteral("Open a shot, or drop it into RUDRA"), QStringLiteral("Stills · Numbered frame sequences · Movies"), QStringLiteral("Choose the source curve"), QStringLiteral("In Reconstruct → Source, start with BT.1886 for graded or broadcast SDR."), QStringLiteral("Working with CG or games?"), QStringLiteral("Choose ACES, filmic or AgX when it matches the curve used to create your footage.")},
    {QStringLiteral("Calibrate"), QStringLiteral("Set a brightness reference"), QStringLiteral("Give brightness a reference"), QStringLiteral("Use known brightness values or a graded HDR reference to place your reconstruction on a meaningful scale. This step is optional."), QStringLiteral("Calibrate in the Reconstruct tab"), QStringLiteral("Use up to three known-brightness patches, or an HDR reference."), QStringLiteral("Use known patches"), QStringLiteral("Pick a black, grey or highlight patch and enter its brightness in nits."), QStringLiteral("Keep moving"), QStringLiteral("No reference available? Continue with the starting reconstruction and assess the image in the viewer.")},
    {QStringLiteral("Reconstruct"), QStringLiteral("Shape the highlights"), QStringLiteral("Shape your HDR image"), QStringLiteral("Set the overall exposure, then refine clipped highlights. Keep the result grounded in the shot."), QStringLiteral("Start with exposure, then highlight fill"), QStringLiteral("Exposure changes the master in stops."), QStringLiteral("Refine clipped areas"), QStringLiteral("Use Fill clipped highlights. Set a region’s peak or reduce Strength where the result goes too far."), QStringLiteral("AI is optional"), QStringLiteral("Reconstruct → Use the model adds learned reconstruction. You can work with the built-in reconstruction alone.")},
    {QStringLiteral("Check"), QStringLiteral("Compare with confidence"), QStringLiteral("Check what changed"), QStringLiteral("Compare the reconstruction with its baseline and inspect the bright areas before you deliver."), QStringLiteral("Compare, wipe and inspect"), QStringLiteral("Press F to fit the view. Hold B for the baseline. Use W to wipe."), QStringLiteral("Use the viewing tools"), QStringLiteral("Adjust the view peak and inspect False colour and scopes. The view peak changes the display, not the master."), QStringLiteral("Review reconstructed detail"), QStringLiteral("Invented shows where clipped areas were reconstructed. For movies, check several frames and use Clip → Hold steady.")},
    {QStringLiteral("Deliver"), QStringLiteral("Export your master"), QStringLiteral("Deliver the right format"), QStringLiteral("Choose a master for grading or an HDR movie for delivery. Save the project so you can return to the same settings."), QStringLiteral("Choose an output in the Deliver tab"), QStringLiteral("EXR master · HDR10 · HLG · ProRes. Metadata support depends on the selected delivery format."), QStringLiteral("For further grading"), QStringLiteral("Export an ACES 2065-1 EXR master. Keep your project alongside the output."), QStringLiteral("For movie delivery"), QStringLiteral("Choose the HDR format and review the export settings. The session autosaves. Reopen this guide from Help → Getting started; F1 opens the user guide.")}
}};
QLabel* label(const QString& text, const char* role, QWidget* parent) {
    auto* l = new QLabel(text, parent);
    l->setProperty("guideRole", role);
    l->setTextFormat(Qt::PlainText);
    l->setWordWrap(true);
    l->setTextInteractionFlags(Qt::TextSelectableByMouse);
    return l;
}
}
QDialog* make_getting_started(QWidget* parent) {
    auto* d = new QDialog(parent);
    d->setAttribute(Qt::WA_DeleteOnClose);
    d->setObjectName("gettingStarted");
    d->setWindowTitle("Getting started");
    auto* outer = new QVBoxLayout(d);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->setSpacing(0);
    auto* header = new QWidget(d);
    header->setObjectName("guideHeader");
    auto* head = new QHBoxLayout(header);
    head->setContentsMargins(30, 20, 24, 20);
    head->addWidget(label("RUDRA", "brand", header));
    head->addSpacing(18);
    head->addWidget(label("Getting started", "muted", header));
    head->addStretch();
    auto* close = new QPushButton(QString::fromUtf8("×"), header);
    close->setObjectName("guideClose");
    close->setAccessibleName("Close getting started");
    close->setToolTip("Close (Esc)");
    close->setAutoDefault(false);
    close->setFixedSize(32, 32);
    QObject::connect(close, &QPushButton::clicked, d, &QDialog::reject);
    head->addWidget(close);
    outer->addWidget(header);
    auto* body = new QHBoxLayout;
    body->setSpacing(0);
    auto* sidebar = new QWidget(d);
    sidebar->setObjectName("guideSidebar");
    sidebar->setFixedWidth(246);
    auto* nav = new QVBoxLayout(sidebar);
    nav->setContentsMargins(18, 28, 18, 24);
    nav->setSpacing(8);
    nav->addWidget(label("YOUR FIRST HDR SHOT", "eyebrow", sidebar));
    nav->addSpacing(12);
    auto* group = new QButtonGroup(d);
    auto* pages = new QStackedWidget(d);
    pages->setObjectName("guidePages");
    for (int i = 0; i < int(steps.size()); ++i) {
        const auto& s = steps[std::size_t(i)];
        auto* button = new QPushButton(sidebar);
        button->setObjectName(QString("guideStep%1").arg(i));
        button->setAccessibleName(QString("Step %1: %2. %3").arg(i + 1).arg(s[0], s[1]));
        button->setCheckable(true);
        button->setAutoDefault(false);
        button->setMinimumHeight(66);
        auto* stepLayout = new QHBoxLayout(button);
        stepLayout->setContentsMargins(12, 8, 12, 8);
        stepLayout->setSpacing(12);
        auto* number = label(QString::number(i + 1), "number", button);
        number->setFixedSize(26, 26);
        number->setAlignment(Qt::AlignCenter);
        number->setAttribute(Qt::WA_TransparentForMouseEvents);
        stepLayout->addWidget(number);
        auto* stepText = new QVBoxLayout;
        stepText->setSpacing(4);
        for (int j : {0, 1}) {
            auto* text = label(s[std::size_t(j)], j == 0 ? "stepTitle" : "stepSubtitle", button);
            text->setAttribute(Qt::WA_TransparentForMouseEvents);
            stepText->addWidget(text);
        }
        stepLayout->addLayout(stepText, 1);
        group->addButton(button, i);
        nav->addWidget(button);
        auto* scroll = new QScrollArea(pages);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        auto* page = new QWidget(scroll);
        auto* content = new QVBoxLayout(page);
        content->setContentsMargins(36, 30, 36, 24);
        content->setSpacing(14);
        content->addWidget(label(QString("STEP 0%1 / 05").arg(i + 1), "eyebrow", page));
        content->addWidget(label(s[2], "title", page));
        content->addWidget(label(s[3], "intro", page));
        content->addSpacing(10);
        auto* card = new QWidget(page);
        card->setObjectName("guideCard");
        auto* cardLayout = new QVBoxLayout(card);
        cardLayout->setContentsMargins(24, 22, 24, 22);
        cardLayout->setSpacing(9);
        cardLayout->addWidget(label(s[4], "heading", card));
        cardLayout->addWidget(label(s[5], "muted", card));
        content->addWidget(card);
        content->addSpacing(10);
        auto* details = new QHBoxLayout;
        details->setSpacing(24);
        for (int j : {6, 8}) {
            auto* column = new QVBoxLayout;
            column->setSpacing(8);
            column->addWidget(label(s[std::size_t(j)], "heading", page));
            column->addWidget(label(s[std::size_t(j + 1)], "body", page));
            column->addStretch();
            details->addLayout(column, 1);
        }
        content->addLayout(details);
        content->addStretch();
        scroll->setWidget(page);
        pages->addWidget(scroll);
    }
    nav->addStretch();
    nav->addWidget(label("Start with the built-in reconstruction.\nThe AI model is optional.", "note", sidebar));
    body->addWidget(sidebar);
    body->addWidget(pages, 1);
    outer->addLayout(body, 1);
    auto* footer = new QWidget(d);
    footer->setObjectName("guideFooter");
    auto* actions = new QHBoxLayout(footer);
    actions->setContentsMargins(28, 18, 28, 18);
    actions->setSpacing(10);
    actions->addWidget(label(QKeySequence(QKeySequence::Save).toString(QKeySequence::NativeText) + " saves the project.", "note", footer), 1);
    auto* guide = new QPushButton("User guide", footer);
    guide->setObjectName("guideHelp");
    guide->setAutoDefault(false);
    QObject::connect(guide, &QPushButton::clicked, d, [] { QDesktopServices::openUrl(QUrl(MainWindow::user_guide_url())); });
    auto* back = new QPushButton("Back", footer);
    back->setObjectName("guideBack");
    back->setAutoDefault(false);
    auto* next = new QPushButton("Next step", footer);
    next->setObjectName("guideNext");
    next->setAutoDefault(false);
    actions->addWidget(guide); actions->addWidget(back); actions->addWidget(next);
    outer->addWidget(footer);
    const auto select = [pages, group, back, next](int i) {
        pages->setCurrentIndex(i);
        group->button(i)->setChecked(true);
        for (auto* button : group->buttons()) {
            for (auto* l : button->findChildren<QLabel*>()) {
                if (l->property("guideRole").toString() != "number") continue;
                l->setProperty("active", button == group->button(i));
                l->style()->unpolish(l);
                l->style()->polish(l);
            }
        }
        back->setEnabled(i > 0);
        next->setText(i == 4 ? "Start working" : "Next step");
    };
    QObject::connect(group, &QButtonGroup::idClicked, d, select);
    QObject::connect(back, &QPushButton::clicked, d, [pages, select] { select(std::max(0, pages->currentIndex() - 1)); });
    QObject::connect(next, &QPushButton::clicked, d, [d, pages, select] {
        if (pages->currentIndex() == 4) d->accept();
        else select(pages->currentIndex() + 1);
    });
    select(0);
    const auto available = (parent ? parent->screen() : QGuiApplication::primaryScreen())->availableGeometry().size();
    d->resize(std::min(980, available.width() - 40), std::min(640, available.height() - 80));
    next->setFocus();
    return d;
}
}
