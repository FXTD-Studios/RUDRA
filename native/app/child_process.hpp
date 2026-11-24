#pragma once
// 1.3 (24 Nov 2026, audit): a long-lived child (the AI fill's Python and its preview, the learned engine,
// the master CLI, the AI fill set-up) ends with RUDRA, also after a crash or an end from Task Manager:
// it joins the job of rudra/platform/process.hpp (bind_child_to_this_process) as soon as it has started.

#include <QObject>
#include <QProcess>

#include <cstdint>

#include "rudra/platform/process.hpp"

namespace rudra::app {

// Call before p->start(...); a QProcess started again keeps its one connection.
inline void end_with_rudra(QProcess* p) {
    if (!p || p->property("rudraEndsWithRudra").toBool()) return;
    p->setProperty("rudraEndsWithRudra", true);
    QObject::connect(p, &QProcess::started, p, [p] { rudra::bind_child_to_this_process(std::int64_t(p->processId())); });
}

}  // namespace rudra::app
