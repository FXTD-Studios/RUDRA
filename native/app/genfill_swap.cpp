#include "genfill_swap.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>

namespace rudra::app {

GenFillSwap genfill_replace_in_master(const QString& master_dir, const QString& fill_dir) {
    GenFillSwap r;
    const QDir master(master_dir), fill(fill_dir);
    if (!master.exists() || !fill.exists()) {
        r.error = QStringLiteral("the master or the fill folder is missing");
        return r;
    }
    const QStringList names = fill.entryList({QStringLiteral("*.exr")}, QDir::Files, QDir::Name);
    QString base = QDir::cleanPath(master.absolutePath()) + QStringLiteral("_before_aifill");
    r.backup = base;
    for (int k = 2; QFileInfo::exists(r.backup) && k < 1000; ++k) r.backup = base + QStringLiteral("_") + QString::number(k);
    bool made = false;
    for (const QString& n : names) {
        const QString m = master.filePath(n), f = fill.filePath(n);
        if (!QFileInfo::exists(m)) {
            ++r.skipped;
            continue;
        }
        if (!made) {
            if (!QDir().mkpath(r.backup)) {
                r.error = QStringLiteral("could not make %1").arg(QDir::toNativeSeparators(r.backup));
                return r;
            }
            made = true;
        }
        const QString b = QDir(r.backup).filePath(n);
        if (!QFile::rename(m, b)) {
            r.error = QStringLiteral("could not move %1 aside").arg(n);
            return r;
        }
        if (!QFile::rename(f, m)) {
            QFile::rename(b, m);   // put the original back: never a frame missing
            r.error = QStringLiteral("could not move the fill of %1 into the master").arg(n);
            return r;
        }
        ++r.replaced;
    }
    if (!made) r.backup.clear();
    return r;
}

}  // namespace rudra::app
