#pragma once
// A child process's stdout as JSON objects, one per line (28 Oct 2026 review: the
// engine and the AI fill read it the same way). Bytes arrive in any pieces; a line
// is used when its '\n' has come; "\r\n" is fine; anything that is not one JSON
// object (a library's own print) is skipped.

#include <QByteArray>
#include <QJsonDocument>
#include <QJsonObject>

namespace rudra::app {

class JsonLines {
public:
    // Calls `on_object` for every whole line that is a JSON object.
    template <class F>
    void feed(const QByteArray& bytes, F&& on_object) {
        pending_ += bytes;
        qsizetype nl;
        while ((nl = pending_.indexOf('\n')) >= 0) {
            QByteArray line = pending_.left(nl);
            pending_.remove(0, nl + 1);
            if (line.endsWith('\r')) line.chop(1);
            const QJsonDocument d = QJsonDocument::fromJson(line);
            if (d.isObject()) on_object(d.object());
        }
    }
    void clear() { pending_.clear(); }

private:
    QByteArray pending_;
};

}  // namespace rudra::app
