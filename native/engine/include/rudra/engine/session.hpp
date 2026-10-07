#pragma once
// The session (Phase 3 step 3): the Studio page's grading state in C++, with
// its undo and redo, and the handlers that change it -- a port of `state`,
// params(), snapshot(), pushUndo(), undo(), redo() and the handlers of the
// mode buttons, the strength and peak sliders, Preserve, the Region EV drag
// and double click, the reset actions, the wipe keys and the keydown map in
// ui/app.js. After the same gestures it gives the same params() JSON, byte
// for byte, and the same undo and redo depths (tests/golden/session, from
// the page itself in headless Chromium, tools/emit_session_golden.py).
//
// No Qt: the app's widgets call these and draw from the state.

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rudra/core/composite.hpp"
#include "rudra/core/js_format.hpp"

namespace rudra {

struct RegionState {
    std::string label;
    double low_nits = 0.0, high_nits = 0.0, ev = 0.0;
};

// The part of the state undo takes back (snapshot()).
struct GradeSnapshot {
    std::string mode = "all";
    double strength = 1.0;
    bool preserve = true;
    std::vector<RegionState> regions;
    // Roadmap 3.1: the curve the artist says made the SDR (source_curve.hpp
    // ids). Undone with the grade; "unknown" is the ACES inverse and is left
    // out of params() so every golden before 3.1 reads the same.
    std::string source = "unknown";
    // Roadmap 3.2: the artist's anchors over that curve, by slot (black, grey,
    // highlight); a slot with nits 0 is empty. Undone with the grade; left out
    // of params() when every slot is empty.
    std::vector<CalibrationPoint> calibration;
    // Roadmap 3.4: the reference match (core/reference_fit.hpp). Undone with
    // the grade; left out of params() when empty. Loading one clears the
    // anchors: a reference is an anchor at every code.
    ReferenceFit reference;
    // Roadmap 3.3: the painted masks (core/masks.hpp), one stroke one undo
    // step (planes are shared, a stroke copies the one it paints). Left out
    // of params() when none; the data travels as a file with the master.
    std::shared_ptr<const MaskSet> masks;
};

// The page's defaultRegions(): highlights 400 to 2 000, speculars 2 000 to
// 8 000, shadows 0.05 to 12 nits, all at 0 EV.
std::vector<RegionState> default_regions();

// js_number (core/js_format.hpp) prints a number as the page's JSON.stringify does.

class Session {
public:
    Session();

    // ---- the state (the page's names) -------------------------------------
    GradeSnapshot grade;              // mode, strength, preserve, regions
    double peak_ev = 0.0;             // the view peak slider: 203 * 2^peak_ev nits
    bool anchor = true, carry_chroma = true;
    double anchor_knee = 0.9;         // Deliver: the anchor's knee on the SDR's max code (3.3); live on the viewer
    std::string container = "aces";   // "aces" or "linear"
    std::optional<double> wipe;       // off, or 0..1 across the plate
    bool flip_held = false;
    int region_sel = -1;
    std::string show = "model";       // Compare: "model" (RUDRA) or "baseline"
    int view_layer = 0;               // Layer: 0 image, 1 false colour, 2 difference
    bool show_changes = false;        // Changes: tint what the model changed (view only, not in params())
    bool rail_left = true, rail_right = true, scopes_open = true;   // Window

    double display_nits() const;
    // JSON.stringify(params()), the settings a render and a master take.
    std::string params_json() const;
    // The composite the viewer draws with these settings.
    CompositeParams composite_params() const;
    // The source curve of the moment (3.1), the grade's; as a core enum.
    SourceCurve source_curve() const;

    std::size_t undo_depth() const { return undo_.size(); }
    std::size_t redo_depth() const { return redo_.size(); }

    // Called after every change, with what changed.
    enum Change : std::uint32_t { Grade = 1, Peak = 2, Wipe = 4, Delivery = 8, Flip = 16, Window = 32, View = 64 };
    void on_change(std::function<void(std::uint32_t)> cb) { changed_ = std::move(cb); }

    // ---- actions (engine/actions ids) --------------------------------------
    // Runs one of the session's actions (the modes, preserve, strength,
    // resets, undo, redo, containers, wipe); false for any other id.
    bool run(std::string_view action);
    static bool owns(std::string_view action);

    // ---- the page's handlers ----------------------------------------------
    void set_mode(std::string_view mode);        // setMode: no-op when unchanged
    void nudge_strength(double delta);           // nudgeStrength
    void strength_press();                       // the slider's pointerdown: pushUndo
    void strength_input(double value);           // its input event
    void peak_input(double peak_ev);             // the peak slider: not undone
    void toggle_preserve();                      // the Preserve check
    void reset_recon();
    void reset_regions();
    void undo();
    void redo();
    void set_container(std::string_view kind);
    void set_source(std::string_view id);          // the Source picker (3.1): no-op when unchanged or unknown id
    // The Calibrate rows (3.2): slot 0..2; code -1 keeps the slot's code,
    // nits <= 0 empties the slot. No-op when nothing changes.
    void set_calibration(int slot, int code, double nits);
    void clear_calibration();
    // The usable anchors (nits > 0), what the composite takes.
    std::vector<CalibrationPoint> calibration_points() const;
    // The Reference block (3.4): a fit the app made from a loaded frame
    // (reference_fit.hpp fit_reference) replaces the anchors; Clear takes it
    // out. No-op when nothing changes.
    void set_reference(ReferenceFit fit);
    void clear_reference();
    // Painted masks (3.3). arm_mask picks the band being painted (-1 none; a
    // view change, not undone); the brush is a tool setting. A stroke:
    // stroke_begin (one undo step), then strokes in frame pixels of a
    // width x height frame (the set takes that size; a set of another size is
    // dropped first), then stroke_end. Invert and clear are steps of their own.
    void arm_mask(int band);
    int painted_band() const { return paint_band_; }
    void set_brush(const Brush& brush);
    const Brush& brush() const { return brush_; }
    void stroke_begin();
    void stroke(int width, int height, double x0, double y0, double x1, double y1, bool erase);
    void stroke_end();
    void invert_mask();
    void clear_mask(int band);
    void clear_masks();
    bool show_masks = true;   // the tint on the viewer while a band is armed
    void toggle_wipe();                          // ACTIONS.wipe
    void set_wipe(double x);                     // a drag on the plate, clamped 0..1
    void set_show(std::string_view source);      // the Compare buttons: also ends a wipe
    void set_view_layer(int layer);              // the Layer buttons
    void toggle_changes();                       // the Changes button
    void toggle_anchor();                        // Deliver: anchor to the source exposure
    void set_anchor_knee(double knee);           // Deliver: the knee, 0.5 to 0.99; not undone (a delivery setting)
    void toggle_carry_chroma();                  // Deliver: carry the source chroma

    // A Region EV value: press, move, release (the pointer's x in pixels),
    // and the double click that zeroes it.
    void region_press(int index, double x);
    void region_move(double x, bool shift);
    void region_release();
    void region_zero(int index);
    // A press on a row away from its value: selects it (drawRegions' .sel).
    void select_region(int index);

    // The window's keydown and keyup (KeyboardEvent.key). Returns the action
    // it ran, or the one the app must run ("open", "play", "prev" ... which
    // are not the session's), or empty. Modified keys do nothing.
    std::string key_down(std::string_view key, bool shift, bool modified = false, bool repeat = false);
    void key_up(std::string_view key);

private:
    void push_undo();
    void restore(const GradeSnapshot& s);
    void notify(std::uint32_t what) const;

    std::vector<GradeSnapshot> undo_, redo_;
    int paint_band_ = -1;
    Brush brush_;
    bool in_stroke_ = false;
    std::function<void(std::uint32_t)> changed_;
    // The Region EV drag.
    int drag_ = -1;
    double drag_x0_ = 0.0, drag_ev0_ = 0.0;
    bool drag_moved_ = false;
};

}  // namespace rudra
