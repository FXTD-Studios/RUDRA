#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "rudra/engine/headroom_presentation.hpp"
#include "rudra/engine/project.hpp"

using namespace rudra;

TEST(HeadroomSession, LegacyRefusalDoesNotChangeStateOrUndo) {
    Session s;
    s.nudge_exposure(1);
    s.undo();
    const auto before = s.params_json();
    const auto depth = s.undo_depth(), redo = s.redo_depth();
    int notifications = 0;
    s.on_change([&](auto) { ++notifications; });
    EXPECT_FALSE(s.set_headroom_policy(HeadroomPolicy::Bt1886V1));
    auto saved = s.grade; saved.headroom_policy = HeadroomPolicy::Bt1886V1;
    EXPECT_FALSE(s.load_state(saved, 5, false, false, .6, "linear"));
    EXPECT_EQ(s.params_json(), before);
    EXPECT_EQ(s.undo_depth(), depth); EXPECT_EQ(s.redo_depth(), redo);
    EXPECT_TRUE(s.anchor); EXPECT_TRUE(s.carry_chroma);
    EXPECT_EQ(s.container, "aces"); EXPECT_EQ(notifications, 0);
}

TEST(HeadroomSession, PolicyUndoRedoAndLegacySerialization) {
    Session s(Session::SourceSupport::HeadroomV1);
    const auto legacy = s.params_json();
    EXPECT_EQ(nlohmann::json::parse(project_json(project_from_session(s), "test.rudra"))["rudra_project"], 1);
    ASSERT_TRUE(s.set_headroom_policy(HeadroomPolicy::Bt1886V1));
    const auto enabled = s.params_json();
    EXPECT_FALSE(master_request_from_json(enabled));
    EXPECT_EQ(project_from_session(s).grade.headroom_policy, HeadroomPolicy::Bt1886V1);
    ASSERT_TRUE(s.set_headroom_policy(HeadroomPolicy::Bt1886V1));
    EXPECT_EQ(s.undo_depth(), 1u); // selecting the same policy adds no undo
    s.nudge_exposure(1);
    s.undo(); EXPECT_EQ(s.params_json(), enabled);
    s.undo(); EXPECT_EQ(s.params_json(), legacy);
    s.redo(); EXPECT_EQ(s.params_json(), enabled);
    EXPECT_FALSE(s.set_headroom_policy(static_cast<HeadroomPolicy>(255)));
    EXPECT_EQ(s.params_json(), enabled); EXPECT_EQ(s.redo_depth(), 1u);
    s.redo(); EXPECT_EQ(s.grade.exposure_ev, 1);
}

TEST(HeadroomSession, CapableProjectRestoreKeepsPolicyAndUndo) {
    Session s(Session::SourceSupport::HeadroomV1);
    auto p = project_from_session(s);
    p.grade.headroom_policy = HeadroomPolicy::Bt1886V1;
    p.grade.clip_fill = p.anchor = p.carry_chroma = p.settle_grain = false;
    p.source_kind = "movie"; p.sources = {"source.mkv"};
    auto parsed = parse_project(project_json(p, "test.rudra"), "test.rudra", ProjectReadSupport::HeadroomV1);
    ASSERT_TRUE(parsed) << parsed.error().message;
    ASSERT_TRUE(s.load_state(parsed->grade, parsed->peak_ev, parsed->anchor, parsed->carry_chroma,
                             parsed->anchor_knee, parsed->container));
    EXPECT_EQ(s.grade.headroom_policy, HeadroomPolicy::Bt1886V1);
    auto roundtrip = project_from_session(s);
    roundtrip.source_kind = p.source_kind; roundtrip.sources = p.sources;
    roundtrip.settle_grain = false; // explicit delivery setting, not a grade undo field
    EXPECT_EQ(project_json(roundtrip, "test.rudra"), project_json(p, "test.rudra"));
    s.undo(); EXPECT_EQ(s.grade.headroom_policy, HeadroomPolicy::Disabled);
    s.redo(); EXPECT_EQ(s.grade.headroom_policy, HeadroomPolicy::Bt1886V1);
}

TEST(HeadroomSession, UnsupportedControlsAreRefusedWithoutChangingTheArtistGrade) {
    Session s(Session::SourceSupport::HeadroomV1);
    ASSERT_TRUE(s.set_headroom_policy(HeadroomPolicy::Bt1886V1));
    auto bridge = HeadroomPresentation::create(); ASSERT_TRUE(bridge);
    const auto original = s.params_json();
    EXPECT_FALSE((*bridge)->set_session(s, true, false, false));
    EXPECT_EQ(s.params_json(), original);
    s.grade.clip_fill = false; s.anchor = s.carry_chroma = false;
    EXPECT_FALSE((*bridge)->set_session(s, false, true, false));
    EXPECT_FALSE((*bridge)->set_session(s, false, false, true));
    EXPECT_TRUE((*bridge)->set_session(s, false, false, false));
    for (int bad = 0; bad < 8; ++bad) {
        Session other = s;
        switch (bad) {
        case 0: other.grade.source = "aces"; break;
        case 1: other.grade.clip_peaks = {{.5, .5, 1000}}; break;
        case 2: other.grade.calibration = {{128, 100}}; break;
        case 3: other.grade.strength = .5; break;
        case 4: other.anchor = true; break;
        case 5: other.carry_chroma = true; break;
        case 6: other.grade.mode = "off"; break;
        case 7: other.grade.headroom_policy = HeadroomPolicy::Disabled; break;
        }
        const auto before = other.params_json();
        EXPECT_FALSE((*bridge)->set_session(other, false, false, false));
        EXPECT_EQ(other.params_json(), before);
        EXPECT_FALSE((*bridge)->render_next());
    }
}
