// Conform by shot (roadmap 5.4): shots from an EDL, an FCPXML or the cuts in
// the movie, onto the programme's frames.
#include <gtest/gtest.h>

#include "rudra/deliver/conform.hpp"

using namespace rudra;

namespace {

FrameRate rate(const char* text) {
    auto r = parse_frame_rate(text);
    EXPECT_TRUE(r) << text;
    return r ? *r : FrameRate{};
}

constexpr const char* kEdl = R"(TITLE: REEL 1 CONFORM
FCM: NON-DROP FRAME

001  A001C003 V     C        11:02:10:00 11:02:12:00 01:00:00:00 01:00:02:00
* FROM CLIP NAME: sh010_plate.mov
002  A001C003 A     C        11:02:10:00 11:02:12:00 01:00:00:00 01:00:02:00
003  B002C001 V     C        09:00:00:00 09:00:01:12 01:00:02:00 01:00:03:12
* FROM CLIP NAME: sh020_plate.mov
* COMMENT: an audio event above is ignored
004  C003C007 V     C        10:00:00:00 10:00:00:00 01:00:03:12 01:00:03:12
004  D004C002 V     D    012 14:00:00:00 14:00:02:00 01:00:03:12 01:00:05:12
* FROM CLIP NAME: sh030_dissolve_in.mov
005  BL       V     C        00:00:00:00 00:00:00:12 01:00:05:12 01:00:06:00
)";

}  // namespace

TEST(Conform, FrameRatesInTheFormsEditorsWrite) {
    EXPECT_EQ(rate("24000/1001").num, 24000);
    EXPECT_EQ(rate("23.976").den, 1001);
    EXPECT_EQ(rate("29.97").nominal(), 30);
    EXPECT_EQ(rate("25").num, 25);
    EXPECT_EQ(rate("25").den, 1);
    EXPECT_FALSE(parse_frame_rate("fast"));
    EXPECT_FALSE(parse_frame_rate("24.5"));
}

TEST(Conform, TimecodeToFrames) {
    EXPECT_EQ(*timecode_frames("01:00:00:00", rate("24")), 86400);
    EXPECT_EQ(*timecode_frames("00:00:01:12", rate("24")), 36);
    EXPECT_EQ(*timecode_frames("00:00:01:12", rate("23.976")), 36);   // 23.976 counts as 24
    // Drop frame at 29.97: frames 0 and 1 of each minute are skipped except every tenth.
    EXPECT_EQ(*timecode_frames("00:01:00;02", rate("29.97")), 1800);
    EXPECT_EQ(*timecode_frames("00:10:00;00", rate("29.97")), 17982);
    EXPECT_EQ(*timecode_frames("01:00:00;00", rate("29.97")), 107892);
    EXPECT_EQ(*timecode_frames("01:00:00:00", rate("29.97"), true), 107892);
    EXPECT_EQ(*timecode_frames("00:01:00:00", rate("29.97")), 1800);   // non-drop
    EXPECT_FALSE(timecode_frames("00:00:00:24", rate("24")));
    EXPECT_FALSE(timecode_frames("1:00:00:00", rate("24")));
}

TEST(Conform, AnEdlsVideoEventsBecomeShots) {
    auto edit = parse_edl(kEdl, rate("24"));
    ASSERT_TRUE(edit) << edit.error().message << " " << edit.error().detail;
    EXPECT_EQ(edit->title, "REEL 1 CONFORM");
    ASSERT_EQ(edit->events.size(), 4u);   // the audio and the zero-length dissolve half are not shots
    EXPECT_EQ(edit->events[0].name, "sh010_plate.mov");
    EXPECT_EQ(edit->events[1].name, "sh020_plate.mov");
    EXPECT_EQ(edit->events[2].name, "sh030_dissolve_in.mov");   // from the dissolve's start
    EXPECT_EQ(edit->events[3].name, "event 005 (black)");
    auto c = conform_shots(*edit, 144);
    ASSERT_TRUE(c) << c.error().message;
    const auto& s = c->shots.shots;
    ASSERT_EQ(s.size(), 4u);
    EXPECT_EQ(s[0].first, 0);
    EXPECT_EQ(s[0].last, 47);
    EXPECT_EQ(s[1].first, 48);
    EXPECT_EQ(s[1].last, 83);
    EXPECT_EQ(s[2].first, 84);
    EXPECT_EQ(s[2].last, 131);
    EXPECT_EQ(s[3].first, 132);
    EXPECT_EQ(s[3].last, 143);
    EXPECT_EQ(s[0].source, "edl");
    EXPECT_TRUE(c->notes.empty());
}

TEST(Conform, TheRecordStartPlacesTheProgramme) {
    auto edit = parse_edl(kEdl, rate("24"));
    ASSERT_TRUE(edit);
    // The movie starts 00:00:01:00 before the first event: shots move 24 frames later.
    auto c = conform_shots(*edit, 200, {}, *timecode_frames("00:59:59:00", rate("24")));
    ASSERT_TRUE(c) << c.error().message;
    EXPECT_EQ(c->shots.shots[0].first, 24);
    EXPECT_EQ(shot_index(c->shots, 10), -1);   // before the first shot: the base grade
}

TEST(Conform, AMovieShorterThanTheEditCutsOrRefuses) {
    auto edit = parse_edl(kEdl, rate("24"));
    ASSERT_TRUE(edit);
    auto shorter = conform_shots(*edit, 140);   // the last event, 132 to 143, is cut at 139
    ASSERT_TRUE(shorter) << shorter.error().message;
    ASSERT_EQ(shorter->shots.shots.size(), 4u);
    EXPECT_EQ(shorter->shots.shots[3].last, 139);
    ASSERT_EQ(shorter->notes.size(), 1u);
    EXPECT_NE(shorter->notes[0].find("runs past the end"), std::string::npos);
    // ...and the event that starts after the end refuses it.
    auto far = conform_shots(*edit, 60);
    ASSERT_FALSE(far);
    EXPECT_NE(far.error().message.find("past the end"), std::string::npos);
}

TEST(Conform, DropFrameEdlsAtTwentyNineNineSeven) {
    const char* edl = R"(TITLE: NTSC
FCM: DROP FRAME
001  AX V C 00:00:00;00 00:01:00;02 00:00:00;00 00:01:00;02
002  AX V C 00:00:00;00 00:00:10;00 00:01:00;02 00:01:10;02
)";
    auto edit = parse_edl(edl, rate("29.97"));
    ASSERT_TRUE(edit) << edit.error().message;
    auto c = conform_shots(*edit, 3000);
    ASSERT_TRUE(c) << c.error().message;
    EXPECT_EQ(c->shots.shots[1].first, 1800);   // ;00 and ;01 of minute 1 do not exist
    EXPECT_EQ(c->shots.shots[1].last, 1800 + 300 - 1);
}

TEST(Conform, TheEditSaysWhetherItsTimecodeIsDropFrame) {
    const char* edl = R"(FCM: DROP FRAME
001  AX V C 00:00:00:00 00:00:10:00 00:01:00:02 00:01:10:02
)";
    auto edit = parse_edl(edl, rate("29.97"));
    ASSERT_TRUE(edit) << edit.error().message;
    EXPECT_TRUE(edit->drop_frame);
    // A record start written with colons is read in the edit's drop frame timecode.
    auto start = timecode_frames("00:01:00:02", rate("29.97"), edit->drop_frame);
    ASSERT_TRUE(start);
    EXPECT_EQ(*start, 1800);
    auto c = conform_shots(*edit, 400, {}, *start);
    ASSERT_TRUE(c) << c.error().message;
    EXPECT_EQ(c->shots.shots[0].first, 0);
    EXPECT_FALSE(parse_edl("FCM: NON-DROP FRAME\n001  AX V C 00:00:00:00 00:00:01:00 01:00:00:00 01:00:01:00\n",
                           rate("29.97"))->drop_frame);
}

TEST(Conform, AnEdlWithoutVideoIsRefused) {
    EXPECT_FALSE(parse_edl("TITLE: X\n001  AX A C 00:00:00:00 00:00:01:00 01:00:00:00 01:00:01:00\n", rate("24")));
    EXPECT_FALSE(parse_edl("001  AX V C 00:00:00:00 00:00:01:00 01:00:00\n", rate("24")));
}

TEST(Conform, AnFcpxmlStorylineBecomesShots) {
    const char* xml = R"(<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.10">
  <resources><format id="r1" frameDuration="1/24s" width="1920" height="1080"/></resources>
  <library><event name="Day 1"><project name="Trailer &amp; Teaser">
    <sequence format="r1" duration="120/24s" tcStart="3600s" tcFormat="NDF">
      <spine>
        <asset-clip ref="r2" name="sh010" offset="3600s" duration="48/24s" start="0s">
          <asset-clip ref="r9" lane="1" name="connected title" offset="10/24s" duration="12/24s"/>
          <adjust-volume amount="-6dB"/>
        </asset-clip>
        <gap name="Gap" offset="86448/24s" duration="12/24s"/>
        <!-- <asset-clip name="commented out" offset="0s" duration="1s"/> -->
        <clip name="sh020 &quot;hero&quot;" offset="86460/24s" duration="60/24s">
          <video ref="r3" offset="0s" duration="60/24s"/>
        </clip>
      </spine>
    </sequence>
  </project></event></library>
</fcpxml>)";
    auto edit = parse_fcpxml(xml, rate("24"));
    ASSERT_TRUE(edit) << edit.error().message << " " << edit.error().detail;
    EXPECT_EQ(edit->title, "Trailer & Teaser");
    ASSERT_EQ(edit->events.size(), 2u);   // the gap and the connected clip are not shots
    EXPECT_EQ(edit->events[0].name, "sh010");
    EXPECT_EQ(edit->events[0].record_in, 0);   // less tcStart
    EXPECT_EQ(edit->events[0].record_out, 48);
    EXPECT_EQ(edit->events[1].name, "sh020 \"hero\"");
    EXPECT_EQ(edit->events[1].record_in, 60);
    auto c = conform_shots(*edit, 120, {}, 0);
    ASSERT_TRUE(c) << c.error().message;
    ASSERT_EQ(c->shots.shots.size(), 2u);
    EXPECT_EQ(c->shots.shots[1].first, 60);
    EXPECT_EQ(c->shots.shots[1].last, 119);
    EXPECT_EQ(c->shots.shots[1].source, "fcpxml");
    EXPECT_EQ(shot_index(c->shots, 50), -1);   // the gap takes the base grade
}

TEST(Conform, ACompoundClipsSpineInResourcesIsNotTheProgramme) {
    const char* xml = R"(<fcpxml version="1.10">
  <resources>
    <format id="r1" frameDuration="1/24s"/>
    <media id="r5" name="compound">
      <sequence format="r1" duration="10s" tcStart="0s">
        <spine><asset-clip name="inside the compound" offset="0s" duration="10s"/></spine>
      </sequence>
    </media>
  </resources>
  <library><event><project name="Programme">
    <sequence format="r1" duration="4s" tcStart="3600s" tcFormat="DF">
      <spine>
        <ref-clip ref="r5" name="sh010" offset="3600s" duration="2s"/>
        <asset-clip name="sh020" offset="3602s" duration="2s"/>
      </spine>
    </sequence>
  </project></event></library>
</fcpxml>)";
    auto edit = parse_fcpxml(xml, rate("24"));
    ASSERT_TRUE(edit) << edit.error().message << " " << edit.error().detail;
    EXPECT_EQ(edit->title, "Programme");
    EXPECT_TRUE(edit->drop_frame);
    ASSERT_EQ(edit->events.size(), 2u);
    EXPECT_EQ(edit->events[0].name, "sh010");
    EXPECT_EQ(edit->events[0].record_in, 0);   // less the programme's tcStart, not the compound's
    EXPECT_EQ(edit->events[1].record_in, 48);
}

TEST(Conform, OtherXmlIsRefused) {
    EXPECT_FALSE(parse_fcpxml("<xmeml><sequence/></xmeml>", rate("24")));
    EXPECT_FALSE(parse_fcpxml("<fcpxml><library/></fcpxml>", rate("24")));
    EXPECT_FALSE(parse_fcpxml("<fcpxml><sequence><spine><clip name='x'/></spine></sequence></fcpxml>", rate("24")));
    // A storyline outside any project is not a programme.
    EXPECT_FALSE(parse_fcpxml(
        "<fcpxml><sequence tcStart='0s'><spine><clip name='x' offset='0s' duration='1s'/></spine></sequence></fcpxml>",
        rate("24")));
    auto in_project = parse_fcpxml(
        "<fcpxml><sequence tcStart='0s'><spine><clip name='loose' offset='0s' duration='1s'/></spine></sequence>"
        "<project name='P'><sequence tcStart='10s'><spine><clip name='x' offset='10s' duration='1s'/></spine></sequence></project>"
        "</fcpxml>",
        rate("24"));
    ASSERT_TRUE(in_project) << in_project.error().message;
    ASSERT_EQ(in_project->events.size(), 1u);
    EXPECT_EQ(in_project->events[0].name, "x");
    EXPECT_EQ(in_project->events[0].record_in, 0);
}
