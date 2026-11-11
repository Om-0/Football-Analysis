"""Comprehensive unit tests for PITCHMIND Canonical Data Schemas (Hardened v2.0).

Verifies:
1. Schema versioning across models (SCHEMA_VERSION == '2.0.0').
2. Fixed pitch coordinate convention (origin centre, raw coordinates vs derived attacking features).
3. Observation semantics: 'detected' is strictly observed; 'estimated', 'interpolated',
   'extrapolated', 'lost', and 'invalid' cannot have observed_flag=True.
4. Configurable ball-height ground tolerance and unknown state support.
5. Explicitly sourced pitch dimensions (measured, supplied, assumed).
6. Nested immutability (frozen Pydantic models with immutable tuple collections).
7. Complete nine-dimensional VideoQualityReport validation.
8. Homography 3x3 matrix verification.
"""

from datetime import date
import pytest
from pydantic import ValidationError

from pitchmind.data_engine.schema import (
    SCHEMA_VERSION,
    BallFrame,
    BallState,
    CalibrationQualityMetrics,
    CameraClass,
    DetectionQualityMetrics,
    EventRecord,
    MatchMetadata,
    Period,
    PitchDimensionSource,
    PlayerFrame,
    PoseQualityMetrics,
    PreflightQualityMetrics,
    PreflightReport,
    TaskReadiness,
    TeamAttackingDirection,
    TrackingQualityMetrics,
    TrackingStatus,
    VideoMetadata,
    VideoQualityReport,
    VideoSegment,
    VisibilityStatus,
)


# ============================================================================
# 1. Schema Versioning & MatchMetadata Tests
# ============================================================================


def test_schema_version_constant():
    assert SCHEMA_VERSION == "2.0.0"


def test_match_metadata_assumed_defaults():
    meta = MatchMetadata(
        match_id="match-2026-001",
        home_team="Arsenal",
        away_team="Chelsea",
        source_provider="pitchmind_video_cv",
    )
    assert meta.schema_version == "2.0.0"
    assert meta.pitch_length_m == 105.0
    assert meta.pitch_width_m == 68.0
    assert meta.pitch_dimension_source == PitchDimensionSource.ASSUMED
    assert meta.coordinate_origin == "center"


def test_match_metadata_explicit_source():
    meta = MatchMetadata(
        match_id="match-2026-002",
        home_team="Bayern",
        away_team="Dortmund",
        pitch_length_m=104.5,
        pitch_width_m=67.8,
        pitch_dimension_source=PitchDimensionSource.MEASURED,
        source_provider="idsse",
        source_rights_record_id="dataset-idsse-bundesliga",
        match_date=date(2023, 4, 1),
        competition="Bundesliga",
    )
    assert meta.pitch_dimension_source == PitchDimensionSource.MEASURED
    assert meta.pitch_length_m == 104.5
    assert meta.pitch_width_m == 67.8


def test_match_metadata_invalid_dimensions():
    with pytest.raises(ValidationError):
        MatchMetadata(
            match_id="match-001",
            home_team="TeamA",
            away_team="TeamB",
            pitch_length_m=-105.0,  # Invalid: <= 0
            source_provider="test",
        )


def test_match_metadata_immutability():
    meta = MatchMetadata(
        match_id="match-001",
        home_team="TeamA",
        away_team="TeamB",
        source_provider="test",
    )
    with pytest.raises(ValidationError):
        meta.home_team = "NewTeam"  # Frozen model cannot be mutated


# ============================================================================
# 2. VideoMetadata & PreflightReport Tests
# ============================================================================


def test_video_metadata_valid():
    valid_sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    meta = VideoMetadata(
        video_id="vid-100",
        file_path="uploads/tactical_clip.mp4",
        sha256_checksum=valid_sha256,
        duration_seconds=300.5,
        total_frames=7512,
        fps=25.0,
        width=1920,
        height=1080,
        codec="h264",
        container="mp4",
        audio_streams=1,
    )
    assert meta.schema_version == "2.0.0"
    assert meta.video_id == "vid-100"


def test_preflight_report_nested_immutability():
    report = PreflightReport(
        video_id="vid-100",
        passed=True,
        checks_run=("file_extension_support", "sha256_checksum", "opencv_decode_sampling"),
        decode_success=True,
        timestamp_continuity=True,
        duration_seconds=120.0,
        fps_measured=25.0,
        dropped_frames_detected=0,
        aspect_ratio_standard=True,
        failure_reasons=(),
    )
    assert isinstance(report.checks_run, tuple)
    assert isinstance(report.failure_reasons, tuple)
    with pytest.raises(AttributeError):
        report.checks_run.append("new_check")  # Tuples are immutable


# ============================================================================
# 3. VideoQualityReport Tests (Complete 9-Dimensional Vector)
# ============================================================================


def test_video_quality_report_nine_dimensions():
    preflight = PreflightQualityMetrics(
        video_integrity_score=0.98,
        aspect_ratio=1.777,
        aspect_ratio_standard=True,
        mean_brightness=112.5,
        contrast_score=48.2,
        blur_score_laplacian=240.5,
        image_quality_score=0.86,
        preliminary_pitch_green_fraction=0.65,
    )
    detection = DetectionQualityMetrics(
        pitch_coverage_fraction=0.88,
        player_visibility_fraction=0.91,
        detection_reliability_score=0.85,
    )
    tracking = TrackingQualityMetrics(
        identity_continuity_score=0.79,
        ball_observability_score=0.72,
    )
    calibration = CalibrationQualityMetrics(
        calibration_reliability_score=0.92,
        calibration_reprojection_error_m=0.82,
    )
    pose = PoseQualityMetrics(
        pose_quality_score=0.75,
        mean_keypoint_confidence=0.82,
        crop_resolution_adequate=True,
    )
    q = VideoQualityReport(
        report_id="qrep-001",
        video_id="vid-100",
        segment_id="seg-01",
        camera_class=CameraClass.WIDE_TACTICAL,
        preflight_quality=preflight,
        detection_quality=detection,
        tracking_quality=tracking,
        calibration_quality=calibration,
        pose_quality=pose,
        pressuregraph_readiness=TaskReadiness.READY,
        tactical_shift_readiness=TaskReadiness.READY,
        roleshift_readiness=TaskReadiness.PARTIAL,
        scan2play_readiness=TaskReadiness.READY,
        reasons_suppressed=(),
    )
    assert q.schema_version == "2.0.0"
    assert q.preflight_quality.video_integrity_score == 0.98
    assert q.preflight_quality.image_quality_score == 0.86
    assert q.detection_quality.pitch_coverage_fraction == 0.88
    assert q.tracking_quality.identity_continuity_score == 0.79
    assert q.calibration_quality.calibration_reliability_score == 0.92
    assert q.pose_quality.pose_quality_score == 0.75
    assert isinstance(q.reasons_suppressed, tuple)


def test_video_quality_report_bounded_dimensions():
    with pytest.raises(ValidationError):
        PreflightQualityMetrics(
            video_integrity_score=-0.1,  # Invalid: < 0.0
            aspect_ratio=1.777,
            aspect_ratio_standard=True,
            mean_brightness=112.5,
            contrast_score=48.2,
            blur_score_laplacian=100.0,
            image_quality_score=0.5,
            preliminary_pitch_green_fraction=0.5,
        )


# ============================================================================
# 4. VideoSegment & Homography Tests
# ============================================================================


def test_video_segment_immutable_homography():
    h_matrix: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    seg = VideoSegment(
        segment_id="seg-001",
        match_id="match-001",
        video_id="vid-100",
        start_time_seconds=0.0,
        end_time_seconds=45.0,
        start_frame=0,
        end_frame=1125,
        camera_class=CameraClass.WIDE_TACTICAL,
        camera_confidence=0.95,
        is_replay=False,
        homography_matrix=h_matrix,
        calibration_reprojection_error_m=0.65,
    )
    assert seg.homography_matrix == h_matrix
    assert isinstance(seg.homography_matrix, tuple)


def test_video_segment_temporal_ordering_violation():
    with pytest.raises(ValidationError, match="end_time_seconds"):
        VideoSegment(
            segment_id="seg-bad",
            match_id="match-001",
            video_id="vid-100",
            start_time_seconds=50.0,
            end_time_seconds=40.0,  # Invalid: end < start
            start_frame=1250,
            end_frame=1000,
            camera_class=CameraClass.MAIN_BROADCAST,
            camera_confidence=0.8,
        )


# ============================================================================
# 5. PlayerFrame Tests (Observation Semantics & Fixed Coordinates)
# ============================================================================


def test_player_frame_valid_observed_and_derived_attack():
    pf = PlayerFrame(
        match_id="match-001",
        period=Period.FIRST_HALF,
        timestamp_seconds=12.4,
        frame_index=310,
        player_id="player-007",
        team_id="0",
        # Fixed raw pitch coordinates (centre origin)
        x=15.2,
        y=-4.5,
        # Derived attacking coordinates pinned to team direction (+1)
        team_attacking_direction=1,
        x_attack=15.2,
        y_attack=-4.5,
        bbox_xyxy=(500.0, 300.0, 560.0, 420.0),
        vx=2.1,
        vy=-0.8,
        speed=2.25,
        observed_flag=True,
        visibility_status=VisibilityStatus.VISIBLE,
        tracking_status=TrackingStatus.DETECTED,
        detection_confidence=0.92,
        tracking_confidence=0.88,
        calibration_uncertainty_m=0.35,
        camera_segment_id="seg-001",
    )
    assert pf.player_id == "player-007"
    assert pf.x == 15.2
    assert pf.y == -4.5
    assert pf.x_attack == 15.2
    assert pf.team_attacking_direction == 1
    assert pf.observed_flag is True


def test_player_frame_missing_coordinates_as_none():
    """Missing observations MUST be represented as None, never 0.0."""
    pf = PlayerFrame(
        match_id="match-001",
        period=1,
        timestamp_seconds=20.0,
        player_id="player-010",
        team_id="1",
        x=None,
        y=None,
        observed_flag=False,
        visibility_status=VisibilityStatus.OUT_OF_VIEW,
        tracking_status=TrackingStatus.LOST,
    )
    assert pf.x is None
    assert pf.y is None
    assert pf.x != 0.0
    assert pf.y != 0.0


@pytest.mark.parametrize(
    "non_detected_status",
    [
        TrackingStatus.ESTIMATED,
        TrackingStatus.INTERPOLATED,
        TrackingStatus.EXTRAPOLATED,
        TrackingStatus.LOST,
        TrackingStatus.INVALID,
    ],
)
def test_player_frame_observation_semantics_rejection(non_detected_status):
    """Observation Semantics: non-detected statuses MUST NOT have observed_flag=True."""
    with pytest.raises(ValidationError, match="cannot have observed_flag=True"):
        PlayerFrame(
            match_id="match-001",
            period=1,
            timestamp_seconds=15.0,
            player_id="player-005",
            team_id="0",
            x=10.0 if non_detected_status not in (TrackingStatus.LOST, TrackingStatus.INVALID) else None,
            y=5.0 if non_detected_status not in (TrackingStatus.LOST, TrackingStatus.INVALID) else None,
            observed_flag=True,  # VIOLATION: non-detected status cannot be marked observed
            visibility_status=VisibilityStatus.OCCLUDED,
            tracking_status=non_detected_status,
        )


def test_player_frame_detected_must_be_observed():
    """Observation Semantics: tracking_status='detected' MUST have observed_flag=True."""
    with pytest.raises(ValidationError, match="must have observed_flag=True"):
        PlayerFrame(
            match_id="match-001",
            period=1,
            timestamp_seconds=12.0,
            player_id="player-001",
            team_id="0",
            x=0.0,
            y=0.0,
            observed_flag=False,  # VIOLATION: detected must have observed_flag=True
            visibility_status=VisibilityStatus.VISIBLE,
            tracking_status=TrackingStatus.DETECTED,
        )


def test_player_frame_incomplete_attacking_coordinates():
    with pytest.raises(ValidationError, match="Incomplete attacking coordinate pair"):
        PlayerFrame(
            match_id="match-001",
            period=1,
            timestamp_seconds=10.0,
            player_id="player-001",
            team_id="0",
            x=10.0,
            y=5.0,
            team_attacking_direction=1,
            x_attack=10.0,
            y_attack=None,  # Incomplete
            observed_flag=True,
            visibility_status=VisibilityStatus.VISIBLE,
            tracking_status=TrackingStatus.DETECTED,
        )


def test_player_frame_invalid_attacking_direction():
    with pytest.raises(ValidationError, match="team_attacking_direction"):
        PlayerFrame(
            match_id="match-001",
            period=1,
            timestamp_seconds=10.0,
            player_id="player-001",
            team_id="0",
            x=10.0,
            y=5.0,
            team_attacking_direction=0,  # Invalid: must be +1 or -1
            x_attack=10.0,
            y_attack=5.0,
            observed_flag=True,
            visibility_status=VisibilityStatus.VISIBLE,
            tracking_status=TrackingStatus.DETECTED,
        )


def test_player_frame_lost_cannot_have_coordinates():
    with pytest.raises(ValidationError, match="cannot have valid pitch coordinates"):
        PlayerFrame(
            match_id="match-001",
            period=1,
            timestamp_seconds=30.0,
            player_id="player-002",
            team_id="0",
            x=20.0,  # VIOLATION
            y=10.0,
            observed_flag=False,
            visibility_status=VisibilityStatus.OUT_OF_VIEW,
            tracking_status=TrackingStatus.LOST,
        )


def test_player_frame_out_of_view_cannot_have_bbox():
    with pytest.raises(ValidationError, match="cannot have a camera bbox_xyxy"):
        PlayerFrame(
            match_id="match-001",
            period=1,
            timestamp_seconds=40.0,
            player_id="player-003",
            team_id="1",
            x=None,
            y=None,
            bbox_xyxy=(100.0, 100.0, 150.0, 200.0),  # VIOLATION
            observed_flag=False,
            visibility_status=VisibilityStatus.OUT_OF_VIEW,
            tracking_status=TrackingStatus.LOST,
        )


# ============================================================================
# 6. BallFrame Tests (Ground Tolerance & Unknown State)
# ============================================================================


def test_ball_frame_ground_within_tolerance():
    bf = BallFrame(
        match_id="match-001",
        period=1,
        timestamp_seconds=12.4,
        x=2.5,
        y=-1.0,
        z=0.2,  # <= 0.5 m default tolerance
        ball_state=BallState.GROUND,
        observed_flag=True,
        visibility_status=VisibilityStatus.VISIBLE,
        tracking_status=TrackingStatus.DETECTED,
        detection_confidence=0.88,
    )
    assert bf.ball_state == BallState.GROUND
    assert bf.z == 0.2


def test_ball_frame_unknown_state_when_uncalibrated():
    """Ball state can be UNKNOWN when elevation evidence is insufficient."""
    bf = BallFrame(
        match_id="match-001",
        period=1,
        timestamp_seconds=12.8,
        x=10.0,
        y=5.0,
        z=None,  # Planar homography cannot reconstruct height
        ball_state=BallState.UNKNOWN,
        observed_flag=True,
        visibility_status=VisibilityStatus.VISIBLE,
        tracking_status=TrackingStatus.DETECTED,
    )
    assert bf.ball_state == BallState.UNKNOWN
    assert bf.z is None


def test_ball_frame_custom_tolerance_valid():
    """With configurable tolerance of 1.2m, a ball at 1.0m can still be classified ground (e.g. bounce)."""
    bf = BallFrame(
        match_id="match-001",
        period=1,
        timestamp_seconds=13.0,
        x=15.0,
        y=8.0,
        z=1.0,
        ground_height_tolerance_m=1.2,
        ball_state=BallState.GROUND,
        observed_flag=True,
        visibility_status=VisibilityStatus.VISIBLE,
        tracking_status=TrackingStatus.DETECTED,
    )
    assert bf.ground_height_tolerance_m == 1.2
    assert bf.ball_state == BallState.GROUND


def test_ball_frame_exceeds_tolerance_cannot_be_ground():
    """Ball at 2.5m exceeds default 0.5m tolerance and cannot be classified as GROUND."""
    with pytest.raises(ValidationError, match="exceeds ground tolerance"):
        BallFrame(
            match_id="match-001",
            period=1,
            timestamp_seconds=14.0,
            x=20.0,
            y=10.0,
            z=2.5,
            ball_state=BallState.GROUND,  # Inconsistent with z=2.5m > 0.5m
            observed_flag=True,
            visibility_status=VisibilityStatus.VISIBLE,
            tracking_status=TrackingStatus.DETECTED,
        )


def test_ball_frame_airborne_valid():
    bf = BallFrame(
        match_id="match-001",
        period=1,
        timestamp_seconds=14.0,
        x=20.0,
        y=10.0,
        z=2.5,
        ball_state=BallState.AIRBORNE,
        observed_flag=True,
        visibility_status=VisibilityStatus.VISIBLE,
        tracking_status=TrackingStatus.DETECTED,
    )
    assert bf.ball_state == BallState.AIRBORNE
    assert bf.z == 2.5


# ============================================================================
# 7. EventRecord Tests (Immutable Collections)
# ============================================================================


def test_event_record_immutable_secondary_players():
    ev = EventRecord(
        event_id="ev-press-101",
        match_id="match-001",
        period=1,
        timestamp_seconds=120.0,
        end_timestamp_seconds=125.4,
        event_type="pressing_episode",
        primary_player_id="player-009",
        secondary_player_ids=("player-007", "player-010"),
        team_id="1",
        outcome="regain",
        pitch_x=34.0,
        pitch_y=14.0,
        source="pressuregraph_model",
        confidence=0.87,
        evidence_metadata={"closing_speed_mps": 4.2},
    )
    assert ev.schema_version == "2.0.0"
    assert isinstance(ev.secondary_player_ids, tuple)
    with pytest.raises(AttributeError):
        ev.secondary_player_ids.append("player-011")


def test_event_record_inverted_interval():
    with pytest.raises(ValidationError, match="end_timestamp_seconds"):
        EventRecord(
            event_id="ev-bad-interval",
            match_id="match-001",
            period=1,
            timestamp_seconds=150.0,
            end_timestamp_seconds=140.0,  # Invalid: end < start
            event_type="tactical_shift",
        )


def test_event_record_deep_immutability():
    nested_data = {
        "metrics": {"closing_speed": 4.5, "angle": 35.0},
        "contributors": ["player-001", "player-002"],
        "tags": {"zone": "defensive_third"},
    }
    ev = EventRecord(
        event_id="ev-press-deep",
        match_id="match-001",
        period=1,
        timestamp_seconds=120.0,
        event_type="pressing_action",
        evidence_metadata=nested_data,
    )
    # 1. Top-level mapping is immutable
    with pytest.raises(TypeError):
        ev.evidence_metadata["new_key"] = 123  # MappingProxyType prevents item assignment

    # 2. Nested dictionary is immutable
    with pytest.raises(TypeError):
        ev.evidence_metadata["metrics"]["closing_speed"] = 9.9

    # 3. Nested list was converted to immutable tuple
    assert isinstance(ev.evidence_metadata["contributors"], tuple)
    with pytest.raises(AttributeError):
        ev.evidence_metadata["contributors"].append("player-003")


# ============================================================================
# 8. Frame Source Synchronization & Attacking Direction Tests
# ============================================================================


def test_match_metadata_period_attacking_directions():
    meta = MatchMetadata(
        match_id="match-directions",
        home_team="Arsenal",
        away_team="Chelsea",
        source_provider="skillcorner",
        period_attacking_directions={
            1: {"home": 1, "away": -1},
            2: {"home": -1, "away": 1},
        },
    )
    assert meta.period_attacking_directions[1]["home"] == 1
    assert meta.period_attacking_directions[2]["home"] == -1


def test_player_and_ball_frame_source_synchronization():
    pf = PlayerFrame(
        match_id="match-sync",
        period=1,
        timestamp_seconds=10.04,
        frame_index=251,
        source_pts=1004000,
        player_id="player-10",
        team_id="1",
        x=12.5,
        y=-5.0,
        team_attacking_direction=TeamAttackingDirection.TOWARDS_POSITIVE_X,
        x_attack=12.5,
        y_attack=-5.0,
        observed_flag=True,
        visibility_status=VisibilityStatus.VISIBLE,
        tracking_status=TrackingStatus.DETECTED,
        bbox_xyxy=(100.0, 200.0, 150.0, 320.0),
    )
    assert pf.frame_index == 251
    assert pf.source_pts == 1004000
    assert pf.team_attacking_direction == TeamAttackingDirection.TOWARDS_POSITIVE_X

    bf = BallFrame(
        match_id="match-sync",
        period=1,
        timestamp_seconds=10.04,
        frame_index=251,
        source_pts=1004000,
        x=14.0,
        y=-2.0,
        z=0.1,
        team_attacking_direction=TeamAttackingDirection.TOWARDS_POSITIVE_X,
        x_attack=14.0,
        y_attack=-2.0,
        ball_state=BallState.GROUND,
        observed_flag=True,
        visibility_status=VisibilityStatus.VISIBLE,
        tracking_status=TrackingStatus.DETECTED,
    )
    assert bf.frame_index == 251
    assert bf.source_pts == 1004000

