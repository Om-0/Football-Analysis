"""Unit and integration tests for Tracking Provider Adapters (Metrica, SkillCorner, IDSSE).

Verifies:
1. Metric coordinate normalization to fixed pitch coordinate standard (origin (0, 0),
   X in [-pitch_length/2, +pitch_length/2], Y in [-pitch_width/2, +pitch_width/2]).
2. Observation semantics: detected -> observed_flag=True; missing/extrapolated -> observed_flag=False.
   Missing coordinates must be None, never 0.0.
3. Preservation of native sampling rates (25 Hz for Metrica/IDSSE, 10 Hz for SkillCorner)
   without silent resampling.
4. Correct ball state classification from elevation (GROUND <= 0.5m, AIRBORNE > 0.5m, UNKNOWN when None).
5. Generation of structured TrackingIngestionReport for auditing and provenance.
"""

import json
import tempfile
from pathlib import Path
from typing import Tuple
import pytest

from pitchmind.data_engine.schema import (
    BallState,
    PitchDimensionSource,
    TeamAttackingDirection,
    TrackingStatus,
    VisibilityStatus,
)
from pitchmind.ingestion.tracking import (
    IDSSETrackingAdapter,
    MetricaTrackingAdapter,
    SkillCornerTrackingAdapter,
    TrackingDataset,
    TrackingIngestionReport,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


# ============================================================================
# 1. Metrica Adapter Tests
# ============================================================================


@pytest.fixture
def metrica_sample_csvs(temp_dir: Path) -> Tuple[Path, Path]:
    """Generate small synthetic Metrica Home and Away CSV fixtures."""
    home_csv = temp_dir / "Sample_Game_1_RawTrackingData_Home_Team.csv"
    away_csv = temp_dir / "Sample_Game_1_RawTrackingData_Away_Team.csv"

    # Home CSV with 3-row header
    home_content = (
        ",,,Home,Home,Home,Home,,,\n"
        ",,,Player11,Player11,Player12,Player12,Ball,Ball,\n"
        "Period,Frame,Time [s],Player11_x,Player11_y,Player12_x,Player12_y,Ball_x,Ball_y\n"
        "1,1,0.04,0.50,0.50,0.10,0.20,0.50,0.50\n"
        "1,2,0.08,0.51,0.52,NaN,NaN,0.52,0.51\n"
        "2,2501,100.04,0.48,0.49,0.85,0.75,NaN,NaN\n"
    )
    home_csv.write_text(home_content, encoding="utf-8")

    # Away CSV with 3-row header
    away_content = (
        ",,,Away,Away,Away,Away,,,\n"
        ",,,Player21,Player21,Player22,Player22,Ball,Ball,\n"
        "Period,Frame,Time [s],Player21_x,Player21_y,Player22_x,Player22_y,Ball_x,Ball_y\n"
        "1,1,0.04,0.90,0.80,0.70,0.60,0.50,0.50\n"
        "1,2,0.08,0.89,0.79,0.69,0.59,0.52,0.51\n"
        "2,2501,100.04,0.15,0.25,NaN,NaN,NaN,NaN\n"
    )
    away_csv.write_text(away_content, encoding="utf-8")

    return home_csv, away_csv


def test_metrica_adapter_coordinate_normalization(metrica_sample_csvs: Tuple[Path, Path]):
    home_csv, away_csv = metrica_sample_csvs
    adapter = MetricaTrackingAdapter(pitch_length_m=105.0, pitch_width_m=68.0)
    dataset = adapter.load(home_csv, away_csv_path=away_csv, match_id="metrica-test-01")

    assert dataset.match_metadata.match_id == "metrica-test-01"
    assert dataset.match_metadata.pitch_length_m == 105.0
    assert dataset.match_metadata.pitch_width_m == 68.0
    assert dataset.ingestion_report.native_sampling_rate_hz == 25.0
    assert dataset.ingestion_report.resampling_applied is False
    assert dataset.ingestion_report.total_frames == 3

    # Check Frame 1: Player 11 is at (0.50, 0.50) -> maps exactly to pitch center (0.0, 0.0)
    f1_p11 = [p for p in dataset.player_frames if p.frame_index == 1 and p.player_id == "H_Player11"][0]
    assert f1_p11.x == 0.0
    assert f1_p11.y == 0.0
    assert f1_p11.observed_flag is True
    assert f1_p11.tracking_status == TrackingStatus.DETECTED
    assert f1_p11.visibility_status == VisibilityStatus.VISIBLE

    # Check Frame 1: Ball at (0.50, 0.50) -> maps to (0.0, 0.0)
    f1_ball = [b for b in dataset.ball_frames if b.frame_index == 1][0]
    assert f1_ball.x == 0.0
    assert f1_ball.y == 0.0
    assert f1_ball.z is None
    assert f1_ball.ball_state == BallState.UNKNOWN
    assert f1_ball.observed_flag is True
    assert f1_ball.tracking_status == TrackingStatus.DETECTED

    # Check Frame 2: Player 12 has NaN coordinates -> must be None, observed_flag=False, tracking_status=LOST
    f2_p12 = [p for p in dataset.player_frames if p.frame_index == 2 and p.player_id == "H_Player12"][0]
    assert f2_p12.x is None
    assert f2_p12.y is None
    assert f2_p12.observed_flag is False
    assert f2_p12.tracking_status == TrackingStatus.LOST
    assert f2_p12.visibility_status == VisibilityStatus.OUT_OF_VIEW

    # Check Frame 3: Ball has NaN coordinates -> must be None, observed_flag=False, tracking_status=LOST
    f3_ball = [b for b in dataset.ball_frames if b.frame_index == 2501][0]
    assert f3_ball.x is None
    assert f3_ball.y is None
    assert f3_ball.observed_flag is False
    assert f3_ball.tracking_status == TrackingStatus.LOST


def test_metrica_period_attacking_directions(metrica_sample_csvs: Tuple[Path, Path]):
    home_csv, away_csv = metrica_sample_csvs
    adapter = MetricaTrackingAdapter()
    dataset = adapter.load(home_csv, away_csv_path=away_csv)

    # In Period 1, Home attacks +1
    f1_home = [p for p in dataset.player_frames if p.period == 1 and p.team_id == "0" and p.x is not None][0]
    assert f1_home.team_attacking_direction == 1
    assert f1_home.x_attack == f1_home.x * 1

    # In Period 2, Home attacks -1 (teams switch ends)
    f2_home = [p for p in dataset.player_frames if p.period == 2 and p.team_id == "0" and p.x is not None][0]
    assert f2_home.team_attacking_direction == -1
    assert f2_home.x_attack == f2_home.x * -1


# ============================================================================
# 2. SkillCorner Adapter Tests
# ============================================================================


@pytest.fixture
def skillcorner_sample_json(temp_dir: Path) -> Tuple[Path, Path]:
    """Generate small synthetic SkillCorner match and structured tracking data."""
    match_file = temp_dir / "match_123_match_data.json"
    struct_file = temp_dir / "match_123_structured_data.json"

    match_meta = {
        "id": 123,
        "home_team": {"id": 10, "name": "Sydney FC"},
        "away_team": {"id": 20, "name": "Melbourne City"},
        "pitch_length": 105.0,
        "pitch_width": 68.0,
    }
    match_file.write_text(json.dumps(match_meta), encoding="utf-8")

    frames_data = [
        {
            "frame": 0,
            "period": 1,
            "timestamp": "00:00:00.000",
            "data": [
                {
                    "trackable_object": 101,
                    "group_name": "home team",
                    "x": -15.5,
                    "y": 10.2,
                    "is_detected": True,  # Directly observed
                },
                {
                    "trackable_object": 102,
                    "group_name": "home team",
                    "x": -35.0,
                    "y": -20.0,
                    "is_detected": False,  # Extrapolated off-screen!
                },
                {
                    "trackable_object": 201,
                    "group_name": "away team",
                    "x": 20.0,
                    "y": -5.0,
                    "is_detected": True,
                },
            ],
            "ball": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.2,  # Ground ball (<= 0.5m)
                "is_detected": True,
            },
        },
        {
            "frame": 1,
            "period": 1,
            "timestamp": "00:00:00.100",
            "data": [
                {
                    "trackable_object": 101,
                    "group_name": "home team",
                    "x": -15.3,
                    "y": 10.1,
                    "is_detected": True,
                },
            ],
            "ball": {
                "x": 2.5,
                "y": 1.0,
                "z": 2.4,  # Airborne ball (> 0.5m)
                "is_detected": True,
            },
        },
    ]
    struct_file.write_text(json.dumps(frames_data), encoding="utf-8")

    return struct_file, match_file


def test_skillcorner_adapter_observation_semantics(skillcorner_sample_json: Tuple[Path, Path]):
    struct_file, match_file = skillcorner_sample_json
    adapter = SkillCornerTrackingAdapter()
    dataset = adapter.load(struct_file, match_data_path=match_file)

    assert dataset.match_metadata.match_id == "skillcorner-123"
    assert dataset.match_metadata.home_team == "Sydney FC"
    assert dataset.match_metadata.pitch_dimension_source == PitchDimensionSource.SUPPLIED
    assert dataset.ingestion_report.native_sampling_rate_hz == 10.0
    assert dataset.ingestion_report.total_frames == 2

    # 1. Directly observed player (is_detected == True)
    p_obs = [p for p in dataset.player_frames if p.player_id == "101" and p.frame_index == 0][0]
    assert p_obs.observed_flag is True
    assert p_obs.tracking_status == TrackingStatus.DETECTED
    assert p_obs.visibility_status == VisibilityStatus.VISIBLE
    assert p_obs.x == -15.5
    assert p_obs.y == 10.2

    # 2. Extrapolated player (is_detected == False) -> MUST NOT have observed_flag=True!
    p_extrap = [p for p in dataset.player_frames if p.player_id == "102" and p.frame_index == 0][0]
    assert p_extrap.observed_flag is False
    assert p_extrap.tracking_status == TrackingStatus.EXTRAPOLATED
    assert p_extrap.visibility_status == VisibilityStatus.OUT_OF_VIEW
    assert p_extrap.x == -35.0
    assert p_extrap.y == -20.0

    # 3. Ball state classification
    b_ground = [b for b in dataset.ball_frames if b.frame_index == 0][0]
    assert b_ground.z == 0.2
    assert b_ground.ball_state == BallState.GROUND
    assert b_ground.observed_flag is True

    b_airborne = [b for b in dataset.ball_frames if b.frame_index == 1][0]
    assert b_airborne.z == 2.4
    assert b_airborne.ball_state == BallState.AIRBORNE
    assert b_airborne.observed_flag is True


# ============================================================================
# 3. IDSSE Adapter Tests
# ============================================================================


@pytest.fixture
def idsse_sample_json(temp_dir: Path) -> Path:
    """Generate small synthetic IDSSE JSON fixture with optical 25 Hz coordinates."""
    json_path = temp_dir / "idsse_buli_match_01.json"
    data = {
        "match_id": "buli-2023-01",
        "home_team": "Bayern",
        "away_team": "Dortmund",
        "pitch_length": 105.0,
        "pitch_width": 68.0,
        "sampling_rate": 25.0,
        "frames": [
            {
                "frame_id": 1,
                "period": 1,
                "timestamp": 0.04,
                "players": [
                    {"player_id": "bay_09", "team_id": "home", "x": 12.0, "y": -4.0},
                    {"player_id": "bvb_05", "team_id": "away", "x": 14.5, "y": -2.0},
                ],
                "ball": {"x": 12.2, "y": -3.8, "z": 0.1},
            },
            {
                "frame_id": 2,
                "period": 1,
                "timestamp": 0.08,
                "players": [
                    {"player_id": "bay_09", "team_id": "home", "x": None, "y": None},
                ],
                "ball": {"x": 12.5, "y": -3.5, "z": 1.2},
            },
        ],
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return json_path


@pytest.fixture
def idsse_sample_csv_bottom_left(temp_dir: Path) -> Path:
    """Generate IDSSE CSV fixture with bottom-left origin [0, 105] x [0, 68]."""
    csv_path = temp_dir / "idsse_match_bl.csv"
    content = (
        "period,frame_id,timestamp,player_id,team_id,x,y,z\n"
        "1,1,0.04,p1,home,52.5,34.0,\n"  # Center (52.5, 34.0) in [0, 105]x[0, 68]
        "1,1,0.04,p2,away,105.0,68.0,\n" # Top right (105, 68) -> (52.5, 34.0)
        "1,1,0.04,ball,ball,52.5,34.0,0.1\n"
    )
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def test_idsse_adapter_json_parsing(idsse_sample_json: Path):
    adapter = IDSSETrackingAdapter()
    dataset = adapter.load(idsse_sample_json)

    assert dataset.match_metadata.match_id == "buli-2023-01"
    assert dataset.match_metadata.home_team == "Bayern"
    assert dataset.ingestion_report.native_sampling_rate_hz == 25.0
    assert dataset.ingestion_report.total_frames == 2

    # Frame 1: Detected player
    p1 = [p for p in dataset.player_frames if p.frame_index == 1 and p.player_id == "bay_09"][0]
    assert p1.x == 12.0
    assert p1.y == -4.0
    assert p1.observed_flag is True
    assert p1.tracking_status == TrackingStatus.DETECTED

    # Frame 2: Missing player -> None coordinates
    p2 = [p for p in dataset.player_frames if p.frame_index == 2 and p.player_id == "bay_09"][0]
    assert p2.x is None
    assert p2.y is None
    assert p2.observed_flag is False
    assert p2.tracking_status == TrackingStatus.LOST

    # Frame 1 ball: ground (z=0.1)
    b1 = [b for b in dataset.ball_frames if b.frame_index == 1][0]
    assert b1.z == 0.1
    assert b1.ball_state == BallState.GROUND

    # Frame 2 ball: airborne (z=1.2)
    b2 = [b for b in dataset.ball_frames if b.frame_index == 2][0]
    assert b2.z == 1.2
    assert b2.ball_state == BallState.AIRBORNE


def test_idsse_adapter_bottom_left_origin_normalization(idsse_sample_csv_bottom_left: Path):
    adapter = IDSSETrackingAdapter()
    dataset = adapter.load(idsse_sample_csv_bottom_left)

    # p1 at (52.5, 34.0) in bottom-left origin should map to (0.0, 0.0) center origin
    p1 = [p for p in dataset.player_frames if p.player_id == "p1"][0]
    assert p1.x == 0.0
    assert p1.y == 0.0

    # p2 at (105.0, 68.0) should map to (+52.5, +34.0)
    p2 = [p for p in dataset.player_frames if p.player_id == "p2"][0]
    assert p2.x == 52.5
    assert p2.y == 34.0

    # ball at (52.5, 34.0) should map to (0.0, 0.0)
    b = dataset.ball_frames[0]
    assert b.x == 0.0
    assert b.y == 0.0


# ============================================================================
# 4. Ingestion Report & Robustness Tests
# ============================================================================


def test_metrica_load_from_directory(metrica_sample_csvs: Tuple[Path, Path], temp_dir: Path):
    """Test loading Metrica tracking when pointing directly to directory."""
    home_csv, _ = metrica_sample_csvs
    adapter = MetricaTrackingAdapter()
    dataset = adapter.load(home_csv.parent)

    assert dataset.match_metadata.source_provider == "metrica"
    assert dataset.ingestion_report.total_frames == 3
    assert len(dataset.player_frames) > 0


def test_skillcorner_load_without_metadata_file(temp_dir: Path):
    """Test SkillCorner adapter fallback when match_data.json is omitted."""
    struct_file = temp_dir / "match_999_structured_data.json"
    frames = [
        {
            "frame": 0,
            "period": 1,
            "timestamp": "00:00:01.000",
            "data": [
                {"trackable_object": 55, "group_name": "home team", "x": 0.0, "y": 0.0, "is_detected": True}
            ],
            "ball": {"x": 0.0, "y": 0.0, "is_detected": True},
        }
    ]
    struct_file.write_text(json.dumps(frames), encoding="utf-8")

    adapter = SkillCornerTrackingAdapter()
    dataset = adapter.load(struct_file)

    assert dataset.match_metadata.pitch_dimension_source == PitchDimensionSource.ASSUMED
    assert dataset.match_metadata.pitch_length_m == 105.0
    assert dataset.match_metadata.pitch_width_m == 68.0
    assert "999" in dataset.match_metadata.match_id
    assert dataset.ingestion_report.total_frames == 1


def test_tracking_adapters_missing_files():
    """Verify that all adapters raise FileNotFoundError on missing files."""
    with pytest.raises(FileNotFoundError):
        MetricaTrackingAdapter().load("non_existent_dir")

    with pytest.raises(FileNotFoundError):
        SkillCornerTrackingAdapter().load("non_existent_structured.json")

    with pytest.raises(FileNotFoundError):
        IDSSETrackingAdapter().load("non_existent_idsse.json")


def test_idsse_unsupported_format(temp_dir: Path):
    """Verify that IDSSE rejects unsupported file extensions."""
    txt_file = temp_dir / "data.txt"
    txt_file.write_text("invalid")
    with pytest.raises(ValueError, match="Unsupported file format"):
        IDSSETrackingAdapter().load(txt_file)


def test_ingestion_report_provenance_and_counts(skillcorner_sample_json: Tuple[Path, Path]):
    """Verify that TrackingIngestionReport contains accurate provenance and counts."""
    struct_file, match_file = skillcorner_sample_json
    adapter = SkillCornerTrackingAdapter()
    dataset = adapter.load(struct_file, match_data_path=match_file)
    rep = dataset.ingestion_report

    assert rep.provider_name == "skillcorner"
    assert rep.source_rights_record_id == "dataset-skillcorner-opendata"
    assert rep.native_sampling_rate_hz == 10.0
    assert rep.resampling_applied is False
    assert rep.observed_records_count == 5  # (2 players + 1 ball in f0) + (1 player + 1 ball in f1)
    assert rep.extrapolated_records_count == 1  # 1 extrapolated player in f0
    assert rep.missing_records_count == 0
    assert rep.coordinate_bounds_normalized["min_x"] <= -35.0
    assert rep.coordinate_bounds_normalized["max_x"] >= 20.0
    assert rep.periods_detected == (1,)

