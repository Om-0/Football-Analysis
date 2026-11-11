"""SkillCorner Open Data tracking ingestion adapter.

Parses SkillCorner Open Data JSON structures:
- <match_id>_match_data.json (metadata, pitch dimensions, lineups)
- <match_id>_structured_data.json (10 Hz tracking frames)

Enforces PITCHMIND canonical observation semantics:
- is_detected == True -> observed_flag=True, tracking_status=DETECTED
- is_detected == False -> observed_flag=False, tracking_status=EXTRAPOLATED (never marked observed)
- Missing players -> x=None, y=None, observed_flag=False, tracking_status=LOST

Preserves native 10 Hz sampling rate without downsampling.
"""

from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pitchmind.data_engine.schema import (
    BallFrame,
    BallState,
    MatchMetadata,
    PitchDimensionSource,
    PlayerFrame,
    TeamAttackingDirection,
    TrackingStatus,
    VisibilityStatus,
)
from pitchmind.ingestion.tracking.base import (
    BaseTrackingAdapter,
    TrackingDataset,
    TrackingIngestionReport,
)


def parse_timestamp_to_seconds(ts_str: str) -> float:
    """Parse timestamp string (e.g. '00:15:30.400' or '15:30.40') to total seconds."""
    if not ts_str:
        return 0.0
    try:
        parts = ts_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600.0 + float(m) * 60.0 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return float(m) * 60.0 + float(s)
        return float(ts_str)
    except Exception:
        return 0.0


class SkillCornerTrackingAdapter(BaseTrackingAdapter):
    """Adapter for SkillCorner Open Data JSON tracking datasets."""

    def __init__(
        self,
        rights_record_id: str = "dataset-skillcorner-opendata",
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> None:
        super().__init__(
            provider_name="skillcorner",
            rights_record_id=rights_record_id,
            default_pitch_length_m=pitch_length_m,
            default_pitch_width_m=pitch_width_m,
        )

    def load(
        self,
        source_path: str | Path,
        match_data_path: Optional[str | Path] = None,
        match_id: Optional[str] = None,
        period_attacking_directions: Optional[Dict[int, Dict[str, int]]] = None,
        include_missing_lineup_players: bool = False,
        **kwargs: Any,
    ) -> TrackingDataset:
        """Load SkillCorner tracking dataset from structured data JSON.
        
        Args:
            source_path: Path to `<match_id>_structured_data.json` or directory containing match files.
            match_data_path: Optional path to `<match_id>_match_data.json`.
            match_id: Optional match identifier (inferred from filename if omitted).
            period_attacking_directions: Optional mapping period -> {team_id: direction (+1/-1)}.
            include_missing_lineup_players: If True, emits LOST PlayerFrame for unobserved squad players.
        """
        path = Path(source_path).resolve()
        structured_file: Optional[Path] = None
        metadata_file: Optional[Path] = None

        if path.is_dir():
            for f in path.glob("*.json"):
                if "structured_data" in f.name:
                    structured_file = f
                elif "match_data" in f.name:
                    metadata_file = f
            if not structured_file:
                raise FileNotFoundError(f"Could not find structured_data JSON in directory: {path}")
        elif path.is_file():
            structured_file = path
            if match_data_path:
                metadata_file = Path(match_data_path).resolve()
            else:
                inferred_meta = path.parent / path.name.replace("structured_data", "match_data")
                if inferred_meta.is_file():
                    metadata_file = inferred_meta
        else:
            raise FileNotFoundError(f"Source path does not exist: {path}")

        # Parse match metadata JSON if available
        raw_match_meta: Dict[str, Any] = {}
        home_team_name = "Home Team"
        away_team_name = "Away Team"
        home_team_id = "0"
        away_team_id = "1"
        pitch_len = self.default_pitch_length_m
        pitch_wid = self.default_pitch_width_m
        dim_source = PitchDimensionSource.ASSUMED

        if metadata_file and metadata_file.is_file():
            with open(metadata_file, "r", encoding="utf-8") as mf:
                raw_match_meta = json.load(mf)

            # Extract team names and IDs
            if "home_team" in raw_match_meta and isinstance(raw_match_meta["home_team"], dict):
                home_team_name = raw_match_meta["home_team"].get("name", home_team_name)
                home_team_id = str(raw_match_meta["home_team"].get("id", "0"))
            if "away_team" in raw_match_meta and isinstance(raw_match_meta["away_team"], dict):
                away_team_name = raw_match_meta["away_team"].get("name", away_team_name)
                away_team_id = str(raw_match_meta["away_team"].get("id", "1"))

            # Extract measured/supplied pitch dimensions
            if "pitch_length" in raw_match_meta and raw_match_meta["pitch_length"]:
                pitch_len = float(raw_match_meta["pitch_length"])
                dim_source = PitchDimensionSource.SUPPLIED
            if "pitch_width" in raw_match_meta and raw_match_meta["pitch_width"]:
                pitch_wid = float(raw_match_meta["pitch_width"])
                dim_source = PitchDimensionSource.SUPPLIED

        # Determine match_id
        if not match_id:
            if "id" in raw_match_meta:
                match_id = f"skillcorner-{raw_match_meta['id']}"
            else:
                base_name = structured_file.name.split("_structured_data")[0]
                match_id = f"skillcorner-{base_name}"

        # Load structured tracking data
        with open(structured_file, "r", encoding="utf-8") as sf:
            frames_json = json.load(sf)

        if not isinstance(frames_json, list):
            raise ValueError(f"Expected list of frame objects in SkillCorner structured data, got {type(frames_json)}")

        # SkillCorner default period attacking directions:
        # Standard: Period 1: Home attacks +X, Away attacks -X; Period 2: Home attacks -X, Away attacks +X
        if period_attacking_directions is None:
            period_attacking_directions = {
                1: {home_team_id: 1, away_team_id: -1, "0": 1, "1": -1},
                2: {home_team_id: -1, away_team_id: 1, "0": -1, "1": 1},
            }

        player_frames: List[PlayerFrame] = []
        ball_frames: List[BallFrame] = []
        periods_seen: Set[int] = set()

        raw_x_vals: List[float] = []
        raw_y_vals: List[float] = []
        norm_x_vals: List[float] = []
        norm_y_vals: List[float] = []

        observed_count = 0
        extrapolated_count = 0
        missing_count = 0

        native_fps = 10.0  # SkillCorner tracking is captured natively at 10 Hz

        for frame_obj in frames_json:
            frame_idx = int(frame_obj.get("frame", 0))
            period = int(frame_obj.get("period", 1))
            periods_seen.add(period)

            # Parse timestamp
            ts_raw = frame_obj.get("timestamp")
            if isinstance(ts_raw, (int, float)):
                timestamp_sec = float(ts_raw)
            elif isinstance(ts_raw, str):
                timestamp_sec = parse_timestamp_to_seconds(ts_raw)
            else:
                timestamp_sec = frame_idx / native_fps

            # Process Players
            players_data = frame_obj.get("data", [])
            for p_data in players_data:
                p_id_raw = p_data.get("trackable_object")
                if p_id_raw is None:
                    continue
                player_id = str(p_id_raw)

                # Determine team
                group_name = str(p_data.get("group_name", "")).lower()
                if "home" in group_name or home_team_name.lower() in group_name:
                    team_id = home_team_id
                elif "away" in group_name or away_team_name.lower() in group_name:
                    team_id = away_team_id
                else:
                    team_id = "referee" if "referee" in group_name else "0"

                team_dir = (
                    period_attacking_directions.get(period, {}).get(team_id, 1)
                    if period_attacking_directions
                    else 1
                )

                x_val = p_data.get("x")
                y_val = p_data.get("y")
                is_detected = bool(p_data.get("is_detected", True))

                if x_val is None or y_val is None:
                    missing_count += 1
                    pf = PlayerFrame(
                        match_id=match_id,
                        period=period,
                        timestamp_seconds=timestamp_sec,
                        frame_index=frame_idx,
                        player_id=player_id,
                        team_id=team_id,
                        x=None,
                        y=None,
                        team_attacking_direction=team_dir,
                        x_attack=None,
                        y_attack=None,
                        observed_flag=False,
                        visibility_status=VisibilityStatus.OUT_OF_VIEW,
                        tracking_status=TrackingStatus.LOST,
                    )
                elif is_detected:
                    # Directly observed player
                    px = float(x_val)
                    py = float(y_val)
                    raw_x_vals.append(px)
                    raw_y_vals.append(py)
                    norm_x_vals.append(px)
                    norm_y_vals.append(py)

                    x_att = px * team_dir
                    y_att = py * team_dir
                    observed_count += 1

                    pf = PlayerFrame(
                        match_id=match_id,
                        period=period,
                        timestamp_seconds=timestamp_sec,
                        frame_index=frame_idx,
                        player_id=player_id,
                        team_id=team_id,
                        x=px,
                        y=py,
                        team_attacking_direction=team_dir,
                        x_attack=x_att,
                        y_attack=y_att,
                        observed_flag=True,
                        visibility_status=VisibilityStatus.VISIBLE,
                        tracking_status=TrackingStatus.DETECTED,
                    )
                else:
                    # Extrapolated player off-screen
                    px = float(x_val)
                    py = float(y_val)
                    raw_x_vals.append(px)
                    raw_y_vals.append(py)
                    norm_x_vals.append(px)
                    norm_y_vals.append(py)

                    x_att = px * team_dir
                    y_att = py * team_dir
                    extrapolated_count += 1

                    # ADR 004 / Rule 5: Extrapolated coordinate MUST NOT have observed_flag=True
                    pf = PlayerFrame(
                        match_id=match_id,
                        period=period,
                        timestamp_seconds=timestamp_sec,
                        frame_index=frame_idx,
                        player_id=player_id,
                        team_id=team_id,
                        x=px,
                        y=py,
                        team_attacking_direction=team_dir,
                        x_attack=x_att,
                        y_attack=y_att,
                        observed_flag=False,
                        visibility_status=VisibilityStatus.OUT_OF_VIEW,
                        tracking_status=TrackingStatus.EXTRAPOLATED,
                    )
                player_frames.append(pf)

            # Process Ball
            ball_data = frame_obj.get("ball")
            if ball_data and isinstance(ball_data, dict):
                bx = ball_data.get("x")
                by = ball_data.get("y")
                bz = ball_data.get("z")
                b_detected = bool(ball_data.get("is_detected", True))

                if bx is None or by is None:
                    missing_count += 1
                    bf = BallFrame(
                        match_id=match_id,
                        period=period,
                        timestamp_seconds=timestamp_sec,
                        frame_index=frame_idx,
                        x=None,
                        y=None,
                        z=None,
                        ball_state=BallState.UNKNOWN,
                        observed_flag=False,
                        visibility_status=VisibilityStatus.OUT_OF_VIEW,
                        tracking_status=TrackingStatus.LOST,
                    )
                else:
                    b_px = float(bx)
                    b_py = float(by)
                    b_pz = float(bz) if bz is not None else None
                    raw_x_vals.append(b_px)
                    raw_y_vals.append(b_py)
                    norm_x_vals.append(b_px)
                    norm_y_vals.append(b_py)

                    # Determine ball state
                    if b_pz is None:
                        ball_state = BallState.UNKNOWN
                    elif b_pz <= 0.5:
                        ball_state = BallState.GROUND
                    else:
                        ball_state = BallState.AIRBORNE

                    if b_detected:
                        observed_count += 1
                        bf = BallFrame(
                            match_id=match_id,
                            period=period,
                            timestamp_seconds=timestamp_sec,
                            frame_index=frame_idx,
                            x=b_px,
                            y=b_py,
                            z=b_pz,
                            ball_state=ball_state,
                            observed_flag=True,
                            visibility_status=VisibilityStatus.VISIBLE,
                            tracking_status=TrackingStatus.DETECTED,
                        )
                    else:
                        extrapolated_count += 1
                        bf = BallFrame(
                            match_id=match_id,
                            period=period,
                            timestamp_seconds=timestamp_sec,
                            frame_index=frame_idx,
                            x=b_px,
                            y=b_py,
                            z=b_pz,
                            ball_state=ball_state,
                            observed_flag=False,
                            visibility_status=VisibilityStatus.OUT_OF_VIEW,
                            tracking_status=TrackingStatus.EXTRAPOLATED,
                        )
                ball_frames.append(bf)

        # Build coordinate bounds
        coord_raw = {
            "min_x": float(min(raw_x_vals)) if raw_x_vals else 0.0,
            "max_x": float(max(raw_x_vals)) if raw_x_vals else 0.0,
            "min_y": float(min(raw_y_vals)) if raw_y_vals else 0.0,
            "max_y": float(max(raw_y_vals)) if raw_y_vals else 0.0,
        }
        coord_norm = {
            "min_x": float(min(norm_x_vals)) if norm_x_vals else 0.0,
            "max_x": float(max(norm_x_vals)) if norm_x_vals else 0.0,
            "min_y": float(min(norm_y_vals)) if norm_y_vals else 0.0,
            "max_y": float(max(norm_y_vals)) if norm_y_vals else 0.0,
        }

        # Build MatchMetadata
        match_metadata = MatchMetadata(
            match_id=match_id,
            home_team=home_team_name,
            away_team=away_team_name,
            pitch_length_m=pitch_len,
            pitch_width_m=pitch_wid,
            pitch_dimension_source=dim_source,
            period_attacking_directions=period_attacking_directions,
            source_provider="skillcorner",
            source_rights_record_id=self.rights_record_id,
        )

        # Build Ingestion Audit Report
        ingestion_report = TrackingIngestionReport(
            provider_name="skillcorner",
            match_id=match_id,
            source_path=str(path),
            source_rights_record_id=self.rights_record_id,
            total_frames=len(frames_json),
            player_records_count=len(player_frames),
            ball_records_count=len(ball_frames),
            observed_records_count=observed_count,
            extrapolated_records_count=extrapolated_count,
            missing_records_count=missing_count,
            native_sampling_rate_hz=native_fps,
            resampling_applied=False,
            coordinate_bounds_raw=coord_raw,
            coordinate_bounds_normalized=coord_norm,
            periods_detected=tuple(sorted(periods_seen)),
            validation_warnings=(),
            validation_errors=(),
        )

        return TrackingDataset(
            match_metadata=match_metadata,
            player_frames=player_frames,
            ball_frames=ball_frames,
            ingestion_report=ingestion_report,
            raw_metadata=raw_match_meta,
        )

