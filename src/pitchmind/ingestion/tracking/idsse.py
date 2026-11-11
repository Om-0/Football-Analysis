"""IDSSE (Bundesliga / 2. Bundesliga) tracking data ingestion adapter.

Parses IDSSE tracking data (JSON or CSV formats) at native 25 Hz sampling rate.
Supports multi-camera optical tracking with 3D ball coordinates (x, y, z).
Normalizes pitch origin to PITCHMIND standard (center origin (0, 0),
longitudinal X in [-pitch_length/2, +pitch_length/2], lateral Y in [-pitch_width/2, +pitch_width/2]),
and enforces canonical observation semantics.

Preserves native 25 Hz sampling rate without downsampling.
"""

from pathlib import Path
import csv
import json
import math
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


class IDSSETrackingAdapter(BaseTrackingAdapter):
    """Adapter for IDSSE Bundesliga / 2. Bundesliga optical tracking datasets."""

    def __init__(
        self,
        rights_record_id: str = "dataset-idsse-bundesliga",
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> None:
        super().__init__(
            provider_name="idsse",
            rights_record_id=rights_record_id,
            default_pitch_length_m=pitch_length_m,
            default_pitch_width_m=pitch_width_m,
        )

    def _normalize_coordinates(
        self,
        x_raw: float,
        y_raw: float,
        pitch_length_m: float,
        pitch_width_m: float,
        is_bottom_left_origin: bool,
    ) -> Tuple[float, float]:
        """Convert coordinates to centre-origin metric standard."""
        if is_bottom_left_origin:
            return x_raw - (pitch_length_m / 2.0), y_raw - (pitch_width_m / 2.0)
        return x_raw, y_raw

    def _is_nan_or_empty(self, val: Any) -> bool:
        if val is None:
            return True
        s = str(val).strip().lower()
        if not s or s in ("nan", "null", "none"):
            return True
        try:
            return math.isnan(float(s))
        except ValueError:
            return True

    def load(
        self,
        source_path: str | Path,
        match_id: Optional[str] = None,
        home_team: Optional[str] = None,
        away_team: Optional[str] = None,
        pitch_length_m: Optional[float] = None,
        pitch_width_m: Optional[float] = None,
        period_attacking_directions: Optional[Dict[int, Dict[str, int]]] = None,
        **kwargs: Any,
    ) -> TrackingDataset:
        """Load IDSSE tracking data from JSON or CSV file.
        
        Args:
            source_path: Path to IDSSE JSON or CSV tracking file.
            match_id: Optional match identifier.
            home_team: Optional home team name override.
            away_team: Optional away team name override.
            pitch_length_m: Optional pitch length override.
            pitch_width_m: Optional pitch width override.
            period_attacking_directions: Optional period attacking directions.
        """
        path = Path(source_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"IDSSE tracking file does not exist: {path}")

        p_len = pitch_length_m or self.default_pitch_length_m
        p_wid = pitch_width_m or self.default_pitch_width_m

        if path.suffix.lower() == ".json":
            return self._load_json(
                path,
                match_id=match_id,
                home_team=home_team,
                away_team=away_team,
                pitch_length_m=p_len,
                pitch_width_m=p_wid,
                period_attacking_directions=period_attacking_directions,
            )
        elif path.suffix.lower() == ".csv":
            return self._load_csv(
                path,
                match_id=match_id,
                home_team=home_team,
                away_team=away_team,
                pitch_length_m=p_len,
                pitch_width_m=p_wid,
                period_attacking_directions=period_attacking_directions,
            )
        else:
            raise ValueError(f"Unsupported file format for IDSSE adapter: {path.suffix}. Expected .json or .csv.")

    def _load_json(
        self,
        json_path: Path,
        match_id: Optional[str],
        home_team: Optional[str],
        away_team: Optional[str],
        pitch_length_m: float,
        pitch_width_m: float,
        period_attacking_directions: Optional[Dict[int, Dict[str, int]]],
    ) -> TrackingDataset:
        with open(json_path, "r", encoding="utf-8") as jf:
            root_data = json.load(jf)

        # Handle either dict with metadata or raw list of frames
        raw_meta: Dict[str, Any] = {}
        if isinstance(root_data, dict):
            raw_meta = root_data
            frames_list = root_data.get("frames", [])
            mid = match_id or root_data.get("match_id", f"idsse-{json_path.stem}")
            ht = home_team or root_data.get("home_team", "Home Team")
            at = away_team or root_data.get("away_team", "Away Team")
            p_len = float(root_data.get("pitch_length", pitch_length_m))
            p_wid = float(root_data.get("pitch_width", pitch_width_m))
            native_fps = float(root_data.get("sampling_rate", 25.0))
            dim_source = PitchDimensionSource.SUPPLIED if "pitch_length" in root_data else PitchDimensionSource.ASSUMED
        elif isinstance(root_data, list):
            frames_list = root_data
            mid = match_id or f"idsse-{json_path.stem}"
            ht = home_team or "Home Team"
            at = away_team or "Away Team"
            p_len = pitch_length_m
            p_wid = pitch_width_m
            native_fps = 25.0
            dim_source = PitchDimensionSource.ASSUMED
        else:
            raise ValueError(f"Invalid JSON root structure in {json_path}: expected dict or list.")

        # Check origin standard from sample coordinates
        sample_x: List[float] = []
        for f in frames_list[:50]:
            for p in f.get("players", []):
                if p.get("x") is not None:
                    sample_x.append(float(p["x"]))
            if f.get("ball", {}).get("x") is not None:
                sample_x.append(float(f["ball"]["x"]))

        # If all sample X are positive and max > 60, it's [0, 105] bottom-left origin
        is_bottom_left = bool(sample_x and min(sample_x) >= 0.0 and max(sample_x) > 60.0)

        if period_attacking_directions is None:
            period_attacking_directions = {
                1: {"0": 1, "1": -1, "home": 1, "away": -1},
                2: {"0": -1, "1": 1, "home": -1, "away": 1},
            }

        player_frames: List[PlayerFrame] = []
        ball_frames: List[BallFrame] = []
        periods_seen: Set[int] = set()

        raw_x_vals: List[float] = []
        raw_y_vals: List[float] = []
        norm_x_vals: List[float] = []
        norm_y_vals: List[float] = []

        observed_count = 0
        missing_count = 0
        extrapolated_count = 0

        for frame_obj in frames_list:
            period = int(frame_obj.get("period", 1))
            periods_seen.add(period)
            frame_idx = int(frame_obj.get("frame_id", frame_obj.get("frame", 0)))
            ts = float(frame_obj.get("timestamp", frame_idx / native_fps))

            # Players
            for p_dict in frame_obj.get("players", []):
                p_id = str(p_dict.get("player_id", p_dict.get("id", "")))
                team_raw = str(p_dict.get("team_id", p_dict.get("team", "0")))
                team_id = "0" if "home" in team_raw.lower() or team_raw == "0" else "1"

                team_dir = (
                    period_attacking_directions.get(period, {}).get(team_id, 1)
                    if period_attacking_directions
                    else 1
                )

                rx_val = p_dict.get("x")
                ry_val = p_dict.get("y")

                if self._is_nan_or_empty(rx_val) or self._is_nan_or_empty(ry_val):
                    missing_count += 1
                    pf = PlayerFrame(
                        match_id=mid,
                        period=period,
                        timestamp_seconds=ts,
                        frame_index=frame_idx,
                        player_id=p_id,
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
                else:
                    rx = float(rx_val)
                    ry = float(ry_val)
                    raw_x_vals.append(rx)
                    raw_y_vals.append(ry)

                    nx, ny = self._normalize_coordinates(rx, ry, p_len, p_wid, is_bottom_left)
                    norm_x_vals.append(nx)
                    norm_y_vals.append(ny)

                    x_att = nx * team_dir
                    y_att = ny * team_dir
                    observed_count += 1

                    pf = PlayerFrame(
                        match_id=mid,
                        period=period,
                        timestamp_seconds=ts,
                        frame_index=frame_idx,
                        player_id=p_id,
                        team_id=team_id,
                        x=nx,
                        y=ny,
                        team_attacking_direction=team_dir,
                        x_attack=x_att,
                        y_attack=y_att,
                        observed_flag=True,
                        visibility_status=VisibilityStatus.VISIBLE,
                        tracking_status=TrackingStatus.DETECTED,
                    )
                player_frames.append(pf)

            # Ball
            b_dict = frame_obj.get("ball")
            if b_dict and isinstance(b_dict, dict):
                bx_val = b_dict.get("x")
                by_val = b_dict.get("y")
                bz_val = b_dict.get("z")

                if self._is_nan_or_empty(bx_val) or self._is_nan_or_empty(by_val):
                    missing_count += 1
                    bf = BallFrame(
                        match_id=mid,
                        period=period,
                        timestamp_seconds=ts,
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
                    brx = float(bx_val)
                    bry = float(by_val)
                    brz = float(bz_val) if not self._is_nan_or_empty(bz_val) else None
                    raw_x_vals.append(brx)
                    raw_y_vals.append(bry)

                    nbx, nby = self._normalize_coordinates(brx, bry, p_len, p_wid, is_bottom_left)
                    norm_x_vals.append(nbx)
                    norm_y_vals.append(nby)

                    # Determine ball state from 3D elevation
                    if brz is None:
                        ball_state = BallState.UNKNOWN
                    elif brz <= 0.5:
                        ball_state = BallState.GROUND
                    else:
                        ball_state = BallState.AIRBORNE

                    observed_count += 1
                    bf = BallFrame(
                        match_id=mid,
                        period=period,
                        timestamp_seconds=ts,
                        frame_index=frame_idx,
                        x=nbx,
                        y=nby,
                        z=brz,
                        ball_state=ball_state,
                        observed_flag=True,
                        visibility_status=VisibilityStatus.VISIBLE,
                        tracking_status=TrackingStatus.DETECTED,
                    )
                ball_frames.append(bf)

        return self._build_dataset(
            match_id=mid,
            home_team=ht,
            away_team=at,
            pitch_length_m=p_len,
            pitch_width_m=p_wid,
            dim_source=dim_source,
            period_attacking_directions=period_attacking_directions,
            source_path=json_path,
            total_frames=len(frames_list),
            player_frames=player_frames,
            ball_frames=ball_frames,
            observed_count=observed_count,
            extrapolated_count=extrapolated_count,
            missing_count=missing_count,
            native_fps=native_fps,
            periods_seen=periods_seen,
            raw_x_vals=raw_x_vals,
            raw_y_vals=raw_y_vals,
            norm_x_vals=norm_x_vals,
            norm_y_vals=norm_y_vals,
            raw_metadata=raw_meta,
        )

    def _load_csv(
        self,
        csv_path: Path,
        match_id: Optional[str],
        home_team: Optional[str],
        away_team: Optional[str],
        pitch_length_m: float,
        pitch_width_m: float,
        period_attacking_directions: Optional[Dict[int, Dict[str, int]]],
    ) -> TrackingDataset:
        """Parse IDSSE tabular CSV format."""
        mid = match_id or f"idsse-{csv_path.stem}"
        ht = home_team or "Home Team"
        at = away_team or "Away Team"
        p_len = pitch_length_m
        p_wid = pitch_width_m
        native_fps = 25.0

        if period_attacking_directions is None:
            period_attacking_directions = {
                1: {"0": 1, "1": -1, "home": 1, "away": -1},
                2: {"0": -1, "1": 1, "home": -1, "away": 1},
            }

        rows: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        # Check origin standard
        sample_x: List[float] = []
        for r in rows[:100]:
            x_val = r.get("x")
            if x_val and not self._is_nan_or_empty(x_val):
                sample_x.append(float(x_val))

        is_bottom_left = bool(sample_x and min(sample_x) >= 0.0 and max(sample_x) > 60.0)

        player_frames: List[PlayerFrame] = []
        ball_frames: List[BallFrame] = []
        periods_seen: Set[int] = set()
        frames_seen: Set[int] = set()

        raw_x_vals: List[float] = []
        raw_y_vals: List[float] = []
        norm_x_vals: List[float] = []
        norm_y_vals: List[float] = []

        observed_count = 0
        missing_count = 0
        extrapolated_count = 0

        for r in rows:
            period = int(float(r.get("period", 1)))
            periods_seen.add(period)

            frame_idx = int(float(r.get("frame_id", r.get("frame", 0))))
            frames_seen.add(frame_idx)

            ts_raw = r.get("timestamp", r.get("time"))
            ts = float(ts_raw) if ts_raw and not self._is_nan_or_empty(ts_raw) else frame_idx / native_fps

            entity_id = str(r.get("player_id", r.get("id", "")))
            is_ball = "ball" in entity_id.lower() or "ball" in r.get("entity_type", "").lower()

            rx_val = r.get("x")
            ry_val = r.get("y")
            rz_val = r.get("z")

            if is_ball:
                if self._is_nan_or_empty(rx_val) or self._is_nan_or_empty(ry_val):
                    missing_count += 1
                    bf = BallFrame(
                        match_id=mid,
                        period=period,
                        timestamp_seconds=ts,
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
                    brx = float(rx_val)
                    bry = float(ry_val)
                    brz = float(rz_val) if not self._is_nan_or_empty(rz_val) else None
                    raw_x_vals.append(brx)
                    raw_y_vals.append(bry)

                    nbx, nby = self._normalize_coordinates(brx, bry, p_len, p_wid, is_bottom_left)
                    norm_x_vals.append(nbx)
                    norm_y_vals.append(nby)

                    if brz is None:
                        ball_state = BallState.UNKNOWN
                    elif brz <= 0.5:
                        ball_state = BallState.GROUND
                    else:
                        ball_state = BallState.AIRBORNE

                    observed_count += 1
                    bf = BallFrame(
                        match_id=mid,
                        period=period,
                        timestamp_seconds=ts,
                        frame_index=frame_idx,
                        x=nbx,
                        y=nby,
                        z=brz,
                        ball_state=ball_state,
                        observed_flag=True,
                        visibility_status=VisibilityStatus.VISIBLE,
                        tracking_status=TrackingStatus.DETECTED,
                    )
                ball_frames.append(bf)
            else:
                team_raw = str(r.get("team_id", r.get("team", "0")))
                team_id = "0" if "home" in team_raw.lower() or team_raw == "0" else "1"

                team_dir = (
                    period_attacking_directions.get(period, {}).get(team_id, 1)
                    if period_attacking_directions
                    else 1
                )

                if self._is_nan_or_empty(rx_val) or self._is_nan_or_empty(ry_val):
                    missing_count += 1
                    pf = PlayerFrame(
                        match_id=mid,
                        period=period,
                        timestamp_seconds=ts,
                        frame_index=frame_idx,
                        player_id=entity_id,
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
                else:
                    rx = float(rx_val)
                    ry = float(ry_val)
                    raw_x_vals.append(rx)
                    raw_y_vals.append(ry)

                    nx, ny = self._normalize_coordinates(rx, ry, p_len, p_wid, is_bottom_left)
                    norm_x_vals.append(nx)
                    norm_y_vals.append(ny)

                    x_att = nx * team_dir
                    y_att = ny * team_dir
                    observed_count += 1

                    pf = PlayerFrame(
                        match_id=mid,
                        period=period,
                        timestamp_seconds=ts,
                        frame_index=frame_idx,
                        player_id=entity_id,
                        team_id=team_id,
                        x=nx,
                        y=ny,
                        team_attacking_direction=team_dir,
                        x_attack=x_att,
                        y_attack=y_att,
                        observed_flag=True,
                        visibility_status=VisibilityStatus.VISIBLE,
                        tracking_status=TrackingStatus.DETECTED,
                    )
                player_frames.append(pf)

        return self._build_dataset(
            match_id=mid,
            home_team=ht,
            away_team=at,
            pitch_length_m=p_len,
            pitch_width_m=p_wid,
            dim_source=PitchDimensionSource.ASSUMED,
            period_attacking_directions=period_attacking_directions,
            source_path=csv_path,
            total_frames=len(frames_seen),
            player_frames=player_frames,
            ball_frames=ball_frames,
            observed_count=observed_count,
            extrapolated_count=extrapolated_count,
            missing_count=missing_count,
            native_fps=native_fps,
            periods_seen=periods_seen,
            raw_x_vals=raw_x_vals,
            raw_y_vals=raw_y_vals,
            norm_x_vals=norm_x_vals,
            norm_y_vals=norm_y_vals,
            raw_metadata={"csv_rows_count": len(rows)},
        )

    def _build_dataset(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        pitch_length_m: float,
        pitch_width_m: float,
        dim_source: PitchDimensionSource,
        period_attacking_directions: Optional[Dict[int, Dict[str, int]]],
        source_path: Path,
        total_frames: int,
        player_frames: List[PlayerFrame],
        ball_frames: List[BallFrame],
        observed_count: int,
        extrapolated_count: int,
        missing_count: int,
        native_fps: float,
        periods_seen: Set[int],
        raw_x_vals: List[float],
        raw_y_vals: List[float],
        norm_x_vals: List[float],
        norm_y_vals: List[float],
        raw_metadata: Dict[str, Any],
    ) -> TrackingDataset:
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

        match_metadata = MatchMetadata(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            pitch_dimension_source=dim_source,
            period_attacking_directions=period_attacking_directions,
            source_provider="idsse",
            source_rights_record_id=self.rights_record_id,
        )

        ingestion_report = TrackingIngestionReport(
            provider_name="idsse",
            match_id=match_id,
            source_path=str(source_path),
            source_rights_record_id=self.rights_record_id,
            total_frames=total_frames,
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
            raw_metadata=raw_metadata,
        )

