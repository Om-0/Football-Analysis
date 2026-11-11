"""Metrica Sports tracking data ingestion adapter.

Parses Metrica Sports CSV tracking files (Home and Away), normalizes coordinates
from the Metrica [0, 1] x [0, 1] unit square to PITCHMIND metric pitch standard
(center origin (0, 0), longitudinal X in [-pitch_length/2, +pitch_length/2],
lateral Y in [-pitch_width/2, +pitch_width/2]), enforces observation semantics
for missing/NaN coordinates, and produces canonical Pydantic models.

Preserves native 25 Hz sampling rate without downsampling.
"""

from pathlib import Path
import csv
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


class MetricaTrackingAdapter(BaseTrackingAdapter):
    """Adapter for Metrica Sports CSV tracking datasets."""

    def __init__(
        self,
        rights_record_id: str = "dataset-metrica-sample",
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
        pitch_dimension_source: PitchDimensionSource = PitchDimensionSource.ASSUMED,
    ) -> None:
        super().__init__(
            provider_name="metrica",
            rights_record_id=rights_record_id,
            default_pitch_length_m=pitch_length_m,
            default_pitch_width_m=pitch_width_m,
        )
        self.pitch_dimension_source = pitch_dimension_source

    def _normalize_x(self, x_raw: float, pitch_length_m: float) -> float:
        """Convert Metrica x [0, 1] to metric pitch X [-pitch_length/2, +pitch_length/2]."""
        return (x_raw - 0.5) * pitch_length_m

    def _normalize_y(self, y_raw: float, pitch_width_m: float) -> float:
        """Convert Metrica y [0, 1] to metric pitch Y [-pitch_width/2, +pitch_width/2].
        
        In Metrica, y=0 is the top touchline (far touchline in broadcast view, +Y in PITCHMIND),
        and y=1 is the bottom touchline (near touchline, -Y in PITCHMIND).
        """
        return (0.5 - y_raw) * pitch_width_m

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

    def _parse_metrica_csv(
        self,
        csv_path: Path,
        team_id: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[str, str]], Optional[Tuple[str, str]]]:
        """Parse Metrica tracking CSV into rows, player column mappings, and ball columns.
        
        Handles both 3-row Metrica header structure and flat 1-row header structure.
        """
        rows: List[Dict[str, Any]] = []
        player_columns: Dict[str, Tuple[str, str]] = {}  # player_id -> (col_x, col_y)
        ball_columns: Optional[Tuple[str, str]] = None

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            first_rows = [next(reader) for _ in range(3)]

            # Check if Row 2 contains 'Period' (standard Metrica 3-row header)
            if len(first_rows) >= 3 and any("period" in cell.lower() for cell in first_rows[2]):
                row0 = first_rows[0]
                row1 = first_rows[1]
                header = first_rows[2]

                # Identify players from row1 and header
                for i in range(3, len(header)):
                    col_name = header[i].strip()
                    player_name = row1[i].strip() if i < len(row1) else ""
                    if not player_name:
                        player_name = col_name.rsplit("_", 1)[0]

                    if "ball" in col_name.lower() or "ball" in player_name.lower():
                        if col_name.endswith("_x") or col_name.endswith("x"):
                            b_x = col_name
                            b_y = header[i + 1].strip() if i + 1 < len(header) else ""
                            ball_columns = (b_x, b_y)
                    elif col_name.endswith("_x") or col_name.endswith("x"):
                        p_id = player_name.strip() or col_name.rsplit("_", 1)[0]
                        p_x = col_name
                        p_y = header[i + 1].strip() if i + 1 < len(header) else ""
                        player_columns[p_id] = (p_x, p_y)

                dict_reader = csv.DictReader(f, fieldnames=header)
                for r in dict_reader:
                    rows.append(r)

            else:
                # Flat single-row header format
                header = first_rows[0]
                # Reset reader from beginning
                f.seek(0)
                dict_reader = csv.DictReader(f)
                header = dict_reader.fieldnames or []

                for i, col in enumerate(header):
                    c_lower = col.lower()
                    if "ball" in c_lower and (c_lower.endswith("_x") or c_lower.endswith("x")):
                        b_x = col
                        b_y = header[i + 1] if i + 1 < len(header) else ""
                        ball_columns = (b_x, b_y)
                    elif (c_lower.endswith("_x") or c_lower.endswith("x")) and not any(
                        k in c_lower for k in ("period", "frame", "time")
                    ):
                        p_id = col.rsplit("_", 1)[0]
                        p_x = col
                        p_y = header[i + 1] if i + 1 < len(header) else ""
                        player_columns[p_id] = (p_x, p_y)

                for r in dict_reader:
                    rows.append(r)

        return rows, player_columns, ball_columns

    def load(
        self,
        source_path: str | Path,
        away_csv_path: Optional[str | Path] = None,
        match_id: str = "metrica-game",
        home_team: str = "Home Team",
        away_team: str = "Away Team",
        pitch_length_m: Optional[float] = None,
        pitch_width_m: Optional[float] = None,
        period_attacking_directions: Optional[Dict[int, Dict[str, int]]] = None,
        **kwargs: Any,
    ) -> TrackingDataset:
        """Load Metrica tracking dataset from CSV file(s) or directory.
        
        Args:
            source_path: Path to Home CSV, or directory containing Metrica CSV files.
            away_csv_path: Optional path to Away CSV if source_path is Home CSV.
            match_id: Unique match identifier.
            home_team: Home team name.
            away_team: Away team name.
            pitch_length_m: Optional override for pitch length in metres.
            pitch_width_m: Optional override for pitch width in metres.
            period_attacking_directions: Optional mapping period -> {team_id: direction (+1/-1)}.
        """
        p_len = pitch_length_m or self.default_pitch_length_m
        p_wid = pitch_width_m or self.default_pitch_width_m

        path = Path(source_path).resolve()
        home_file: Optional[Path] = None
        away_file: Optional[Path] = None

        if path.is_dir():
            # Find Home and Away files in directory
            for f in path.glob("*.csv"):
                if "home" in f.name.lower():
                    home_file = f
                elif "away" in f.name.lower():
                    away_file = f
            if not home_file:
                raise FileNotFoundError(f"Could not find Home tracking CSV in directory: {path}")
        elif path.is_file():
            home_file = path
            if away_csv_path:
                away_file = Path(away_csv_path).resolve()
            else:
                # Infer away file by replacing 'Home' with 'Away'
                inferred_away = path.parent / path.name.replace("Home", "Away").replace("home", "away")
                if inferred_away.is_file() and inferred_away != path:
                    away_file = inferred_away
        else:
            raise FileNotFoundError(f"Source path does not exist: {path}")

        # Parse Home CSV
        home_rows, home_players, home_ball = self._parse_metrica_csv(home_file, team_id="0")

        # Parse Away CSV if available
        away_rows: List[Dict[str, Any]] = []
        away_players: Dict[str, Tuple[str, str]] = {}
        away_ball: Optional[Tuple[str, str]] = None
        if away_file and away_file.is_file():
            away_rows, away_players, away_ball = self._parse_metrica_csv(away_file, team_id="1")

        # Coordinate bounds and counters
        raw_x_vals: List[float] = []
        raw_y_vals: List[float] = []
        norm_x_vals: List[float] = []
        norm_y_vals: List[float] = []

        observed_count = 0
        missing_count = 0
        extrapolated_count = 0

        player_frames: List[PlayerFrame] = []
        ball_frames: List[BallFrame] = []
        periods_seen: Set[int] = set()

        # In Metrica, standard sampling rate is 25 Hz
        native_fps = 25.0

        # Default period attacking directions if not supplied:
        # Standard: Period 1: Home attacks +X, Away attacks -X; Period 2: Home attacks -X, Away attacks +X
        if period_attacking_directions is None:
            period_attacking_directions = {
                1: {"0": 1, "1": -1},
                2: {"0": -1, "1": 1},
            }

        num_frames = max(len(home_rows), len(away_rows))

        for idx in range(num_frames):
            h_row = home_rows[idx] if idx < len(home_rows) else None
            a_row = away_rows[idx] if idx < len(away_rows) else None
            ref_row = h_row if h_row is not None else a_row
            if ref_row is None:
                continue

            period_raw = ref_row.get("Period") or ref_row.get("period", "1")
            period = int(float(period_raw))
            periods_seen.add(period)

            frame_raw = ref_row.get("Frame") or ref_row.get("frame", str(idx + 1))
            frame_idx = int(float(frame_raw))

            time_raw = ref_row.get("Time [s]") or ref_row.get("time [s]") or ref_row.get("Time", "")
            if time_raw and not self._is_nan_or_empty(time_raw):
                timestamp_sec = float(time_raw)
            else:
                timestamp_sec = frame_idx / native_fps

            # Process Home Players
            if h_row:
                team_dir = (
                    period_attacking_directions.get(period, {}).get("0", 1)
                    if period_attacking_directions
                    else 1
                )
                for p_id, (col_x, col_y) in home_players.items():
                    val_x = h_row.get(col_x)
                    val_y = h_row.get(col_y)

                    if self._is_nan_or_empty(val_x) or self._is_nan_or_empty(val_y):
                        missing_count += 1
                        pf = PlayerFrame(
                            match_id=match_id,
                            period=period,
                            timestamp_seconds=timestamp_sec,
                            frame_index=frame_idx,
                            player_id=f"H_{p_id}",
                            team_id="0",
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
                        rx = float(val_x)
                        ry = float(val_y)
                        raw_x_vals.append(rx)
                        raw_y_vals.append(ry)

                        px = self._normalize_x(rx, p_len)
                        py = self._normalize_y(ry, p_wid)
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
                            player_id=f"H_{p_id}",
                            team_id="0",
                            x=px,
                            y=py,
                            team_attacking_direction=team_dir,
                            x_attack=x_att,
                            y_attack=y_att,
                            observed_flag=True,
                            visibility_status=VisibilityStatus.VISIBLE,
                            tracking_status=TrackingStatus.DETECTED,
                        )
                    player_frames.append(pf)

            # Process Away Players
            if a_row:
                team_dir = (
                    period_attacking_directions.get(period, {}).get("1", -1)
                    if period_attacking_directions
                    else -1
                )
                for p_id, (col_x, col_y) in away_players.items():
                    val_x = a_row.get(col_x)
                    val_y = a_row.get(col_y)

                    if self._is_nan_or_empty(val_x) or self._is_nan_or_empty(val_y):
                        missing_count += 1
                        pf = PlayerFrame(
                            match_id=match_id,
                            period=period,
                            timestamp_seconds=timestamp_sec,
                            frame_index=frame_idx,
                            player_id=f"A_{p_id}",
                            team_id="1",
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
                        rx = float(val_x)
                        ry = float(val_y)
                        raw_x_vals.append(rx)
                        raw_y_vals.append(ry)

                        px = self._normalize_x(rx, p_len)
                        py = self._normalize_y(ry, p_wid)
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
                            player_id=f"A_{p_id}",
                            team_id="1",
                            x=px,
                            y=py,
                            team_attacking_direction=team_dir,
                            x_attack=x_att,
                            y_attack=y_att,
                            observed_flag=True,
                            visibility_status=VisibilityStatus.VISIBLE,
                            tracking_status=TrackingStatus.DETECTED,
                        )
                    player_frames.append(pf)

            # Process Ball (Home row prioritised, then away row)
            ball_col_pair = home_ball or away_ball
            ball_row = h_row if (home_ball and h_row) else a_row

            if ball_col_pair and ball_row:
                bx_col, by_col = ball_col_pair
                b_val_x = ball_row.get(bx_col)
                b_val_y = ball_row.get(by_col)

                if self._is_nan_or_empty(b_val_x) or self._is_nan_or_empty(b_val_y):
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
                    brx = float(b_val_x)
                    bry = float(b_val_y)
                    raw_x_vals.append(brx)
                    raw_y_vals.append(bry)

                    bpx = self._normalize_x(brx, p_len)
                    bpy = self._normalize_y(bry, p_wid)
                    norm_x_vals.append(bpx)
                    norm_y_vals.append(bpy)
                    observed_count += 1

                    bf = BallFrame(
                        match_id=match_id,
                        period=period,
                        timestamp_seconds=timestamp_sec,
                        frame_index=frame_idx,
                        x=bpx,
                        y=bpy,
                        z=None,  # Metrica 2D does not supply airborne z elevation
                        ball_state=BallState.UNKNOWN,  # Unknown without verified z elevation
                        observed_flag=True,
                        visibility_status=VisibilityStatus.VISIBLE,
                        tracking_status=TrackingStatus.DETECTED,
                    )
                ball_frames.append(bf)

        # Coordinate bounds dictionary
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
            home_team=home_team,
            away_team=away_team,
            pitch_length_m=p_len,
            pitch_width_m=p_wid,
            pitch_dimension_source=self.pitch_dimension_source,
            period_attacking_directions=period_attacking_directions,
            source_provider="metrica",
            source_rights_record_id=self.rights_record_id,
        )

        # Build Ingestion Audit Report
        ingestion_report = TrackingIngestionReport(
            provider_name="metrica",
            match_id=match_id,
            source_path=str(path),
            source_rights_record_id=self.rights_record_id,
            total_frames=num_frames,
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
            raw_metadata={
                "home_file": str(home_file),
                "away_file": str(away_file) if away_file else None,
                "home_players_detected": list(home_players.keys()),
                "away_players_detected": list(away_players.keys()),
                "total_csv_rows": num_frames,
            },
        )
