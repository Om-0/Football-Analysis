"""PITCHMIND Canonical Data Schemas (Pydantic v2 - Hardened Baseline v2.0).

This module defines the canonical tracking data model serving as the hard,
immutable boundary between Computer Vision perception and downstream Tactical Analytics.

Key Architectural Decisions (v2.0):
1. Fixed Metric Pitch Frame:
   - Origin (0.0, 0.0) at pitch centre circle.
   - Longitudinal axis (X): length in metres [-pitch_length_m/2, +pitch_length_m/2].
     +X points towards right-hand goal as viewed from main broadcast camera.
     -X points towards left-hand goal.
   - Lateral axis (Y): width in metres [-pitch_width_m/2, +pitch_width_m/2].
     +Y points towards far touchline (top of broadcast screen).
     -Y points towards near touchline (bottom of broadcast screen).
   - Raw pitch coordinates (x, y) are permanently preserved and immutable.
   - Team-relative attacking coordinates (x_attack, y_attack) are derived features
     pinned to a fixed period-level team_attacking_direction (+1 or -1) and cannot
     silently flip on individual possession changes.
2. Staged Quality Engine:
   - Separates preflight, detection, tracking, calibration, and pose quality
     measurements according to when they become available during processing.
3. Observation Semantics (ADR 004):
   - 'detected' indicates direct measurement by detector (observed_flag=True).
   - 'estimated', 'interpolated', 'extrapolated', 'lost', and 'invalid' represent
     algorithmic generation, reconstruction, or absence (observed_flag=False).
4. Ball Elevation & Physical Modeling:
   - Ball height (z) in metres represents vertical elevation above the pitch plane.
   - Planar homography assumes Z = 0 and produces geometric parallax error for airborne balls;
     planar homography CANNOT reconstruct 3D airborne ball elevation.
   - The 0.5 m ground tolerance is an unvalidated design candidate baseline that requires
     empirical calibration; unknown state is supported whenever height evidence is ambiguous.
5. Deep Immutability:
   - Frozen Pydantic models with recursively immutable mappings and tuples for all nested collections.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION: str = "2.0.0"


def freeze_value(val: Any) -> Any:
    """Recursively convert nested dicts to MappingProxyType and lists/sets to tuples."""
    if isinstance(val, (dict, Mapping)):
        return MappingProxyType({k: freeze_value(v) for k, v in val.items()})
    elif isinstance(val, (list, tuple, set)):
        return tuple(freeze_value(v) for v in val)
    return val


# ============================================================================
# Enumerations
# ============================================================================


class CameraClass(str, Enum):
    """Classification of video camera view angles and shot types."""

    WIDE_TACTICAL = "wide_tactical"
    MAIN_BROADCAST = "main_broadcast"
    CLOSE_UP = "close_up"
    BEHIND_GOAL = "behind_goal"
    REPLAY = "replay"
    UNKNOWN = "unknown"


class VisibilityStatus(str, Enum):
    """Observation visibility status for players and ball."""

    VISIBLE = "visible"
    OCCLUDED = "occluded"
    OUT_OF_VIEW = "out_of_view"
    UNKNOWN = "unknown"


class TrackingStatus(str, Enum):
    """Continuity and generation status of player/ball tracklets."""

    DETECTED = "detected"
    ESTIMATED = "estimated"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"
    LOST = "lost"
    INVALID = "invalid"


class TaskReadiness(str, Enum):
    """Task-specific analytical readiness states."""

    READY = "ready"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class BallState(str, Enum):
    """Physical state of the ball relative to the pitch plane."""

    GROUND = "ground"
    AIRBORNE = "airborne"
    UNKNOWN = "unknown"


class PitchDimensionSource(str, Enum):
    """Provenance and certainty of pitch metric dimensions."""

    MEASURED = "measured"    # Measured on site via laser / survey / high-accuracy calibration
    SUPPLIED = "supplied"    # Provided by competition or club metadata
    ASSUMED = "assumed"      # FIFA standard 105.0 x 68.0 m applied as fallback


class TeamAttackingDirection(int, Enum):
    """Longitudinal pitch attacking direction for a team in a specific match period."""

    TOWARDS_POSITIVE_X = 1   # Attacking towards +X (right goal from broadcast camera)
    TOWARDS_NEGATIVE_X = -1  # Attacking towards -X (left goal from broadcast camera)


class Period(int, Enum):
    """Match playing periods."""

    FIRST_HALF = 1
    SECOND_HALF = 2
    EXTRA_TIME_1 = 3
    EXTRA_TIME_2 = 4
    PENALTIES = 5


# Type alias for immutable 3x3 homography matrix: 3 rows of 3 floats
HomographyMatrix = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]


# ============================================================================
# Staged Quality Measurement Schemas (Refinement 1)
# ============================================================================


class PreflightQualityMetrics(BaseModel):
    """Quality metrics evaluated during video ingestion and preflight decode.
    
    Available immediately after container inspection without requiring CV inference.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    video_integrity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dimension 1: Container integrity, PTS continuity, and decode health [0.0, 1.0].",
    )
    aspect_ratio: float = Field(..., gt=0.0, description="Measured width / height ratio.")
    aspect_ratio_standard: bool = Field(..., description="Whether aspect ratio conforms to 16:9 widescreen (~1.777).")
    mean_brightness: float = Field(..., ge=0.0, le=255.0, description="Mean pixel intensity across luma channel.")
    contrast_score: float = Field(..., ge=0.0, description="Standard deviation of pixel intensity across luma channel.")
    blur_score_laplacian: float = Field(..., ge=0.0, description="Variance of Laplacian sharpness metric.")
    image_quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dimension 8: Normalized visual clarity composite score [0.0, 1.0].",
    )
    preliminary_pitch_green_fraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Unvalidated candidate estimate of green grass chromaticity in sample frames [0.0, 1.0].",
    )


class DetectionQualityMetrics(BaseModel):
    """Quality metrics evaluated during the object detection stage."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    pitch_coverage_fraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dimension 2: Fraction of standard 2D pitch area visible in camera frustum [0.0, 1.0].",
    )
    player_visibility_fraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dimension 3: Ratio of observed outfield players to expected 22 [0.0, 1.0].",
    )
    detection_reliability_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dimension 4: Mean detector confidence score across detected entities [0.0, 1.0].",
    )


class TrackingQualityMetrics(BaseModel):
    """Quality metrics evaluated during the multi-object tracking stage."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    identity_continuity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dimension 5: Tracklet continuity penalizing fragmentation and ID switches [0.0, 1.0].",
    )
    ball_observability_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dimension 7: Coverage, continuity, and confidence score for ball tracking [0.0, 1.0].",
    )


class CalibrationQualityMetrics(BaseModel):
    """Quality metrics evaluated during pitch camera calibration."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    calibration_reliability_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dimension 6: Pitch calibration reliability score based on reprojection error [0.0, 1.0].",
    )
    calibration_reprojection_error_m: Optional[float] = Field(
        None,
        ge=0.0,
        description="Reprojection root-mean-square error in pitch metres.",
    )


class PoseQualityMetrics(BaseModel):
    """Quality metrics evaluated during pose estimation (conditional on SCAN2PLAY)."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    pose_quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dimension 9: Keypoint detection confidence and crop resolution [0.0, 1.0].",
    )
    mean_keypoint_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean joint detection confidence.")
    crop_resolution_adequate: bool = Field(..., description="Whether player bounding box resolution exceeds 80x80 px.")


# ============================================================================
# Metadata & Video Ingestion Schemas
# ============================================================================


class MatchMetadata(BaseModel):
    """Match-level metadata, pitch geometry, and source rights record.
    
    Fixed Pitch Coordinate Convention:
    - Origin (0.0, 0.0) at pitch centre circle.
    - Longitudinal axis (X): length in metres [-pitch_length_m/2, +pitch_length_m/2].
      +X points towards right-hand goal, -X towards left-hand goal.
    - Lateral axis (Y): width in metres [-pitch_width_m/2, +pitch_width_m/2].
      +Y points towards far touchline, -Y towards near touchline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    schema_version: str = Field(SCHEMA_VERSION, description="Canonical schema version.")
    match_id: str = Field(..., min_length=1, description="Unique, stable match identifier.")
    home_team: str = Field(..., min_length=1, description="Home team name.")
    away_team: str = Field(..., min_length=1, description="Away team name.")
    home_score: Optional[int] = Field(None, ge=0, description="Final home score if match completed.")
    away_score: Optional[int] = Field(None, ge=0, description="Final away score if match completed.")
    match_date: Optional[date] = Field(None, description="Date of the match.")
    competition: Optional[str] = Field(None, description="Competition or tournament name.")
    pitch_length_m: float = Field(105.0, gt=0.0, description="Pitch length in metres.")
    pitch_width_m: float = Field(68.0, gt=0.0, description="Pitch width in metres.")
    pitch_dimension_source: PitchDimensionSource = Field(
        PitchDimensionSource.ASSUMED,
        description="Provenance of pitch dimensions ('measured', 'supplied', 'assumed').",
    )
    period_attacking_directions: Optional[Mapping[int, Mapping[str, int]]] = Field(
        None,
        description="Optional period-level attacking directions: period -> {team_id: direction (+1/-1)}.",
    )
    source_provider: str = Field(
        ...,
        description="Source tracking provider or 'pitchmind_video_cv' for internal video reconstruction.",
    )
    source_rights_record_id: Optional[str] = Field(
        None,
        description="Reference identifier in datasets/manifests/data_rights_register.json.",
    )
    coordinate_origin: str = Field(
        "center",
        description="Fixed coordinate origin definition (always 'center' at (0.0, 0.0)).",
    )
    notes: Optional[str] = Field(None, description="Additional context or experimental notes.")

    @field_validator("period_attacking_directions", mode="before")
    @classmethod
    def freeze_attacking_directions(cls, v: Any) -> Any:
        return freeze_value(v) if v is not None else None


class VideoMetadata(BaseModel):
    """Technical container and stream properties of uploaded match footage.
    
    Preserves original container timing, time-base, and stream properties for exact synchronization.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    schema_version: str = Field(SCHEMA_VERSION, description="Canonical schema version.")
    video_id: str = Field(..., min_length=1, description="Unique identifier for the video file.")
    file_path: str = Field(..., min_length=1, description="Relative or absolute path to video file.")
    sha256_checksum: str = Field(
        ...,
        min_length=64,
        max_length=64,
        pattern=r"^[a-fA-F0-9]{64}$",
        description="Cryptographic SHA-256 hash of the video container.",
    )
    duration_seconds: float = Field(..., gt=0.0, description="Total duration in seconds.")
    total_frames: int = Field(..., gt=0, description="Total frame count in video stream.")
    fps: float = Field(..., gt=0.0, description="Nominal frame rate in frames per second (FPS).")
    width: int = Field(..., gt=0, description="Frame width in pixels (e.g. 1920).")
    height: int = Field(..., gt=0, description="Frame height in pixels (e.g. 1080).")
    codec: str = Field(..., min_length=1, description="Video codec name (e.g. 'h264', 'hevc').")
    pixel_format: str = Field("yuv420p", description="Pixel chroma format (e.g. 'yuv420p').")
    container: str = Field(..., min_length=1, description="Container format (e.g. 'mp4', 'mov', 'mkv').")
    time_base: Optional[str] = Field(None, description="Original container stream time-base (e.g. '1/90000').")
    start_pts: Optional[int] = Field(None, description="Initial Presentation Timestamp integer from video stream.")
    start_time_seconds: float = Field(0.0, ge=0.0, description="Container initial start time in seconds.")
    audio_streams: int = Field(0, ge=0, description="Number of audio streams detected.")


class PreflightReport(BaseModel):
    """Preflight inspection report generated prior to heavy CV processing."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    schema_version: str = Field(SCHEMA_VERSION, description="Canonical schema version.")
    video_id: str = Field(..., min_length=1, description="Target video identifier.")
    passed: bool = Field(..., description="Whether video passed all preflight integrity checks.")
    checks_run: Tuple[str, ...] = Field(..., min_length=1, description="Immutable tuple of checks executed.")
    decode_success: bool = Field(..., description="Whether sample frames decoded successfully.")
    timestamp_continuity: bool = Field(..., description="Whether container timestamps are continuous.")
    duration_seconds: float = Field(..., gt=0.0, description="Verified duration in seconds.")
    fps_measured: float = Field(..., gt=0.0, description="Measured frame rate from frame timestamps.")
    dropped_frames_detected: int = Field(0, ge=0, description="Count of detected frame drops or PTS gaps.")
    aspect_ratio_standard: bool = Field(True, description="Whether aspect ratio conforms to 16:9 widescreen.")
    quality_metrics: Optional[PreflightQualityMetrics] = Field(
        None,
        description="Staged preflight quality measurements available at ingestion time.",
    )
    failure_reasons: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Immutable tuple of reasons for preflight failure, if any.",
    )
    report_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when preflight was executed.",
    )


class VideoQualityReport(BaseModel):
    """Multidimensional video quality vector and downstream task readiness.
    
    Refinement 1: Separates measurements into staged components:
    - preflight_quality: Available at ingestion/preflight
    - detection_quality: Available after detector pass
    - tracking_quality: Available after multi-object tracker
    - calibration_quality: Available after pitch calibration
    - pose_quality: Available after pose estimation
    
    Architectural Status:
    The nine quality dimensions and readiness thresholds are proposed design candidates
    requiring empirical calibration and validation against ground truth.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    schema_version: str = Field(SCHEMA_VERSION, description="Canonical schema version.")
    report_id: str = Field(..., min_length=1, description="Unique identifier for quality report.")
    video_id: str = Field(..., min_length=1, description="Associated video identifier.")
    segment_id: Optional[str] = Field(None, description="Associated camera segment ID if segment-scoped.")
    camera_class: CameraClass = Field(..., description="Camera view classification.")

    # Staged Quality Sub-Components
    preflight_quality: PreflightQualityMetrics = Field(
        ...,
        description="Staged preflight quality metrics (Dimension 1: integrity, Dimension 8: image quality).",
    )
    detection_quality: Optional[DetectionQualityMetrics] = Field(
        None,
        description="Staged detection metrics (Dimension 2: pitch coverage, 3: player visibility, 4: reliability).",
    )
    tracking_quality: Optional[TrackingQualityMetrics] = Field(
        None,
        description="Staged tracking metrics (Dimension 5: identity continuity, 7: ball observability).",
    )
    calibration_quality: Optional[CalibrationQualityMetrics] = Field(
        None,
        description="Staged calibration metrics (Dimension 6: calibration reliability).",
    )
    pose_quality: Optional[PoseQualityMetrics] = Field(
        None,
        description="Staged pose metrics (Dimension 9: pose quality, conditional on SCAN2PLAY).",
    )

    # Task-Specific Readiness States
    pressuregraph_readiness: TaskReadiness = Field(
        ...,
        description="Readiness state for PRESSUREGRAPH pressing analysis.",
    )
    tactical_shift_readiness: TaskReadiness = Field(
        ...,
        description="Readiness state for TACTICAL SHIFT shape change-point detection.",
    )
    roleshift_readiness: TaskReadiness = Field(
        ...,
        description="Readiness state for ROLESHIFT archetype clustering.",
    )
    scan2play_readiness: TaskReadiness = Field(
        ...,
        description="Readiness state for SCAN2PLAY pre-reception orientation proxy.",
    )
    reasons_suppressed: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Immutable tuple documenting rationale for downgraded or suppressed modules.",
    )

    # Convenience Properties to access the Nine Quality Dimensions directly
    @property
    def video_integrity_score(self) -> float:
        return self.preflight_quality.video_integrity_score

    @property
    def pitch_coverage_fraction(self) -> Optional[float]:
        return self.detection_quality.pitch_coverage_fraction if self.detection_quality else None

    @property
    def player_visibility_fraction(self) -> Optional[float]:
        return self.detection_quality.player_visibility_fraction if self.detection_quality else None

    @property
    def detection_reliability_score(self) -> Optional[float]:
        return self.detection_quality.detection_reliability_score if self.detection_quality else None

    @property
    def identity_continuity_score(self) -> Optional[float]:
        return self.tracking_quality.identity_continuity_score if self.tracking_quality else None

    @property
    def calibration_reliability_score(self) -> Optional[float]:
        return self.calibration_quality.calibration_reliability_score if self.calibration_quality else None

    @property
    def ball_observability_score(self) -> Optional[float]:
        return self.tracking_quality.ball_observability_score if self.tracking_quality else None

    @property
    def image_quality_score(self) -> float:
        return self.preflight_quality.image_quality_score

    @property
    def pose_quality_score(self) -> Optional[float]:
        return self.pose_quality.pose_quality_score if self.pose_quality else None


class VideoSegment(BaseModel):
    """Temporal camera shot segment with view classification and calibration.
    
    Homography Documentation:
    - homography_matrix H is a 3x3 transform mapping image pixel coordinates [u, v, 1]^T
      to metric ground-plane pitch coordinates [x, y, 1]^T (image_to_pitch).
    - Planar homography assumes points lie strictly on the flat pitch plane (Z = 0).
      It CANNOT estimate elevation or reconstruct 3D airborne ball trajectories.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    schema_version: str = Field(SCHEMA_VERSION, description="Canonical schema version.")
    segment_id: str = Field(..., min_length=1, description="Unique segment identifier.")
    match_id: str = Field(..., min_length=1, description="Associated match identifier.")
    video_id: str = Field(..., min_length=1, description="Associated video identifier.")
    start_time_seconds: float = Field(..., ge=0.0, description="Start timestamp in video (seconds).")
    end_time_seconds: float = Field(..., gt=0.0, description="End timestamp in video (seconds).")
    start_frame: int = Field(..., ge=0, description="Starting frame index (0-based).")
    end_frame: int = Field(..., ge=0, description="Ending frame index (0-based).")
    camera_class: CameraClass = Field(..., description="Classified camera view type.")
    camera_confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence [0.0, 1.0].")
    is_replay: bool = Field(False, description="True if segment is a slow-motion or broadcast replay.")
    homography_matrix: Optional[HomographyMatrix] = Field(
        None,
        description="Immutable 3x3 planar homography matrix (image_to_pitch metres).",
    )
    calibration_reprojection_error_m: Optional[float] = Field(
        None,
        ge=0.0,
        description="Calibration reprojection error in metres.",
    )
    quality_report: Optional[VideoQualityReport] = Field(
        None,
        description="Associated multidimensional quality assessment report.",
    )

    @model_validator(mode="after")
    def validate_temporal_ordering(self) -> VideoSegment:
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError(
                f"end_time_seconds ({self.end_time_seconds}) must be strictly greater than "
                f"start_time_seconds ({self.start_time_seconds})"
            )
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"end_frame ({self.end_frame}) must be >= start_frame ({self.start_frame})"
            )
        return self


# ============================================================================
# Canonical Tracking Schemas (Core CV / Analytics Boundary)
# ============================================================================


class PlayerFrame(BaseModel):
    """Canonical frame-level observation for a single player.
    
    Fixed Pitch Coordinate Convention:
    - Pitch centre is (0.0, 0.0) in metres.
    - Raw pitch coordinates (x, y) are immutable and never flip with possession changes.
    - x in [-pitch_length_m/2, +pitch_length_m/2], +X towards right-hand goal.
    - y in [-pitch_width_m/2, +pitch_width_m/2], +Y towards far touchline.
    - Team-relative attacking coordinates (x_attack, y_attack) are derived features
      pinned to team_attacking_direction (+1 or -1 for the playing period).
    
    Observation Semantics (ADR 004):
    - observed_flag=True ONLY when tracking_status is 'detected'.
    - 'estimated', 'interpolated', 'extrapolated', 'lost', and 'invalid' MUST have observed_flag=False.
    - Missing coordinates are represented as None, NEVER 0.0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    schema_version: str = Field(SCHEMA_VERSION, description="Canonical schema version.")
    match_id: str = Field(..., min_length=1, description="Match identifier.")
    period: int = Field(..., ge=1, le=5, description="Playing period (1: 1H, 2: 2H, etc.).")
    timestamp_seconds: float = Field(..., ge=0.0, description="Match clock timestamp in seconds from period start.")
    frame_index: Optional[int] = Field(None, ge=0, description="Source video frame index (0-based).")
    source_pts: Optional[int] = Field(None, description="Original presentation timestamp integer from container.")
    player_id: str = Field(..., min_length=1, description="Internal track identity or verified player ID.")
    team_id: str = Field(..., min_length=1, description="Team identifier ('0', '1', or 'referee').")

    # Raw Metric Pitch Coordinates (Fixed Centre Origin (0,0), Metres)
    x: Optional[float] = Field(None, description="Longitudinal pitch coordinate in metres (center = 0.0).")
    y: Optional[float] = Field(None, description="Lateral pitch coordinate in metres (center = 0.0).")

    # Team-Relative Attacking Coordinates (Derived feature, pinned per team per period)
    team_attacking_direction: Optional[int] = Field(
        None,
        description="Fixed period attacking direction for team_id: +1 (towards +X) or -1 (towards -X).",
    )
    x_attack: Optional[float] = Field(
        None,
        description="Team-relative longitudinal coordinate (attacks +X) in metres.",
    )
    y_attack: Optional[float] = Field(
        None,
        description="Team-relative lateral coordinate in metres.",
    )

    # 2D Image Space Bounding Box in Camera Pixels: (xmin, ymin, xmax, ymax)
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = Field(
        None,
        description="Bounding box in image pixels (xmin, ymin, xmax, ymax) when visible.",
    )

    # Derived Kinematics (Metres/sec, Metres/sec^2)
    vx: Optional[float] = Field(None, description="Longitudinal velocity in m/s.")
    vy: Optional[float] = Field(None, description="Lateral velocity in m/s.")
    speed: Optional[float] = Field(None, ge=0.0, description="Instantaneous scalar speed in m/s (>= 0.0).")
    ax: Optional[float] = Field(None, description="Longitudinal acceleration in m/s^2.")
    ay: Optional[float] = Field(None, description="Lateral acceleration in m/s^2.")
    orientation_radians: Optional[float] = Field(
        None,
        ge=-3.141592653589793,
        le=3.141592653589793,
        description="Estimated body orientation angle in radians [-pi, pi].",
    )

    # Observation Status & Provenance Flags
    observed_flag: bool = Field(
        ...,
        description="True ONLY if position was directly detected; False if estimated/interpolated/extrapolated.",
    )
    visibility_status: VisibilityStatus = Field(
        ...,
        description="Visibility status: visible, occluded, out_of_view, or unknown.",
    )
    tracking_status: TrackingStatus = Field(
        ...,
        description="Tracking status: detected, estimated, interpolated, extrapolated, lost, invalid.",
    )
    detection_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Computer-vision detector confidence score [0.0, 1.0].",
    )
    tracking_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Multi-object tracking association confidence score [0.0, 1.0].",
    )
    calibration_uncertainty_m: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated spatial uncertainty in metres resulting from homography error.",
    )
    camera_segment_id: Optional[str] = Field(
        None,
        description="FK to video_segments if observation originated from video CV.",
    )

    @model_validator(mode="after")
    def validate_data_integrity(self) -> PlayerFrame:
        # Rule 1: Directly observed requires tracking_status == DETECTED
        if self.tracking_status == TrackingStatus.DETECTED:
            if not self.observed_flag:
                raise ValueError(
                    "Observation Semantics Violation: tracking_status='detected' must have observed_flag=True."
                )
        else:
            # Rule 2: All other tracking states (estimated, interpolated, extrapolated, lost, invalid)
            # represent algorithmic generation or absence and MUST have observed_flag=False.
            if self.observed_flag:
                raise ValueError(
                    f"Observation Semantics Violation: tracking_status='{self.tracking_status.value}' "
                    f"cannot have observed_flag=True. Only directly detected measurements may be marked observed."
                )

        # Rule 3: Lost or invalid tracking cannot have pitch coordinates
        if self.tracking_status in (TrackingStatus.LOST, TrackingStatus.INVALID):
            if self.x is not None or self.y is not None:
                raise ValueError(
                    f"Data Integrity Violation: tracking_status='{self.tracking_status.value}' "
                    f"cannot have valid pitch coordinates (x={self.x}, y={self.y}). Coordinates must be None."
                )

        # Rule 4: Out of view players cannot have an image bounding box
        if self.visibility_status == VisibilityStatus.OUT_OF_VIEW and self.bbox_xyxy is not None:
            raise ValueError(
                "Data Integrity Violation: visibility_status='out_of_view' cannot have a camera bbox_xyxy."
            )

        # Rule 5: If raw coordinates are provided, both x and y must be provided
        if (self.x is None and self.y is not None) or (self.x is not None and self.y is None):
            raise ValueError(
                f"Incomplete coordinate pair: both x and y must be provided or both must be None (x={self.x}, y={self.y})."
            )

        # Rule 6: If team-relative attacking coordinates are provided, both must be present
        if (self.x_attack is None and self.y_attack is not None) or (
            self.x_attack is not None and self.y_attack is None
        ):
            raise ValueError(
                f"Incomplete attacking coordinate pair: both x_attack and y_attack must be provided or both None."
            )

        # Rule 7: Attacking direction must be +1 or -1 if provided
        if self.team_attacking_direction is not None and self.team_attacking_direction not in (-1, 1):
            raise ValueError(
                f"Invalid team_attacking_direction ({self.team_attacking_direction}): must be +1 or -1."
            )

        # Rule 8: Validate bounding box geometry (xmin < xmax, ymin < ymax)
        if self.bbox_xyxy is not None:
            xmin, ymin, xmax, ymax = self.bbox_xyxy
            if xmax <= xmin or ymax <= ymin:
                raise ValueError(
                    f"Invalid bbox_xyxy dimensions: ({xmin}, {ymin}, {xmax}, {ymax}). "
                    f"Require xmin < xmax and ymin < ymax."
                )

        return self


class BallFrame(BaseModel):
    """Canonical frame-level observation for the match ball.
    
    Data Integrity & Physical Modeling (ADR 004):
    - Ball height (z) in metres represents vertical elevation above the pitch plane.
    - Planar homography assumes Z = 0; it produces severe parallax error for airborne balls.
      Therefore, planar homography alone CANNOT measure elevation (z remains None unless
      multi-view or 3D ball trajectory modeling is verified).
    - Configurable ground-height tolerance (default: 0.5 m) is an unvalidated design candidate baseline.
    - When evidence is insufficient or ambiguous, ball_state must be 'unknown'.
    - Missing coordinates are represented as None, NEVER 0.0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    schema_version: str = Field(SCHEMA_VERSION, description="Canonical schema version.")
    match_id: str = Field(..., min_length=1, description="Match identifier.")
    period: int = Field(..., ge=1, le=5, description="Playing period (1: 1H, 2: 2H, etc.).")
    timestamp_seconds: float = Field(..., ge=0.0, description="Match clock timestamp in seconds.")
    frame_index: Optional[int] = Field(None, ge=0, description="Source video frame index.")
    source_pts: Optional[int] = Field(None, description="Original presentation timestamp integer from container.")

    # Pitch Ground-Plane Coordinates (Metres, fixed centre origin (0.0, 0.0))
    x: Optional[float] = Field(None, description="Longitudinal pitch coordinate in metres.")
    y: Optional[float] = Field(None, description="Lateral pitch coordinate in metres.")
    z: Optional[float] = Field(
        None,
        ge=0.0,
        description="Physical elevation above pitch plane in metres (None if uncalibrated/planar).",
    )
    ground_height_tolerance_m: float = Field(
        0.5,
        ge=0.0,
        description=(
            "Configurable engineering heuristic threshold in metres below which ball may be classified as "
            "'ground'. Note: 0.5 m is an unvalidated design candidate baseline and must be calibrated "
            "empirically against 3D ground truth tracking in Phase 6."
        ),
    )

    # Team-Relative Attacking Coordinates (Derived feature)
    team_attacking_direction: Optional[int] = Field(
        None,
        description="Fixed period attacking direction for team in possession (+1 or -1).",
    )
    x_attack: Optional[float] = Field(None, description="Team-relative longitudinal x in metres.")
    y_attack: Optional[float] = Field(None, description="Team-relative lateral y in metres.")

    # Image Space Bounding Box (xmin, ymin, xmax, ymax) in camera pixels
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = Field(
        None,
        description="Ball bounding box in image pixels.",
    )

    # Ball Kinematics
    vx: Optional[float] = Field(None, description="Ball velocity in x direction (m/s).")
    vy: Optional[float] = Field(None, description="Ball velocity in y direction (m/s).")
    vz: Optional[float] = Field(None, description="Ball vertical velocity in z direction (m/s).")
    speed: Optional[float] = Field(None, ge=0.0, description="Ball scalar speed in m/s.")

    # Ball Physical State and Observation Flags
    ball_state: BallState = Field(
        BallState.UNKNOWN,
        description="Physical state: ground, airborne, or unknown when evidence is insufficient.",
    )
    observed_flag: bool = Field(..., description="True if directly detected; False if estimated/predicted.")
    visibility_status: VisibilityStatus = Field(..., description="Visibility status: visible, occluded, etc.")
    tracking_status: TrackingStatus = Field(..., description="Tracking status: detected, interpolated, lost, etc.")
    detection_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Detector confidence [0.0, 1.0].")
    calibration_uncertainty_m: Optional[float] = Field(None, ge=0.0, description="Reprojection uncertainty in metres.")

    @model_validator(mode="after")
    def validate_ball_integrity(self) -> BallFrame:
        # Rule 1: Observation Semantics - only detected can be marked observed_flag=True
        if self.tracking_status == TrackingStatus.DETECTED:
            if not self.observed_flag:
                raise ValueError(
                    "Observation Semantics Violation: ball tracking_status='detected' must have observed_flag=True."
                )
        else:
            if self.observed_flag:
                raise ValueError(
                    f"Observation Semantics Violation: ball tracking_status='{self.tracking_status.value}' "
                    f"cannot have observed_flag=True."
                )

        # Rule 2: Lost or invalid tracking cannot have pitch coordinates
        if self.tracking_status in (TrackingStatus.LOST, TrackingStatus.INVALID):
            if self.x is not None or self.y is not None:
                raise ValueError(
                    f"Data Integrity Violation: tracking_status='{self.tracking_status.value}' "
                    f"cannot possess valid pitch coordinates."
                )

        # Rule 3: Coordinate pairs must be complete
        if (self.x is None and self.y is not None) or (self.x is not None and self.y is None):
            raise ValueError(
                f"Incomplete ball coordinate pair: both x and y must be provided or both None (x={self.x}, y={self.y})."
            )

        # Rule 4: Configurable ground-height tolerance and physical consistency
        # If z exceeds ground_height_tolerance_m, ball cannot be classified as GROUND
        # (It can be AIRBORNE or UNKNOWN).
        if self.z is not None and self.z > self.ground_height_tolerance_m:
            if self.ball_state == BallState.GROUND:
                raise ValueError(
                    f"Physical Inconsistency: Ball elevation z={self.z:.2f} m exceeds ground tolerance "
                    f"({self.ground_height_tolerance_m:.2f} m). ball_state cannot be 'ground'. "
                    f"Must be 'airborne' or 'unknown'."
                )

        return self


# ============================================================================
# Events & Model Annotations
# ============================================================================


class EventRecord(BaseModel):
    """Match event, tactical interval, or human-reviewed annotation.
    
    Refinement 2: Enforces deep immutability on evidence_metadata, recursively
    freezing nested dictionaries into MappingProxyType and lists into tuples.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    schema_version: str = Field(SCHEMA_VERSION, description="Canonical schema version.")
    event_id: str = Field(..., min_length=1, description="Unique event identifier.")
    match_id: str = Field(..., min_length=1, description="Match identifier.")
    period: int = Field(..., ge=1, le=5, description="Match period (1: 1H, 2: 2H, etc.).")
    timestamp_seconds: float = Field(..., ge=0.0, description="Start timestamp of the event (seconds).")
    end_timestamp_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description="End timestamp of interval events (e.g. pressing episode, tactical shift).",
    )
    event_type: str = Field(
        ...,
        min_length=1,
        description="Event classification (e.g. 'pass', 'shot', 'pressing_episode', 'tactical_shift').",
    )
    primary_player_id: Optional[str] = Field(
        None,
        description="Primary actor or primary presser ID.",
    )
    secondary_player_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Immutable tuple of supporting defenders, pass receivers, or secondary actors.",
    )
    team_id: Optional[str] = Field(None, description="Acting team identifier ('0' or '1').")
    outcome: Optional[str] = Field(
        None,
        description="Outcome classification (e.g. 'successful', 'regain', 'escape', 'forced_pass').",
    )
    pitch_x: Optional[float] = Field(None, description="Location of event on pitch (longitudinal metres).")
    pitch_y: Optional[float] = Field(None, description="Location of event on pitch (lateral metres).")
    source: str = Field(
        "manual_annotation",
        description="Provenance of label: 'manual_annotation', 'provider_event', 'pressuregraph_model', etc.",
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence or adjudicator agreement score [0.0, 1.0].",
    )
    evidence_metadata: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Deeply immutable structured tactical evidence.",
    )

    @field_validator("evidence_metadata", mode="after")
    @classmethod
    def ensure_deep_immutability(cls, v: Any) -> Any:
        return freeze_value(v) if v is not None else MappingProxyType({})

    @model_validator(mode="after")
    def validate_interval_ordering(self) -> EventRecord:
        if self.end_timestamp_seconds is not None:
            if self.end_timestamp_seconds < self.timestamp_seconds:
                raise ValueError(
                    f"end_timestamp_seconds ({self.end_timestamp_seconds}) cannot be less than "
                    f"timestamp_seconds ({self.timestamp_seconds})"
                )
        return self
