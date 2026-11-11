"""Base tracking provider adapter architecture and common data structures.

Enforces:
1. PITCHMIND fixed pitch coordinate standard: origin (0, 0) at center circle,
   longitudinal X in [-pitch_length/2, +pitch_length/2] (attacks +X),
   lateral Y in [-pitch_width/2, +pitch_width/2] (+Y towards far touchline).
2. Observation semantics: directly observed measurements must have observed_flag=True,
   while missing, interpolated, or extrapolated data must have observed_flag=False.
3. Native sampling rates preserved without silent downsampling (e.g. 25 Hz remains 25 Hz).
4. Provenance tracking with structured ingestion audit reports.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from pydantic import BaseModel, ConfigDict, Field

from pitchmind.data_engine.schema import (
    SCHEMA_VERSION,
    BallFrame,
    BallState,
    MatchMetadata,
    PitchDimensionSource,
    PlayerFrame,
    TeamAttackingDirection,
    TrackingStatus,
    VisibilityStatus,
)


class TrackingIngestionReport(BaseModel):
    """Audit report generated during tracking ingestion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(SCHEMA_VERSION, description="Canonical schema version.")
    provider_name: str = Field(..., description="Source tracking provider identifier.")
    match_id: str = Field(..., description="Unique match identifier.")
    source_path: str = Field(..., description="Path to source file or directory.")
    source_rights_record_id: Optional[str] = Field(
        None,
        description="Reference identifier in datasets/manifests/data_rights_register.json.",
    )
    total_frames: int = Field(..., ge=0, description="Total unique frames ingested.")
    player_records_count: int = Field(..., ge=0, description="Total individual player-frame rows.")
    ball_records_count: int = Field(..., ge=0, description="Total ball-frame rows.")
    observed_records_count: int = Field(..., ge=0, description="Count of directly observed entities.")
    extrapolated_records_count: int = Field(..., ge=0, description="Count of extrapolated entities.")
    missing_records_count: int = Field(..., ge=0, description="Count of missing/lost entities.")
    native_sampling_rate_hz: float = Field(..., gt=0.0, description="Native capture frequency in Hertz.")
    resampling_applied: bool = Field(
        False,
        description="Whether temporal resampling was applied (False by default; native rate preserved).",
    )
    coordinate_bounds_raw: Mapping[str, float] = Field(
        default_factory=dict,
        description="Min and max of raw source coordinates before normalization.",
    )
    coordinate_bounds_normalized: Mapping[str, float] = Field(
        default_factory=dict,
        description="Min and max of normalized pitch coordinates in metres.",
    )
    periods_detected: Tuple[int, ...] = Field(
        default_factory=tuple,
        description="Playing periods present in the dataset (e.g. (1, 2)).",
    )
    validation_warnings: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Non-fatal warnings encountered during ingestion.",
    )
    validation_errors: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Errors encountered during ingestion.",
    )
    ingestion_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of ingestion.",
    )


class TrackingDataset(BaseModel):
    """Container holding canonical tracking structures and source provenance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    match_metadata: MatchMetadata
    player_frames: List[PlayerFrame]
    ball_frames: List[BallFrame]
    ingestion_report: TrackingIngestionReport
    raw_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw provider metadata preserved for auditing.",
    )


class BaseTrackingAdapter(ABC):
    """Abstract base class for external tracking data providers."""

    def __init__(
        self,
        provider_name: str,
        rights_record_id: Optional[str] = None,
        default_pitch_length_m: float = 105.0,
        default_pitch_width_m: float = 68.0,
    ) -> None:
        self.provider_name = provider_name
        self.rights_record_id = rights_record_id
        self.default_pitch_length_m = default_pitch_length_m
        self.default_pitch_width_m = default_pitch_width_m

    @abstractmethod
    def load(self, source_path: str | Path, **kwargs: Any) -> TrackingDataset:
        """Parse provider files, normalize coordinates, and return canonical TrackingDataset."""
        pass

