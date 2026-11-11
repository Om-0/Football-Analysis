"""Tracking ingestion adapters: SkillCorner, IDSSE, and Metrica normalization to canonical schema."""

from pitchmind.ingestion.tracking.base import (
    BaseTrackingAdapter,
    TrackingDataset,
    TrackingIngestionReport,
)
from pitchmind.ingestion.tracking.idsse import IDSSETrackingAdapter
from pitchmind.ingestion.tracking.metrica import MetricaTrackingAdapter
from pitchmind.ingestion.tracking.skillcorner import SkillCornerTrackingAdapter

__all__ = [
    "BaseTrackingAdapter",
    "TrackingDataset",
    "TrackingIngestionReport",
    "MetricaTrackingAdapter",
    "SkillCornerTrackingAdapter",
    "IDSSETrackingAdapter",
]
