"""PITCHMIND Video Ingestion Preflight & Container Inspection Engine.

Performs automatic preflight quality inspection on user-uploaded football videos
prior to computer-vision inference and tracking reconstruction.

Key Capabilities:
1. Video Validation:
   - Supports MP4, MOV, and MKV containers.
   - Validates file existence, non-empty payload, container integrity, and decode health.
   - Emits structured, actionable failure reasons.
2. Metadata Extraction:
   - Integrates FFprobe to extract stream properties, codec, resolution, nominal/average FPS,
     duration, frame count, pixel format (e.g. yuv420p), time-base, start_pts, and audio streams.
3. Memory-Efficient Frame Sampling:
   - Configurable sampling interval (by frame count or fixed second interval).
   - Incremental frame extraction via OpenCV seeking; never loads full matches into RAM.
   - Graceful handling of individual frame decode failures.
4. Preliminary Quality Assessment:
   - Blur measurement via variance of Laplacian.
   - Brightness (luma mean) and contrast (luma standard deviation).
   - Aspect ratio verification against 16:9 standard.
   - Preliminary pitch-green chromaticity ratio (clearly identified as an unvalidated candidate estimate).
   - Staged PreflightQualityMetrics emission.
5. Visual and Machine-Readable Artifacts:
   - Saves sampled frames to disk.
   - Generates an annotated multi-tile contact sheet image.
   - Emits machine-readable preflight_report.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pitchmind.data_engine.schema import (
    PreflightQualityMetrics,
    PreflightReport,
    VideoMetadata,
)

logger = logging.getLogger(__name__)


def find_executable(name: str) -> Optional[str]:
    """Locate an executable binary in system PATH or standard candidate directories."""
    resolved = shutil.which(name)
    if resolved and os.path.exists(resolved):
        return resolved

    # Fallback search in known Windows WinGet and user directories
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Microsoft"
        / "WinGet"
        / "Packages"
        / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        / "ffmpeg-9.0.2-full_build"
        / "bin",
        Path(r"C:\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg\bin"),
        Path(os.environ.get("USERPROFILE", "")) / "Practice" / "ffmpeg-2025-09-22-git-c9168717bf-full_build" / "bin",
    ]
    exe_name = f"{name}.exe" if not name.endswith(".exe") else name
    for candidate_dir in candidates:
        candidate_path = candidate_dir / exe_name
        if candidate_path.is_file():
            return str(candidate_path)

    return None


def calculate_sha256(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Calculate the cryptographic SHA-256 checksum of a file in streaming chunks."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def probe_container_ffprobe(file_path: str | Path) -> Dict[str, Any]:
    """Extract stream and container metadata using FFprobe."""
    ffprobe_bin = find_executable("ffprobe")
    if not ffprobe_bin:
        raise FileNotFoundError("ffprobe executable was not found on system PATH or candidate directories.")

    cmd = [
        ffprobe_bin,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed with returncode {result.returncode}: {result.stderr}")

    return json.loads(result.stdout)


def parse_fraction(fraction_str: str) -> float:
    """Safely parse FFprobe fractional frame rate or time-base string (e.g. '25/1', '30000/1001')."""
    if "/" in fraction_str:
        num, den = fraction_str.split("/", 1)
        denom_val = float(den)
        return float(num) / denom_val if denom_val != 0.0 else 0.0
    return float(fraction_str)


def measure_frame_quality(frame_bgr: np.ndarray) -> Dict[str, float]:
    """Compute visual quality indicators on a single video frame.
    
    Metrics:
    - blur_laplacian: Variance of Laplacian (higher indicates sharper edges).
    - brightness: Mean intensity in grayscale / luma channel.
    - contrast: Standard deviation of intensity in luma channel.
    - pitch_green_fraction: Fraction of pixels matching green grass chromaticity in HSV.
      Note: This is an unvalidated preliminary candidate estimate of pitch visibility.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Blur measurement
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 2. Brightness and Contrast
    mean_brightness = float(np.mean(gray))
    contrast_score = float(np.std(gray))

    # 3. Preliminary Pitch-Green Chromaticity Ratio (Unvalidated Candidate Estimate)
    # Green grass hue in OpenCV HSV [0..180]: typically 35 to 85 (H=25..75 inclusive)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40], dtype=np.uint8)
    upper_green = np.array([80, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixel_count = int(np.count_nonzero(green_mask))
    total_pixels = frame_bgr.shape[0] * frame_bgr.shape[1]
    pitch_green_fraction = green_pixel_count / max(total_pixels, 1)

    return {
        "blur_laplacian": laplacian_var,
        "brightness": mean_brightness,
        "contrast": contrast_score,
        "pitch_green_fraction": pitch_green_fraction,
    }


def create_contact_sheet(
    sampled_frames: List[Tuple[int, float, np.ndarray, Dict[str, float]]],
    tile_width: int = 320,
    tile_height: int = 180,
    columns: int = 4,
) -> np.ndarray:
    """Compose sampled frames into an annotated contact sheet grid.
    
    Args:
        sampled_frames: List of tuples (frame_index, timestamp_sec, frame_bgr, quality_dict).
        tile_width: Width of each tile in pixels.
        tile_height: Height of each tile in pixels.
        columns: Number of tile columns in the grid.
        
    Returns:
        Annotated BGR composite image.
    """
    if not sampled_frames:
        blank = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        cv2.putText(blank, "No frames decoded", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return blank

    num_tiles = len(sampled_frames)
    rows = math.ceil(num_tiles / columns)
    grid_width = columns * tile_width
    grid_height = rows * tile_height

    contact_sheet = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for i, (f_idx, t_sec, frame, q_dict) in enumerate(sampled_frames):
        r = i // columns
        c = i % columns
        y1 = r * tile_height
        y2 = y1 + tile_height
        x1 = c * tile_width
        x2 = x1 + tile_width

        # Resize tile
        resized = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)

        # Semi-transparent footer banner for readability
        banner_h = 28
        footer = resized[tile_height - banner_h : tile_height, :].copy()
        black_bg = np.zeros_like(footer)
        blended = cv2.addWeighted(footer, 0.4, black_bg, 0.6, 0)
        resized[tile_height - banner_h : tile_height, :] = blended

        # Annotations text
        label = f"#{f_idx:05d} | {t_sec:.1f}s | B:{q_dict.get('blur_laplacian', 0):.0f} | G:{q_dict.get('pitch_green_fraction', 0):.0%}"
        cv2.putText(
            resized,
            label,
            (6, tile_height - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        contact_sheet[y1:y2, x1:x2] = resized

    return contact_sheet


class VideoPreflightInspector:
    """Inspects video container integrity, metadata, and decoding continuity."""

    def __init__(self, supported_extensions: Optional[List[str]] = None) -> None:
        self.supported_extensions = supported_extensions or [".mp4", ".mov", ".mkv"]

    def inspect(
        self,
        file_path: str | Path,
        video_id: Optional[str] = None,
        sample_frame_count: int = 16,
        sample_interval_seconds: Optional[float] = None,
        output_dir: Optional[str | Path] = None,
        save_sampled_frames: bool = True,
        save_contact_sheet: bool = True,
    ) -> Tuple[VideoMetadata, PreflightReport]:
        """Perform comprehensive preflight inspection on a video file.
        
        Args:
            file_path: Path to the target video container.
            video_id: Optional unique identifier (generated from filename if omitted).
            sample_frame_count: Number of frames to sample across the video timeline (default 16).
            sample_interval_seconds: Optional interval in seconds to sample frames.
            output_dir: Optional directory to write sampled frames, contact sheet, and JSON report.
            save_sampled_frames: If True and output_dir provided, saves individual sample frames.
            save_contact_sheet: If True and output_dir provided, generates and saves contact_sheet.jpg.
            
        Returns:
            Tuple of (VideoMetadata, PreflightReport).
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")

        # Check for zero-byte or empty file
        if path.stat().st_size == 0:
            vid_id = video_id or f"vid-{path.stem}"
            report = PreflightReport(
                video_id=vid_id,
                passed=False,
                checks_run=("file_existence", "file_non_empty"),
                decode_success=False,
                timestamp_continuity=False,
                duration_seconds=1.0,
                fps_measured=25.0,
                dropped_frames_detected=0,
                aspect_ratio_standard=False,
                quality_metrics=None,
                failure_reasons=("Video file is empty (0 bytes).",),
                report_timestamp=datetime.now(timezone.utc),
            )
            meta = VideoMetadata(
                video_id=vid_id,
                file_path=str(path),
                sha256_checksum="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                duration_seconds=1.0,
                total_frames=1,
                fps=25.0,
                width=1920,
                height=1080,
                codec="unknown",
                pixel_format="unknown",
                container=path.suffix.lstrip(".").lower() or "unknown",
            )
            return meta, report

        vid_id = video_id or f"vid-{path.stem}"
        checks_run: List[str] = []
        failure_reasons: List[str] = []

        # --------------------------------------------------------------------
        # 1. Extension Check
        # --------------------------------------------------------------------
        checks_run.append("file_extension_support")
        if path.suffix.lower() not in self.supported_extensions:
            failure_reasons.append(
                f"Unsupported container extension '{path.suffix}'. Supported: {self.supported_extensions}"
            )

        # --------------------------------------------------------------------
        # 2. SHA-256 Checksum
        # --------------------------------------------------------------------
        checks_run.append("sha256_checksum")
        sha256 = calculate_sha256(path)

        # --------------------------------------------------------------------
        # 3. FFprobe Stream Analysis
        # --------------------------------------------------------------------
        checks_run.append("ffprobe_stream_probe")
        probe_data: Dict[str, Any] = {}
        video_stream: Dict[str, Any] = {}
        audio_stream_count = 0
        try:
            probe_data = probe_container_ffprobe(path)
            streams = probe_data.get("streams", [])
            for st in streams:
                if st.get("codec_type") == "video" and not video_stream:
                    video_stream = st
                elif st.get("codec_type") == "audio":
                    audio_stream_count += 1
            if not video_stream:
                failure_reasons.append("No video stream found in container.")
        except Exception as e:
            failure_reasons.append(f"FFprobe inspection failed: {e}")

        # Extract stream properties
        fmt = probe_data.get("format", {})
        codec = video_stream.get("codec_name", "unknown")
        pixel_format = video_stream.get("pix_fmt", "yuv420p")
        container = fmt.get("format_name", path.suffix.lstrip(".").lower()).split(",")[0]
        time_base = video_stream.get("time_base")
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # Start time and start PTS
        try:
            start_pts = int(video_stream.get("start_pts")) if video_stream.get("start_pts") else None
        except (ValueError, TypeError):
            start_pts = None

        try:
            start_time_seconds = float(video_stream.get("start_time") or fmt.get("start_time", 0.0))
        except (ValueError, TypeError):
            start_time_seconds = 0.0

        # Nominal FPS
        r_fps = video_stream.get("r_frame_rate", "25/1")
        try:
            fps = parse_fraction(r_fps)
        except Exception:
            fps = 25.0
        if fps <= 0.0:
            fps = 25.0

        # Duration
        try:
            duration_seconds = float(video_stream.get("duration") or fmt.get("duration", 0.0))
        except (ValueError, TypeError):
            duration_seconds = 0.0

        # Total frames
        try:
            total_frames = int(video_stream.get("nb_frames") or round(duration_seconds * fps))
        except (ValueError, TypeError):
            total_frames = int(round(duration_seconds * fps))

        # --------------------------------------------------------------------
        # 4. OpenCV Incremental Decode & Frame Sampling (Memory Efficient)
        # --------------------------------------------------------------------
        checks_run.append("opencv_decode_sampling")
        decode_success = False
        measured_fps = fps
        dropped_frames = 0
        aspect_ratio_standard = True
        sampled_frame_records: List[Tuple[int, float, np.ndarray, Dict[str, float]]] = []

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            failure_reasons.append("OpenCV failed to open video file for decoding.")
        else:
            try:
                cap_fps = cap.get(cv2.CAP_PROP_FPS)
                if cap_fps > 0:
                    measured_fps = cap_fps
                cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if cap_frames > 0 and total_frames <= 0:
                    total_frames = cap_frames

                # Fallback duration if missing
                if duration_seconds <= 0.0 and total_frames > 0 and measured_fps > 0:
                    duration_seconds = total_frames / measured_fps

                # Determine sample frame indices
                if sample_interval_seconds and sample_interval_seconds > 0:
                    interval_frames = max(int(round(sample_interval_seconds * measured_fps)), 1)
                    sample_indices = list(range(0, max(total_frames, 1), interval_frames))
                    if not sample_indices:
                        sample_indices = [0]
                else:
                    if total_frames > 0:
                        sample_indices = [
                            int(i * (total_frames - 1) / max(sample_frame_count - 1, 1))
                            for i in range(sample_frame_count)
                        ]
                    else:
                        sample_indices = [0]

                # Setup output directories if requested
                frames_dir = None
                if output_dir:
                    out_path = Path(output_dir)
                    out_path.mkdir(parents=True, exist_ok=True)
                    if save_sampled_frames:
                        frames_dir = out_path / "sampled_frames"
                        frames_dir.mkdir(parents=True, exist_ok=True)

                decoded_count = 0
                for idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        decoded_count += 1
                        t_sec = idx / measured_fps if measured_fps > 0 else 0.0
                        if width == 0 or height == 0:
                            height, width = frame.shape[:2]

                        # Compute visual quality metrics for this frame
                        q_dict = measure_frame_quality(frame)
                        sampled_frame_records.append((idx, t_sec, frame, q_dict))

                        # Save individual sample frame incrementally to disk
                        if frames_dir and save_sampled_frames:
                            frame_fname = f"frame_{idx:06d}_t{t_sec:.2f}s.jpg"
                            cv2.imwrite(str(frames_dir / frame_fname), frame)
                    else:
                        dropped_frames += 1

                decode_success = decoded_count > 0 and dropped_frames == 0
                if decoded_count == 0:
                    failure_reasons.append("Failed to decode any frames from video container.")
                elif dropped_frames > 0:
                    failure_reasons.append(
                        f"Sample decode failed on {dropped_frames}/{len(sample_indices)} test frames."
                    )
            finally:
                cap.release()

        # Aspect ratio check (16:9 standard: ~1.777)
        aspect_ratio = (width / height) if height > 0 else 1.0
        aspect_ratio_standard = 1.70 <= aspect_ratio <= 1.85
        if not aspect_ratio_standard and width > 0 and height > 0:
            failure_reasons.append(
                f"Non-standard aspect ratio ({width}x{height}, ratio {aspect_ratio:.2f}). Standard 16:9 expected."
            )

        # Ensure non-zero fallbacks for invalid containers
        if duration_seconds <= 0.0:
            duration_seconds = 1.0
            if "Video duration could not be determined." not in failure_reasons:
                failure_reasons.append("Video duration could not be determined.")

        if total_frames <= 0:
            total_frames = int(round(duration_seconds * measured_fps))
        if width <= 0:
            width = 1920
        if height <= 0:
            height = 1080

        # --------------------------------------------------------------------
        # 5. Preliminary Quality Assessment Computation
        # --------------------------------------------------------------------
        quality_metrics: Optional[PreflightQualityMetrics] = None
        if sampled_frame_records:
            avg_blur = float(np.mean([r[3]["blur_laplacian"] for r in sampled_frame_records]))
            avg_brightness = float(np.mean([r[3]["brightness"] for r in sampled_frame_records]))
            avg_contrast = float(np.mean([r[3]["contrast"] for r in sampled_frame_records]))
            avg_green = float(np.mean([r[3]["pitch_green_fraction"] for r in sampled_frame_records]))

            # Normalized video integrity: decode ratio penalizing dropped frames
            req_count = len(sampled_frame_records) + dropped_frames
            integrity_score = (len(sampled_frame_records) / req_count) if req_count > 0 else 0.0

            # Normalized image quality: sharpness (sigmoid on laplacian) and contrast
            # Blur variance >= 200 is generally crisp 1080p; < 50 is blurry
            sharpness_norm = min(max(avg_blur / 300.0, 0.0), 1.0)
            contrast_norm = min(max(avg_contrast / 70.0, 0.0), 1.0)
            image_quality_score = float(0.6 * sharpness_norm + 0.4 * contrast_norm)

            quality_metrics = PreflightQualityMetrics(
                video_integrity_score=integrity_score,
                aspect_ratio=aspect_ratio,
                aspect_ratio_standard=aspect_ratio_standard,
                mean_brightness=avg_brightness,
                contrast_score=avg_contrast,
                blur_score_laplacian=avg_blur,
                image_quality_score=image_quality_score,
                preliminary_pitch_green_fraction=avg_green,
            )

        # --------------------------------------------------------------------
        # 6. Artifact Generation (Contact Sheet & JSON Report)
        # --------------------------------------------------------------------
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Generate contact sheet
            if save_contact_sheet and sampled_frame_records:
                contact_sheet = create_contact_sheet(sampled_frame_records)
                sheet_path = out_path / "contact_sheet.jpg"
                cv2.imwrite(str(sheet_path), contact_sheet)

        passed = len(failure_reasons) == 0

        # Construct PreflightReport
        preflight_report = PreflightReport(
            video_id=vid_id,
            passed=passed,
            checks_run=tuple(checks_run),
            decode_success=decode_success,
            timestamp_continuity=dropped_frames == 0,
            duration_seconds=duration_seconds,
            fps_measured=measured_fps,
            dropped_frames_detected=dropped_frames,
            aspect_ratio_standard=aspect_ratio_standard,
            quality_metrics=quality_metrics,
            failure_reasons=tuple(failure_reasons),
            report_timestamp=datetime.now(timezone.utc),
        )

        # Construct VideoMetadata
        video_metadata = VideoMetadata(
            video_id=vid_id,
            file_path=str(path),
            sha256_checksum=sha256,
            duration_seconds=duration_seconds,
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            codec=codec,
            pixel_format=pixel_format,
            container=container,
            time_base=time_base,
            start_pts=start_pts,
            start_time_seconds=start_time_seconds,
            audio_streams=audio_stream_count,
        )

        # Write machine-readable JSON report if output_dir provided
        if output_dir:
            report_payload = {
                "schema_version": "2.0.0",
                "video_metadata": video_metadata.model_dump(mode="json"),
                "preflight_report": preflight_report.model_dump(mode="json"),
            }
            json_path = Path(output_dir) / "preflight_report.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(report_payload, jf, indent=2)

        return video_metadata, preflight_report
