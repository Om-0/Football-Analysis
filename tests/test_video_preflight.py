"""Unit and integration tests for Video Ingestion Preflight Inspection."""

import json
import subprocess
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

from pitchmind.ingestion.video import (
    VideoPreflightInspector,
    calculate_sha256,
    create_contact_sheet,
    find_executable,
    measure_frame_quality,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def synthetic_valid_video(temp_dir: Path) -> Path:
    """Generate a valid synthetic 1280x720 25 FPS MP4 video clip."""
    video_path = temp_dir / "valid_clip.mp4"
    width, height, fps, num_frames = 1280, 720, 25.0, 50
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    try:
        for i in range(num_frames):
            # Draw synthetic green football pitch with moving white marker
            frame = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)  # Forest Green
            # White pitch line
            cv2.line(frame, (100, 360), (1180, 360), (255, 255, 255), 2)
            # Center circle
            cv2.circle(frame, (640, 360), 80, (255, 255, 255), 2)
            # Moving ball marker
            ball_x = int(100 + i * 20)
            cv2.circle(frame, (ball_x, 360), 10, (255, 255, 255), -1)
            out.write(frame)
    finally:
        out.release()

    return video_path


@pytest.fixture
def synthetic_nonstandard_video(temp_dir: Path) -> Path:
    """Generate a 500x500 square video clip (ratio 1:1, non-standard for broadcast/tactical)."""
    video_path = temp_dir / "square_clip.mp4"
    width, height, fps, num_frames = 500, 500, 25.0, 25
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    try:
        for _ in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            out.write(frame)
    finally:
        out.release()

    return video_path


def test_find_executable():
    """Verify that find_executable resolves ffmpeg and ffprobe."""
    ffmpeg = find_executable("ffmpeg")
    ffprobe = find_executable("ffprobe")
    assert ffmpeg is not None, "ffmpeg executable should be found."
    assert ffprobe is not None, "ffprobe executable should be found."
    assert Path(ffmpeg).is_file()
    assert Path(ffprobe).is_file()


def test_calculate_sha256(synthetic_valid_video: Path):
    """Verify SHA-256 calculation matches cryptographic standard."""
    checksum = calculate_sha256(synthetic_valid_video)
    assert len(checksum) == 64
    assert all(c in "0123456789abcdef" for c in checksum.lower())


def test_preflight_valid_video(synthetic_valid_video: Path):
    """Test preflight inspection on a valid standard 16:9 video container."""
    inspector = VideoPreflightInspector()
    metadata, report = inspector.inspect(synthetic_valid_video, video_id="vid-test-01")

    assert metadata.video_id == "vid-test-01"
    assert metadata.width == 1280
    assert metadata.height == 720
    assert metadata.fps == 25.0
    assert metadata.duration_seconds >= 1.5
    assert len(metadata.sha256_checksum) == 64

    assert report.video_id == "vid-test-01"
    assert report.passed is True
    assert report.decode_success is True
    assert report.aspect_ratio_standard is True
    assert report.dropped_frames_detected == 0
    assert len(report.failure_reasons) == 0


def test_preflight_nonstandard_aspect_ratio(synthetic_nonstandard_video: Path):
    """Test that a non-standard aspect ratio flags a failure reason."""
    inspector = VideoPreflightInspector()
    _, report = inspector.inspect(synthetic_nonstandard_video)

    assert report.passed is False
    assert report.aspect_ratio_standard is False
    assert any("aspect ratio" in r.lower() for r in report.failure_reasons)


def test_preflight_unsupported_extension(temp_dir: Path):
    """Test rejection of unsupported file extensions."""
    bad_file = temp_dir / "match_notes.txt"
    bad_file.write_text("tactical notes")

    inspector = VideoPreflightInspector()
    _, report = inspector.inspect(bad_file)

    assert report.passed is False
    assert any("unsupported container extension" in r.lower() for r in report.failure_reasons)


def test_preflight_corrupt_file(temp_dir: Path):
    """Test graceful failure reporting when given a corrupt video container."""
    corrupt_file = temp_dir / "corrupted.mp4"
    corrupt_file.write_bytes(b"\x00\x01\x02\x03CORRUPT_HEADER_BYTES")

    inspector = VideoPreflightInspector()
    _, report = inspector.inspect(corrupt_file)

    assert report.passed is False
    assert report.decode_success is False
    assert len(report.failure_reasons) > 0


def test_preflight_missing_file():
    """Test that inspecting a non-existent file raises FileNotFoundError."""
    inspector = VideoPreflightInspector()
    with pytest.raises(FileNotFoundError):
        inspector.inspect("non_existent_file.mp4")


def test_preflight_output_artifacts(synthetic_valid_video: Path, temp_dir: Path):
    """Test generation of contact sheet, sample frames, and JSON report."""
    out_dir = temp_dir / "preflight_output"
    inspector = VideoPreflightInspector()
    metadata, report = inspector.inspect(
        synthetic_valid_video,
        video_id="vid-artifact-test",
        sample_frame_count=8,
        output_dir=out_dir,
        save_sampled_frames=True,
        save_contact_sheet=True,
    )

    assert report.passed is True

    # 1. Verify sampled frames on disk
    frames_dir = out_dir / "sampled_frames"
    assert frames_dir.is_dir()
    saved_frames = list(frames_dir.glob("*.jpg"))
    assert len(saved_frames) == 8
    for f in saved_frames:
        assert f.stat().st_size > 0

    # 2. Verify contact sheet
    sheet_file = out_dir / "contact_sheet.jpg"
    assert sheet_file.is_file()
    assert sheet_file.stat().st_size > 0
    sheet_img = cv2.imread(str(sheet_file))
    assert sheet_img is not None
    # 8 tiles, 4 columns -> 2 rows: height = 2 * 180 = 360, width = 4 * 320 = 1280
    assert sheet_img.shape == (360, 1280, 3)

    # 3. Verify machine-readable JSON report
    json_file = out_dir / "preflight_report.json"
    assert json_file.is_file()
    with open(json_file, "r", encoding="utf-8") as jf:
        data = json.load(jf)

    assert data["schema_version"] == "2.0.0"
    assert data["video_metadata"]["video_id"] == "vid-artifact-test"
    assert data["preflight_report"]["video_id"] == "vid-artifact-test"
    assert data["preflight_report"]["passed"] is True
    assert data["preflight_report"]["aspect_ratio_standard"] is True
    assert data["preflight_report"]["quality_metrics"]["video_integrity_score"] == 1.0


def test_preflight_sampling_interval_vs_count(synthetic_valid_video: Path, temp_dir: Path):
    """Test frame sampling configured via time interval instead of fixed count."""
    out_dir = temp_dir / "interval_output"
    inspector = VideoPreflightInspector()
    _, report = inspector.inspect(
        synthetic_valid_video,
        sample_interval_seconds=0.5,
        output_dir=out_dir,
        save_sampled_frames=True,
    )

    assert report.passed is True
    frames_dir = out_dir / "sampled_frames"
    saved_frames = list(frames_dir.glob("*.jpg"))
    # For a 2.0s clip (50 frames at 25 fps), interval 0.5s gives frames at approx 0.0s, 0.52s, 1.04s, 1.56s (4 frames)
    assert len(saved_frames) >= 3


def test_preflight_mov_container(temp_dir: Path):
    """Test preflight inspection on a QuickTime MOV container."""
    ffmpeg_bin = find_executable("ffmpeg")
    assert ffmpeg_bin is not None, "FFmpeg required for MOV generation test."

    mov_path = temp_dir / "clip.mov"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=1280x720:rate=25",
        "-c:v",
        "libx264",
        str(mov_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    inspector = VideoPreflightInspector()
    metadata, report = inspector.inspect(mov_path)

    assert report.passed is True
    assert report.decode_success is True
    assert "mov" in metadata.container.lower()


def test_preflight_mkv_container(temp_dir: Path):
    """Test preflight inspection on a Matroska MKV container."""
    ffmpeg_bin = find_executable("ffmpeg")
    assert ffmpeg_bin is not None, "FFmpeg required for MKV generation test."

    mkv_path = temp_dir / "clip.mkv"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=1280x720:rate=25",
        "-c:v",
        "libx264",
        str(mkv_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    inspector = VideoPreflightInspector()
    metadata, report = inspector.inspect(mkv_path)

    assert report.passed is True
    assert report.decode_success is True
    assert "matroska" in metadata.container.lower()


def test_preflight_zero_byte_file(temp_dir: Path):
    """Test graceful failure reporting when given an empty 0-byte file."""
    empty_file = temp_dir / "empty.mp4"
    empty_file.touch()

    inspector = VideoPreflightInspector()
    _, report = inspector.inspect(empty_file)

    assert report.passed is False
    assert any("empty" in r.lower() for r in report.failure_reasons)


def test_preflight_quality_metrics_values(synthetic_valid_video: Path):
    """Verify calculated preflight quality metrics match expected pitch features."""
    inspector = VideoPreflightInspector()
    _, report = inspector.inspect(synthetic_valid_video)

    assert report.quality_metrics is not None
    q = report.quality_metrics
    # Synthetic pitch is largely green
    assert q.preliminary_pitch_green_fraction > 0.60
    assert q.blur_score_laplacian > 10.0
    assert 0.0 <= q.mean_brightness <= 255.0
    assert q.contrast_score >= 0.0
    assert 0.0 <= q.image_quality_score <= 1.0
    assert q.aspect_ratio_standard is True


def test_create_contact_sheet_empty():
    """Verify that create_contact_sheet handles empty frame lists gracefully."""
    blank = create_contact_sheet([])
    assert blank.shape == (180, 320, 3)
    assert isinstance(blank, np.ndarray)


def test_measure_frame_quality_pure_pitch():
    """Verify measure_frame_quality on pure football green image."""
    pure_green = np.full((100, 100, 3), (34, 139, 34), dtype=np.uint8)  # Forest Green
    res = measure_frame_quality(pure_green)
    assert res["pitch_green_fraction"] > 0.95
    assert res["blur_laplacian"] == 0.0  # Homogeneous texture


