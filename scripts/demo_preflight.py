"""PITCHMIND Phase 2A — End-to-End Preflight Pipeline Demonstration.

Generates a synthetic tactical football sequence, runs the complete
VideoPreflightInspector pipeline, and verifies output artifacts (contact sheet,
sample frames, and JSON report).
"""

from pathlib import Path
import json
import cv2
import numpy as np

from pitchmind.ingestion.video import VideoPreflightInspector


def generate_sample_clip(file_path: Path, num_frames: int = 75, fps: float = 25.0) -> None:
    """Generate a realistic synthetic tactical camera clip (1280x720, 25 FPS)."""
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))

    try:
        for f in range(num_frames):
            # 1. Pitch canvas: Grass green
            frame = np.full((height, width, 3), (38, 128, 42), dtype=np.uint8)

            # 2. White touchlines and penalty box lines
            cv2.rectangle(frame, (80, 60), (1200, 660), (240, 240, 240), 3)  # Boundary
            cv2.line(frame, (640, 60), (640, 660), (240, 240, 240), 2)       # Halfway line
            cv2.circle(frame, (640, 360), 90, (240, 240, 240), 2)             # Center circle
            cv2.circle(frame, (640, 360), 4, (240, 240, 240), -1)             # Center spot

            # Penalty boxes
            cv2.rectangle(frame, (80, 210), (280, 510), (240, 240, 240), 2)
            cv2.rectangle(frame, (1000, 210), (1200, 510), (240, 240, 240), 2)

            # 3. Red Team Outfield Players (Team 0)
            red_color = (30, 30, 220)
            red_positions = [
                (200, 360), (320, 240), (320, 480), (450, 300),
                (450, 420), (550, 360), (600, 200), (600, 520)
            ]
            for rx, ry in red_positions:
                dx = int(10 * np.sin(f * 0.1 + rx))
                dy = int(5 * np.cos(f * 0.1 + ry))
                cv2.circle(frame, (rx + dx, ry + dy), 9, red_color, -1)
                cv2.circle(frame, (rx + dx, ry + dy), 9, (255, 255, 255), 1)

            # 4. Blue Team Outfield Players (Team 1)
            blue_color = (220, 80, 30)
            blue_positions = [
                (1080, 360), (960, 240), (960, 480), (830, 300),
                (830, 420), (730, 360), (680, 200), (680, 520)
            ]
            for bx, by in blue_positions:
                dx = int(8 * np.sin(f * 0.12 + bx))
                dy = int(6 * np.cos(f * 0.12 + by))
                cv2.circle(frame, (bx + dx, by + dy), 9, blue_color, -1)
                cv2.circle(frame, (bx + dx, by + dy), 9, (255, 255, 255), 1)

            # 5. Match Ball (Moving towards right goal)
            ball_x = int(550 + f * 4.5)
            ball_y = int(360 + 20 * np.sin(f * 0.2))
            cv2.circle(frame, (ball_x, ball_y), 5, (255, 255, 255), -1)
            cv2.circle(frame, (ball_x, ball_y), 5, (0, 0, 0), 1)

            out.write(frame)
    finally:
        out.release()


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    demo_dir = project_root / "artifacts" / "preflight_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    clip_path = demo_dir / "synthetic_tactical_sample.mp4"
    print(f"[1/4] Generating synthetic tactical match clip: {clip_path.name}...")
    generate_sample_clip(clip_path, num_frames=75, fps=25.0)
    print(f"      Size: {clip_path.stat().st_size:,} bytes | Resolution: 1280x720 | 75 frames (3.0s)")

    print(f"[2/4] Running VideoPreflightInspector pipeline...")
    inspector = VideoPreflightInspector()
    output_artifacts_dir = demo_dir / "pipeline_outputs"
    video_meta, preflight_rep = inspector.inspect(
        clip_path,
        video_id="demo-match-tactical-01",
        sample_frame_count=12,
        output_dir=output_artifacts_dir,
        save_sampled_frames=True,
        save_contact_sheet=True,
    )

    print(f"[3/4] Inspection completed:")
    print(f"      Passed Integrity: {preflight_rep.passed}")
    print(f"      Codec / Container: {video_meta.codec} / {video_meta.container}")
    print(f"      Duration / FPS: {video_meta.duration_seconds:.2f}s / {video_meta.fps:.1f} FPS")
    print(f"      SHA-256 Checksum: {video_meta.sha256_checksum[:16]}...")
    print(f"      Checks Run: {', '.join(preflight_rep.checks_run)}")

    if preflight_rep.quality_metrics:
        qm = preflight_rep.quality_metrics
        print("\n--- Preliminary Quality Metrics (Unvalidated Design Candidates) ---")
        print(f"      Video Integrity Score:      {qm.video_integrity_score:.3f}")
        print(f"      Composite Image Quality:    {qm.image_quality_score:.3f}")
        print(f"      Laplacian Sharpness/Blur:   {qm.blur_score_laplacian:.1f}")
        print(f"      Mean Luma Brightness:       {qm.mean_brightness:.1f} / 255")
        print(f"      Luma Contrast Score:        {qm.contrast_score:.1f}")
        print(f"      Pitch Green Chromaticity:   {qm.preliminary_pitch_green_fraction:.1%}")
        print(f"      Standard 16:9 Aspect Ratio: {qm.aspect_ratio_standard} ({qm.aspect_ratio:.2f})")

    print("\n[4/4] Output Artifacts Verification:")
    contact_sheet = output_artifacts_dir / "contact_sheet.jpg"
    print(f"      Contact Sheet: {contact_sheet.name} ({contact_sheet.stat().st_size:,} bytes)")
    json_report = output_artifacts_dir / "preflight_report.json"
    print(f"      JSON Report:   {json_report.name} ({json_report.stat().st_size:,} bytes)")
    saved_frames = list((output_artifacts_dir / "sampled_frames").glob("*.jpg"))
    print(f"      Sample Frames: {len(saved_frames)} saved to disk")

    # Display JSON report snippet
    with open(json_report, "r", encoding="utf-8") as jf:
        rep_json = json.load(jf)

    print("\n--- preflight_report.json (Preview) ---")
    print(json.dumps(rep_json["preflight_report"], indent=2)[:800] + "\n      ...")


if __name__ == "__main__":
    main()

