# PITCHMIND — ARCHITECTURAL & TECHNICAL DECISIONS (ADRs)

**Document Version**: 2.0  
**Status**: Architecture Baseline  
**Date**: 11 November 2026  

This document records the binding architectural decisions, engineering rationales, and research methodologies established for PITCHMIND v2.0.

---

### ADR 001: Video-First Product Contract with Internal Tracking Generation
- **Context**: Football tactical analytics platforms traditionally depend on proprietary optical tracking providers (Second Spectrum, ChyronHego, SkillCorner). This limits accessibility for lower-tier clubs, academies, and independent analysts who only have raw video.
- **Decision**: The primary user-facing input to PITCHMIND is raw match video (broadcast or tactical camera). Player and ball tracking are generated internally by computer-vision algorithms.
- **Consequence**: Structured tracking datasets (SkillCorner, IDSSE, Metrica) are ingested strictly via canonical provider adapters for research, model training, and benchmark comparisons. They are not a parallel user-facing ingestion route.

---

### ADR 002: Rights-Based Footage Policy
- **Context**: Football video broadcasts are intellectual property protected by copyright. Indiscriminate scraping or commercial processing of broadcast streams creates severe legal liability.
- **Decision**: PITCHMIND enforces a rights-based footage framework:
  1. Users must explicitly affirm they hold lawful authorization or valid rights to process uploaded footage.
  2. The software prohibits unauthorized distribution of processed video, extracted frames, or commercial tracking datasets.
  3. All test footage used for benchmarking must have its provenance and usage rights documented in `datasets/manifests/data_rights_register.json`.
- **Consequence**: Blanket commercial broadcast scraping is rejected. The platform functions as an analysis tool for authorized footage.

---

### ADR 003: Proposed Quality Dimensions as Calibrated Design Candidates
- **Context**: Compressing video quality into a single global scalar score obscures whether footage is suitable for specific downstream tasks (e.g. good enough for team shape, but inadequate for pose estimation).
- **Decision**: Quality assessment evaluates nine candidate dimensions: *Video Integrity, Pitch Coverage, Player Visibility, Detection Reliability, Identity Continuity, Calibration Reliability, Ball Observability, Image Quality, and Pose Quality*. Each analytical module defines task-specific readiness thresholds (`ready`, `partial`, `insufficient`).
- **Research Policy**: These nine quality dimensions and readiness thresholds are **proposed architectural design candidates**. They require empirical calibration, ablation testing, and validation against ground truth during development, rather than being treated as pre-established scientific facts.

---

### ADR 004: Strict Canonical Tracking Data Boundary
- **Context**: Allowing downstream analytical models (e.g. GNNs, change-point detectors) to directly consume detector bounding boxes, pixel coordinates, or provider-specific JSONs creates brittle coupling and leaks implementation details.
- **Decision**: The canonical tracking layer (`player_frames`, `ball_frames`) is the hard boundary between CV and analytics.
- **Binding Data Integrity Rules**:
  1. A missing value is **not** zero.
  2. An out-of-view player is **not** a stationary player.
  3. Generated, interpolated, or extrapolated coordinates must have `observed_flag=False`.
  4. Coordinate convention: Metric pitch in metres ($[-52.5, 52.5]$ m by $[-34.0, 34.0]$ m), origin at pitch center. Analytical grid normalized to 10 Hz.

---

### ADR 005: Independent Analytical Modules Prior to Multi-Task Fusion
- **Context**: Multi-task learning (a single neural backbone predicting pressing, team shape, and player archetypes simultaneously) sounds elegant but often suffers from severe negative transfer and difficult debugging in early development.
- **Decision**: Develop each module independently:
  - **PRESSUREGRAPH**: Heuristic baseline $\to$ XGBoost $\to$ GINEConv/GATv2 + GRU spatio-temporal graph model.
  - **TACTICAL SHIFT**: Rolling spatial geometry $\to$ PELT penalized change-point detection.
  - **ROLESHIFT**: Movement profile aggregation $\to$ K-Means / GMM clustering.
  - **SCAN2PLAY**: Orientation proxy $\to$ gated pilot.
- **Consequence**: Shared multi-task backbones are deferred until independent baselines demonstrate measurable positive transfer.

---

### ADR 006: Local-First Web Application Architecture
- **Context**: The research prototype must provide an intuitive visual interface without incurring cloud hosting costs, complex Kubernetes infrastructure, or distributed storage overhead.
- **Decision**: Build a local-first web architecture:
  - **Backend**: FastAPI with Python async job execution.
  - **Metadata & Job Store**: SQLite.
  - **Columnar Analytical Store**: Apache Parquet files queried via DuckDB for zero-copy, memory-efficient windowed SQL queries.
  - **Frontend**: React + TypeScript with an interactive Canvas/SVG pitch radar. Video and tracking data stream in synchronized chunks.
- **Consequence**: Defer multi-tenant authentication, cloud object storage, and native EXE packaging until later stages.

---

### ADR 007: Hardware Mitigation Strategy for RTX 3050 (4 GB VRAM) & 16 GB RAM
- **Context**: Computer vision models (YOLOX, ByteTrack, HRNet/pose) running on full 1080p video easily exceed 4 GB VRAM and 16 GB system memory if not carefully constrained.
- **Decision**: Enforce strict hardware mitigation rules:
  1. **Chunked Processing**: Process matches in sliding 10–30 second windows rather than decoding entire matches into RAM.
  2. **Frame Sampling**: Run heavy detection at 5–10 Hz rather than full native 60 FPS; tracklets interpolate intermediate frames.
  3. **Intermediate Caching**: Cache detector bounding boxes, tracklets, and extracted features to disk (`.parquet`) to prevent repeated expensive CV passes during analytics training.
  4. **Small Batch Sizes**: Keep GPU neural inference batch size to 1–2.
  5. **Role Separation**: CPU handles FFmpeg decoding and DuckDB queries; GPU is reserved for PyTorch/PyG neural inference.

---

### ADR 008: Core Research Experimental Design & Ablation Matrix
- **Context**: The central scientific contribution is evaluating whether explicit quality modeling improves tactical predictions under incomplete/noisy observations.
- **Decision**: Execute the structured ablation experiment matrix:
  - **E1 (Clean Baseline)**: Tracking-only baseline on clean, high-confidence tracking data.
  - **E2 (+ Quality Features)**: Model augmented with explicit tracking quality and confidence indicators.
  - **E3 (Structured Dropout)**: Performance degradation under simulated camera cuts, occlusions, and off-screen player dropout.
  - **E4 (Spatial & Identity Noise)**: Robustness under coordinate perturbation (homography error) and identity-switch noise.
  - **E5 (Temporal Jitter)**: Resilience to timestamp drift and variable frame rates.
  - **E6 (Quality-Aware Training)**: Training models with quality masks and noise injection to evaluate recovery.
  - **E7 (Cross-Provider Transfer)**: Train on SkillCorner (10 Hz CV-derived), evaluate on IDSSE (25 Hz optical tracking).

---

### ADR 009: Dependency & Licensing Governance
- **Context**: Open-source tools and datasets possess distinct, binding licenses. Inventing dependencies or assuming public availability equals permission violates open-source compliance.
- **Decision**: Maintain a verified Third-Party Notices and License Register (`third_party_notices/LICENSES.md`).
  - Do not invent citations or license terms.
  - Mark any unverified repository entries explicitly as unverified until pinned.
  - Strictly enforce virtual environment (`.venv`) isolation; do not install unverified ML dependencies during Phase 0.

