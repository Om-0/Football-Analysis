# PITCHMIND — SYSTEM ARCHITECTURE SPECIFICATION

**Document Version**: 2.0  
**Status**: Architecture Baseline  
**Date**: 11 November 2026  

---

## 1. System Architecture at a Glance

PITCHMIND is an end-to-end, video-first football tactical intelligence platform. The architecture decouples computer-vision perception from downstream tactical modeling via a strict **Canonical Tracking Data Layer**.

```
                        ┌──────────────────────────────────────────────┐
                        │          User Uploads Football Video         │
                        │   (Standard Broadcast or Tactical Camera)    │
                        └──────────────────────┬───────────────────────┘
                                               │
                        ┌──────────────────────▼───────────────────────┐
                        │  Stage A-B: Upload Ingestion & Preflight     │
                        │  (FFmpeg/FFprobe decode, integrity check)    │
                        └──────────────────────┬───────────────────────┘
                                               │
                        ┌──────────────────────▼───────────────────────┐
                        │  Stage C-D: Shot Segmentation & Camera Class │
                        │  (Broadcast cuts, wide tactical, replays)    │
                        └──────────────────────┬───────────────────────┘
                                               │
                        ┌──────────────────────▼───────────────────────┐
                        │  Stage E: Multidimensional Quality Engine    │
                        │  (Proposed 9-D vector, task-readiness gating)│
                        └──────────────────────┬───────────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               │                                                               │
    ┌──────────▼──────────┐                                         ┌──────────▼──────────┐
    │  Broadcast Route    │                                         │ Tactical Camera     │
    │  - Segment at cuts  │                                         │ - Full-pitch view   │
    │  - Per-shot calib   │                                         │ - Persistent calib  │
    │  - Out-of-view mask │                                         │ - 18-22 players     │
    └──────────┬──────────┘                                         └──────────┬──────────┘
               │                                                               │
               └───────────────────────────────┬───────────────────────────────┘
                                               │
                        ┌──────────────────────▼───────────────────────┐
                        │  Stage F-I: Common CV Engine                 │
                        │  - Detection (YOLOX baseline)                │
                        │  - Tracking (ByteTrack baseline)             │
                        │  - Team Assignment (Kit appearance)          │
                        │  - Pitch Calibration (Landmark homography)   │
                        └──────────────────────┬───────────────────────┘
                                               │
  ═════════════════════════════════════════════▼═════════════════════════════════════════════
                         CANONICAL TRACKING DATA ENGINE (Strict Boundary)
    - match_metadata    - video_segments    - player_frames    - ball_frames
    - Quality & visibility masks            - 10 Hz unified analytical grid
  ═════════════════════════════════════════════┬═════════════════════════════════════════════
                                               │  (External Provider Adapters:
                                               │   SkillCorner, IDSSE, Metrica)
                        ┌──────────────────────▼───────────────────────┐
                        │  Stage K: Football Feature Engine            │
                        │  - Kinematics & Possession Context           │
                        │  - Team Geometry (Width, Length, Centroid)   │
                        │  - Dynamic Interaction Graph (GINEConv/GATv2)│
                        └──────────────────────┬───────────────────────┘
                                               │
         ┌──────────────────┬──────────────────┴─────────────┬──────────────────┐
         │                  │                                │                  │
┌────────▼────────┐ ┌───────▼────────┐              ┌────────▼────────┐ ┌───────▼────────┐
│ PRESSUREGRAPH   │ │ TACTICAL SHIFT │              │   ROLESHIFT     │ │   SCAN2PLAY    │
│ Pressing GNN    │ │ PELT Detection │              │ Archetype Clust.│ │ Pose Proxy     │
│ (XGBoost -> GRU)│ │ (Team States)  │              │ (Within-Provider│ │ (Conditional)  │
└────────┬────────┘ └───────┬────────┘              └────────┬────────┘ └───────┬────────┘
         │                  │                                │                  │
         └──────────────────┴──────────────────┬─────────────┴──────────────────┘
                                               │
                        ┌──────────────────────▼───────────────────────┐
                        │  Stage L: Application & API Layer            │
                        │  - FastAPI Service & SQLite Jobs             │
                        │  - DuckDB Windowed Parquet Analytics         │
                        │  - React Frontend (Chunked Streaming Replay) │
                        └──────────────────────────────────────────────┘
```

---

## 2. Stage-by-Stage Processing Pipeline

| Stage | Name | Responsibilities | Key Outputs / Artifacts |
| :--- | :--- | :--- | :--- |
| **A** | Upload & Manifest | Store input video file, calculate SHA-256 checksum, record rights confirmation and metadata. | Video manifest, SHA-256, container metadata. |
| **B** | Preflight Decode | Sample initial frames via FFmpeg/OpenCV, verify timestamp continuity, frame dimensions, and FPS. | Preflight decode report, sampled keyframes. |
| **C** | Shot Segmentation | Detect camera cuts, transitions, slow-motion replays, and close-ups. | `video_segments` table with frame intervals. |
| **D** | Camera Classification | Classify each segment: Wide Tactical, Main Broadcast, Close-Up, Behind-Goal, Replay. | Segment camera class + classification confidence. |
| **E** | Quality Assessment | Evaluate proposed 9-dimensional quality vector per segment and derive task-readiness decisions. | Segment quality vector, task-readiness flags. |
| **F** | Object Detection | Detect players, goalkeepers, referees, and the ball on sampled frames (YOLOX baseline). | Bounding boxes, class labels, detector scores. |
| **G** | Multi-Object Tracking | Associate detections across frames into consistent tracklets (ByteTrack baseline). | Local track IDs, association confidence. |
| **H** | Team Identification | Cluster player appearance vectors by kit color and apply user kit hints. | Team assignments ($0, 1$) + confidence score. |
| **I** | Pitch Calibration | Detect pitch markings and compute 2D image-to-pitch homography matrix per segment. | Homography matrix $H$, reprojection error. |
| **J** | Canonicalization | Project observations to metric pitch coordinates; enforce canonical schema and masks. | Parquet tables: `player_frames`, `ball_frames`. |
| **K** | Feature Generation | Compute rolling kinematics, team shape geometry, passing lanes, and interaction graphs. | Windowed feature matrix, PyG graph objects. |
| **L** | Analytics & Serving | Run analytical model heads, evaluate uncertainty, serve windowed payloads via DuckDB. | Predictions, evidence intervals, UI report. |

---

## 3. Dual Route Video Strategy

### 3.1 Standard Broadcast Route
- **Dynamic Framing**: The broadcast camera pans, tilts, and zooms to follow the ball. Many off-ball players are frequently outside the field of view.
- **Segment-Level Processing**: Homography matrices are estimated per continuous camera shot. Never assume calibration persists across a cut or transition.
- **Explicit Out-of-View Masking**: Players outside the camera frustum are marked with `visibility_status="out_of_view"`. They are **never** treated as stationary or falsely interpolated across shots.
- **Replay Exclusion**: Broadcast replays and slow-motion sequences are tagged and excluded from the live analytical timeline.

### 3.2 Wide Tactical Camera Route
- **Full-Pitch Observability**: Elevated, stable camera angle capturing 18–22 players simultaneously.
- **Global Team Geometry**: Continuous pitch coverage enables reliable convex hull, compactness, and line-height calculations across the entire match.
- **Calibration Persistence**: Camera parameters remain relatively static; calibration drift is checked periodically rather than re-estimated every shot.
- **Airborne Ball Handling**: Planar homography is valid only on the pitch plane. Airborne balls have distinct physical trajectory handling ($z > 0$).

---

## 4. Multidimensional Quality Engine & Routing

### 4.1 Proposed Nine Quality Dimensions (Design Candidates)
The quality engine evaluates nine candidate dimensions per camera segment and sliding time window. **Note**: These dimensions are proposed architectural design candidates that require empirical calibration and validation against ground truth:

1. **Video Integrity (`video_integrity_score`)**: Container decoding continuity, dropped frames, PTS consistency $[0.0, 1.0]$.
2. **Pitch Coverage (`pitch_coverage_fraction`)**: Fraction of the standard 2D pitch polygon visible in camera frustum $[0.0, 1.0]$.
3. **Player Visibility (`player_visibility_fraction`)**: Ratio of observed outfield players to expected 22 $[0.0, 1.0]$.
4. **Detection Reliability (`detection_reliability_score`)**: Mean detector class confidence across players and ball $[0.0, 1.0]$.
5. **Identity Continuity (`identity_continuity_score`)**: Tracklet continuity penalizing fragmentation and ID switches $[0.0, 1.0]$.
6. **Calibration Reliability (`calibration_reliability_score`)**: Pitch calibration reliability score based on reprojection error $[0.0, 1.0]$ (with `calibration_reprojection_error_m`).
7. **Ball Observability (`ball_observability_score`)**: Ball detection coverage, continuity, and tracking confidence $[0.0, 1.0]$.
8. **Image Quality (`image_quality_score`)**: Visual sharpness composite score $[0.0, 1.0]$ (with `blur_score_laplacian`).
9. **Pose Quality (`pose_quality_score`)**: Keypoint detection confidence and crop resolution $[0.0, 1.0]$ (conditional on SCAN2PLAY).

### 4.2 Task-Specific Readiness Gating
Each analytical module defines minimum evidence criteria and receives one of three operational states:

| Module | Minimum Evidence Required | Possible Readiness States | Fallback Behavior |
| :--- | :--- | :--- | :--- |
| **PRESSUREGRAPH** | Ball tracked, possession inferred, $\ge 3$ defending players tracked within 15 m of ball carrier. | `ready`, `partial`, `insufficient` | Suppress pressing attribution if presser is out-of-view; report episode with reduced confidence. |
| **TACTICAL SHIFT** | $\ge 8$ outfield players tracked per team, continuous pitch coverage $\ge 60\%$, stable calibration over 30 s. | `ready`, `partial`, `insufficient` | Suspend change-point detection during long camera cuts or close-ups; flag timeline gap. |
| **ROLESHIFT** | Consistent player identity tracked across $\ge 15$ continuous minutes in valid camera shots. | `ready`, `limited`, `insufficient` | Restrict archetype profile to high-confidence match windows; report sample duration. |
| **SCAN2PLAY** | Player crop $\ge 80 \times 80$ px, keypoint confidence $\ge 0.7$ on neck/shoulders across 2 s prior to reception. | `ready`, `pilot-only`, `insufficient` | Suppress orientation proxy; mark reception event as unobserved pose. |

### 4.3 Quality-Aware Model Architecture
- **Pre-Softmax Masking**: In attention heads or graph edges, missing or out-of-view nodes are masked **prior** to softmax normalization.
- **Selective Prediction**: Analytical heads produce an uncertainty/evidence score. Predictions below calibrated thresholds are withheld from summary metrics.

---

## 5. Canonical Tracking Data Model

The canonical schema represents the strict decoupling boundary between computer vision and tactical analytics.

### 5.1 Core Tables & Grain

```
┌─────────────────────────────────┐           ┌─────────────────────────────────┐
│         match_metadata          │           │         video_segments          │
├─────────────────────────────────┤           ├─────────────────────────────────┤
│ match_id (PK)                   │1         *│ segment_id (PK)                 │
│ home_team_id, away_team_id      ├───────────┤ match_id (FK)                   │
│ pitch_length, pitch_width       │           │ start_timestamp, end_timestamp  │
│ source_provider, source_rights  │           │ camera_class, quality_vector    │
└────────────────┬────────────────┘           └─────────────────────────────────┘
                 │1
                 ├─────────────────────────────┐
                 │*                            │*
┌────────────────▼────────────────┐           ┌▼────────────────────────────────┐
│         player_frames           │           │          ball_frames            │
├─────────────────────────────────┤           ├─────────────────────────────────┤
│ match_id, period, timestamp (PK)│           │ match_id, period, timestamp (PK)│
│ player_id, team_id              │           │ x, y, z                         │
│ x, y (pitch metres)             │           │ visible_flag, tracking_status   │
│ vx, vy, ax, ay                  │           │ detection_confidence            │
│ visibility_status, observed_flag│           │ ball_state (ground/air/unknown) │
│ tracking_status, quality_score  │           └─────────────────────────────────┘
└─────────────────────────────────┘
```

### 5.2 Enumerations and Data Policies

#### Visibility Status
- `visible`: Player directly detected in camera frame.
- `occluded`: Player temporarily occluded by another player or referee.
- `out_of_view`: Player located outside the current camera frustum.
- `unknown`: Visibility could not be determined.

#### Tracking Status & Observation Semantics
- `detected`: Position directly measured by detector (`observed_flag=True`).
- `estimated`: Short-term algorithmic motion estimation (`observed_flag=False`).
- `interpolated`: Gap filled between verified detections (`observed_flag=False`).
- `extrapolated`: Position projected beyond camera view or gap (`observed_flag=False`).
- `lost`: Track terminated or unrecoverable (`observed_flag=False`, coordinates must be `None`).
- `invalid`: Erroneous or filtered track (`observed_flag=False`, coordinates must be `None`).

#### Fixed Coordinate Standards & Provenance
- **Fixed Metric Pitch Frame**: Origin $(0.0, 0.0)$ at centre circle.
  - Longitudinal axis ($X$): Along pitch length in metres $[-L/2, +L/2]$. $+X$ points towards the right-hand goal as viewed from the main broadcast camera; $-X$ towards the left-hand goal.
  - Lateral axis ($Y$): Across pitch width in metres $[-W/2, +W/2]$. $+Y$ points towards the far touchline (top of broadcast screen); $-Y$ towards the near touchline.
  - Raw coordinates $(x, y)$ are permanently preserved and immutable.
- **Team-Relative Attacking Coordinates**: Derived features $(x_{attack}, y_{attack})$ pinned to a fixed period attacking direction (`team_attacking_direction` $+1$ or $-1$). They do **not** flip on individual possession turnovers.
- **Pitch Dimension Provenance**: Explicitly tracked via `pitch_dimension_source` as `measured` (laser/survey), `supplied` (competition metadata), or `assumed` (FIFA standard $105 \times 68$ m default).
- **Homography Direction & Planar Constraint**:
  - The homography matrix $H \in \mathbb{R}^{3 \times 3}$ maps image pixels $[u, v, 1]^T$ to pitch ground plane $[x, y, 1]^T$ (`image_to_pitch`).
  - Planar homography assumes all transformed points lie on the flat pitch surface ($Z = 0$). Airborne balls experience geometric parallax error; planar homography CANNOT reconstruct 3D ball elevation ($Z$).
- **Analytical Grid**: Tracking streams resampled to a uniform **10 Hz** grid using timestamp alignment.

---

## 6. Football Feature Engine

The feature engine generates multi-scale spatial and kinematic representations from canonical tracking data:

1. **Kinematics**: Player speed ($v = \sqrt{v_x^2 + v_y^2}$), instantaneous acceleration, directional heading angle, change of direction rate.
2. **Possession & Ball Context**: Euclidean distance to ball, candidate ball carrier, team in possession, ball speed and trajectory heading.
3. **Team Geometry**:
   - Convex hull area (team spread).
   - Team centroid $(C_x, C_y)$ and geometric dispersion.
   - Defensive line height ($x$-coordinate of the second-deepest defender).
   - Inter-line distances (defensive line to midfield line, midfield to forward line).
   - Team width (max $y$ - min $y$) and team length (max $x$ - min $x$).
4. **Local Interaction Graph**:
   - Nodes: 22 players + 1 ball node. Node features include normalized coordinates, velocities, team identity, and visibility status.
   - Edges: Dynamic spatial proximity ($\le 15$ m) for same-team coordination and opponent marking.
   - Edge attributes: Relative distance vector $(\Delta x, \Delta y)$, relative speed, passing lane obstruction flag.

---

## 7. Analytical Modules

### 7.1 PRESSUREGRAPH — Collective Pressing Intelligence
- **Objective**: Detect coordinated pressing episodes and identify the primary presser and supporting defenders.
- **Model Progression**:
  1. *Baseline 1 (Heuristic)*: Distance to ball carrier $< 3$ m, closing speed $> 3.5$ m/s, team compactness $< 400$ m$^2$.
  2. *Baseline 2 (XGBoost)*: Tabular feature vector over 3-second rolling window.
  3. *Advanced (GNN + Temporal)*: PyTorch Geometric GINEConv or GATv2 spatial layers combined with a GRU temporal encoder.
- **Annotation Codebook**: Episode start (initiation of closing movement), episode end (regain, escape, foul, dead ball), primary presser (most direct pressure applied), supporting defenders (closing outlets/lanes).

### 7.2 TACTICAL SHIFT — Tactical Behavior & Shape Changes
- **Objective**: Identify sustained structural adjustments in team shape and line height across match phases.
- **Methodology**:
  - Compute rolling 30-second team geometry vectors (compactness, line heights, centroid).
  - Apply **PELT** (Pruned Exact Linear Time) penalized change-point detection to segment match into structural phases.
  - Cluster recurring structural states using Gaussian Mixture Models (GMM) to identify distinct defensive blocks (low block, mid block, high press).

### 7.3 ROLESHIFT — Behavioral Player Archetypes
- **Objective**: Discover functional player archetypes from trajectory movement profiles and provide similarity retrieval.
- **Methodology**:
  - Aggregate spatial heatmaps, defensive action zones, pressing participation, and progressive movement into a fixed-length player-match embedding.
  - Apply K-Means and GMM clustering to identify empirical archetypes (e.g. inverted fullback, box-to-box midfielder).
  - Cosine similarity retrieval constrained to within-provider comparisons to prevent coordinate-system confounds.

### 7.4 SCAN2PLAY — Orientation Proxy Before Ball Reception (Conditional Pilot)
- **Objective**: Measure observable head and body orientation adjustments during the 2–3 seconds prior to receiving a pass.
- **Go/No-Go Gate**: Executes **only** on clips where player pose keypoints pass quality and coverage thresholds. If coverage is insufficient, this module is suppressed and documented as infeasible for that segment.

---

## 8. Storage, Query, and Application Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             FastAPI Application                             │
├────────────────────────────────┬───────────────────────────┬────────────────┤
│          API Routers           │     Background Jobs       │ Data Services  │
│  - /api/matches (CRUD)         │  - Video Ingestion Worker │ - DuckDB Engine│
│  - /api/quality (Readiness)    │  - CV Inference Queue     │ - Parquet I/O  │
│  - /api/analytics (Modules)    │  - SQLite Job Queue       │ - Feature Store│
└────────────────┬───────────────┴─────────────┬─────────────┴────────────────┘
                 │                             │
    ┌────────────▼──────────┐     ┌────────────▼──────────┐
    │  SQLite Database      │     │  Parquet & DuckDB     │
    │  - match_metadata     │     │  - player_frames.pq   │
    │  - video_segments     │     │  - ball_frames.pq     │
    │  - processing_jobs    │     │  - Fast windowed SQL  │
    └───────────────────────┘     └───────────────────────┘
```

- **Metadata Tier (SQLite)**: Stores match metadata, video container properties, camera segments, and job execution states.
- **Analytical Storage Tier (Apache Parquet + DuckDB)**: High-frequency tracking data stored in columnar Parquet files partitioned by `match_id`. DuckDB executes zero-copy windowed analytical queries directly over Parquet files.
- **Web Frontend (React + TypeScript)**: Interactive tactical pitch visualization using Canvas/SVG. Streams windowed tracking chunks on demand rather than loading entire matches into browser memory.

