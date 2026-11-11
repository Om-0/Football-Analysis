# PITCHMIND — PROJECT BRIEF

**Document Version**: 2.0  
**Status**: Architecture Baseline  
**Date**: 11 November 2026  

---

## 1. Product Statement

PITCHMIND is a **video-first football tactical intelligence platform**. A user uploads a match recording (either standard broadcast footage or wide/full-pitch tactical footage). The platform automatically classifies camera shots, assesses video quality, detects and tracks players and the ball, projects player coordinates onto standard pitch dimensions, constructs a canonical tracking representation, and executes four tactical and behavioral analytical modules:

1. **PRESSUREGRAPH**: Identification of coordinated pressing episodes, primary presser attribution, and supporting defensive structure.
2. **TACTICAL SHIFT**: Detection of sustained collective team-shape changes, defensive line shifts, and compactness transitions.
3. **ROLESHIFT**: Player behavioral archetype exploration and similarity retrieval within available datasets.
4. **SCAN2PLAY**: Pre-reception orientation adjustments and scan proxies (strictly conditional on verified pose coverage).

PITCHMIND is architected as a **local-first web application** (FastAPI backend, React frontend, Parquet/DuckDB analytics engine) engineered to operate within resource-constrained hardware environments.

---

## 2. Primary Users & Jobs-to-be-Done (JTBD)

| User Persona | Primary Job-to-be-Done | Key PITCHMIND Output |
| :--- | :--- | :--- |
| **Tactical Analyst / Coach** | Review opponent pressing schemes, defensive blocks, and transition triggers from match video without access to optical tracking providers. | Pressing episode timeline, primary/support defender attribution, pitch replay with uncertainty highlights, and tactical shift segmentations. |
| **Performance Analyst** | Inspect physical structure, team compactness, line heights, and defensive spacing across match phases. | Rolling team geometry metrics (width, length, centroid, inter-line distance), tracking overlays, and coverage indicators. |
| **Scout / Recruitment Analyst** | Discover player movement profiles and identify stylistically similar players within tracked competitions. | Multi-dimensional behavioral archetypes, radar profiles, and cosine/metric similarity retrieval with explicit sample constraints. |
| **Sports Science Researcher** | Investigate football analytics models and evaluate algorithm robustness under missing, occluded, or noisy tracking. | Reproducible datasets, open benchmarking pipelines, ablation experiment suites (E1–E7), and quality-calibrated selective outputs. |

---

## 3. Scope Priority & Development Boundaries

| Priority Level | Capability / Module | Initial Scope Commitment |
| :--- | :--- | :--- |
| **Core Product** | Video ingestion, preflight decode, shot segmentation, and CV tracking | Required end-to-end product baseline for both broadcast and tactical footage. |
| **Core Research** | Quality-aware learning under partial observability | Primary research question: evaluate selective prediction and robustness under degradation (Matrix E1–E7). |
| **Core Analytics** | PRESSUREGRAPH | Heuristic baseline $\to$ XGBoost baseline $\to$ Spatio-temporal GNN (GINEConv/GATv2 + GRU). |
| **Core Analytics** | TACTICAL SHIFT | Rolling team-state features $\to$ PELT penalized change-point detection $\to$ clustering of recurring states. |
| **Limited / Exploratory** | ROLESHIFT | Within-provider player archetype clustering (K-Means/GMM) and similarity retrieval. No transfer claims. |
| **Conditional Pilot** | SCAN2PLAY | Pre-reception head/body orientation proxy. Proceeds **only** if pose keypoint coverage passes the readiness gate. |
| **Deferred** | End-to-end multimodal video-tracking fusion | Optional future experiment; not a dependency for baseline v2.0 deliverables. |
| **Deferred** | Multi-tenant cloud infrastructure & native installers | Local-first prototype prioritizes researcher usability and low overhead over enterprise packaging. |

---

## 4. Video Input Contract & Rights-Based Footage Policy

### 4.1 Required Inputs & Formats
- Supported containers: MP4, MOV, MKV (subject to installed FFmpeg codecs).
- Minimum resolution: 720p (1080p preferred).
- Frame rate: 25–60 FPS.
- Continuous match footage preferred; edited highlight reels are restricted to localized action analysis and rejected for full-match structural claims.

### 4.2 Footage Classification Policy
- **Full-Pitch Tactical Camera**: Wide, elevated view capturing 18–22 players simultaneously. Preferred for full-pitch team shape, compactness, and collective pressing analysis.
- **Standard Broadcast Footage**: Dynamic pan-tilt-zoom camera tracking the ball carrier. Players off-screen are unobservable; analysis coverage is segment-dependent.
- **Multi-Camera / Replays**: Replays and slow-motion cuts are isolated via shot segmentation and excluded from the live match timeline.

### 4.3 Rights-Based Footage Policy
PITCHMIND enforces a strict **rights-based footage policy**:
1. Users must confirm they hold lawful authorization or valid rights to process uploaded footage.
2. Public availability of video streams or highlights does **not** confer processing, reproduction, or redistribution rights.
3. The platform strictly prohibits unauthorized redistribution of processed video, extracted frames, or proprietary tracking data.
4. All third-party test footage and research datasets must have documented provenance in the project Data Rights Register.

---

## 5. Product Outputs

1. **Match Overview & Integrity Report**: Container metadata, duration, frame rate, camera segment breakdown, and overall pitch visibility profile.
2. **Interactive Match Replay**: Pitch radar view synchronized with video playback, showing tracked players, ball position, motion vectors, and data confidence indicators.
3. **PRESSUREGRAPH Dashboard**: Pressing episode scrubber, primary presser identification, supporting defenders, passing-lane obstruction evidence, and outcome classification (regain, escape, forced pass).
4. **TACTICAL SHIFT Dashboard**: Team-state timeline, detected structural change points, line-height visualizations, and before/after phase comparisons.
5. **ROLESHIFT Profile View**: Player behavioral archetype classifications and similarity retrieval across tracked player-matches.
6. **SCAN2PLAY Event Viewer** *(Conditional)*: Orientation adjustment events leading into ball receptions, displayed with pose confidence masks.
7. **Exportable Intelligence Dossier**: Structured JSON and CSV exports documenting predictions, model versions, evidence intervals, and quality limitations.

---

## 6. Explicit Non-Goals for Research Prototype v2.0

To ensure engineering focus and research rigor, the following capabilities are explicitly declared **non-goals**:

- **No Perfect All-22 Tracking from Broadcast**: Broadcast video cuts and close-ups cannot observe off-screen players. The system will never hallucinate or invent unobserved player trajectories.
- **No Causal Attribution of Tactical Outcomes**: Models identify statistical association and observable coordination, not counterfactual or causal tactical blame.
- **No Gaze Tracking or Awareness Claims**: Body and head orientation proxies do not measure visual attention, gaze fixation, or mental awareness.
- **No Automated Commercial Player Scouting / Valuation**: ROLESHIFT evaluates descriptive behavioral similarity within available datasets, not predictive transfer success or monetary valuation.
- **No Unrestricted Public Scraping**: The project will not implement automated scrapers or bypass DRM/licensing mechanisms for commercial broadcasts.
- **No Monolithic Multi-Task Neural Backbone**: Independent module heads will be proven first; a shared backbone will not be introduced without positive transfer ablation evidence.

---

## 7. Definition of Done for v2.0 Research Prototype

The PITCHMIND v2.0 research prototype is defined as complete when:

1. **End-to-End Pipeline Execution**: An authorized user can upload a video and receive an automatic quality inspection, camera segmentation, and canonical tracking dataset.
2. **Dual-Route Verification**: At least one wide tactical clip and one standard broadcast clip have been successfully processed end-to-end, with explicit documentation of coverage differences.
3. **Core Analytical Baselines**: PRESSUREGRAPH and TACTICAL SHIFT have established heuristic/ML baselines evaluated on match-level held-out splits.
4. **Research Matrix Execution**: Robustness experiments E1–E7 are executed (or omitted with documented rationale), demonstrating the impact of quality-aware training on degraded inputs.
5. **Calibrated Selective Output**: Downstream analytical modules suppress or downgrade outputs during low-quality intervals, demonstrating a superior risk-coverage trade-off.
6. **Licensing & Provenance Compliance**: All code, model weights, and benchmark datasets have verified entries in the license and data rights registers.
7. **Comprehensive Technical Dossier**: A final technical report documents both positive and negative results, failure cases, and explicit operational limitations.

