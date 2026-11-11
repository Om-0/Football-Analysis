# PITCHMIND — DATA STRATEGY, RIGHTS & LICENSING (docs/DATA_AND_LICENSING.md)

**Document Version**: 2.0  
**Status**: Architecture Baseline  
**Date**: 11 November 2026  

---

## 1. Rights-Based Video Footage Policy

PITCHMIND operates under a strict **rights-based video footage framework**:

1. **User Authorization**: Any user uploading video to the platform must affirm they hold lawful authorization, broadcast rights, or internal club permissions to process the footage.
2. **No Unrestricted Public Scraping**: PITCHMIND will not implement automated scraping, streaming extraction, or DRM-bypass tools for commercial broadcast television.
3. **No Unauthorized Redistribution**: The system will never redistribute raw match footage, extracted image frames, or proprietary commercial tracking data. Model weights, benchmark manifests, and open code are distributed; raw commercial media is kept local and private.
4. **Data Rights Register**: All video clips and external tracking files utilized during development, integration testing, and evaluation must have a documented entry in `datasets/manifests/data_rights_register.json`.

---

## 2. Research & Benchmark Datasets

The table below records the verified status of the external research datasets used for development, canonical adapter integration, and model benchmarking. **Note**: License terms reflect verified repository snapshots; unverified details are marked explicitly.

| Dataset / Resource | Description & Content | Provenance / URL | Verified License Status | Permitted Use in PITCHMIND | Limitations & Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SkillCorner Open Data** [R1, R2] | 10 A-League 2024/25 matches of 10 Hz broadcast-derived tracking, dynamic events, and match phases. | `github.com/SkillCorner/opendata` | **MIT License** (README requests attribution/credit). | Primary tracking adapter development; feature engineering; PRESSUREGRAPH & TACTICAL SHIFT baseline training. | Small match sample (10 games); tracking is CV-derived with extrapolated flags; provider event labels require independent validation. |
| **IDSSE Data** [R3] | 7 complete Bundesliga/2.Bundesliga 2022/23 matches; all-player and ball tracking at 25 FPS with synchronized events. | `huggingface.co/datasets/pysport/idsse-data` | **CC BY 4.0** | Cross-provider transfer testing (Ablation E7); evaluation of coordinate normalization across provider systems. | Different provider capture technology and sampling rate (25 Hz downsampled to 10 Hz canonical grid); keep held-out from model tuning. |
| **Metrica Sports Sample Data** [R4] | Three anonymized sample matches with synchronized tracking and event data. | `github.com/metrica-sports/sample-data` | **Unverified License** in repository root; README requests public acknowledgement. | Adapter testing, pitch coordinate sanity checks, and 2D replay prototyping. | Redistribution prohibited without explicit permission; verify terms before any commercial application. |
| **SoccerNet Tracking Kit** [R5, R6] | 12 complete games from main camera; benchmark of 100 30-second 1080p clips with tracklet annotations. | `soccer-net.org/tasks/tracking`, `github.com/SoccerNet/sn-tracking` | **Academic Benchmark Terms**; devkit code license unverified at root. | Benchmarking player/ball detection and multi-object tracking accuracy (HOTA metric). | Benchmark data access requires registration; data terms are separate from devkit code; not for commercial redistribution. |
| **SoccerNet-GAR** [R7] | 87,939 annotated group activities across 64 FIFA World Cup 2022 matches; 4.5 s synchronized windows. | `huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR` | **CC BY-NC 4.0** (Non-Commercial) | Optional group activity reference; benchmark comparison for collective actions. | Strictly non-commercial; user must agree to access terms; event classes do not match pressing/shift definitions. |
| **SkillCorner Body Pose** [R8, R9] | Two-match initial release; 29 joints per detected player at 25 FPS. | `huggingface.co/datasets/SkillCorner/opendata-bodypose` | **MIT License** (Dataset Card) | Conditional pilot for SCAN2PLAY orientation proxy analysis. | Highly limited sample (2 matches); 25 FPS pose vs 10 FPS tracking; only detected players have pose; $z$-joint relative to centroid, not pitch elevation. |

---

## 3. Dataset Integrity & Splitting Discipline

1. **Match-Level Split Rule**: Never randomly split frames, windows, or episodes from the same match across training and testing splits. Split strictly by **match identity** to prevent data leakage from shared tactical styles, camera conditions, and opponent dynamics.
2. **Grouped Cross-Validation**: Use Leave-One-Match-Out (LOMO) or grouped K-fold cross-validation across the available 10 SkillCorner matches.
3. **Cross-Provider Generalization**: The 7 IDSSE matches are treated as an external held-out test suite (Ablation E7) to evaluate how well models trained on CV-derived tracking transfer to optical tracking.
4. **Storage Separation**:
   - Raw video files and tracking CSVs/JSONs are stored in `datasets/raw/` (ignored by Git via `.gitignore`).
   - Intermediate processed parquet files are stored in `datasets/processed/` (ignored by Git).
   - Only schema definitions, ingestion adapters, and metadata manifests (`datasets/manifests/`) are tracked in version control.

---

## 4. Academic & Pretrained Model Discipline

The project baseline assumes that **pretrained perception backbones** (e.g. YOLOX detector weights, ByteTrack association heuristics) may be used as engineering starting points for the computer-vision engine, while PITCHMIND's tactical intelligence models (PRESSUREGRAPH, TACTICAL SHIFT) are trained, ablated, and evaluated directly by this project.

*Academic Constraint Check*: If an academic institution or evaluation panel mandates that perception models must also be trained from scratch, that represents a major scope expansion requiring explicit adjustment of the CV milestones.

