# PITCHMIND — THIRD-PARTY NOTICES & LICENSE REGISTER

**Document Version**: 2.0  
**Status**: Architecture Baseline  
**Date**: 11 November 2026  

This register documents all identifiable open-source software libraries, pretrained models, datasets, and systems utilized by or referenced in PITCHMIND. In accordance with project policy, all license details reflect verified repository snapshots as of 10 November 2026; unverified entries are marked explicitly.

---

## 1. Core Software Libraries & Tooling

| Software / Component | Repository / Source | Verified License | Intended Use in PITCHMIND | License Notes & Conditions |
| :--- | :--- | :--- | :--- | :--- |
| **Kloppy** | `https://github.com/PySport/kloppy` | **BSD-3-Clause** | Tracking and event data loading, standardization, and normalization. | Permissive; retain copyright notice and disclaimer in redistributed source. |
| **YOLOX** | `https://github.com/Megvii-BaseDetection/YOLOX` | **Apache-2.0** | Object detection baseline (players, ball, referees). | Permissive; include Apache license and notice. Checkpoint weights require separate verification. |
| **ByteTrack** | `https://github.com/FoundationVision/ByteTrack` | **MIT License** | Multi-object tracking baseline (tracklet association). | Permissive; retain copyright and permission notice. |
| **TVCalib** | `https://github.com/MM4SPA/tvcalib` | **MIT License** | Camera calibration baseline for soccer pitch homography estimation. | Permissive source code; bundled semantic segmentation weights have separate provenance. |
| **PyTorch Geometric** | `https://github.com/pyg-team/pytorch_geometric` | **MIT License** | Graph neural network implementation (GINEConv, GATv2, temporal graph models). | Permissive; standard PyTorch ecosystem library. |
| **Accessible Space** | `https://github.com/jonas-bischofberger/accessible-space` | **MIT License** | Reference implementation for player accessible space and pitch control features. | Permissive; document underlying spatial modeling assumptions. |
| **mplsoccer** | `https://github.com/andrewRowlinson/mplsoccer` | **MIT License** | Pitch visualization, coordinate mapping, heatmaps, and tactical radar plotting. | Permissive; standard football analytics visualization library. |
| **OpenCV** | `https://github.com/opencv/opencv` | **Apache-2.0** | Image and video manipulation, keyframe rendering, geometric transformations. | Permissive; review build configuration for patented codec flags. |
| **FFmpeg** | `https://ffmpeg.org/` | **LGPL-2.1+ / GPL-2.0+** | Video decoding, container inspection, keyframe extraction, and transcoding. | License depends on build flags (e.g. `--enable-gpl`, `--enable-nonfree`). Review build specifics. |
| **DuckDB** | `https://github.com/duckdb/duckdb` | **MIT License** | Analytical SQL engine for zero-copy windowed queries over tracking Parquet files. | Permissive; embedded columnar database. |
| **MLflow** | `https://github.com/mlflow/mlflow` | **Apache-2.0** | Experiment tracking, model metrics logging, and artifact management. | Permissive; used for tracking ablation experiment runs. |

---

## 2. Research Datasets & External Resources

| Dataset / Source | Origin & Access URL | Verified License Status | Permitted Scope in PITCHMIND | Critical Compliance Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **SkillCorner Open Data** | `https://github.com/SkillCorner/opendata` | **MIT License** | Research adapter development, feature engineering, and model training. | README requests attribution/credit. Tracking is CV-derived (10 Hz). |
| **IDSSE Tracking Data** | `https://huggingface.co/datasets/pysport/idsse-data` | **CC BY 4.0** | Cross-provider transfer testing (Ablation E7) and coordinate normalization. | Attribution required. 25 Hz optical tracking downsampled to 10 Hz canonical grid. |
| **Metrica Sports Sample Data** | `https://github.com/metrica-sports/sample-data` | **Unverified License** (No root LICENSE confirmed in snapshot) | Initial adapter testing and coordinate sanity checks. | README requests public acknowledgement. Do not redistribute without explicit permission. |
| **SoccerNet Tracking Kit** | `https://github.com/SoccerNet/sn-tracking` | **Unverified License** (Code license unconfirmed; benchmark terms) | Multi-object tracking evaluation (HOTA metric). | Benchmark data terms are separate from devkit code. Requires academic benchmark registration. |
| **SoccerNet-GAR** | `https://github.com/OpenSportsLab/SoccerNet-GAR` | **CC BY-NC 4.0** (Non-Commercial) | Optional group activity reference baseline. | Restricted to non-commercial research use only; user must accept access terms. |
| **SkillCorner Body Pose** | `https://huggingface.co/datasets/SkillCorner/opendata-bodypose` | **MIT License** (Dataset Card) | Conditional pilot for SCAN2PLAY orientation proxy analysis. | Initial two-match test release. 25 FPS pose vs 10 FPS tracking; coverage gaps expected. |

---

## 3. Compliance Rules for Contributors

1. **Verification Before Adoption**: Any new dependency added during later phases must have its repository URL, pinned commit/tag, and license file verified before inclusion.
2. **Attribution File Maintenance**: When building the frontend and backend release artifacts, retain third-party notices in distribution packages.
3. **No Commercial Redistribution of Media**: Never bundle or redistribute third-party match footage, extracted image frames, or non-permissive datasets.

