# PITCHMIND — DEVELOPMENT ROADMAP & GATES

**Document Version**: 2.0  
**Status**: Architecture Baseline  
**Date**: 11 November 2026  

---

## 1. 28-Week Master Execution Roadmap

The 28-week roadmap is engineered as an ambitious, disciplined research-prototype schedule. It prioritizes the end-to-end video-to-tracking-to-analysis path from the outset, establishing reproducible baselines before introducing advanced neural architectures.

```
Weeks: 01-02   03-05   06-08   09-12   13-16   17-20   21-23   24-26   27-28
       ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
Phase: │  1  │ │  2  │ │  3  │ │  4  │ │  5  │ │  6  │ │  7  │ │  8  │ │  9  │
       └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘
Gate:    G1      G2      G3      G4      G5      G6      G7       -     Final
```

| Phase | Weeks | Focus Area | Key Deliverables & Exit Criteria |
| :--- | :--- | :--- | :--- |
| **1** | 1–2 | **Rights, Environment & Schemas** | - Verify video source rights and dataset licenses.<br>- Establish virtual environment (`.venv`) and verify hardware CUDA availability.<br>- Define and test canonical tracking schemas (`player_frames`, `ball_frames`).<br>- Exit Gate: **G1 (Rights & Access)**. |
| **2** | 3–5 | **Video Ingestion & Preflight** | - FFmpeg/FFprobe decode preflight and container integrity scanner.<br>- Camera cut detection and initial shot segmentation.<br>- Build SkillCorner and IDSSE provider adapters with 10 Hz canonical resampling.<br>- Generate initial video quality report.<br>- Exit Gate: **G2 (Ingestion Integrity)**. |
| **3** | 6–8 | **CV Prototype & Pitch Replay** | - Integrate YOLOX detection baseline (players, ball, referees).<br>- Integrate ByteTrack multi-object tracker.<br>- Implement initial landmark-based pitch calibration homography.<br>- Build lightweight Canvas/SVG pitch replay in React to verify CV tracks.<br>- Exit Gate: **G3 (Tracking Baseline)**. |
| **4** | 9–12 | **Feature Engine & Heuristic Baselines** | - Implement kinematic, team geometry, and spatial control feature pipelines.<br>- Write PRESSUREGRAPH annotation codebook and label pilot episode set.<br>- Implement heuristic and XGBoost baselines for PRESSUREGRAPH.<br>- Implement rolling team geometry features and PELT baseline for TACTICAL SHIFT.<br>- Exit Gate: **G4 (Calibration Accuracy)** & **G5 (Baseline Verification)**. |
| **5** | 13–16 | **Core Analytical Models** | - Implement interaction graph construction (PyTorch Geometric).<br>- Train and evaluate GINEConv / GATv2 + GRU temporal graph models.<br>- Evaluate TACTICAL SHIFT change-point detection on full matches.<br>- Adjudicate multi-annotator disagreements on test splits.<br>- Exit Gate: **G5 (Model Performance)**. |
| **6** | 17–20 | **Robustness & Quality-Aware Research** | - Execute the core research matrix: Experiments **E1 through E7**.<br>- Evaluate model degradation under simulated dropout, spatial noise, and cuts.<br>- Implement selective prediction thresholds and evaluate risk-coverage curves.<br>- Perform cross-provider generalization testing (SkillCorner $\leftrightarrow$ IDSSE).<br>- Exit Gate: **G6 (Robustness Validation)**. |
| **7** | 21–23 | **Secondary Modules & Pilots** | - Evaluate ROLESHIFT behavioral archetype clustering and similarity retrieval.<br>- Evaluate SCAN2PLAY readiness gate: check pose keypoint coverage on reception windows.<br>- If pose coverage is insufficient, document SCAN2PLAY as infeasible and defer.<br>- Exit Gate: **G7 (Secondary Module Readiness)**. |
| **8** | 24–26 | **Product Hardening & Integration** | - Connect FastAPI backend, SQLite job queue, and DuckDB analytical queries.<br>- Implement chunked tracking streaming in React frontend.<br>- Integrate quality and uncertainty warning overlays across the UI.<br>- Implement structured JSON and CSV report export. |
| **9** | 27–28 | **Final Evaluation & Demonstration** | - Complete held-out match-level cross-validation.<br>- Compile research technical paper and ablation report (documenting negative results).<br>- Prepare end-to-end demonstration using one tactical clip and one broadcast clip.<br>- Final deliverable archive and reproducibility package. |

---

## 2. Formal Go/No-Go Exit Gates

Each milestone contains an explicit gate. If a gate's conditions are not met, the predefined fallback or scope reduction must be enacted immediately.

| Gate | Title | Proceed When... | If Not Met (Fallback Action) |
| :--- | :--- | :--- | :--- |
| **G1** | **Rights & Access** | All video test clips and external datasets have documented permissions in the Data Rights Register. | Replace source footage with verified open clips; exclude unverified sources from distribution. |
| **G2** | **Ingestion Integrity** | Video decode and data adapters produce repeatable timestamps, frame dimensions, and canonical records without silent drift. | Fix resampling and container timestamp handling before initiating any ML or CV tasks. |
| **G3** | **Tracking Baseline** | Detection and tracking baselines produce usable player tracklets and pitch coordinates on benchmark clips with quantified error. | Narrow supported camera classes to wide tactical views; do not build tactical claims on broken tracks. |
| **G4** | **Calibration Accuracy** | Landmark reprojection error is $< 1.5$ m on key camera segments with valid field lines. | Fall back to image-space tracking overlays; strictly suppress metric pitch features (metres, speeds). |
| **G5** | **Baseline Performance** | PRESSUREGRAPH and TACTICAL SHIFT beat trivial random/majority baselines on held-out matches. | Revise feature definitions, tighten annotation codebooks, or narrow the prediction task scope. |
| **G6** | **Robustness Research** | Quality-aware training demonstrates superior calibration or improved risk-coverage trade-off on degraded inputs. | Honestly document the negative result in the research report; do not claim unwarranted improvement. |
| **G7** | **Secondary Modules** | SCAN2PLAY passes pose coverage check ($\ge 80 \times 80$ px crops, high keypoint confidence across receptions). | Formally defer SCAN2PLAY as infeasible; document limitations and allocate remaining time to core modules. |

---

## 3. First Milestone Deliverable (Immediate Exit Gate)

The foundational milestone of PITCHMIND is:

> **Milestone 1 Deliverable**: A video file can be uploaded, automatically inspected for decode integrity and camera cuts, processed into an initial canonical tracking representation, and replayed on a 2D pitch radar with visible confidence and coverage indicators.

Advanced neural networks (GNNs, multi-task backbones) are barred until this end-to-end foundational loop is operating reliably.

