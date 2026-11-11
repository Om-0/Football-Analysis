# PITCHMIND

**Video-first football tactical intelligence and player behaviour analysis platform**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Architecture: v2.0](https://img.shields.io/badge/architecture-v2.0--baseline-green.svg)](docs/ARCHITECTURE.md)

---

## Overview

PITCHMIND is an advanced football analytics platform designed around a **video-first** architecture. Rather than relying exclusively on costly proprietary optical tracking data, PITCHMIND allows users to upload standard broadcast or wide tactical camera footage. The platform automatically assesses video quality, detects and tracks players and the ball, reconstructs 2D pitch coordinates, and produces deep tactical intelligence across four distinct analytical modules:

1. **PRESSUREGRAPH** — Collective pressing intelligence, primary presser identification, and supporting defensive structure.
2. **TACTICAL SHIFT** — Detection of sustained team-shape changes, defensive line adjustments, and compactness shifts.
3. **ROLESHIFT** — Behavioral player archetypes and similarity retrieval across matches (within-provider exploratory analysis).
4. **SCAN2PLAY** — Pre-reception orientation proxy analysis (conditional on verified pose keypoint coverage).

### Core Research Objective
Traditional football tactical analytics assume complete, clean, 25 Hz all-player tracking. In reality, broadcast and tactical video introduce severe challenges: **occlusions, out-of-view players, camera cuts, identification switches, and calibration drift**. PITCHMIND's central research objective is:

> *Does explicitly modeling observation quality and partial observability improve football tactical predictions and their reliability compared with conventional tracking-only models?*

---

## System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │          User Uploads Football Video         │
                    │   (Standard Broadcast or Tactical Camera)    │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │  1. Video Ingestion & Preflight Inspection   │
                    │   (FFmpeg metadata, frame rate, continuity)  │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │  2. Shot Segmentation & Camera Classification│
                    │   (Wide Tactical, Broadcast, Replay, Close)  │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │  3. Multidimensional Quality Assessment      │
                    │ (Proposed 9-D candidates requiring validation│
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │  4. Computer Vision Reconstruction Engine    │
                    │   (Detection -> Tracking -> Pitch Calib)     │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │  5. Canonical Tracking Engine (Boundary)     │
                    │  (PlayerFrame, BallFrame, Quality Masks)     │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │  6. Football Feature Engineering             │
                    │   (Kinematics, Team Geometry, Graph Edges)   │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌──────────────────┬──────────────┴─────┬──────────────────┐
         │                  │                    │                  │
┌────────▼────────┐ ┌───────▼────────┐  ┌────────▼────────┐ ┌───────▼────────┐
│ PRESSUREGRAPH   │ │ TACTICAL SHIFT │  │   ROLESHIFT     │ │   SCAN2PLAY    │
│ Pressing GNN    │ │ PELT Detection │  │ Archetype Clust.│ │ Pose Proxy     │
└────────┬────────┘ └───────┬────────┘  └────────┬────────┘ └───────┬────────┘
         │                  │                    │                  │
         └──────────────────┴──────────────┬─────┴──────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │  7. FastAPI Service & React Visualization    │
                    │  (Windowed DuckDB queries, interactive pitch)│
                    └──────────────────────────────────────────────┘
```

---

## Analytical Modules

| Module | Purpose | Model Baseline | Advanced Architecture | Readiness Gate |
| :--- | :--- | :--- | :--- | :--- |
| **PRESSUREGRAPH** | Coordinated pressing episodes & defender involvement | Heuristic rules + XGBoost | Spatio-temporal GNN (GINEConv / GATv2 + GRU) | Ball + local players tracked |
| **TACTICAL SHIFT** | Sustained changes in team structure & lines | Rolling spatial features + PELT | Penalized change-point segmentation | Wide pitch coverage |
| **ROLESHIFT** | Behavioral archetype discovery & similarity | K-Means & GMM | Metric learning embeddings | Multi-match player tracks |
| **SCAN2PLAY** | Pre-reception orientation adjustments | Rule-based orientation | Temporal 1D CNN / GRU | Verified pose keypoint coverage |

---

## Quality Assessment & Readiness Strategy

Quality assessment in PITCHMIND is **multidimensional** and **task-specific**:
- Rather than calculating a single opaque score, PITCHMIND evaluates nine candidate quality dimensions: *Video Integrity, Pitch Coverage, Player Visibility, Detection Reliability, Identity Continuity, Calibration Reliability, Ball Observability, Image Quality, and Pose Quality*.
- **Important Design Note**: These quality dimensions and their downstream readiness thresholds (`ready`, `partial`, `insufficient`) represent **proposed architectural design candidates** that require calibration and empirical validation against ground truth during development.
- When evidence is insufficient, the system selectively suppresses or downgrades downstream metrics rather than fabricating certainty.

---

## Rights-Based Footage Policy & Licensing

PITCHMIND operates under a strict **rights-based footage framework**:
- Users must confirm they have the necessary rights or authorization to process uploaded video.
- Public availability of video does not imply reuse, redistribution, or commercial rights.
- Unauthorized redistribution of match video or proprietary tracking datasets is prohibited.
- All external dependencies and benchmark datasets are tracked in a verified register (`third_party_notices/LICENSES.md` and `datasets/manifests/data_rights_register.json`).

---

## Hardware Baseline & Local Execution

PITCHMIND is engineered to run locally on resource-constrained hardware:
- **GPU**: NVIDIA RTX 3050 Laptop GPU (4 GB VRAM)
- **CPU**: Intel Core i5-12500H (12 cores, 16 threads)
- **RAM**: 16 GB DDR4/DDR5
- **Isolated Environment**: Dedicated virtual environment (`.venv`) isolating all project dependencies.

### Optimization Principles
1. **Windowed Chunking**: Process video in short sliding intervals rather than decoding entire matches into memory.
2. **Intermediate Caching**: Cache detections, tracklets, and feature matrices to avoid recomputing expensive CV stages.
3. **Small Batch Sizes**: Constrain neural inference batch size to 1–2 with optional gradient accumulation.
4. **Selective Querying**: Use DuckDB and Parquet for zero-copy, memory-efficient columnar queries over tracking data.

---

## Repository Structure

```
pitchmind/
├── AGENTS.md                  # Instructions and constraints for coding agents
├── README.md                  # Project overview and quickstart
├── pyproject.toml             # Python build configuration and metadata
├── configs/                   # Configuration YAMLs for CV and feature engines
├── datasets/
│   └── manifests/             # Dataset checksums and data-rights register
├── docs/                      # Core architectural documentation
│   ├── PROJECT_BRIEF.md       # Scope, personas, and definition of done
│   ├── ARCHITECTURE.md        # Technical architecture specifications
│   ├── ROADMAP.md             # 28-week development timeline & exit gates
│   ├── TECHNICAL_DECISIONS.md # Architectural Decision Records (ADRs)
│   └── DATA_AND_LICENSING.md  # Rights register, dataset terms & compliance
├── src/
│   └── pitchmind/
│       ├── ingestion/         # Video preflight and tracking data adapters
│       ├── video_intelligence/# Detection, tracking, calibration, quality
│       ├── data_engine/       # Canonical schema, feature engine, graphs
│       ├── models/            # Analytical module implementations
│       ├── evaluation/        # Metrics, ablations, and validation suites
│       └── visualization/     # Pitch rendering and uncertainty overlays
├── backend/                   # FastAPI endpoints, background jobs, storage
├── frontend/                  # React + TypeScript web application
├── tests/                     # Unit and integration test suites
└── third_party_notices/       # License register and attribution notices
```

---

## Documentation Links

- **[Project Brief](docs/PROJECT_BRIEF.md)** — Personas, scope boundaries, non-goals, and definition of done.
- **[System Architecture](docs/ARCHITECTURE.md)** — In-depth technical specifications and data schemas.
- **[Development Roadmap](docs/ROADMAP.md)** — Phased 28-week roadmap and formal go/no-go exit gates.
- **[Technical Decisions](docs/TECHNICAL_DECISIONS.md)** — Architecture Decision Records (ADRs) and design rationale.
- **[Data & Licensing](docs/DATA_AND_LICENSING.md)** — Rights-based footage policy and verified dataset registers.
- **[Third-Party Notices](third_party_notices/LICENSES.md)** — Open-source license register and dependency provenance.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details. Third-party components and datasets retain their respective licenses (see `third_party_notices/LICENSES.md`).
