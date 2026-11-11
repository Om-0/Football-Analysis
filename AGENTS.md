# PITCHMIND — AGENT & CONTRIBUTOR GUIDELINES (AGENTS.md)

This document establishes the binding rules of engagement, engineering standards, and architectural constraints for any automated agent or human contributor working on the **PITCHMIND** codebase.

---

## 1. Project Mission & Fixed Architecture Baseline

PITCHMIND is a **video-first football tactical intelligence platform**. Users upload match video (standard broadcast or wide tactical camera). The platform automatically evaluates video quality, tracks players and the ball, projects coordinates onto the pitch plane, and generates tactical insights through four analytical modules:
1. **PRESSUREGRAPH**: Collective pressing intelligence, primary presser identification, and support structure.
2. **TACTICAL SHIFT**: Detection of sustained collective shape/compactness changes via change-point analysis.
3. **ROLESHIFT**: Player behavioural archetype clustering and similarity retrieval (exploratory, within-provider).
4. **SCAN2PLAY**: Pre-reception orientation proxy analysis (conditional on verified pose coverage).

### Core Research Objective
**Quality-aware football tactical intelligence under incomplete, noisy, and unreliable observations.**  
We explicitly model observation confidence, visibility masks, and tracking status. Models must produce calibrated, selective predictions rather than fabricating certainty from missing data.

---

## 2. Binding Architectural Decisions

All contributions must strictly respect these decisions (see `docs/TECHNICAL_DECISIONS.md`):

1. **Video-First Product Contract**: Video is the user-facing input. Tracking is an internal output produced by CV. Tracking datasets (SkillCorner, IDSSE, Metrica) enter strictly through provider adapters at the canonical layer for benchmarking and research.
2. **Rights-Based Footage Policy**: We operate under an explicit rights-based footage framework. Users must confirm they hold appropriate permissions to process uploaded footage. Public availability does not confer reuse or redistribution rights. Source-specific permissions are recorded in the data rights register, and unauthorized distribution of footage or proprietary tracking data is strictly prohibited.
3. **Quality Design Candidates (Requiring Empirical Validation)**: The nine video quality dimensions and downstream task-readiness thresholds (`ready`, `partial`, `insufficient`) are proposed architectural design candidates. They must be calibrated, ablated, and validated against ground truth—not treated as already established performance claims.
4. **Canonical Data Boundary**: The canonical tracking layer (`src/pitchmind/data_engine/schema/`) is the hard boundary between CV and tactical analytics. Downstream models must never depend directly on detector formats, camera pixels, or provider-specific JSONs.
5. **Data Integrity Rule**:
   - A missing observation is **not** zero.
   - An out-of-view player is **not** a stationary player.
   - An interpolated or extrapolated coordinate is **never** marked as directly observed (`observed_flag=False`).
   - Ball height ($z$) is physically distinct from ground-plane projection ($x, y$). Planar homography fails on airborne balls.
6. **Independent Analytical Modules**: Modules are implemented as independent models before considering any shared multi-task representation. Prevent negative transfer.
7. **Local-First Web Architecture**: Backend is FastAPI + DuckDB/Parquet + SQLite. Frontend is React + TypeScript. Multi-tenant cloud infra and native desktop packaging are deferred non-goals.

---

## 3. Git Repository Discipline & Boundary Rule

- **Explicit Project Pathing**: The project Git repository root is `C:\Users\ombag\Documents\GitHub\Football-Analysis`.
- **Parent Git Boundary**: Do **not** modify, re-initialize, or delete the parent directory repository at `C:\Users\ombag\.git`. All Git commands must explicitly operate within the project repository (`git -C "C:\Users\ombag\Documents\GitHub\Football-Analysis"` or with the current working directory anchored to the project).
- **Clean Commits**: Commit messages must follow conventional format: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.

---

## 4. Isolated Python Environment Policy

- **Virtual Environment Isolation**: All Python operations, package installations, and tests must execute within the isolated project virtual environment (`.venv`), located at `C:\Users\ombag\Documents\GitHub\Football-Analysis\.venv`.
- **No Global Site-Packages Pollution**: Never use global Python site-packages as the project environment.
- **Dependency Compatibility Verification**: Before installing machine learning dependencies (PyTorch, PyTorch Geometric, OpenCV, XGBoost), verify binary wheel availability and compatibility against the runtime Python version. Do not install heavy ML dependencies during Phase 0.

---

## 5. Hardware Constraints & Execution Discipline

Development and execution occur on an **NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), Intel i5-12500H, and 16 GB RAM**:

- **Batch Size Limit**: Keep batch size to 1–2 for GPU perception tasks. Use gradient accumulation only when training requires it.
- **Frame Sampling**: Never decode full-resolution 1080p full-match video in bulk into memory. Sample keyframes, chunk processing into discrete time windows, and write intermediate results to disk.
- **Cache Intermediates**: Cache detector bounding boxes, tracklets, and visual embeddings to avoid rerunning expensive CV models during tactical analytics development.
- **VRAM Conservation**: CPU performs FFmpeg decoding and analytical DuckDB queries. The GPU is reserved for neural inference and PyTorch Geometric models.

---

## 6. Coding & Engineering Standards

- **Language & Types**: Python >= 3.10. Use explicit type hints everywhere (`typing`, `typing_extensions`). Enforce strict Pydantic v2 validation on public schemas and API boundaries.
- **Coordinate Conventions**:
  - Pitch coordinate standard: metric pitch in metres ($x \in [-52.5, 52.5]$ or $[0, 105]$, $y \in [-34, 34]$ or $[0, 68]$). Always document origin $(0, 0)$ and orientation.
  - Normalized direction: when computing direction-dependent features, normalize so the possession team attacks in the $+x$ direction, while preserving the raw coordinate record.
- **Docstrings**: Every module, class, and public function must have docstrings specifying inputs, outputs, units (e.g. metres, m/s, seconds, Hz), coordinate frame, and failure behavior.
- **Testing Requirements**:
  - Every coordinate transform, resampling logic, and schema parser must have unit tests in `tests/`.
  - Ingestion adapters must have regression tests verifying coordinate normalization and timestamp alignment.
  - Test suites must run cleanly via `pytest tests/`.
- **Verified Licensing & Third-Party Integrity**:
  - Respect `third_party_notices/LICENSES.md` and `datasets/manifests/data_rights_register.json`.
  - Do not fabricate or invent repository citations or license terms. Maintain verified license details and explicitly mark any unverified entries.
  - Never commit raw match footage, proprietary tracking data, model weights, or private credentials to Git.

---

## 7. Agent Workflow Rules

- **Read Before Writing**: Inspect existing schemas and utilities before adding duplicate functions.
- **Preserve Documentation Integrity**: Do not remove existing comments, docstrings, or architecture notices.
- **No Hallucinated Tools**: Use only supported tools and verified environment capabilities.
- **Verification First**: Verify syntax and unit tests locally before declaring a phase complete.
