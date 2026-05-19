# Movement Analytics

[![CI](https://github.com/dl1683/Movement-Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/dl1683/Movement-Analytics/actions/workflows/ci.yml)

**Computational movement quality analysis for physical AI**

A pipeline for generating synthetic human gait, extracting kinematic signals from video, computing movement quality metrics grounded in biomechanics literature, and visualizing results in real time. Built to establish the scientific and engineering foundation for a Movement Quality Score — a multidimensional scoring model for evaluating human and robotic movement from video.

---

## Why This Exists

Physical AI has a measurement problem. Every humanoid robotics company ships policies optimized against internal benchmarks with no common scoring standard. The text-AI world solved this with evaluation companies (Arize, Braintrust, Weights & Biases). Physical AI has no equivalent.

This repository builds the technical proof that movement quality can be:
1. **Defined** — grounded in 90+ peer-reviewed biomechanics and motor-control citations
2. **Measured** — extracted computationally from video via MediaPipe pose estimation
3. **Scored** — quantified across 6 biomechanical domains using literature-derived reference ranges

---

## What's Here

### Research Foundation (`research/`)

A comprehensive literature review covering:

| Domain | Key Metrics | Key Citations |
|---|---|---|
| **Gait Cycle** | Spatiotemporal parameters, phase timing, speed | Perry & Burnfield 2010, Winter 2009 |
| **Joint Kinematics** | Hip/knee/ankle ROM, pelvis motion, deviations | Schwartz & Rozumalski 2008 |
| **Quality Indices** | Gait Deviation Index (GDI), Gait Profile Score (GPS) | Baker et al. 2009 |
| **Smoothness** | SPARC, Log Dimensionless Jerk, Normalized Jerk | Balasubramanian et al. 2012 |
| **Symmetry** | SI, Symmetry Ratio, Normalized Symmetry Index | Robinson 1987, Shorter 2020 |
| **Stability** | Margin of Stability, Lyapunov Exponents | Hof et al. 2005 |
| **Variability** | Stride CV, entropy, DFA | Hausdorff et al. 2001 |
| **Expert vs. Novice** | Motor variability, coordination, trained movers | Marineau 2024, Tanabe 2023 |

### Analysis Pipeline (`src/movement_analytics/`)

```
generators/     Procedural biomechanical gait synthesis (9 profiles)
kinematics/     Joint angle computation + gait quality metrics
visualization/  Real-time dashboard with time-series plots and gauges
pose/           Pose estimation integration (MediaPipe)
```

### 9 Gait Profiles

| Profile | Description | Key Feature |
|---|---|---|
| `normal` | Healthy adult at comfortable speed | Baseline reference |
| `slow` | Cautious elderly/post-injury | Reduced ROM, shorter stride |
| `fast` | Fast walk approaching jog | Increased ROM and cadence |
| `limp` | Asymmetric gait | 35% left-right asymmetry |
| `stiff_knee` | Reduced knee flexion in swing | Spastic gait pattern |
| `trendelenburg` | Excessive pelvic drop + trunk lean | Hip abductor weakness |
| `model_runway` | Trained fashion model walk | Exaggerated pelvis, controlled trunk |
| `noisy` | High motor variability | 4° noise on all joints |
| `parkinsonian` | Shuffling gait (Parkinson's) | Reduced ROM, diminished arm swing, short stride |

### Movement Quality Score (MQS)

A composite **0–100 score** computed from 6 weighted biomechanical domains:

| Domain | Weight | Signals Used |
|---|---|---|
| **Kinematics** | 25% | Hip/knee/ankle ROM (bilateral) vs. clinical norms |
| **Smoothness** | 18% | SPARC of hip velocity (spectral arc length) |
| **Symmetry** | 18% | Hip/knee/ankle Symmetry Index (left vs. right) |
| **Coordination** | 14% | Continuous Relative Phase consistency (bilateral hip) |
| **Variability** | 13% | Stride time coefficient of variation |
| **Temporal** | 12% | Cadence and stride time vs. normal ranges |

MQS differentiates across profiles: normal gait scores highest across all domains, stiff-knee gait is penalized in kinematics (reduced knee ROM), noisy gait is penalized in smoothness and variability. Reference ranges sourced from Perry & Burnfield 2010, Winter 2009, Balasubramanian et al. 2012, Hausdorff et al. 2001, and Hamill et al. 1999.

### Computed Metrics

For each gait profile, the pipeline computes **50+ metrics** in real time:

- **Movement Quality Score** — composite 0–100 with 6-domain breakdown
- **Joint ROM** — hip, knee, ankle (bilateral)
- **Peak angular velocity** — per joint
- **SPARC smoothness** — spectral arc length per joint velocity
- **Normalized Jerk** — per joint
- **Symmetry Index** — hip, knee, ankle (left vs. right)
- **Symmetry Ratio** — bilateral comparison
- **CRP Coordination** — inter-limb phase coupling consistency (Hilbert transform)
- **Cadence** — steps per minute
- **Stride time** — mean and coefficient of variation
- **Gait phase** — stance/swing detection per frame

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/dl1683/Movement-Analytics.git
cd Movement-Analytics
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -e .

# Run with live dashboard
python -m movement_analytics --profile normal

# Render specific profile to video
python -m movement_analytics --profile limp --output output/limp.mp4

# Render all profiles (headless)
python -m movement_analytics --all-profiles --output output/gait --no-display

# Analyze real video (downloads MediaPipe model on first run)
python -m movement_analytics --video path/to/walking.mp4 --output output/analysis.mp4
```

### CLI Options

```
--profile, -p     Gait profile (normal, slow, fast, limp, stiff_knee,
                  trendelenburg, model_runway, noisy, parkinsonian)
--output, -o      Output video path (MP4) or image path (PNG for --compare)
--no-display      Skip live window (headless rendering)
--fps             Frames per second (default: 30)
--cycles, -c      Number of gait cycles (default: 6)
--all-profiles    Generate all profiles
--compare         Generate MQS comparison report across all profiles
--video, -v       Input video file for pose estimation analysis
--benchmark       Output MQS benchmark JSON across all profiles
--sensitivity     Generate MQS sensitivity analysis plots (PNG)
```

### Sensitivity Analysis

```bash
# Generate sensitivity plots showing MQS response to parameter variation
python -m movement_analytics --sensitivity --output output/mqs_sensitivity.png
```

The sensitivity report sweeps three parameters (knee ROM, noise level, asymmetry) and plots how MQS and each domain score respond. All curves are monotonic — the score degrades continuously as movement quality worsens, with no discontinuities or inversions.

### Profile Comparison

```bash
# Generate MQS comparison chart
python -m movement_analytics --compare --output output/mqs_comparison.png
```

The comparison report shows Movement Quality Score breakdowns across all 9 gait profiles, revealing how each domain contributes to the overall score.

### Reproducible Benchmark

```bash
# Output MQS benchmark as JSON (suitable for CI regression testing)
python -m movement_analytics --benchmark --output output/benchmark.json
```

The benchmark computes MQS and all domain/metric scores across all 9 profiles in a deterministic, reproducible format. Use `--cycles` to control the number of gait cycles evaluated.

### Python API

```python
from movement_analytics import (
    GaitParameters, GAIT_PROFILES, generate_frames,
    compute_gait_summary, mqs_domain_scores,
)

# Score a built-in gait profile
profile = GAIT_PROFILES["parkinsonian"]
_, angles_right, angles_left, _ = generate_frames(profile.params, n_cycles=6)
summary = compute_gait_summary(angles_right, angles_left, fps=30)

print(f"MQS: {summary['movement_quality_score']:.1f}")
for domain, score in mqs_domain_scores(summary).items():
    print(f"  {domain}: {score:.1f}")

# Custom gait parameters
custom = GaitParameters(hip_rom=25, knee_rom=35, cadence=85)
_, ar, al, _ = generate_frames(custom, n_cycles=6)
custom_summary = compute_gait_summary(ar, al, fps=30)
print(f"Custom MQS: {custom_summary['movement_quality_score']:.1f}")
```

---

## Architecture

```
Input: Gait Parameters OR Video File
  ↓
Gait Model / MediaPipe Pose Estimation → Joint angle trajectories
  ↓
Stick Figure Renderer → Sagittal-plane animation with color-coded joints
  ↓
Joint Angle Computation → Per-frame bilateral angle extraction
  ↓
Gait Metrics Engine → SPARC, NJ, SI, CRP, ROM, CV, cadence, phase detection
  ↓
Movement Quality Score → 6-domain composite (0–100)
  ↓
Real-Time Dashboard → MQS gauge + bilateral plots + metric panels
  ↓
Output: Video file and/or live display
```

---

## Movement Quality Signal Framework

The research document identifies **15 signals** across 6 domains that form the basis for the Movement Quality Score:

| # | Signal | Domain | Clinical Reference |
|---|---|---|---|
| 1 | Hip flexion/extension ROM | Kinematics | 40–45° normal |
| 2 | Knee flexion ROM | Kinematics | 60–65° normal |
| 3 | Ankle dorsiflexion ROM | Kinematics | ~30° normal |
| 4 | Pelvic obliquity amplitude | Kinematics | <7° normal |
| 5 | Hip angular velocity (peak) | Dynamics | — |
| 6 | SPARC (hip velocity) | Smoothness | -1.5 to -1.7 (smooth) |
| 7 | SPARC (knee velocity) | Smoothness | -1.5 to -1.7 (smooth) |
| 8 | Hip flexion Symmetry Index | Symmetry | <10% normal |
| 9 | Knee flexion Symmetry Index | Symmetry | <10% normal |
| 10 | Stride time CV | Variability | 1–3% normal |
| 11 | Cadence | Temporal | 100–120 spm |
| 12 | Double support time | Temporal | ~20% of cycle |
| 13 | Trunk lateral lean | Kinematics | <5° normal |
| 14 | Inter-limb coordination (CRP) | Coordination | — |
| 15 | Gait Deviation Index (GDI) | Composite | ≥100 normal |

---

## Dependencies

- Python ≥ 3.10
- NumPy, SciPy, OpenCV, MediaPipe, Matplotlib

---

## Project Status

| Component | Status |
|---|---|
| Research document | Complete (90+ citations, 10 sections) |
| Gait model (9 profiles) | Complete |
| Stick-figure renderer | Complete |
| Joint angle computation | Complete |
| Gait metrics engine | Complete (98% test coverage) |
| Movement Quality Score | Complete (6-domain composite with CRP coordination) |
| Real-time dashboard | Complete (bilateral overlays, MQS gauge, 6-domain breakdown) |
| Pose estimation on external video | Functional (MediaPipe PoseLandmarker, sagittal plane) |
| CI/CD | Complete (GitHub Actions, 70 tests, ruff lint, coverage) |
| Reproducible benchmark | Complete (JSON output for regression testing) |
| Movement Quality Score model | Planned (learned weights from expert raters) |

---

## License

Proprietary. All rights reserved.
