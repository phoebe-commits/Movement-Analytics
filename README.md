# Movement Analytics

**Computational movement quality analysis for physical AI**

A research-grade pipeline for generating synthetic human gait, extracting kinematic signals, computing biomechanically validated movement quality metrics, and visualizing them in real time. Built to establish the scientific and engineering foundation for a Movement Quality Score — a multidimensional scoring model for evaluating human and robotic movement.

---

## Why This Exists

Physical AI has a measurement problem. Every humanoid robotics company ships policies optimized against internal benchmarks with no common scoring standard. The text-AI world solved this with evaluation companies (Arize, Braintrust, Weights & Biases). Physical AI has no equivalent.

This repository builds the technical proof that movement quality can be:
1. **Defined** — grounded in 70+ peer-reviewed biomechanics and motor-control citations
2. **Measured** — extracted computationally from video in real time
3. **Scored** — quantified across 15 orthogonal signal dimensions with clinical validation

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
generators/     Procedural biomechanical gait synthesis (8 profiles)
kinematics/     Joint angle computation + gait quality metrics
visualization/  Real-time dashboard with time-series plots and gauges
pose/           Pose estimation integration (MediaPipe)
```

### 8 Gait Profiles

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

### Movement Quality Score (MQS)

A composite **0–100 score** computed from weighted biomechanical domains:

| Domain | Weight | Signals Used |
|---|---|---|
| **Kinematics** | 30% | Hip/knee/ankle ROM (bilateral) vs. clinical norms |
| **Smoothness** | 20% | SPARC of hip velocity (spectral arc length) |
| **Symmetry** | 20% | Hip/knee/ankle Symmetry Index (left vs. right) |
| **Variability** | 15% | Stride time coefficient of variation |
| **Temporal** | 15% | Cadence and stride time vs. normal ranges |

MQS differentiates across profiles: normal gait scores 100, stiff-knee gait 83.7 (reduced ROM detected), noisy gait 57.3 (degraded smoothness and variability).

### Computed Metrics

For each gait profile, the pipeline computes **50+ metrics** in real time:

- **Movement Quality Score** — composite 0–100 with 5-domain breakdown
- **Joint ROM** — hip, knee, ankle (bilateral)
- **Peak angular velocity** — per joint
- **SPARC smoothness** — spectral arc length per joint velocity
- **Normalized Jerk** — per joint
- **Symmetry Index** — hip, knee, ankle (left vs. right)
- **Symmetry Ratio** — bilateral comparison
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
```

### CLI Options

```
--profile, -p     Gait profile (normal, slow, fast, limp, stiff_knee,
                  trendelenburg, model_runway, noisy)
--output, -o      Output video path (MP4) or image path (PNG for --compare)
--no-display      Skip live window (headless rendering)
--fps             Frames per second (default: 30)
--cycles, -c      Number of gait cycles (default: 6)
--all-profiles    Generate all profiles
--compare         Generate MQS comparison report across all profiles
```

### Profile Comparison

```bash
# Generate MQS comparison chart
python -m movement_analytics --compare --output output/mqs_comparison.png
```

The comparison report shows Movement Quality Score breakdowns across all 8 gait profiles, revealing how each domain (kinematics, smoothness, symmetry, variability, temporal) contributes to the overall score. Clinical reference ranges are displayed for context.

---

## Architecture

```
Input: Gait Parameters (cadence, ROM, asymmetry, noise, etc.)
  ↓
Gait Model → Biomechanically accurate joint angle trajectories
  ↓
Stick Figure Renderer → Sagittal-plane animation with color-coded joints
  ↓
Joint Angle Computation → Per-frame bilateral angle extraction
  ↓
Gait Metrics Engine → SPARC, NJ, SI, ROM, CV, cadence, phase detection
  ↓
Movement Quality Score → 5-domain composite (0–100)
  ↓
Real-Time Dashboard → MQS gauge + bilateral plots + metric panels
  ↓
Output: Video file and/or live display
```

---

## Movement Quality Signal Framework

The research document identifies **15 signals** across 5 domains that form the basis for a Movement Quality Score:

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
| Research document | Complete (90+ citations, v2.0) |
| Gait model (8 profiles) | Complete |
| Stick-figure renderer | Complete |
| Joint angle computation | Complete |
| Gait metrics engine | Complete |
| Movement Quality Score | Complete (5-domain composite) |
| Real-time dashboard | Complete (bilateral overlays, MQS gauge) |
| Pose estimation on external video | In progress |
| Movement Quality Score model | Planned |

---

## License

Proprietary. All rights reserved.
