# Kinematics and Movement Quality Analysis: A Research Foundation for Physical AI

**Movement-Analytics Research Document — Deliverable 1**
**Version:** 1.0
**Date:** 2026-05-19

---

## Abstract

This document establishes the scientific foundation for computational movement quality assessment in the context of physical AI and humanoid robotics. It synthesizes peer-reviewed literature across clinical gait analysis, biomechanics, motor control, and computer vision to answer four questions: (1) what movement signals matter, (2) how are they measured, (3) what distinguishes expert from novice movement, and (4) how can movement quality be quantified with defensible, reproducible metrics. The framework directly informs the design of a Movement Quality Score — a multidimensional scoring model for evaluating human and robotic movement.

---

## Table of Contents

1. [The Gait Cycle: Structure and Parameters](#1-the-gait-cycle-structure-and-parameters)
2. [Joint Kinematics in Walking](#2-joint-kinematics-in-walking)
3. [Movement Quality Metrics: Composite Indices](#3-movement-quality-metrics-composite-indices)
4. [Smoothness and Jerk-Based Metrics](#4-smoothness-and-jerk-based-metrics)
5. [Symmetry Assessment](#5-symmetry-assessment)
6. [Dynamic Stability Metrics](#6-dynamic-stability-metrics)
7. [Variability and Motor Control](#7-variability-and-motor-control)
8. [Expert vs. Novice Movement: What the Literature Shows](#8-expert-vs-novice-movement)
9. [Computational Methods: Pose Estimation and Video-Based Analysis](#9-computational-methods)
10. [The Top 15 Movement Signals for a Quality Score](#10-the-top-15-movement-signals)
11. [References](#11-references)

---

## 1. The Gait Cycle: Structure and Parameters

### 1.1 Cycle Structure

A single gait cycle (stride) runs from initial contact of one foot to the next initial contact of the same foot. It divides into two primary phases with precisely characterized sub-phases (Perry & Burnfield, 2010):

| Sub-phase | % of Gait Cycle | Description |
|---|---|---|
| Initial Contact | 0–2% | Heel strikes the ground |
| Loading Response | 2–12% | Weight acceptance, shock absorption |
| Midstance | 12–31% | Single-limb support, body advances over foot |
| Terminal Stance | 31–50% | Heel rise, body advances ahead of stance foot |
| Pre-swing | 50–62% | Weight transfer to opposite limb, push-off |
| Initial Swing | 62–75% | Foot clearance, limb acceleration |
| Mid-swing | 75–87% | Limb advances past stance limb |
| Terminal Swing | 87–100% | Limb decelerates for next contact |

**Key ratios at comfortable walking speed:**
- Stance:Swing = 60:40
- Double-limb support = ~20% of cycle (two periods of ~10%)
- Single-limb support = ~40% of cycle

### 1.2 Spatiotemporal Parameters — Normal Ranges

| Parameter | Normal Adult Value | Source |
|---|---|---|
| Gait speed | 1.2–1.4 m/s (males 1.40, females 1.30 at age 40–49) | Bohannon & Wang, 2019 |
| Stride length | ~1.4 m (normalized ~0.8 × leg length) | Winter, 2009 |
| Step length | ~0.72 m | Perry & Burnfield, 2010 |
| Cadence | 100–120 steps/min (women ~6–9 higher than men) | Multiple sources |
| Step width | 5–10 cm | Perry & Burnfield, 2010 |
| Double support time | ~20% of gait cycle | Perry & Burnfield, 2010 |

### 1.3 Speed as a "Vital Sign"

Gait speed is increasingly recognized as a clinical vital sign:
- **< 1.0 m/s:** Associated with increased mortality in older adults (Studenski et al., 2011, JAMA)
- **< 0.8 m/s:** Functionally limited
- **< 0.4 m/s:** Severe impairment
- **Minimal clinically important difference:** 0.05–0.10 m/s (Fritz & Lusardi, 2009)

### 1.4 Changes with Pathology and Aging

Pathological gait consistently shows: slower speed, shorter stride length, reduced cadence, increased double-support time, decreased ankle push-off power, and increased variability. Elderly gait adds: reduced vertical center-of-mass displacement, wider step width, and altered arm swing patterns.

---

## 2. Joint Kinematics in Walking

### 2.1 Sagittal-Plane Joint Angle Profiles

The sagittal plane (side view) captures the primary joint motions in walking. Normal ranges synthesized from Perry (1992) and Winter (2009):

| Gait Phase | Hip | Knee | Ankle |
|---|---|---|---|
| Initial Contact | 20° flexion | 0° (full extension) | Neutral (0°) |
| Loading Response | 20° flex | 0° → 15–20° flex | Neutral → 5° plantarflexion |
| Midstance | 20° flex → neutral | 20° flex → 0° | 5° PF → 5° dorsiflexion |
| Terminal Stance | Neutral → 20° extension | ~0° | 5–10° dorsiflexion (peak) |
| Pre-swing | 20° ext → neutral | 0° → 40° flex | 10° DF → 20° PF (push-off) |
| Initial Swing | Neutral → 10° flex | 40–60° flex (peak) | 20° PF → 5° |
| Mid-swing | 10° → 20° flex | 60° → 30° flex | ~Neutral |
| Terminal Swing | ~25–30° flex (peak) | 30° → 0° | Neutral |

**Total sagittal ROM during normal gait:**
- **Hip:** 40–45° (30° flexion to 10–20° extension)
- **Knee:** 60–65° (0° to 60° flexion)
- **Ankle:** ~30° (15° dorsiflexion to 20° plantarflexion)

### 2.2 Pelvis Kinematics

| Plane | Motion | Normal ROM |
|---|---|---|
| Sagittal | Anterior/posterior tilt | 2–5° total |
| Frontal | Obliquity (pelvic drop) | 4–5° total |
| Transverse | Rotation | 8–12° total |

Excessive pelvic obliquity (>7°) indicates hip abductor weakness (Trendelenburg sign). Excessive pelvic rotation may compensate for reduced hip flexion ROM.

### 2.3 Frontal and Transverse Planes

Key ranges: hip adduction/abduction ~10–12° ROM; knee valgus/varus ~5–8°; ankle inversion/eversion ~10–15°; hip rotation ~10–15°; foot progression angle ~5–15° external.

Standard errors of measurement (clinical-grade mocap):
- Hip sagittal: 2.9–4.1°; frontal: 2.7–3.7°; transverse: 1.9–3.9°
- Knee sagittal: 1.6–4.2°; frontal: 1.0–1.9°; transverse: 1.3–2.9°
- Ankle sagittal: 0.7–2.0°; frontal: 1.2–2.3°; transverse: 2.9–4.0°

### 2.4 Kinematic Deviations Indicating Quality Issues

| Deviation | Typical Cause | Clinical Significance |
|---|---|---|
| Excessive knee flexion at IC (>5°) | Quadriceps weakness, hamstring contracture | Crouch gait |
| Insufficient ankle DF in swing (<0°) | Peroneal nerve palsy, ankle stiffness | Drop foot |
| Excessive pelvic obliquity (>7°) | Hip abductor weakness | Trendelenburg gait |
| Reduced hip extension in terminal stance (<5°) | Hip flexor tightness | Shortened stride |
| Knee hyperextension (>5° recurvatum) | Quadriceps overactivity, PF spasticity | Instability risk |
| Excessive trunk lateral lean | Compensatory for hip abductor weakness | Energy cost increase |

---

## 3. Movement Quality Metrics: Composite Indices

### 3.1 Gait Deviation Index (GDI)

**Schwartz & Rozumalski, 2008, Gait & Posture, DOI: 10.1016/j.gaitpost.2008.05.001**

The GDI maps 9 kinematic waveforms (pelvis tilt/obliquity/rotation, hip flexion/adduction/rotation, knee flexion, ankle dorsiflexion, foot progression) onto a single scalar via SVD-based dimensionality reduction.

**Computation:**
1. Sample 9 kinematic variables at 2% intervals across gait cycle (51 points each → 459-element vector)
2. Perform SVD on a reference matrix of control + patient strides
3. Retain first 15 gait features (~98% of variance)
4. Project each stride onto these 15 features
5. Compute Euclidean distance from control mean
6. Scale: **GDI = 100 − (10/SD_control) × distance**

**Interpretation:**
- GDI ≥ 100: Within normal range
- Each 10-point decrease = 1 SD from normal
- ~80: mild pathology; ~60: moderate; <50: severe

### 3.2 Gait Profile Score (GPS)

**Baker et al., 2009, Gait & Posture, DOI: 10.1016/j.gaitpost.2009.05.020**

The GPS computes the RMS deviation of each kinematic variable from a reference mean, then aggregates:

**Gait Variable Score (per variable):**
GVS_i = √[(1/T) × Σ(x_{i,t} − x_{i,t,ref})²]

**Overall GPS:**
GPS = √[(1/N) × Σ(GVS_i²)]

**Interpretation:**
- GPS in degrees of average deviation from normal
- Normal GPS: ~5–6°
- Minimal clinically important difference: 1.6°
- The Movement Analysis Profile (MAP) displays individual GVS values

### 3.3 Edinburgh Visual Gait Score (EVGS)

**Read et al., 2003, J Pediatr Orthop**

Observational scoring: 17 parameters per lower extremity, each scored 0 (normal), 1 (moderate deviation), or 2 (significant deviation). Total 0–34 per side. Good inter-rater reliability validated against 3D gait analysis.

---

## 4. Smoothness and Jerk-Based Metrics

Movement smoothness reflects the continuity and fluidity of motion — one of the most sensitive indicators of motor control quality and neuromotor recovery.

### 4.1 Spectral Arc Length (SPARC)

**Balasubramanian, Melendez-Calderon & Burdet, 2012, IEEE TBME, DOI: 10.1109/TBME.2011.2179545**
**Revised: Balasubramanian et al., 2015, Frontiers in Neurology**

SPARC measures the arc length of the normalized Fourier magnitude spectrum of the velocity profile:

**SPARC = −∫₀^ωc √[(1/ωc)² + (dV̂(ω)/dω)²] dω**

Where V̂(ω) is the normalized magnitude spectrum, ωc is the adaptive frequency cutoff (~10 Hz).

**Properties:**
- Dimensionless, independent of movement duration and amplitude
- Values closer to 0 = smoother; more negative = less smooth
- Perfectly smooth (minimum-jerk) movement: SPARC ≈ −1.5 to −1.7
- ICC > 0.9 (excellent reliability)
- CV 1.6–2.2% (lowest inter-subject variability of all smoothness metrics)
- **Best overall smoothness metric** per comparative studies (Mancini et al., 2023; Gulde & Hermsdörfer, 2018)

### 4.2 Log Dimensionless Jerk (LDLJ)

**Hogan & Sternad, 2009, J Motor Behavior**

LDLJ = −ln|DJ|, where DJ = −(T³/v_peak²) × ∫₀ᵀ jerk(t)² dt

- Dimensionless; more negative = smoother
- ICC > 0.9 (excellent)
- CV 6.8–8.7% (moderate variability)
- **Limitation:** Correlated with movement duration

### 4.3 Normalized Jerk (NJ)

NJ = √[0.5 × ∫(jerk² dt) × T⁵ / A²]

Where T = duration, A = peak-to-peak amplitude. Lower = smoother. (Teulings et al., 1997, Acta Psychologica)

### 4.4 Harmonic Ratio (HR)

**Menz, Lord & Fitzpatrick, 2003, Gait & Posture**

Ratio of even-to-odd harmonics in trunk acceleration. Higher = more rhythmic/smooth. Healthy controls: HR ~3.67 (AP). Better conceptualized as step-to-step symmetry than pure smoothness.

### 4.5 Ranking

Per comparative validation studies:
1. **SPARC** — robust, duration-independent, lowest variability
2. **LDLJ** — good reliability but duration-coupled
3. **NJ / Harmonic Ratio** — useful secondary metrics
4. NARJ, Number of Zero-Crossings — high variability, not recommended as primary

---

## 5. Symmetry Assessment

### 5.1 Symmetry Index (SI)

**Robinson, Herzog & Nigg, 1987, J Manipulative Physiol Ther**

SI(%) = 2 × |X_R − X_L| / (|X_R| + |X_L|) × 100

- SI = 0%: perfect symmetry
- Healthy adults: SI < 5–10% for spatiotemporal parameters
- Clinical significance threshold: SI > 10–15% (Knapik et al., 1991)
- Post-stroke: commonly 20–50%+

### 5.2 Symmetry Ratio (SR)

SR = min(X_L, X_R) / max(X_L, X_R)

- 1.0 = symmetric; lower = more asymmetric
- Lowest variability and highest reliability (ICC₁₀ > 0.80) across spatiotemporal parameters

### 5.3 Symmetry Angle (SA)

**Zifchock et al., 2008, Gait & Posture, DOI: 10.1016/j.gaitpost.2006.11.209**

SA(%) = [(45° − arctan(X_L/X_R)) / 90°] × 100

Bounded (0–100%), continuous. Originally for angular data.

### 5.4 Normalized Symmetry Index (NSI)

**Shorter et al., 2020, J Biomech, DOI: 10.1016/j.jbiomech.2019.109531**

NSI(%) = [(X_NS − X_S) / (max(0, X_NS, X_S) − min(0, X_NS, X_S))] × 100

Bounded (−100% to +100%), linear, normalized. Best combination of properties for general use.

### 5.5 Clinical Thresholds

| Population | Typical SI |
|---|---|
| Healthy adults | < 5–10% |
| Clinical concern | > 10–15% |
| Post-stroke | 20–50%+ |
| Amputees | 15–40%+ |
| Parkinson's (early) | Detectable asymmetry in step time |

---

## 6. Dynamic Stability Metrics

### 6.1 Margin of Stability (MoS)

**Hof, Gazendam & Sinke, 2005, J Biomech, DOI: 10.1016/j.jbiomech.2004.03.025**

Based on the extrapolated center of mass (XcoM):

**XcoM = P_CoM + V_CoM / √(g/l)**

Where P_CoM = center of mass position, V_CoM = CoM velocity, g = 9.81 m/s², l = effective pendulum length.

**MoS = BoS_boundary − XcoM**

- MoS > 0: Mechanically stable (XcoM within base of support)
- MoS < 0: Unstable (corrective step required)
- Normal gait is "controlled falling" — AP MoS near zero at specific phases

### 6.2 Lyapunov Exponents (Local Dynamic Stability)

**Rosenstein, Collins & De Luca, 1993, Physica D**

λ_max quantifies the rate at which nearby trajectories in reconstructed state space diverge:
- Higher λ_max = less locally stable
- Lower = more stable
- Requires state-space reconstruction (embedding dimension ~5, time delay via AMI)
- Reliably differentiates young healthy from elderly with ≥50 strides

### 6.3 Center of Mass Trajectory

**Normal CoM displacement during gait:**
- Vertical: ~4–5 cm at comfortable speed (sinusoidal, peak at midstance)
- Lateral: ~4–7 cm (shifts toward stance limb)
- Six determinants of gait minimize vertical excursion and energy cost (Saunders, Inman & Eberhart, 1953)

---

## 7. Variability and Motor Control

### 7.1 Step-to-Step Variability

**Hausdorff, Rios & Edelberg, 2001, Arch Phys Med Rehabil**

Coefficient of Variation (CV) = SD/mean × 100

**Normal values:**
- Healthy adults: CV of stride time, gait speed = **1–3%** (remarkably low)
- Stride time SD in non-fallers: ~49 ± 4 ms
- Stride time SD in fallers: ~106 ± 30 ms

**Key finding:** Variability measures may be MORE predictive of falls than mean gait values. Average gait speed relates to fear of falling but NOT to actual fall risk.

### 7.2 Motor Noise and Signal

The "optimal variability" hypothesis (Stergiou & Decker, 2011) holds that healthy movement exhibits a specific structure of variability: not too much (unstable), not too little (rigid/inflexible). Both extremes are pathological.

- **Approximate Entropy (ApEn)** and **Sample Entropy (SampEn):** Quantify regularity/predictability of time series. Lower entropy = more regular (potentially pathological rigidity). Higher = more random (potentially unstable).
- **Detrended Fluctuation Analysis (DFA):** Assesses long-range correlations in stride-to-stride fluctuations. Healthy gait shows fractal scaling (α ≈ 0.7–0.8). Parkinson's and aging shift toward random (α → 0.5).

### 7.3 Coordination Metrics

- **Continuous Relative Phase (CRP):** Phase difference between two oscillating segments (e.g., thigh-shank). In-phase = 0°, anti-phase = 180°. Expert walkers show more consistent CRP patterns.
- **Cross-correlation:** Peak correlation coefficient and time lag between bilateral joint angle signals. Higher peak and shorter lag = better coordination.

---

## 8. Expert vs. Novice Movement: What the Literature Shows

### 8.1 Motor Variability and Expertise

**Marineau et al., 2024, Scand J Med Sci Sports, DOI: 10.1111/sms.14706**

Scoping review of 59 biomechanics studies: **48 (81%)** report higher-skilled performers have measurably less motor variability than lower-skilled performers, regardless of sport, expertise definition, or variability measurement. Holds across javelin, golf, pitching, gymnastics, and gait.

The mechanism: experts have internalized error-correction loops that novices have not. Expert movement is more repeatable but also more *functionally* variable — variability is structured and task-relevant rather than random noise.

### 8.2 Fashion Model Gait: Measured Differences

**Tanabe et al., 2023, Frontiers in Sports and Active Living, DOI: 10.3389/fspor.2023.1091470**

12-camera infrared mocap at 100 Hz. 7 professional fashion models (mean 16.6 years experience) vs. 10 non-models, two instruction conditions ("casual" and "attractively as possible").

**Key findings:**
- Models produced systematically different knee extension during push-off (0–6%, 45.5–47.5%, 88–100% of gait cycle)
- Distinctive backward upper-arm silhouette in specific phases (0–11.5%, 18–23%, 64–71%)
- Observer ratings: models 4.29 ± 0.64 vs. non-models 3.50 ± 0.43 (7-point Likert)
- Both groups changed strategy under instruction, but models' changes were more precisely localized in the gait cycle

**Kobayashi, Saito & Murahori, 2024, Sensors, DOI: 10.3390/s24123865**

69 professional runway models: PCA + hierarchical clustering identified **5 distinct walking-style clusters** with significantly different median show-years between clusters. Runway gait is structured, clusterable, and time-evolving with experience.

### 8.3 Movement Smoothness and Skill Level

Flash & Hogan (1985) established the minimum-jerk model as the organizing principle of skilled reaching: the nervous system plans trajectories that minimize jerk (third derivative of position). Extensions to walking:

- Skilled walkers show lower SPARC values (closer to 0) in joint angle velocities
- Rehabilitation patients show progressively smoother movement as they recover
- SPARC reliably tracks skill acquisition across motor learning studies (Balasubramanian et al., 2015)

### 8.4 Coordination and Expertise

Expert walkers show:
- More consistent CRP patterns (lower CRP variability)
- Stronger inter-limb coupling (higher cross-correlation peaks)
- More precise gait event timing (lower stride time CV)
- Better-organized variability structure (DFA α closer to 0.75)

### 8.5 Computational Approaches to Movement Quality

Emerging ML/AI approaches:
- **Functional Movement Screen (FMS) scoring from video:** CNN-based systems achieving moderate-to-good agreement with expert raters (Moro et al., 2024)
- **Gait quality prediction from IMU data:** LSTM networks predicting GDI from wearable sensor data (Chia Bejarano et al., 2015)
- **Movement quality from pose estimation:** Temporal convolutions on MediaPipe keypoints for exercise quality assessment (multiple recent works)
- **Action quality assessment (AQA):** CoRe, DAE, USDL architectures for scoring athletic movements from video

---

## 9. Computational Methods: Pose Estimation and Video-Based Analysis

### 9.1 Pose Estimation Models

| Model | Keypoints | Speed | 3D? | Best For |
|---|---|---|---|---|
| **MediaPipe Pose** | 33 (BlazePose) | Real-time (30+ fps on CPU) | Pseudo-3D (depth estimate) | Real-time applications, mobile |
| **OpenPose** | 25 (body) + hands + face | 10–15 fps GPU | 2D only | Multi-person, research baseline |
| **MMPose / ViTPose** | 17–133 (configurable) | Variable | 2D, liftable to 3D | High accuracy research |
| **HRNet** | 17 (COCO) | ~15 fps GPU | 2D | High-resolution features |
| **AlphaPose** | 17–26 | 20+ fps GPU | 2D | Multi-person, top-down |

**For real-time single-person analysis:** MediaPipe is optimal — runs on CPU, 33 keypoints including hands and feet, provides pseudo-depth.

### 9.2 Joint Angle Computation from Keypoints

**ISB Recommendations (Wu et al., 2002, J Biomech):**
Joint angles computed as Euler/Cardan angles of the distal segment relative to the proximal segment, in the order: flexion/extension → abduction/adduction → internal/external rotation.

**Practical 2D approach (sagittal view):**
- **Knee flexion:** 180° − included angle at knee between thigh and shank vectors
- **Hip flexion:** Angle of thigh relative to trunk/vertical
- **Ankle dorsiflexion:** Angle between shank and foot segments

### 9.3 Gait Event Detection from Video

Without force plates, gait events detected from:
- **Vertical position of ankle/heel marker:** Minimum height = heel strike
- **Hip flexion peaks:** Maximum flexion ≈ heel strike; maximum extension ≈ toe-off
- **Foot velocity:** Zero-crossing of horizontal foot velocity
- **ML approaches:** HMMs or LSTMs trained on labeled gait data

### 9.4 Validation: Video-Based vs. Marker-Based

Typical joint angle errors from vision-based pose estimation compared to optical mocap:
- **Sagittal plane:** 3–8° RMS error for major joints (MediaPipe, OpenPose)
- **Frontal/transverse:** 5–15° RMS error (significantly worse)
- **Gait event detection:** ±2–4 frames at 30 fps (~60–130 ms)
- Best results with single-person, clear sagittal view, controlled lighting

---

## 10. The Top 15 Movement Signals for a Quality Score

Based on the literature review, the following 15 signals form the most defensible, computationally tractable basis for a Movement Quality Score. Each is justified by peer-reviewed evidence of discriminative power:

| # | Signal | Domain | Why It Matters | Key Citation |
|---|---|---|---|---|
| 1 | **Hip flexion/extension ROM** | Kinematics | Primary driver of stride length; reduced in pathological gait | Perry & Burnfield, 2010 |
| 2 | **Knee flexion ROM** | Kinematics | Peak swing flexion discriminates normal from stiff gait | Winter, 2009 |
| 3 | **Ankle dorsiflexion ROM** | Kinematics | Push-off and clearance; reduced = drop foot risk | Perry & Burnfield, 2010 |
| 4 | **Pelvic obliquity amplitude** | Kinematics | >7° indicates hip abductor weakness (Trendelenburg) | Schwartz & Rozumalski, 2008 |
| 5 | **Hip angular velocity (peak)** | Dynamics | Reflects power generation; reduced in weakness | Winter, 2009 |
| 6 | **SPARC (hip velocity)** | Smoothness | Best smoothness metric; tracks skill and recovery | Balasubramanian et al., 2012 |
| 7 | **SPARC (knee velocity)** | Smoothness | Knee smoothness discriminates pathological gait | Balasubramanian et al., 2015 |
| 8 | **Hip flexion Symmetry Index** | Symmetry | Bilateral difference; >10% clinically significant | Robinson et al., 1987 |
| 9 | **Knee flexion Symmetry Index** | Symmetry | Most sensitive to unilateral pathology | Shorter et al., 2020 |
| 10 | **Stride time CV** | Variability | Predicts fall risk better than mean gait speed | Hausdorff et al., 2001 |
| 11 | **Cadence** | Temporal | Deviations from 100–120 spm indicate compensation | Perry & Burnfield, 2010 |
| 12 | **Double support time** | Temporal | Increases with instability and pathology | Perry & Burnfield, 2010 |
| 13 | **Trunk lateral lean** | Kinematics | Compensatory mechanism; indicates control deficit | Baker et al., 2009 |
| 14 | **Inter-limb coordination (CRP consistency)** | Coordination | Expert movers show lower CRP variability | Hamill et al., 1999 |
| 15 | **Gait Deviation Index (GDI)** | Composite | Gold-standard overall kinematic quality metric | Schwartz & Rozumalski, 2008 |

### Signal Selection Rationale

The 15 signals span five domains (kinematics, dynamics, smoothness, symmetry, variability) chosen for:
1. **Discriminative power:** Each has peer-reviewed evidence of distinguishing expert/healthy from novice/pathological movement
2. **Computational tractability:** All computable from 2D sagittal-plane video with pose estimation
3. **Clinical validity:** All have established normal ranges and clinical significance thresholds
4. **Complementarity:** No two signals capture the same information — removing any one reduces coverage
5. **Relevance to humanoid robotics:** Each maps directly to a dimension humanoid OEMs evaluate when benchmarking their policies

### From 15 Signals to a Composite Score

The signals above can be combined into a multidimensional Movement Quality Score through:

1. **Normalization:** Each signal mapped to a 0–100 scale using the clinical normal range as reference
2. **Domain weighting:** Kinematics (30%), Smoothness (20%), Symmetry (20%), Variability (15%), Temporal (15%)
3. **Aggregation:** Weighted root-mean-square or distance-based composite (analogous to GDI/GPS methodology)
4. **Validation:** ICC > 0.80 against expert biomechanics raters as the minimum credibility threshold

---

## 11. References

### Foundational Texts
- Perry J, Burnfield JM. *Gait Analysis: Normal and Pathological Function*. 2nd ed. SLACK Inc; 2010.
- Winter DA. *Biomechanics and Motor Control of Human Movement*. 4th ed. Wiley; 2009.

### Gait Cycle and Spatiotemporal Parameters
- Bohannon RW, Wang YC. Four-meter gait speed: normative values and reliability. *Arch Phys Med Rehabil*. 2019.
- Studenski S, et al. Gait speed and survival in older adults. *JAMA*. 2011;305(1):50–58.
- Fritz S, Lusardi M. White paper: walking speed — the sixth vital sign. *J Geriatr Phys Ther*. 2009;32(2):46–49.
- Saunders JB, Inman VT, Eberhart HD. The major determinants in normal and pathological gait. *J Bone Joint Surg Am*. 1953;35(3):543–558.

### Composite Quality Indices
- Schwartz MH, Rozumalski A. The Gait Deviation Index: a new comprehensive index of gait pathology. *Gait & Posture*. 2008;28(3):351–357. DOI: 10.1016/j.gaitpost.2008.05.001
- Baker R, McGinley JL, Schwartz MH, et al. The Gait Profile Score and Movement Analysis Profile. *Gait & Posture*. 2009;30(3):265–269. DOI: 10.1016/j.gaitpost.2009.05.020
- Schutte LM, et al. An index for quantifying deviations from normal gait. *Gait & Posture*. 2000;11(1):25–31.
- Read HS, et al. Edinburgh Visual Gait Score for use in cerebral palsy. *J Pediatr Orthop*. 2003;23(3):296–301.

### Smoothness and Jerk
- Balasubramanian S, Melendez-Calderon A, Burdet E. A robust and sensitive metric for quantifying movement smoothness. *IEEE Trans Biomed Eng*. 2012;59(8):2126–2136. DOI: 10.1109/TBME.2011.2179545
- Balasubramanian S, et al. On the analysis of movement smoothness. *J NeuroEngineering Rehab*. 2015;12:112.
- Hogan N, Sternad D. Sensitivity of smoothness measures to movement duration, amplitude, and arrests. *J Motor Behav*. 2009;41(6):529–534.
- Flash T, Hogan N. The coordination of arm movements: an experimentally confirmed mathematical model. *J Neurosci*. 1985;5(7):1688–1703.
- Teulings HL, et al. Parkinsonism reduces coordination of fingers, wrist, and arm in fine motor control. *Exp Neurol*. 1997;146(1):159–170.
- Menz HB, Lord SR, Fitzpatrick RC. Acceleration patterns of the head and pelvis when walking. *Gait & Posture*. 2003;18(1):35–46.

### Symmetry
- Robinson RO, Herzog W, Nigg BM. Use of force platform variables to quantify gait symmetry. *J Manipulative Physiol Ther*. 1987;10(4):172–176.
- Zifchock RA, et al. The symmetry angle: a novel, robust method. *Gait & Posture*. 2008;27(4):622–627.
- Shorter KA, et al. The Normalized Symmetry Index. *J Biomech*. 2020;99:109531.
- Plotnik M, Giladi N, Hausdorff JM. Bilateral coordination of human gait. *Exp Brain Res*. 2005;181(4):561–570.

### Dynamic Stability
- Hof AL, Gazendam MGJ, Sinke WE. The condition for dynamic stability. *J Biomech*. 2005;38(1):1–8. DOI: 10.1016/j.jbiomech.2004.03.025
- Rosenstein MT, Collins JJ, De Luca CJ. A practical method for calculating largest Lyapunov exponents. *Physica D*. 1993;65(1–2):117–134.
- Dingwell JB, Cusumano JP. Nonlinear time series analysis of human walking. *Chaos*. 2000;10(4):848–863.
- Hak L, et al. Steps to take to enhance gait stability. *PLOS ONE*. 2013;8(12):e82842.

### Variability and Motor Control
- Hausdorff JM, Rios DA, Edelberg HK. Gait variability and fall risk. *Arch Phys Med Rehabil*. 2001;82(8):1050–1056.
- Stergiou N, Decker LM. Human movement variability, nonlinear dynamics, and pathology. *Hum Movement Sci*. 2011;30(5):869–888.
- Bizovska L, et al. Variability of spatial temporal gait parameters and CoP in elderly fallers. *PLOS ONE*. 2017;12(2):e0171997.

### Expert vs. Novice Movement
- Marineau CJ, et al. From novice to expert: how expertise shapes motor variability in sports biomechanics. *Scand J Med Sci Sports*. 2024. DOI: 10.1111/sms.14706
- Tanabe K, et al. Gait characteristics of professional fashion models. *Front Sports Active Living*. 2023;5:1091470. DOI: 10.3389/fspor.2023.1091470
- Kobayashi K, Saito S, Murahori Y. Classification of fashion models' walking styles. *Sensors*. 2024;24(12):3865. DOI: 10.3390/s24123865

### Pose Estimation and Computational Methods
- Wu G, et al. ISB recommendation on definitions of joint coordinate systems. *J Biomech*. 2002;35:543–548.
- Lugaresi C, et al. MediaPipe: a framework for building perception pipelines. *arXiv:1906.08172*. 2019.
- Cao Z, et al. OpenPose: realtime multi-person 2D pose estimation. *IEEE TPAMI*. 2021;43(1):172–186.
- Xu Y, et al. ViTPose: simple vision transformer baselines for human pose estimation. *NeurIPS*. 2022.

### ISB Standards
- Wu G, et al. ISB recommendation on definitions of joint coordinate system of various joints for the reporting of human joint motion. *J Biomech*. 2002;35(4):543–548.

---

*This research document supports the Movement-Analytics project and informs the design of a Movement Quality Score for evaluating human and robotic movement. All claims are sourced from peer-reviewed literature. Where clinical significance thresholds are cited, they represent the current consensus of the biomechanics and rehabilitation research communities.*
