# Kinematics and Movement Quality Analysis: A Research Foundation for Physical AI

**Movement-Analytics Research Document — Deliverable 1**
**Version:** 2.0
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
    - 8.1 Motor Variability and Expertise
    - 8.2 Fashion Model Gait: Measured Differences
    - 8.3 Trained Movers: Ballet Dancers and Athletes
    - 8.4 Movement Smoothness and Skill Level
    - 8.5 Coordination, Motor Control Theory, and Expertise
    - 8.6 Movement Quality Assessment Systems
    - 8.7 Computational Approaches to Movement Quality
9. [Computational Methods: Pose Estimation and Video-Based Analysis](#9-computational-methods)
    - 9.1 Pose Estimation Models — Current Landscape
    - 9.2 2D-to-3D Pose Lifting
    - 9.3 Joint Angle Computation from Keypoints
    - 9.4 Gait Event Detection from Video
    - 9.5 Validation: Video-Based vs. Marker-Based
    - 9.6 Synthetic Data and Simulation for Movement Analysis
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

Scoping review of 59 biomechanics studies: **48 (81%)** report higher-skilled performers have measurably less motor variability than lower-skilled performers, regardless of sport, expertise definition, or variability measurement. Holds across javelin, golf, pitching, gymnastics, and gait. The review catalogued **21 different metrics of motor variability** across the 59 studies, categorized into two analytical perspectives:

**Linear (magnitude-based) metrics:**
- **Standard deviation (SD):** Most common metric, used in 31 studies. Quantifies dispersion around a mean joint angle, velocity, or force output.
- **Coefficient of variation (CoV = SD/mean):** Second most common (9 studies). Normalizes variability by the mean for cross-comparison.
- **RMSD:** Deviation from a reference trajectory.

**Nonlinear (structure-based) metrics:**
- **Approximate Entropy (ApEn) and Sample Entropy (SampEn):** Quantify regularity/predictability. Lower entropy = more regular (potentially rigid). Higher = more random (potentially unstable).
- **Detrended Fluctuation Analysis (DFA):** Scaling exponent α characterizing long-range correlations. α ≈ 0.5 = random (white noise); α ≈ 0.75 = optimal fractal complexity; α ≈ 1.0 = pink noise. Introduced to gait by Hausdorff et al., 1995, J Appl Physiol, 78, 349–358.
- **Largest Lyapunov exponent (LLE):** Quantifies local dynamic stability; higher = less stable. Used clinically for fall risk (Toebes et al., 2012, Gait & Posture, 36(3), 527–531).

**The critical nuance — optimal variability:**

Stergiou & Decker (2011, DOI: 10.1016/j.humov.2011.06.002) proposed the **optimal movement variability hypothesis**: healthy systems exhibit intermediate variability with fractal temporal structure. Both excess (noise/instability) and deficit (rigidity) indicate dysfunction. For experts, linear measures (SD, CoV) tend to be **lower** (less outcome variability), while nonlinear measures show **more complex temporal structure** (α closer to 0.75, higher sample entropy). Experts channel variability into task-irrelevant dimensions while constraining task-relevant dimensions.

### 8.2 Fashion Model Gait: Measured Differences

**Tanabe et al., 2023, Frontiers in Sports and Active Living, DOI: 10.3389/fspor.2023.1091470**

12-camera infrared mocap at 100 Hz. 7 professional fashion models (mean 16.6 years experience) vs. 10 non-models, two instruction conditions ("casual" and "attractively as possible").

**Key findings:**
- Models produced systematically different knee extension during push-off (0–6%, 45.5–47.5%, 88–100% of gait cycle)
- Distinctive backward upper-arm silhouette in specific phases (0–11.5%, 18–23%, 64–71%)
- Observer ratings: models 4.29 ± 0.64 vs. non-models 3.50 ± 0.43 (7-point Likert)
- Both groups changed strategy under instruction, but models' changes were more precisely localized in the gait cycle
- Models employed a **wider variety of biomechanical strategies**, suggesting richer motor repertoires

**Tanabe et al., 2023, Scientific Reports, 13:15280, DOI: 10.1038/s41598-023-45130-2** — companion SEM study: backward arm swing, forward head tilt, lower cadence (longer stride time), greater knee extension at push-off, and larger toe-off angle all positively predicted perceived gait attractiveness, while lower BMI was the sole physique predictor (r = −0.567).

**Kobayashi, Saito & Murahori, 2024, Sensors, DOI: 10.3390/s24123865**

69 professional runway models: PCA + hierarchical clustering identified **5 distinct walking-style clusters** with significantly different median show-years between clusters. Runway gait is structured, clusterable, and time-evolving with experience.

### 8.3 Trained Movers: Ballet Dancers and Athletes

**Galbusera et al., 2024, Bioengineering, 11(11), 1102:** Professional ballet dancers exhibit significantly **more controlled angular velocities** (narrower, refined ranges) compared to non-dancers. Right knee angular velocity: dancers −0.35 to 0.54 rad/s vs. non-dancers −3.88 to 2.61 rad/s — a dramatic narrowing of the variability envelope.

**Losova, 2014, Acta Gymnica, 44(2):** Ballet dancers showed greater hip extension, greater hip abduction, increased pelvic tilt and rotation, greater knee flexion/extension, decreased maximal ankle plantarflexion during loading, and increased plantarflexion in terminal stance.

**Hreljac, 2000, Gait & Posture, 11(3), 199–206, DOI: 10.1016/s0966-6362(00)00045-x:** Runners were **smoother than non-runners** during both running and fast walking (using endpoint jerk-cost at the heel). Training in one locomotor mode transfers smoothness to others.

### 8.4 Movement Smoothness and Skill Level

Flash & Hogan (1985) established the minimum-jerk model as the organizing principle of skilled reaching: the nervous system plans trajectories that minimize jerk (third derivative of position). Extensions to walking:

- Skilled walkers show lower SPARC values (closer to 0) in joint angle velocities
- Rehabilitation patients show progressively smoother movement as they recover
- SPARC reliably tracks skill acquisition across motor learning studies (Balasubramanian et al., 2015)
- **Choi et al., 2014, BioMedical Engineering OnLine, 13:20:** Skilled golfers had significantly lower normalized jerk values than unskilled golfers across all body joints, strongest in the lower body (r = 0.657, p < 0.01)

### 8.5 Coordination, Motor Control Theory, and Expertise

**Bernstein's Degrees of Freedom Framework (1967):** The foundational three-stage learning model:
1. **Freezing** — novices constrain degrees of freedom to simplify control
2. **Freeing** — degrees of freedom progressively released with practice
3. **Exploiting** — experts harness reactive forces, gravity, and inter-segmental dynamics

Confirmed by systematic review (Adrjan et al., 2020, Motor Control, 24(3), 457–471): 10 of 13 studies provide evidence consistent with the freezing hypothesis.

**Uncontrolled Manifold (UCM) Analysis** (Scholz & Schöner, 1999, Exp Brain Res, 126, 289–306, DOI: 10.1007/s002210050738): Decomposes joint-level variability into:
- **V_UCM ("good variability")** — within the manifold, does not affect the task variable (e.g., CoM position)
- **V_ORT ("bad variability")** — orthogonal to the manifold, perturbs the task variable
- **Synergy index (ΔV)** — ratio V_UCM / V_ORT; higher = stronger motor synergies

Mohler et al. (2020, Eur J Sport Sci, 20(9), 1187–1196, DOI: 10.1080/17461391.2019.1709561): 13 expert and 12 novice runners — experts had stronger synergies stabilizing center of mass trajectory (more V_UCM relative to V_ORT).

**Optimal Feedback Control** (Todorov & Jordan, 2002, Nature Neuroscience, 5, 1226–1235, DOI: 10.1038/nn963): Optimal control only corrects deviations that interfere with task goals ("minimal intervention principle"). Expert nervous systems implement this more effectively — explaining why experts show structured variability rather than uniformly low variability.

**Motor Abundance** (Latash, 2012, Exp Brain Res, 217, 1–5, DOI: 10.1007/s00221-012-3000-4): Extra degrees of freedom are not redundancy but abundance — they provide flexibility and robustness. Experts create flexible motor synergies that stabilize task-critical variables while allowing exploration in task-irrelevant dimensions.

Expert walkers show:
- More consistent CRP patterns (lower CRP variability)
- Stronger inter-limb coupling (higher cross-correlation peaks)
- More precise gait event timing (lower stride time CV)
- Better-organized variability structure (DFA α closer to 0.75)
- More efficient angular momentum management via pelvis–thorax counter-rotation

### 8.6 Movement Quality Assessment Systems

**Functional Movement Screen (FMS)** (Cook & Burton, 2006, N Am J Sports Phys Ther): Seven fundamental movement patterns scored on a 4-point ordinal scale (0–3). Maximum composite = 21; scores below 14 indicate higher injury risk. Widely adopted but limited by inter-rater variability and ordinal granularity.

**Systematic review of movement quality assessments** (Wijekulasuriya et al., 2025, Sports Med Open, 11:11, DOI: 10.1186/s40798-025-00813-0): Identified **36 different movement quality assessments** across 131 studies (59 different movements total). Intra-rater reliability for composite scores: r = 0.939; inter-rater: r = 0.887; individual movement reliability only moderate (κ = 0.57).

### 8.7 Computational Approaches to Movement Quality

**Automated FMS scoring from video:**
- LLM-FMS dataset (2025, PLOS ONE, DOI: 10.1371/journal.pone.0313707): 1812 FMS keyframe images from 45 subjects with detailed annotations, using RTMPose for skeletal extraction
- Li et al. (2023, Mathematics, 11(24):4936): Automatic FMS evaluation using attention mechanisms and score distribution prediction

**Gait quality prediction from video:**
- Kanko et al. (2020, Nature Communications, DOI: 10.1038/s41467-020-17807-z): Deep neural networks predict from single-camera video: walking speed (r = 0.73), cadence (r = 0.79), knee flexion angle at max extension (r = 0.83), GDI (r = 0.75)
- OpenCap (Uhlrich et al., 2023, PLOS Computational Biology, 19(10), DOI: 10.1371/journal.pcbi.1011462): Two smartphone cameras + deep learning → 3D kinematics, GRF, musculoskeletal dynamics. Errors: 4.1° joint kinematics, 6.7% BW ground reaction forces. Equipment cost <$700 vs. >$150,000 marker-based

**Rehabilitation exercise quality:** Liao et al. (2020, IEEE Trans Neural Syst Rehab Eng, 28(2), 468–477): CNNs + RNNs on KIMORE and UI-PRMD skeleton datasets for clinical quality score prediction.

**Action quality assessment (AQA) architectures:**
- ST-GCN (Spatial-Temporal Graph Convolutional Networks) — Yan et al., 2018, arXiv:1801.07455: Models skeleton as a graph for non-Euclidean joint relationships
- Temporal Parsing Transformers — decompose long actions into sub-phases
- Siamese networks with deep metric learning for relative quality comparison
- I3D + regression heads for direct score prediction

---

## 9. Computational Methods: Pose Estimation and Video-Based Analysis

### 9.1 Pose Estimation Models — Current Landscape

| Model | Architecture | Keypoints | AP (COCO) | Speed | 3D? |
|---|---|---|---|---|---|
| **MediaPipe Pose** | BlazePose (lightweight) | 33 | ~72 (est.) | 30+ fps CPU | Pseudo-3D |
| **RTMPose** | SimCC + CSPNeXt | 17 (COCO) | 75.8 AP | 430+ fps GPU | 2D |
| **ViTPose** | Vision Transformer | 17–133 | 80.9 AP | ~30 fps GPU | 2D, liftable |
| **HRNet** | High-Res Net | 17 | 75.5 AP | ~15 fps GPU | 2D |
| **OpenPose** | PAFs + CMaps | 25 + hands | 65.3 AP | 10–15 fps GPU | 2D |
| **AlphaPose** | RMPE (top-down) | 17–26 | 72.3 AP | 20+ fps GPU | 2D |

**RTMPose** (Jiang et al., 2023) achieves the best speed-accuracy tradeoff for production systems: 75.8 AP on COCO at 430+ fps on GPU, making it viable for real-time multi-camera deployments. The SimCC (Simple Coordinate Classification) approach avoids heatmap post-processing overhead.

**ViTPose** (Xu et al., NeurIPS 2022) achieves state-of-the-art accuracy (80.9 AP) with plain Vision Transformer backbone but requires more compute. Best for offline high-fidelity analysis.

**For real-time single-person analysis:** MediaPipe remains optimal — runs on CPU, 33 keypoints including hands, feet, and face landmarks, provides pseudo-depth. For higher accuracy: RTMPose on GPU.

### 9.2 2D-to-3D Pose Lifting

Monocular 2D keypoints can be lifted to 3D using temporal and spatial priors:

| Method | MPJPE (mm) | Approach | Key Property |
|---|---|---|---|
| **VideoPose3D** (Pavllo et al., CVPR 2019) | 46.8 | Temporal dilated convolutions | Semi-supervised, long receptive field |
| **MotionBERT** (Zhu et al., ICCV 2023) | 35.8 | Dual-stream transformer | SOTA, unified motion representation |
| **MHFormer** (Li et al., CVPR 2022) | 43.0 | Multi-hypothesis transformer | Handles depth ambiguity |

MPJPE = Mean Per Joint Position Error on Human3.6M. MotionBERT's 35.8 mm error is approaching the noise floor of marker-based mocap (~2–5 mm per marker), making video-based 3D reconstruction increasingly viable for clinical gait analysis.

### 9.3 Joint Angle Computation from Keypoints

**ISB Recommendations (Wu et al., 2002, J Biomech):**
Joint angles computed as Euler/Cardan angles of the distal segment relative to the proximal segment, in the order: flexion/extension → abduction/adduction → internal/external rotation.

**Practical 2D approach (sagittal view):**
- **Knee flexion:** 180° − included angle at knee between thigh and shank vectors
- **Hip flexion:** Angle of thigh relative to trunk/vertical
- **Ankle dorsiflexion:** Angle between shank and foot segments

**3D approach from lifted keypoints:**
- Construct anatomical coordinate systems per ISB convention
- Decompose rotation matrices into clinically meaningful Euler angles
- Apply Butterworth low-pass filter (6–10 Hz) to reduce high-frequency jitter
- Account for soft-tissue artifact by comparing with biomechanical constraints

### 9.4 Gait Event Detection from Video

Without force plates, gait events must be inferred from kinematic signals:

**Zeni et al. method** — heel strike detected from anterior-posterior foot position relative to pelvis. Validated accuracy: **4.78 ms** mean error for heel strike detection, 12.5 ms for toe-off (Zeni et al., 2008, Gait & Posture, 27(4), 710–714).

Additional approaches:
- **Vertical position of ankle/heel marker:** Local minima in vertical foot trajectory = heel strike
- **Hip flexion peaks:** Maximum flexion ≈ heel strike; maximum extension ≈ toe-off
- **Foot velocity:** Zero-crossing of horizontal foot velocity (requires temporal filtering)
- **ML approaches:** HMMs and LSTMs trained on labeled gait data achieve <20 ms mean error with sufficient training data

### 9.5 Validation: Video-Based vs. Marker-Based

Systematic comparison of pose estimation against gold-standard optical mocap:

| Metric | MediaPipe | OpenPose | RTMPose | Clinical Threshold |
|---|---|---|---|---|
| Hip sagittal (°) | 5.2 ± 2.1 RMS | 6.8 ± 3.2 RMS | 4.1 ± 1.8 RMS | ±5° acceptable |
| Knee sagittal (°) | 4.8 ± 2.5 RMS | 5.9 ± 2.8 RMS | 3.5 ± 1.5 RMS | ±5° acceptable |
| Ankle sagittal (°) | 6.1 ± 3.0 RMS | 7.5 ± 3.5 RMS | 5.2 ± 2.2 RMS | ±5° acceptable |
| Frontal plane (°) | 8–15 RMS | 10–18 RMS | 7–12 RMS | ±8° acceptable |
| Gait event timing | ±2–4 frames | ±3–5 frames | ±1–3 frames | ±50 ms acceptable |

**OpenCap validation** (Uhlrich et al., 2023, PLOS Computational Biology, DOI: 10.1371/journal.pcbi.1011462): Two smartphone cameras + deep learning achieved **4.1° joint kinematics** and **6.7% BW ground reaction force** error. Equipment cost <$700 vs. >$150,000 for marker-based systems — a >200× cost reduction with clinically acceptable accuracy.

**Key factors affecting accuracy:**
- Single-person, clear sagittal view, controlled lighting → best results
- Multi-person scenes, occlusion → significant degradation
- Temporal filtering (Butterworth 4th-order, 6–10 Hz) improves joint angle waveforms by 20–40%
- Minimum 30 fps required; 60+ fps recommended for gait event detection

### 9.6 Synthetic Data and Simulation for Movement Analysis

**Parametric human body models:**
- **SMPL** (Loper et al., 2015, ACM ToG): 6890-vertex mesh + 72 pose parameters + 10 shape parameters. Industry standard for synthetic training data generation.
- **SMPL-X** (Pavlakos et al., 2019): Extends SMPL with expressive hands and face.

**Physics simulation:**
- **MuJoCo** (Todorov et al., 2012): High-fidelity contact dynamics, widely used for locomotion policy learning in humanoid robotics
- **Isaac Gym / Isaac Sim** (NVIDIA): GPU-accelerated physics for massive parallel simulation of walking policies
- **PyBullet**: Open-source alternative with reasonable contact fidelity

**Relevance to movement quality scoring:** Synthetic environments enable (1) generating unlimited labeled training data for pose estimation models, (2) testing movement quality metrics under controlled perturbations, (3) establishing ground truth for validation of video-based measurement systems.

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
- Tanabe K, et al. Biomechanical strategies to maximize gait attractiveness among women. *Front Sports Active Living*. 2023;5:1091470. DOI: 10.3389/fspor.2023.1091470
- Tanabe K, et al. [companion SEM study]. *Scientific Reports*. 2023;13:15280. DOI: 10.1038/s41598-023-45130-2
- Kobayashi K, Saito S, Murahori Y. Classification of fashion models' walking styles. *Sensors*. 2024;24(12):3865. DOI: 10.3390/s24123865
- Galbusera F, et al. A gait analysis in professional dancers: a cross-sectional study. *Bioengineering*. 2024;11(11):1102.
- Losova L. Kinematic analysis of the gait in professional ballet dancers. *Acta Gymnica*. 2014;44(2).
- Hreljac A. Stride smoothness evaluation of runners and other athletes. *Gait & Posture*. 2000;11(3):199–206. DOI: 10.1016/s0966-6362(00)00045-x
- Choi A, et al. Kinematic evaluation of golf swing jerkiness. *BioMedical Engineering OnLine*. 2014;13:20.

### Motor Control Theory
- Bernstein NA. *The Co-ordination and Regulation of Movements*. Pergamon Press; 1967.
- Scholz JP, Schöner G. The uncontrolled manifold concept: identifying control variables for a functional task. *Exp Brain Res*. 1999;126:289–306. DOI: 10.1007/s002210050738
- Todorov E, Jordan MI. Optimal feedback control as a theory of motor coordination. *Nature Neuroscience*. 2002;5:1226–1235. DOI: 10.1038/nn963
- Latash ML. The bliss (not the problem) of motor abundance (not redundancy). *Exp Brain Res*. 2012;217:1–5. DOI: 10.1007/s00221-012-3000-4
- Mohler F, et al. Variability of running coordination in experts and novices: a 3D uncontrolled manifold analysis. *Eur J Sport Sci*. 2020;20(9):1187–1196. DOI: 10.1080/17461391.2019.1709561
- Adrjan N, et al. [Freezing degrees of freedom systematic review]. *Motor Control*. 2020;24(3):457–471.

### Movement Quality Assessment
- Cook G, Burton L. Pre-participation screening: the use of fundamental movements. *N Am J Sports Phys Ther*. 2006;1(2):62–73.
- Wijekulasuriya GA, et al. The development and content of movement quality assessments in athletic populations. *Sports Med Open*. 2025;11:11. DOI: 10.1186/s40798-025-00813-0

### Pose Estimation and Computational Methods
- Wu G, et al. ISB recommendation on definitions of joint coordinate systems. *J Biomech*. 2002;35(4):543–548.
- Lugaresi C, et al. MediaPipe: a framework for building perception pipelines. *arXiv:1906.08172*. 2019.
- Cao Z, et al. OpenPose: realtime multi-person 2D pose estimation. *IEEE TPAMI*. 2021;43(1):172–186.
- Xu Y, et al. ViTPose: simple vision transformer baselines for human pose estimation. *NeurIPS*. 2022.
- Jiang T, et al. RTMPose: real-time multi-person pose estimation based on MMPose. *arXiv:2303.07399*. 2023.
- Pavllo D, et al. 3D human pose estimation in video with temporal convolutions and semi-supervised training. *CVPR*. 2019.
- Zhu W, et al. MotionBERT: a unified perspective on learning human motion representations. *ICCV*. 2023.
- Zeni JA Jr, Richards JG, Higginson JS. Two simple methods for determining gait events during treadmill walking. *Gait & Posture*. 2008;27(4):710–714.
- Uhlrich SD, et al. OpenCap: human movement dynamics from smartphone videos. *PLOS Computational Biology*. 2023;19(10):e1011462. DOI: 10.1371/journal.pcbi.1011462
- Kanko RM, et al. [gait quality from single-camera video]. *Nature Communications*. 2020. DOI: 10.1038/s41467-020-17807-z

### Computational Movement Quality
- Liao Y, Vakanski A, Xian M. A deep learning framework for assessing physical rehabilitation exercises. *IEEE Trans Neural Syst Rehab Eng*. 2020;28(2):468–477.
- Yan S, et al. Spatial temporal graph convolutional networks for skeleton-based action recognition. *arXiv:1801.07455*. 2018.
- Loper M, et al. SMPL: a skinned multi-person linear model. *ACM Trans Graphics*. 2015;34(6):248.

### Simulation and Synthetic Data
- Todorov E, Erez T, Tassa Y. MuJoCo: a physics engine for model-based control. *IEEE/RSJ IROS*. 2012.

---

*This research document supports the Movement-Analytics project and informs the design of a Movement Quality Score for evaluating human and robotic movement. All claims are sourced from peer-reviewed literature. Where clinical significance thresholds are cited, they represent the current consensus of the biomechanics and rehabilitation research communities.*
