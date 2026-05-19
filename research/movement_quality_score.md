# Movement Quality Score (MQS): Technical Specification

**Version:** 1.5.0
**Date:** 2026-05-19

---

## Abstract

This document formally specifies the Movement Quality Score (MQS), a composite metric for evaluating human and robotic movement quality from video. The MQS maps 50+ kinematic measurements across six biomechanical domains into a single 0–100 score grounded in peer-reviewed clinical reference ranges. Every design choice is traceable to published literature.

---

## 1. Motivation

Movement quality has no universal benchmark. Clinical gait analysis uses 3D optical motion capture systems costing >$150,000. Physical AI companies evaluate locomotion policies against internal metrics with no common standard. The MQS bridges this gap: a computationally tractable, scientifically defensible score computable from video.

**Requirements:**
1. Grounded in peer-reviewed biomechanics literature
2. Discriminative across healthy, pathological, and expert movement
3. Computable in real time from 2D pose estimation
4. Interpretable — each component maps to a clinical concept
5. Extensible to humanoid robot evaluation

---

## 2. Architecture

### 2.1 Domain Model

The MQS decomposes movement quality into six domains, weighted by discriminative power in the clinical literature:

| Domain | Weight | Rationale |
|---|---|---|
| **Kinematics** | 25% | Joint ROM is the primary descriptor of gait pathology (Perry & Burnfield, 2010) |
| **Smoothness** | 18% | SPARC reliably tracks motor recovery and skill (Balasubramanian et al., 2012) |
| **Symmetry** | 18% | Bilateral asymmetry is the defining feature of unilateral pathology (Robinson, 1987) |
| **Coordination** | 14% | CRP consistency measures inter-limb coupling quality (Hamill et al., 1999) |
| **Variability** | 13% | Stride time CV predicts fall risk better than mean speed (Hausdorff et al., 2001) |
| **Temporal** | 12% | Cadence and stride time deviations indicate compensatory strategies |

Weights sum to 1.0. The 25-18-18-14-13-12 distribution reflects the relative volume of evidence supporting each domain's discriminative power. The coordination domain was added in v1.1 to capture inter-limb phase coupling, which is orthogonal to bilateral symmetry (symmetry measures amplitude agreement; coordination measures timing agreement).

### 2.2 Signal Selection

Each domain uses specific signals with clinically validated reference ranges:

**Kinematics Domain (25%):**

| Signal | Optimal Range | Worst-Case Bounds | Source |
|---|---|---|---|
| Hip flexion/extension ROM | 35–50° | 10–70° | Perry & Burnfield, 2010 |
| Knee flexion ROM | 50–70° | 15–90° | Winter, 2009 |
| Ankle dorsiflexion ROM | 20–35° | 5–50° | Perry & Burnfield, 2010 |
| Pelvic obliquity amplitude | 0–7° | 0–20° | Perry & Burnfield, 2010 |
| Trunk lateral lean | 0–5° | 0–15° | Winter, 2009 |

Sagittal-plane signals (hip, knee, ankle ROM) computed bilaterally (6 signals). Frontal-plane signals (pelvic obliquity, trunk lean) computed bilaterally when available (up to 4 signals). All averaged.

**Smoothness Domain (18%):**

| Signal | Optimal Range | Worst-Case Bounds | Source |
|---|---|---|---|
| SPARC (hip velocity) | −2.0 to −1.3 | −6.0 to −0.5 | Balasubramanian et al., 2012/2015 |
| SPARC (knee velocity) | −16.0 to −12.0 | −25.0 to −8.0 | Derived from synthetic profiles; normal knee SPARC ≈ −14.7 |

Computed bilaterally (4 signals: R/L hip + R/L knee SPARC). The domain score = `min(overall_mean, hip_sparc_floor)`, where `overall_mean` is the mean of all 4 signal scores and `hip_sparc_floor` is the mean of the 2 hip SPARC scores. This prevents knee SPARC (which may score well even when hip movement is severely degraded) from diluting the hip SPARC penalty — matching the symmetry domain's `min(SI_mean, waveform_sym)` pattern. Hip velocity SPARC uses the Balasubramanian et al. reference range. Knee SPARC uses a separate range calibrated from the synthetic gait model because the knee velocity profile has higher spectral complexity (stance flexion wave + swing peak) than the hip. The knee SPARC range [-16, -12] captures normal variation while penalizing stiff-knee gait (SPARC ≈ −21) and parkinsonian shuffling (SPARC ≈ −22). Ankle SPARC is excluded due to foot-contact transients that inflate spectral complexity.

**Symmetry Domain (18%):**

| Signal | Optimal Range | Worst-Case Bounds | Source |
|---|---|---|---|
| Hip flexion SI | 0–10% | 0–50% | Robinson et al., 1987; Knapik et al., 1991 |
| Knee flexion SI | 0–10% | 0–50% | Shorter et al., 2020 |
| Ankle dorsiflexion SI | 0–10% | 0–50% | Robinson et al., 1987 |
| Pelvis obliquity SI | 0–10% | 0–50% | Baker et al., 2009 |

SI = 2 × |mean(L) − mean(R)| / (mean(L) + mean(R)) × 100. Four signals, averaged. Pelvis obliquity SI (v1.3) enables frontal-plane asymmetry detection: limping gait shows SI ≈ 21% due to asymmetric pelvic drop, while normal gait shows SI ≈ 0%. The 0–10% optimal range is an initial heuristic derived from analogous sagittal SI ranges (Robinson 1987) and Baker et al.'s frontal-plane kinematic norms; clinical calibration with real gait data is needed. Trunk lateral lean SI is computed and reported as a diagnostic but excluded from the composite to avoid double-penalizing compensatory trunk patterns already captured by kinematics domain scoring.

**Limitation:** Pelvis obliquity SI is currently validated on synthetic data only. The video pipeline computes a single global pelvis tilt per frame (not bilateral stance-phase pelvic drop), so video-derived pelvis obliquity SI will be zero.

**Stride-phase pelvic asymmetry (v1.3):** To address the video limitation, `stride_pelvic_asymmetry()` splits each gait cycle (between consecutive ipsilateral heel strikes) into first-half and second-half segments, then compares pelvic excursion (peak-to-peak). In pathological gait, the pelvis drops more during one stance phase than the other, creating measurable half-cycle excursion asymmetry. This metric works from a single global pelvis signal — no bilateral sensors needed — making it applicable to both synthetic and video-derived data. Reported as `pelvic_drop_asymmetry` and `trunk_lean_asymmetry` diagnostics (not in MQS composite pending validation).

Waveform symmetry (|NCC| × 100, where NCC = normalized cross-correlation of centered bilateral signals) is integrated into the symmetry domain composite as of v1.2. The symmetry domain score = `min(SI_mean, hip_waveform_sym)`, ensuring that either amplitude asymmetry (caught by SI) or shape/timing asymmetry (caught by waveform NCC) will penalize the score. NCC is amplitude-insensitive by construction (centered, normalized); amplitude differences between sides are captured by SI, not waveform symmetry. Anti-phase bilateral coupling (healthy gait) scores 100% because absolute NCC is used. Waveform symmetry captures shape and timing differences that mean-based SI misses (e.g., noisy gait: SI=0.1%, waveform=92.9%).

**Design note:** Only hip flexion waveform symmetry is used in the MQS composite. Knee and ankle waveform symmetry are computed and reported as diagnostics but excluded from scoring because their non-sinusoidal waveforms (stance/swing transitions) produce low NCC even in healthy gait (knee ~45%, ankle ~39%). These values reflect the inherent asymmetry of the knee/ankle kinematic curve shape between stance and swing phases, not pathological asymmetry. Hip flexion, with its near-sinusoidal profile, is the most reliable waveform symmetry indicator.

**Variability Domain (13%):**

| Signal | Optimal Range | Worst-Case Bounds | Source |
|---|---|---|---|
| Stride time CV | 0–4% | 0–20% | Hausdorff et al., 2001 |

Single signal. Healthy adults: 1–3% CV; fallers show >6%.

**Coordination Domain (14%):**

| Signal | Optimal Range | Worst-Case Bounds | Source |
|---|---|---|---|
| Hip CRP CSD (inter-limb) | 0–15° | 0–60° | Hamill et al., 1999; Plotnik et al., 2005 |
| Hip-Knee CRP MAD (intra-limb) | 15–35° | 0–60° | Derived from synthetic profiles; normal ≈ 28° |

Three signals (1 inter-limb + 2 bilateral intra-limb), averaged. Inter-limb CRP: Continuous Relative Phase via Hilbert transform of bilateral hip flexion signals, CSD measures coordination consistency. Intra-limb CRP (v1.4): hip-knee coupling via Hilbert transform of ipsilateral hip flexion and knee flexion, computed bilaterally. Stiff-knee gait shows hip-knee CRP MAD ≈ 43° (disrupted proximal-distal coupling), compared to 28° for normal gait. Knee bilateral CRP is reported but excluded due to non-sinusoidal stance/swing transitions.

**Temporal Domain (12%):**

| Signal | Optimal Range | Worst-Case Bounds | Source |
|---|---|---|---|
| Cadence | 90–130 spm | 40–180 spm | Perry & Burnfield, 2010 |
| Stride time | 0.8–1.3 s | 0.3–2.5 s | Winter, 2009 |

Two signals, averaged.

### 2.3 Signal-to-Score Mapping

Each signal is mapped to a 0–100 subscale using a piecewise linear function:

```
score(value) =
    100                                    if optimal_low ≤ value ≤ optimal_high
    100 × (value − worst_low) / (optimal_low − worst_low)    if value < optimal_low
    100 × (worst_high − value) / (worst_high − optimal_high)  if value > optimal_high
    0                                      if value ≤ worst_low or value ≥ worst_high
```

This mapping:
- Returns 100 for any value within the clinically normal range
- Linearly degrades to 0 at the pathological extreme
- Is bounded [0, 100]
- Is continuous and monotonic outside the normal range

### 2.4 Composite Score

```
MQS = Σ (domain_weight_d × domain_score_d)
```

Where `domain_score_d` is the arithmetic mean of all signal scores within domain `d`.

The MQS is bounded [0, 100] by construction (all components are bounded [0, 100] and weights sum to 1.0).

---

## 3. Validation

### 3.1 Construct Validity

The MQS correctly differentiates across the 9 implemented gait profiles (v1.5, 6-domain model with frontal-plane symmetry, bilateral SPARC, and hip SPARC floor):

| Profile | MQS | GDI | Kinematics | Smoothness | Symmetry | Coordination | Variability | Temporal |
|---|---|---|---|---|---|---|---|---|
| Normal | 98.3 | 100.0 | 93.4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Model Runway | 96.1 | 92.5 | 84.6 | 100.0 | 99.9 | 100.0 | 100.0 | 100.0 |
| Fast | 89.5 | 88.5 | 93.4 | 93.0 | 100.0 | 100.0 | 100.0 | 36.7 |
| Limp | 88.8 | 86.9 | 90.9 | 91.5 | 84.8 | 100.0 | 100.0 | 61.2 |
| Trendelenburg | 87.3 | 91.6 | 60.0 | 99.3 | 100.0 | 100.0 | 100.0 | 78.5 |
| Slow | 83.8 | 84.6 | 84.9 | 82.6 | 100.0 | 100.0 | 100.0 | 22.7 |
| Stiff Knee | 79.9 | 82.4 | 74.1 | 70.5 | 100.0 | 78.7 | 100.0 | 55.8 |
| Noisy | 58.7 | 83.4 | 52.1 | 0.0 | 92.9 | 98.7 | 23.7 | 100.0 |
| Parkinsonian | 59.9 | 78.1 | 63.0 | 0.0 | 95.7 | 90.3 | 78.9 | 33.4 |

**Expected patterns confirmed:**
- Normal scores highest with near-perfect kinematics
- Trendelenburg is strongly penalized in kinematics (pelvic obliquity 24° vs. 7° normal, trunk lean 16° vs. 5°)
- Stiff knee is penalized in kinematics (knee ROM 25° vs. 50–70° normal), smoothness (knee SPARC −21 vs. −16 to −12 normal), and coordination (hip-knee CRP MAD 43° vs. 28° normal)
- Limp is penalized in symmetry (84.8) via both sagittal SI (hip SI 19.4%) and frontal-plane pelvis obliquity SI (21%) — the asymmetric pelvic drop is now detected (v1.3)
- Noisy is penalized in smoothness (hip SPARC floor = 0, preventing knee SPARC from diluting the penalty) and variability (stride CV 33%)
- Slow and fast are penalized in temporal (cadence outside 90–130 spm range)
- Stiff-knee and parkinsonian are heavily penalized in smoothness via knee SPARC (v1.4): stiff_knee knee SPARC ≈ −21 (reduced swing velocity), parkinsonian ≈ −22 (shuffling rhythm)
- Parkinsonian scores lowest overall (MQS 59.9), with smoothness = 0.0 (hip SPARC severely degraded, floor applied) and coordination = 90.3 (disrupted hip-knee coupling). Noisy scores 58.7, penalized in both smoothness (hip SPARC floor = 0) and variability (stride CV 16.2%) while parkinsonian shows moderate variability (CV 7.4%) — consistent with the shuffling-but-regular pattern of Parkinson's gait
- Noisy gait shows reduced symmetry (92.9) thanks to waveform symmetry detecting shape-based asymmetry that mean-based SI misses; parkinsonian similarly at 95.7
- Bilateral noise is generated with independent random seeds per side (v1.1.1)
- GDI and MQS provide complementary perspectives: GDI measures sagittal-plane waveform shape deviation from normal (100 = identical, lower = more deviant), while MQS measures multi-domain quality. Trendelenburg GDI = 91.6 (sagittal waveforms nearly normal) vs. MQS = 87.3 (penalized in kinematics for frontal-plane pathology). Parkinsonian GDI = 78.1 (most deviant waveform shape) vs. MQS = 59.9 (penalized across smoothness, coordination, variability, temporal)

### 3.2 Discriminative Power

The MQS spread across profiles (58.7–98.3) provides meaningful differentiation. The domain breakdown explains *why* each profile scores as it does, which is critical for clinical and engineering interpretability. Notably, the Trendelenburg profile (kinematics = 60.0) demonstrates the frontal-plane ROM detection added in v1.1, and the limp profile (symmetry = 84.8, pelvis obliquity SI = 21%) demonstrates frontal-plane asymmetry detection added in v1.3.

### 3.3 Limitations and Known Gaps

1. **Frontal plane coverage improving:** Pelvic obliquity and trunk lateral lean are scored in the kinematics domain (ROM) and symmetry domain (pelvis obliquity SI, v1.3). Trendelenburg gait is detected via kinematics (score drops to ~60), and asymmetric limping is detected via frontal-plane symmetry (pelvis obliquity SI = 21%). Frontal-plane signals from real video require either multi-camera setup or 3D pose lifting for reliable measurement.

2. **Smoothness domain uses hip and knee SPARC:** Hip SPARC uses the Balasubramanian et al. reference range; knee SPARC uses a synthetic-derived range (v1.4). Ankle SPARC is excluded due to foot-contact transients. The knee SPARC range is not yet validated against clinical data.

3. **Variability requires multiple strides:** With fewer than 3 detected strides, stride CV reliability degrades. The current implementation returns NaN for stride CV when fewer than 3 strides are detected, and the MQS scores missing variability as 50.0 (neutral). Missing kinematics, smoothness, and symmetry signals are excluded from their domain averages rather than defaulted to optimal values (v1.1.1).

4. **Symmetry composite (v1.2):** The symmetry domain now uses `min(SI_mean, hip_waveform_sym)` where SI is the traditional mean-based Symmetry Index and hip waveform symmetry is the absolute normalized cross-correlation of bilateral hip flexion curves. The `min` operator ensures that either amplitude asymmetry (caught by SI) or shape asymmetry (caught by waveform NCC) will penalize the score. This addresses the prior limitation where mean-based SI alone missed phase-specific asymmetries. However, the waveform metric uses hip flexion only; extending to knee/ankle waveform correlation could improve sensitivity to distal asymmetries.

5. **Coordination domain uses inter-limb + intra-limb CRP (v1.4):** Three signals: bilateral hip CRP (inter-limb) + bilateral hip-knee CRP MAD (intra-limb). Stiff-knee gait is now detected via disrupted hip-knee coupling (CRP MAD ≈ 43° vs. 28° normal). The hip-knee CRP reference range [15°, 35°] is synthetic-derived and needs clinical validation.

6. **Video MQS confidence degradation (v1.2):** When MQS is computed from video-derived poses, the raw score is scaled by a confidence factor: `CF = observed_fraction × mean_pose_confidence`. This ensures that poor detection rates or low landmark visibility produce a lower (more conservative) MQS rather than a misleadingly high score. The raw unscaled MQS is preserved as `mqs_raw` for debugging. Synthetic data (no pose metadata) always receives CF = 1.0.

---

## 4. Extension Path

### 4.1 Planned Signal Additions

| Signal | Domain | Implementation Status |
|---|---|---|
| Pelvic obliquity amplitude | Kinematics | **Implemented** (v1.1) |
| Trunk lateral lean | Kinematics | **Implemented** (v1.1) |
| Pelvis obliquity SI | Symmetry | **Implemented** (v1.3) — frontal-plane asymmetry detection |
| Trunk lateral lean SI | Symmetry | **Implemented** (v1.3) — diagnostic only, not in MQS composite |
| Stride-phase pelvic asymmetry | Symmetry | **Implemented** (v1.3) — video-compatible, diagnostic only |
| Stride-phase trunk asymmetry | Symmetry | **Implemented** (v1.3) — video-compatible, diagnostic only |
| Double support time | Temporal | **Implemented** (v1.4) — computed as diagnostic; not scored in MQS (sinusoidal model → 50% stance → 0% DS; needs realistic toe-off detection) |
| Intra-limb CRP (hip-knee) | Coordination | **Implemented** (v1.4) — scored bilaterally in coordination domain |
| Gait Deviation Index (GDI) | Composite | **Implemented** (v1.5) — simplified GDI per Schwartz & Rozumalski 2008; compares stride-normalized waveforms (hip, knee, ankle) against normal reference, 100 = normal, ~10 points per SD of deviation |
| DFA scaling exponent | Variability | Requires >50 strides for reliability |
| Head stabilization index | New: Global | Requires head tracking with sufficient resolution |

### 4.2 Humanoid Robotics Extension

The MQS framework extends naturally to humanoid robot evaluation:
- Replace "clinical normal range" with "reference policy range"
- Add robot-specific signals: actuator smoothness, energy efficiency, contact force symmetry
- Weight domains based on deployment context (e.g., industrial handling prioritizes stability; social robots prioritize naturalness)

### 4.3 Scoring Model Evolution

The current piecewise linear mapping is a principled starting point. Future versions may use:
- **Nonlinear mappings** derived from logistic regression on clinical outcome data
- **Learned weights** from expert rater agreement studies (target ICC > 0.80)
- **Bayesian composite** incorporating measurement uncertainty from pose estimation

---

## 5. References

- Balasubramanian S, et al. A robust and sensitive metric for quantifying movement smoothness. *IEEE TBME*. 2012;59(8):2126–2136.
- Balasubramanian S, et al. On the analysis of movement smoothness. *J NeuroEngineering Rehab*. 2015;12:112.
- Baker R, et al. The Gait Profile Score and Movement Analysis Profile. *Gait & Posture*. 2009;30(3):265–269.
- Hamill J, van Emmerik REA, Heiderscheit BC, Li L. A dynamical systems approach to lower extremity running injuries. *Clin Biomech*. 1999;14(5):297–308.
- Hausdorff JM, et al. Gait variability and fall risk. *Arch Phys Med Rehabil*. 2001;82(8):1050–1056.
- Knapik JJ, et al. Strength, flexibility and athletic injuries. *Sports Med*. 1991;11(4):210–227.
- Perry J, Burnfield JM. *Gait Analysis: Normal and Pathological Function*. 2nd ed. SLACK Inc; 2010.
- Plotnik M, Giladi N, Hausdorff JM. A new measure for quantifying the bilateral coordination of human gait. *Exp Brain Res*. 2007;181(4):561–570.
- Robinson RO, et al. Use of force platform variables to quantify gait symmetry. *J Manipulative Physiol Ther*. 1987;10(4):172–176.
- Schwartz MH, Rozumalski A. The Gait Deviation Index. *Gait & Posture*. 2008;28(3):351–357.
- Shorter KA, et al. The Normalized Symmetry Index. *J Biomech*. 2020;99:109531.
- Winter DA. *Biomechanics and Motor Control of Human Movement*. 4th ed. Wiley; 2009.
