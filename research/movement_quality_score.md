# Movement Quality Score (MQS): Technical Specification

**Version:** 1.0
**Date:** 2026-05-19

---

## Abstract

This document formally specifies the Movement Quality Score (MQS), a composite metric for evaluating human and robotic movement quality from video. The MQS maps 50+ kinematic measurements across five biomechanical domains into a single 0–100 score grounded in peer-reviewed clinical reference ranges. Every design choice is traceable to published literature.

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

The MQS decomposes movement quality into five orthogonal domains, weighted by discriminative power in the clinical literature:

| Domain | Weight | Rationale |
|---|---|---|
| **Kinematics** | 30% | Joint ROM is the primary descriptor of gait pathology (Perry & Burnfield, 2010) |
| **Smoothness** | 20% | SPARC reliably tracks motor recovery and skill (Balasubramanian et al., 2012) |
| **Symmetry** | 20% | Bilateral asymmetry is the defining feature of unilateral pathology (Robinson, 1987) |
| **Variability** | 15% | Stride time CV predicts fall risk better than mean speed (Hausdorff et al., 2001) |
| **Temporal** | 15% | Cadence and stride time deviations indicate compensatory strategies |

Weights sum to 1.0. The 30-20-20-15-15 distribution reflects the relative volume of evidence supporting each domain's discriminative power.

### 2.2 Signal Selection

Each domain uses specific signals with clinically validated reference ranges:

**Kinematics Domain (30%):**

| Signal | Optimal Range | Worst-Case Bounds | Source |
|---|---|---|---|
| Hip flexion/extension ROM | 35–50° | 10–70° | Perry & Burnfield, 2010 |
| Knee flexion ROM | 50–70° | 15–90° | Winter, 2009 |
| Ankle dorsiflexion ROM | 20–35° | 5–50° | Perry & Burnfield, 2010 |

Computed bilaterally (6 signals total), averaged.

**Smoothness Domain (20%):**

| Signal | Optimal Range | Worst-Case Bounds | Source |
|---|---|---|---|
| SPARC (hip velocity) | −2.0 to −1.3 | −6.0 to −0.5 | Balasubramanian et al., 2012/2015 |

Computed bilaterally (2 signals), averaged. Hip velocity SPARC is used because hip flexion has the smoothest sinusoidal profile in healthy gait, making it the most reliable smoothness indicator. Knee and ankle SPARC are excluded from the composite due to sharp phase transitions that inflate spectral complexity independent of movement quality.

**Symmetry Domain (20%):**

| Signal | Optimal Range | Worst-Case Bounds | Source |
|---|---|---|---|
| Hip flexion SI | 0–10% | 0–50% | Robinson et al., 1987; Knapik et al., 1991 |
| Knee flexion SI | 0–10% | 0–50% | Shorter et al., 2020 |
| Ankle dorsiflexion SI | 0–10% | 0–50% | Robinson et al., 1987 |

SI = 2 × |mean(L) − mean(R)| / (mean(L) + mean(R)) × 100. Three signals, averaged.

**Variability Domain (15%):**

| Signal | Optimal Range | Worst-Case Bounds | Source |
|---|---|---|---|
| Stride time CV | 0–4% | 0–20% | Hausdorff et al., 2001 |

Single signal. Healthy adults: 1–3% CV; fallers show >6%.

**Temporal Domain (15%):**

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

The MQS correctly differentiates across the 8 implemented gait profiles:

| Profile | MQS | Kinematics | Smoothness | Symmetry | Variability | Temporal |
|---|---|---|---|---|---|---|
| Normal | 100 | 100 | 100 | 100 | 100 | 100 |
| Model Runway | 100 | 100 | 100 | 100 | 100 | 100 |
| Trendelenburg | 96.8 | 100 | 100 | 100 | 100 | 78.5 |
| Limp | 90.7 | 95.8 | 100 | 88.9 | 100 | 61.2 |
| Fast | 89.5 | 100 | 95.2 | 100 | 100 | 36.7 |
| Slow | 84.2 | 85.8 | 100 | 100 | 100 | 22.7 |
| Stiff Knee | 83.7 | 67.8 | 100 | 100 | 100 | 55.8 |
| Noisy | 60.2 | 100 | 0 | 100 | 0 | 68.1 |

**Expected patterns confirmed:**
- Normal and model runway score highest (both exhibit clinically optimal kinematics)
- Stiff knee is penalized in kinematics (knee ROM 25° vs. 50–70° normal)
- Limp is penalized in symmetry (Hip SI 19.4%)
- Noisy is penalized in smoothness and variability (SPARC degraded, stride CV 25.6%)
- Slow and fast are penalized in temporal (cadence outside 90–130 spm range)

### 3.2 Discriminative Power

The MQS spread across profiles (60.2–100) provides meaningful differentiation. The domain breakdown explains *why* each profile scores as it does, which is critical for clinical and engineering interpretability.

### 3.3 Limitations and Known Gaps

1. **Frontal plane pathologies underweighted:** Trendelenburg gait (excessive pelvic obliquity) scores 96.8 because the current implementation uses sagittal-plane signals. Adding pelvic obliquity to the kinematics domain would improve detection.

2. **Smoothness domain uses only hip SPARC:** This is by design (knee/ankle SPARC is confounded by phase transitions) but misses smoothness deficits in isolated distal joints.

3. **Variability requires multiple strides:** With fewer than 3 detected strides, stride CV reliability degrades. The current implementation defaults to 0 when no strides are detected, which inflates the variability score.

4. **Symmetry uses mean-based SI:** The SI formula uses absolute mean values, which can miss phase-specific asymmetries (e.g., asymmetric push-off timing with symmetric ROM). Waveform-based symmetry metrics (e.g., comparing full gait cycle curves) would improve sensitivity.

5. **No coordination domain:** Inter-limb coordination (CRP) is identified in the research as signal #14 but not yet implemented in the MQS computation.

---

## 4. Extension Path

### 4.1 Planned Signal Additions

| Signal | Domain | Implementation Status |
|---|---|---|
| Pelvic obliquity amplitude | Kinematics | Data available, scoring not yet integrated |
| Trunk lateral lean | Kinematics | Data available, scoring not yet integrated |
| CRP consistency | New: Coordination | Requires phase computation implementation |
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
- Hausdorff JM, et al. Gait variability and fall risk. *Arch Phys Med Rehabil*. 2001;82(8):1050–1056.
- Knapik JJ, et al. Strength, flexibility and athletic injuries. *Sports Med*. 1991;11(4):210–227.
- Perry J, Burnfield JM. *Gait Analysis: Normal and Pathological Function*. 2nd ed. SLACK Inc; 2010.
- Robinson RO, et al. Use of force platform variables to quantify gait symmetry. *J Manipulative Physiol Ther*. 1987;10(4):172–176.
- Schwartz MH, Rozumalski A. The Gait Deviation Index. *Gait & Posture*. 2008;28(3):351–357.
- Shorter KA, et al. The Normalized Symmetry Index. *J Biomech*. 2020;99:109531.
- Winter DA. *Biomechanics and Motor Control of Human Movement*. 4th ed. Wiley; 2009.
