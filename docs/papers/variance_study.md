---
title: "Quantitative Kinematic Variance Analysis of Professional Runway Walking vs. Unconstrained Internet Walking Video"
author: "Phoebe Richmond --- Elysium Intelligence --- phoebe@joinelysium.ai"
date: "2026"
---

## Abstract {-}

We present a quantitative analysis comparing the kinematic variance of professional runway model walks against heterogeneous control internet walking video. Using a computational pipeline that extracts 24 biomechanical metrics from monocular video via MediaPipe pose estimation, we test the hypothesis that runway walks exhibit significantly lower kinematic variance across movement quality domains. Our analysis of 22 runway videos and 17 control internet walking videos reveals that control walking data exhibits 2.56$\times$ higher median variance (Levene's test), with 10 of 24 metrics showing statistically significant differences ($p < 0.05$), all in the hypothesized direction. The strongest effects appear in movement smoothness (17$\times$) and bilateral symmetry (14$\times$), while coordination (1.3$\times$) and temporal parameters (1.7$\times$) show smaller differences. Three of 24 metrics show the reverse direction (higher runway variance), though none significantly. Multivariate analysis via PCA suggests runway walks occupy a 2.13$\times$ tighter region in kinematic feature space ($\sim$5$\times$ by volume, 25$\times$ by generalized variance), though the multivariate analysis is limited by small complete-case sample sizes ($n = 20$). These findings provide partial but directionally consistent support for the thesis that professional runway walking constitutes a low-variance, high-quality kinematic distribution that motivates evaluation as a training source for robotic movement learning.

**Keywords:** gait analysis, kinematic variance, movement quality, pose estimation, robot learning, biomechanics

---

## Introduction

The development of physical AI systems capable of human-like locomotion requires high-quality training data that captures the essential structure of bipedal walking while minimizing noise and inter-subject variability. While internet-scale video corpora provide enormous volume, they introduce proportional variance: differences in walking speed, terrain, footwear, pathology, camera angle, and individual biomechanics create a training distribution with high entropy across kinematic dimensions.

We hypothesize that professional runway model walks constitute a naturally low-variance kinematic distribution. The fashion modeling profession itself acts as a variance filter through three mechanisms:

1. **Selection pressure** — agencies and designers select models whose baseline gait exhibits specific aesthetic properties (fluid motion, bilateral symmetry, controlled trunk)
2. **Training convergence** — professional models undergo extensive coaching to standardize their walk, reducing inter-individual differences
3. **Environmental control** — runway shows enforce consistent conditions (flat surface, known distance, controlled pace, frontal camera alignment)

This paper tests this hypothesis quantitatively by extracting 24 biomechanical metrics from both runway and internet walking video using a computational pipeline, then applying rigorous statistical analysis to compare distributional properties across groups.

---

## Methods

### Data Collection

**Runway dataset.** 22 video recordings of professional runway model walks (MOV format, 30 fps, 1080p). All videos capture a single model walking toward or away from camera on a straight runway.

**Control dataset.** 20 walking videos sourced from YouTube using `yt-dlp`, selected via diverse search queries ("person walking on sidewalk," "gait assessment walking," "hallway walking exercise," "walking stock footage"). Each video was trimmed to 30 seconds at ≤720p resolution. After quality filtering (requiring ≥30% pose detection rate), 17 control videos remained.

### Pose Estimation

We use MediaPipe PoseLandmarker (heavy model, VIDEO mode) to extract 33 anatomical keypoints per frame. For each frame $t$, the pose estimator returns a set of 3D landmarks:

$$\mathbf{L}_t = \{(x_i^t, y_i^t, z_i^t, v_i^t)\}_{i=1}^{33}$$

where $(x_i, y_i, z_i)$ are normalized coordinates and $v_i \in [0,1]$ is the visibility confidence for landmark $i$.

**Multi-person tracking.** When multiple persons are detected ($|\mathcal{P}_t| > 1$), we track the target subject using pelvis centroid continuity. The pelvis centroid at frame $t$ is:

$$\mathbf{c}_t = \frac{1}{2}\left(\mathbf{p}_{\text{left\_hip}}^t + \mathbf{p}_{\text{right\_hip}}^t\right)$$

The tracked person is selected by minimizing Euclidean displacement from the previous frame:

$$\text{person}^* = \arg\min_{p \in \mathcal{P}_t} \|\mathbf{c}_t^{(p)} - \mathbf{c}_{t-1}\|_2$$

### Joint Angle Computation

Joint angles are computed from 2D landmark positions using the vector dot product formulation. For three landmarks $A$, $B$, $C$ defining a joint at vertex $B$:

$$\theta = \arccos\left(\frac{(\mathbf{A} - \mathbf{B}) \cdot (\mathbf{C} - \mathbf{B})}{\|\mathbf{A} - \mathbf{B}\| \cdot \|\mathbf{C} - \mathbf{B}\|}\right)$$

For joints computed via the three-point included angle method, the **flexion angle** is defined as:

$$\theta_{\text{flex}} = 180° - \theta_{\text{included}}$$

where $\theta_{\text{included}} = \arccos\left(\frac{(\mathbf{A} - \mathbf{B}) \cdot (\mathbf{C} - \mathbf{B})}{\|\mathbf{A} - \mathbf{B}\| \cdot \|\mathbf{C} - \mathbf{B}\|}\right)$

This yields 0° at full extension and increasing values with flexion.

The anatomical angles extracted are:

| Joint | Proximal ($A$) | Vertex ($B$) | Distal ($C$) | Transform | Plane |
|-------|----------|----------|---------|-----------|-------|
| Hip flexion | Shoulder center$^\dagger$ | Hip | Knee | $180° - \theta$ | Sagittal |
| Knee flexion | Hip | Knee | Ankle | $180° - \theta$ | Sagittal |
| Ankle dorsiflexion | Knee | Ankle | Foot index | $90° - \theta$ | Sagittal |
| Shoulder flexion | Neck$^\ddagger$ | Shoulder | Elbow | $180° - \theta$ | Sagittal |
| Elbow flexion | Shoulder | Elbow | Wrist | $180° - \theta$ | Sagittal |
| Pelvis obliquity | — | — | — | See below | Frontal |
| Trunk lean | — | — | — | See below | Frontal |

$^\dagger$ Shoulder center = midpoint of left and right shoulders, representing the trunk reference axis (not the ipsilateral shoulder).

$^\ddagger$ Neck = shoulder center position. In the video pipeline, `neck` is defined as `shoulder_center.copy()` — the midpoint of left and right shoulder landmarks. The resulting angle measures neck-shoulder-elbow flexion.

**Ankle dorsiflexion** uses a 90° offset rather than 180°: $\theta_{\text{ankle}} = 90° - \theta_{\text{included}}(K, A, T)$ where $K$ = knee, $A$ = ankle, $T$ = foot index. This maps the anatomical neutral position (foot perpendicular to shank) to 0°.

**Pelvis obliquity** is computed from the hip-to-hip vector angle relative to horizontal:

$$\theta_{\text{pelvis}} = \left|\arctan2\left(y_{\text{R\_hip}} - y_{\text{L\_hip}},\; x_{\text{R\_hip}} - x_{\text{L\_hip}}\right)\right|$$

The absolute value yields unsigned obliquity (magnitude of pelvic tilt). A signed variant is stored separately for direction-sensitive analysis.

**Trunk lateral lean** uses the segment-to-vertical angle between pelvis and shoulder center:

$$\theta_{\text{trunk}} = \arctan2\left(x_{\text{shoulder\_center}} - x_{\text{pelvis}},\; y_{\text{shoulder\_center}} - y_{\text{pelvis}}\right)$$

Note: in pixel coordinates where $y$ increases downward, the denominator is typically negative (shoulders above pelvis in the image), so positive values indicate lateral deviation from vertical.

### Signal Processing Pipeline

Raw joint angle time series from video-based pose estimation contain noise from landmark jitter, occlusions, and multi-person ambiguity. We apply a 6-stage signal processing pipeline:

**Stage 1: Physiological outlier rejection.** Values outside clinically plausible ranges are replaced with NaN:

$$\theta_t^{\text{clean}} = \begin{cases} \theta_t & \text{if } \theta_{\min}^{\text{physio}} \leq \theta_t \leq \theta_{\max}^{\text{physio}} \\ \text{NaN} & \text{otherwise} \end{cases}$$

Reference ranges: hip flexion $[-50°, 120°]$, knee flexion $[-20°, 160°]$, ankle dorsiflexion $[-90°, 60°]$.

**Stage 2: Median pre-filter.** A 5-sample running median filter removes impulsive noise (single-frame spikes) without distorting genuine movement dynamics:

$$\tilde{\theta}_t = \text{median}\{\theta_{t-2}, \theta_{t-1}, \theta_t, \theta_{t+1}, \theta_{t+2}\}$$

This filter removes spikes up to 2 frames wide. It operates only on contiguous non-NaN segments, preserving gap structure.

**Stage 3: PCHIP interpolation.** Gaps (NaN sequences from occlusions or outlier rejection) are filled using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation. Given observed values at indices $\{t_k, \theta_k\}$, PCHIP constructs a $C^1$-continuous piecewise cubic:

On each interval $[t_k, t_{k+1}]$, with $s = (t - t_k)/h_k$:

$$\hat{\theta}(t) = \theta_k H_{00}(s) + h_k d_k H_{10}(s) + \theta_{k+1} H_{01}(s) + h_k d_{k+1} H_{11}(s)$$

where $H_{ij}$ are the cubic Hermite basis functions, $h_k = t_{k+1} - t_k$, and $d_k$ are the monotonicity-preserving derivatives (Fritsch & Carlson, 1980). PCHIP avoids the Runge oscillation artifacts of standard cubic spline interpolation.

For signals with >50% missing data, we fall back to linear interpolation.

**Stage 4: Post-interpolation outlier rejection.** A second physiological range check removes any artifacts introduced by interpolation across large gaps.

**Stage 5: Confidence-adaptive Butterworth smoothing.** A two-pass low-pass Butterworth filter suppresses remaining high-frequency noise:

$$|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2n}}$$

where $n = 2$ (filter order) and $\omega_c$ is the cutoff angular frequency. We apply `filtfilt` (zero-phase forward-backward filtering) for zero phase distortion:

- **Pass 1 (low-confidence frames):** For frames where MediaPipe confidence $v_t < 0.7$, apply aggressive cutoff $f_c = 3$ Hz
- **Pass 2 (all frames):** Standard cutoff $f_c = 6$ Hz across the full signal

The Nyquist frequency constraint requires $f_c < f_s / 2$ where $f_s$ is the video frame rate.

**Stage 6: Completeness check.** When overall signal completeness drops below 50%, the Movement Quality Score returns NaN rather than a misleading result.

### Gait Metric Extraction

From the processed joint angle time series, we compute 24 biomechanical metrics spanning 8 domains.

#### Range of Motion (ROM)

For each joint angle time series $\theta(t)$ of duration $T$:

$$\text{ROM} = \max_{t \in [0,T]} \theta(t) - \min_{t \in [0,T]} \theta(t)$$

Clinical reference ranges (Perry & Burnfield, 2010): hip flexion 40--45°, knee flexion 60--65°, ankle dorsiflexion ~30°.

#### Spectral Arc Length (SPARC)

SPARC quantifies movement smoothness via the arc length of the normalized frequency spectrum of the velocity profile (Balasubramanian et al., 2012). Given angular velocity $\dot{\theta}(t)$:

1. Compute the discrete Fourier transform:

$$\hat{V}(f) = \text{FFT}\{\dot{\theta}(t)\}, \quad f \in [0, f_s/2]$$

2. Normalize the amplitude spectrum:

$$\hat{V}_{\text{norm}}(f) = \frac{|\hat{V}(f)|}{\max_f |\hat{V}(f)|}$$

3. Determine the adaptive frequency cutoff $f_c^{\text{adapt}}$: the highest frequency where $\hat{V}_{\text{norm}}(f) \geq 0.05$ (amplitude threshold), capped at $f_c = 10$ Hz.

4. Compute the spectral arc length as the negative arc length of the normalized amplitude spectrum curve:

$$\text{SPARC} = -\sum_{k=1}^{K-1} \sqrt{(f_{k+1} - f_k)^2 + (\hat{V}_{\text{norm}}(f_{k+1}) - \hat{V}_{\text{norm}}(f_k))^2}$$

where $K$ is the number of frequency bins up to $f_c^{\text{adapt}}$, and $f_k$ are the discrete frequency values from the FFT. This is a direct arc-length computation over the $(f, \hat{V}_{\text{norm}})$ curve in the frequency-amplitude plane — each term is the Euclidean distance between consecutive points on the curve.

SPARC is always negative; values closer to zero indicate smoother movement. Smooth movements have compact frequency spectra (short arc length); jerky movements spread energy across frequencies (long arc length). For healthy gait: SPARC $\approx -1.5$ to $-1.7$ for hip velocity.

Note: The original Balasubramanian et al. (2012) formulation uses a continuous integral with a frequency normalization factor $(1/f_c)^2$. Our implementation follows the discrete unnormalized form, which is an implementation variant that may change absolute values and can alter ordering when adaptive cutoffs differ across signals. Within-study relative comparisons remain valid as long as groups share similar frequency content ranges.

#### Normalized Jerk (NJ)

The dimensionless normalized jerk metric (Teulings et al., 1997):

$$\text{NJ} = \sqrt{\frac{T^5}{2A^2} \int_0^T \left(\frac{d^3\theta}{dt^3}\right)^2 dt}$$

where $T = N \cdot \Delta t$ is the movement duration, $N$ is the number of samples, $\Delta t = 1/f_s$ is the sampling interval, and $A = \max(\theta) - \min(\theta)$ is the peak-to-peak amplitude. Lower values indicate smoother movement.

The discrete approximation replaces the integral with a Riemann sum:

$$\text{NJ} \approx \sqrt{\frac{T^5}{2A^2} \cdot \Delta t \sum_{t=1}^{N} \left(\dddot{\theta}_t\right)^2}$$

The third derivative $\dddot{\theta}_t$ is computed via cascaded second-order central finite differences (three successive applications of NumPy's `gradient`).

#### Symmetry Index (SI)

The Robinson Symmetry Index (Robinson et al., 1987) quantifies bilateral asymmetry:

$$\text{SI} = \frac{2|X_R - X_L|}{X_R + X_L} \times 100\%$$

where $X_R$ and $X_L$ are the mean absolute values of the right and left joint angle time series respectively:

$$X_R = \frac{1}{T}\sum_{t=1}^{T} |\theta_R(t)|, \quad X_L = \frac{1}{T}\sum_{t=1}^{T} |\theta_L(t)|$$

Perfect symmetry yields $\text{SI} = 0\%$. Clinically, $\text{SI} < 10\%$ is considered normal (Shorter et al., 2008).

#### Waveform Symmetry via Normalized Cross-Correlation (NCC)

Beyond amplitude-based SI, we capture bilateral shape and timing differences using the normalized cross-correlation:

$$\text{NCC}(\theta_L, \theta_R) = \frac{\sum_{t=1}^{T} (\theta_L(t) - \bar{\theta}_L)(\theta_R(t) - \bar{\theta}_R)}{\|\theta_L - \bar{\theta}_L\|_2 \cdot \|\theta_R - \bar{\theta}_R\|_2}$$

$$\text{Waveform Symmetry} = |\text{NCC}| \times 100\%$$

NCC is invariant to amplitude scaling (captured separately by SI) and measures purely shape-based similarity. A value of 100% indicates identical bilateral waveforms. The absolute value is used because anti-phase bilateral coupling (expected in normal gait for hip flexion) represents symmetric coordination.

#### Continuous Relative Phase (CRP)

CRP quantifies inter-limb and intra-limb coordination via the Hilbert transform (Hamill et al., 1999). For two oscillating signals $\theta_a(t)$ and $\theta_b(t)$:

1. Center each signal: $\tilde{\theta}_a(t) = \theta_a(t) - \bar{\theta}_a$

2. Compute the analytic signal via the Hilbert transform:

$$z_a(t) = \tilde{\theta}_a(t) + j\mathcal{H}\{\tilde{\theta}_a(t)\}$$

where $\mathcal{H}$ denotes the Hilbert transform:

$$\mathcal{H}\{x(t)\} = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{x(\tau)}{t - \tau} d\tau$$

3. Extract instantaneous phase: $\phi_a(t) = \text{atan2}(\text{Im}(z_a), \text{Re}(z_a))$

4. Compute the continuous relative phase:

$$\text{CRP}(t) = ((\phi_a(t) - \phi_b(t) + 180°) \bmod 360°) - 180°$$

**CRP consistency** is quantified via the circular standard deviation of the CRP signal, using the mean resultant length $R$:

$$R = \sqrt{\left(\frac{1}{T}\sum_t \cos\phi_t\right)^2 + \left(\frac{1}{T}\sum_t \sin\phi_t\right)^2}$$

$$\text{CSD} = \sqrt{-2\ln R} \quad \text{(in radians, converted to degrees)}$$

Healthy inter-limb coordination: $\text{CSD} < 15°$. Pathological: $\text{CSD} > 30°$. For bilateral hip flexion in normal gait, $\text{CRP} \approx 180°$ (anti-phase coupling).

#### Arm Swing Metrics

Arm swing ROM and symmetry are computed from bilateral shoulder flexion angles:

$$\text{arm\_swing\_ROM} = \frac{\text{ROM}(\theta_{\text{R\_shoulder}}) + \text{ROM}(\theta_{\text{L\_shoulder}})}{2}$$

$$\text{arm\_swing\_SI} = \text{SI}(\theta_{\text{R\_shoulder}}, \theta_{\text{L\_shoulder}})$$

The arm swing ratio normalizes against a clinical reference of 25° shoulder flexion ROM:

$$\text{arm\_swing\_ratio} = \text{clip}\left(\frac{\text{arm\_swing\_ROM}}{25°}, \; 0, \; 2\right)$$

Normal arm swing ratio $\approx 1.0$; values $< 0.7$ suggest diminished arm swing (characteristic of Parkinsonian gait).

#### Coefficient of Variation (CV)

Stride-to-stride variability is quantified by the coefficient of variation:

$$\text{CV} = \frac{\sigma}{\mu} \times 100\%$$

where $\sigma$ and $\mu$ are the standard deviation and mean of the stride intervals (or per-stride ROM values). We compute:

- **Stride time CV:** Variability of inter-heel-strike intervals. Normal: 1--3% (Hausdorff et al., 2001).
- **Kinematic CV:** Mean of per-joint, per-stride ROM coefficients of variation. Normal: 0--5%.

#### Gait Deviation Index (GDI)

Our simplified GDI (after Schwartz & Rozumalski, 2008) compares stride-normalized joint angle waveforms against a normal reference:

1. Segment gait cycles using detected heel strikes $\{HS_1, HS_2, \ldots, HS_{n+1}\}$, yielding $n$ complete strides (requiring at least 3 heel strikes for 2 strides)

2. Normalize each stride to 101 points via linear interpolation:

$$\theta_{\text{norm}}^{(i)}(p) = \text{interp}\left(\theta^{(i)}, \frac{p}{100}\right), \quad p \in \{0, 1, \ldots, 100\}$$

3. Compute RMS deviation from the normal reference for each joint $j$ and stride $i$:

$$d_j^{(i)} = \sqrt{\frac{1}{101}\sum_{p=0}^{100}\left(\theta_{\text{norm},j}^{(i)}(p) - \theta_{\text{ref},j}(p)\right)^2}$$

4. Average across joints and strides:

$$\bar{d} = \frac{1}{n \cdot |\mathcal{J}|} \sum_{i=1}^{n} \sum_{j \in \mathcal{J}} d_j^{(i)}$$

5. Convert to GDI score (100 = normal, decreasing with deviation), clipped at 0:

$$\text{GDI} = \max\left(0, \; 100 - \frac{\bar{d}}{5°} \times 10\right)$$

6. Average bilateral scores: $\text{GDI} = (\text{GDI}_L + \text{GDI}_R) / 2$

Each 5° RMS deviation reduces GDI by approximately 10 points. The 5° constant is calibrated to approximate the standard deviation of normal gait variability in clinical gait lab data: healthy subjects typically show ~5° RMS deviation from the mean waveform across repeated trials, so each additional SD of deviation reduces the score by 10 points. This is a simplified adaptation of the original Schwartz & Rozumalski (2008) GDI, which uses PCA on 9 kinematic features from clinical datasets; our version uses direct RMS comparison against a synthetic normal reference (see Appendix AI for a detailed comparison).

#### Gait Event Detection

Heel strikes are detected as peaks in the low-pass filtered hip flexion signal (maximum hip flexion $\approx$ heel strike):

$$\text{HS} = \{t : \theta_{\text{hip}}^{\text{filt}}(t) \text{ is a local maximum with prominence} > \max(0.15 \cdot \text{ROM}_{\text{hip}}, 1°)\}$$

When video-derived heel Y-coordinates are available, heel strike timing is refined by finding the lowest heel position (maximum pixel Y) within ±0.15s of each hip flexion peak:

$$t_{\text{HS}}^* = \arg\max_{|t - t_{\text{HS}}| \leq 0.15f_s} y_{\text{heel}}(t)$$

**Cadence** is derived from stride timing. Since the detector identifies ipsilateral heel strikes, each stride interval contains two steps:

$$\text{cadence} = \frac{60}{\bar{\Delta t}_{\text{stride}}} \times 2 \quad \text{(steps/min)}$$

Normal cadence: 100--120 steps/min (Perry & Burnfield, 2010).

**Double support percentage** is estimated from the stance phase fraction within each gait cycle. If $S$ denotes the stance fraction (heel-strike to toe-off as a proportion of stride duration), then in normal gait:

- Each leg spends $S$ of the cycle in stance and $(1 - S)$ in swing
- Double support occurs twice per cycle: at initial contact and pre-swing
- Total double support = $2S - 1$ (valid when $S > 0.5$, which is always true for walking)

$$\text{DS}\% = \max\left(0, \; 2S - 1\right) \times 100, \quad S = \frac{t_{\text{toe-off}} - t_{\text{heel-strike}}}{t_{\text{stride}}}$$

Normal double support: ~20% of the gait cycle at comfortable walking speed.

### Movement Quality Score (MQS)

MQS is a composite 0--100 score aggregating 6 biomechanical domains with literature-derived weights:

$$\text{MQS} = \sum_{d \in \mathcal{D}} w_d \cdot S_d$$

where the domain weights $w_d$ sum to 1:

| Domain $d$ | Weight $w_d$ | Signals |
|------------|-------------|---------|
| Kinematics | 0.25 | Hip/knee/ankle ROM, pelvis obliquity, trunk lean |
| Smoothness | 0.18 | Bilateral hip and knee SPARC |
| Symmetry | 0.18 | Hip/knee/ankle/pelvis SI + hip waveform symmetry |
| Coordination | 0.14 | Inter-limb CRP (bilateral hip) + intra-limb CRP (hip-knee) |
| Variability | 0.13 | Stride time CV + kinematic ROM CV |
| Temporal | 0.12 | Cadence + stride time vs. normal ranges |

Each domain score $S_d$ is the mean of its constituent signal scores. Individual signal scores are computed by mapping metric values to [0, 100] via piecewise linear transfer functions anchored to clinical reference ranges:

$$S_{\text{signal}}(v) = \begin{cases}
0 & \text{if } v \leq v_{\text{worst,lo}} \\
100 \cdot \dfrac{v - v_{\text{worst,lo}}}{v_{\text{opt,lo}} - v_{\text{worst,lo}}} & \text{if } v_{\text{worst,lo}} < v < v_{\text{opt,lo}} \\
100 & \text{if } v_{\text{opt,lo}} \leq v \leq v_{\text{opt,hi}} \\
100 \cdot \dfrac{v_{\text{worst,hi}} - v}{v_{\text{worst,hi}} - v_{\text{opt,hi}}} & \text{if } v_{\text{opt,hi}} < v < v_{\text{worst,hi}} \\
0 & \text{if } v \geq v_{\text{worst,hi}}
\end{cases}$$

Reference ranges are sourced from Perry & Burnfield (2010), Winter (2009), Balasubramanian et al. (2012), Hausdorff et al. (2001), and Hamill et al. (1999).

**Confidence weighting (video pipeline).** For video-derived analysis, MQS is scaled by a confidence factor:

$$\text{MQS}_{\text{weighted}} = \text{MQS} \times c_f$$

$$c_f = f_{\text{obs}} \times \bar{v}_{\text{detected}} \times (1 - p_{\text{interp}})$$

$$p_{\text{interp}} = \text{clip}(\lambda \cdot f_{\text{interp}}, \; 0, \; 0.5), \quad \lambda = 0.5$$

where $f_{\text{obs}}$ is the fraction of frames with detected pose, $\bar{v}_{\text{detected}}$ is the mean landmark confidence across detected frames, and $f_{\text{interp}}$ is the fraction of angle data filled by interpolation. The interpolation penalty $p_{\text{interp}}$ is clipped to $[0, 0.5]$ to prevent the confidence factor from dropping below $0.5 \cdot f_{\text{obs}} \cdot \bar{v}_{\text{detected}}$ even for heavily interpolated signals.

**Sufficient evidence gating.** MQS returns NaN when overall signal completeness drops below 50%. Completeness is computed per-domain as the fraction of expected signals that are present and non-NaN, then averaged across all 6 domains. This prevents misleading scores from sparse pose detection data.

### Statistical Analysis

We apply four layers of statistical testing to compare variance between groups, plus multivariate analysis.

#### Levene's Test for Equality of Variances

For each metric $m$, we test $H_0$: $\sigma^2_{\text{runway}} = \sigma^2_{\text{control}}$ using Levene's test (Levene, 1960). Given two groups of sizes $n_1$ and $n_2$, define:

$$Z_{ij} = |X_{ij} - \tilde{X}_{i}|$$

where $\tilde{X}_i$ is the group median. Our implementation uses SciPy's `levene` with its default `center='median'`, which is technically the **Brown-Forsythe variant** — more robust to non-normality than the original mean-centered Levene's test. We follow the common convention of referring to this as "Levene's test" throughout. The test statistic is:

$$W = \frac{(N - k)}{(k - 1)} \cdot \frac{\sum_{i=1}^{k} n_i (\bar{Z}_{i\cdot} - \bar{Z}_{\cdot\cdot})^2}{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_{i\cdot})^2}$$

where $k = 2$ groups, $N = n_1 + n_2$, $\bar{Z}_{i\cdot}$ is the group mean of the absolute deviations, and $\bar{Z}_{\cdot\cdot}$ is the grand mean. Under $H_0$, $W \sim F(k-1, N-k)$.

The variance ratio (effect size) is:

$$R_\sigma = \frac{s_{\text{control}}^2}{s_{\text{runway}}^2}$$

where $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ is the sample variance with Bessel's correction.

#### Mann-Whitney U Test

For group distributional comparison, we use the non-parametric Mann-Whitney U test (Mann & Whitney, 1947), which tests stochastic ordering without assuming normality:

$$U_1 = R_1 - \frac{n_1(n_1+1)}{2}$$

where $R_1$ is the sum of ranks for group 1 in the combined ranked sample. $U_1$ counts the number of $(i,j)$ pairs where observation $i$ from group 1 exceeds observation $j$ from group 2. The effect size is reported as:

**Cohen's d** (standardized mean difference, using unweighted pooled SD):

$$d = \frac{\bar{X}_{\text{runway}} - \bar{X}_{\text{control}}}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{s_1^2 + s_2^2}{2}}$$

**Rank-biserial correlation** (non-parametric effect size):

$$r_{rb} = 1 - \frac{2U}{n_1 n_2}$$

#### Permutation Tests

To validate variance ratio significance without distributional assumptions, we use one-sided permutation testing with 10,000 iterations (testing $H_1$: control variance > runway variance). For each metric:

1. Compute observed variance ratio: $R_{\text{obs}} = s^2_{\text{control}} / s^2_{\text{runway}}$

2. For $B = 10{,}000$ permutations:
   - Randomly shuffle group labels across the pooled sample
   - Compute permuted variance ratio $R_b^*$

3. Compute $p$-value:

$$p = \frac{|\{b : R_b^* \geq R_{\text{obs}}\}| + 1}{B + 1}$$

The $+1$ in numerator and denominator prevents $p = 0$ and ensures the observed statistic is included (Phipson & Smyth, 2010).

#### Bootstrap Confidence Intervals

We construct 95% confidence intervals on the variance ratio using the percentile bootstrap (Efron & Tibshirani, 1993):

1. For $B = 10{,}000$ bootstrap iterations:
   - Resample with replacement from each group: $\theta_R^* \sim \hat{F}_R$, $\theta_C^* \sim \hat{F}_C$
   - Compute bootstrap variance ratio: $R_b^* = s^{*2}_C / s^{*2}_R$

2. Compute percentile CI: $[\hat{R}_{0.025}^*, \hat{R}_{0.975}^*]$

3. Compute bootstrap proportion: $\hat{P}(R > 1) = \frac{1}{B}\sum_{b=1}^{B} \mathbb{1}[R_b^* > 1]$

#### Multiple Comparison Correction

With 24 simultaneous tests, we apply the Bonferroni correction:

$$\alpha_{\text{Bonf}} = \frac{\alpha}{m} = \frac{0.05}{24} \approx 0.0021$$

A metric is considered significant after correction if $p < \alpha_{\text{Bonf}}$.

**Non-independence of tests.** The 24 metrics are not fully independent — several are derived from the same underlying joint angle signals (e.g., hip\_ROM, hip\_SPARC, and hip\_SI all depend on hip flexion angles), and MQS and GDI are composites of other metrics. Bonferroni correction with $m = 24$ is therefore conservative (true effective number of independent tests is likely $m_{\text{eff}} < 24$). For a more precise correction, one could estimate $m_{\text{eff}}$ from the eigenvalues of the inter-metric correlation matrix. We report both uncorrected and Bonferroni-corrected results to bracket the true significance.

#### Principal Component Analysis (PCA)

To assess multivariate distributional differences, we apply PCA to the standardized feature matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ (excluding MQS and GDI composites):

1. Standardize: $\tilde{\mathbf{X}} = (\mathbf{X} - \boldsymbol{\mu}) / \boldsymbol{\sigma}$

2. Eigen-decompose the covariance matrix: $\mathbf{C} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$

3. Project: $\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{V}_{:,1:k}$ where $k$ components are retained

**Spread metrics.** Within each group $g$, the covariance in PC space is $\mathbf{C}_g = \text{Cov}(\mathbf{Z}_g)$. We compare:

- **Trace** (total variance): $\text{tr}(\mathbf{C}_g) = \sum_{i=1}^{k} \lambda_{g,i}$
- **Determinant** (generalized variance / volume): $|\mathbf{C}_g| = \prod_{i=1}^{k} \lambda_{g,i}$
- **Mean centroid distance**: $\bar{d}_g = \frac{1}{n_g}\sum_{i=1}^{n_g} \|\mathbf{z}_i - \bar{\mathbf{z}}_g\|_2$

The spread ratio $\rho = \text{tr}(\mathbf{C}_{\text{control}}) / \text{tr}(\mathbf{C}_{\text{runway}})$ quantifies how much larger the control distribution is in kinematic feature space.

#### Linear Discriminant Analysis (LDA)

LDA finds the linear projection that maximizes class separability:

$$\mathbf{w}^* = \arg\max_{\mathbf{w}} \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$$

where $\mathbf{S}_B = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T$ is the between-class scatter matrix and $\mathbf{S}_W = \sum_{g} \sum_{i \in g} (\mathbf{x}_i - \boldsymbol{\mu}_g)(\mathbf{x}_i - \boldsymbol{\mu}_g)^T$ is the within-class scatter matrix. The solution is $\mathbf{w}^* \propto \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$.

We report both resubstitution accuracy and Leave-One-Out Cross-Validation (LOO-CV) accuracy to assess generalization.

### Detrended Fluctuation Analysis (DFA)

For stride-to-stride interval sequences with ≥16 strides, we compute the DFA scaling exponent $\alpha$ (Hausdorff et al., 2001):

1. Compute the cumulative deviation series: $Y(k) = \sum_{i=1}^{k} (s_i - \bar{s})$

2. For each scale $n$ (logarithmically spaced from 4 to $N/4$), divide $Y$ into $\lfloor N/n \rfloor$ non-overlapping segments of length $n$

3. Within each segment $j$, fit a linear trend $\hat{Y}_n^{(j)}$ and compute the local RMS:

$$F_j(n) = \sqrt{\frac{1}{n}\sum_{k=1}^{n}\left(Y(jn + k) - \hat{Y}_n^{(j)}(k)\right)^2}$$

4. Average across segments to get the fluctuation function:

$$F(n) = \frac{1}{\lfloor N/n \rfloor}\sum_{j=1}^{\lfloor N/n \rfloor} F_j(n)$$

5. The scaling exponent $\alpha$ is the slope of $\log F(n)$ vs. $\log n$, estimated by ordinary least squares:

$$\alpha = \frac{\sum_i (\log n_i - \overline{\log n})(\log F(n_i) - \overline{\log F})}{\sum_i (\log n_i - \overline{\log n})^2}$$

Interpretation: $\alpha \approx 0.5$ (uncorrelated, random walk — pathological), $\alpha \approx 0.75$ (long-range correlations — healthy), $\alpha \approx 1.0$ ($1/f$ noise — over-correlated).

---

## Results

### Dataset Summary

After quality filtering (≥30% pose detection rate), the analysis includes $n_R = 22$ runway videos and $n_C = 17$ control videos. Runway videos achieved near-perfect pose detection (median 100%), reflecting controlled filming conditions. Three control videos were excluded due to insufficient pose detection.

### Univariate Variance Comparison

**Levene's test** identified 10 of 24 metrics with statistically significant variance differences ($p < 0.05$), all showing higher variance in the control group ($R_\sigma > 1$). Zero metrics showed significantly higher variance in the runway group.

The **median variance ratio** across all 24 metrics is $R_\sigma = 2.56\times$, meaning control walking videos exhibit 2.56 times more kinematic variance than runway walks. The mean ratio is $13.43\times$, driven by extreme values in smoothness and symmetry domains.

**Top variance ratios** (with Levene's $p$-values and bootstrap $P(R > 1)$):

| Metric | Domain | $R_\sigma$ | Levene $p$ | Bonferroni | Bootstrap $P(R>1)$ |
|--------|--------|-----------|------------|------------|-------------------|
| arm\_swing\_SI | Symmetry | 100.2$\times$ | 0.0006 | Yes | 100% |
| GDI | Composite | 57.0$\times$ | <0.0001 | Yes | 100% |
| ankle\_SI | Symmetry | 48.4$\times$ | 0.0038 | No | 100% |
| trunk\_lean\_ROM | Kinematics | 22.5$\times$ | 0.1017 | No | 99.9% |
| hip\_SPARC | Smoothness | 20.8$\times$ | <0.0001 | Yes | 100% |
| knee\_SPARC | Smoothness | 13.3$\times$ | <0.0001 | Yes | 100% |
| hip\_SI | Symmetry | 13.9$\times$ | 0.0301 | No | 99.3% |
| stride\_time | Temporal | 11.0$\times$ | 0.0037 | No | 99.9% |
| ankle\_ROM | Kinematics | 8.7$\times$ | <0.0001 | Yes | 100% |
| hip\_ROM | Kinematics | 3.8$\times$ | 0.0008 | Yes | 99.9% |

Six metrics survive Bonferroni correction ($\alpha = 0.0021$): arm\_swing\_SI, GDI, hip\_SPARC, knee\_SPARC, ankle\_ROM, and hip\_ROM.

### Permutation Test Validation

Permutation tests (10,000 iterations) corroborate Levene's results. All 10 Levene-significant metrics also achieve permutation $p < 0.05$. Additionally, trunk\_lean\_ROM ($p_{\text{perm}} = 0.045$), hip\_SI ($p_{\text{perm}} = 0.024$), stride\_time\_CV ($p_{\text{perm}} = 0.017$), and knee\_ROM ($p_{\text{perm}} = 0.024$) reach significance under permutation testing, suggesting Levene's test is conservative for these metrics.

### Bootstrap Confidence Intervals

Bootstrap 95% CIs on the variance ratio exclude 1.0 (indicating $R_\sigma$ is significantly different from equality) for 9 metrics. Eight metrics achieve $P(R > 1) \geq 99\%$. Notably, all bootstrap CIs are right-skewed, consistent with the heavy-tailed nature of variance ratio distributions.

### Group Mean Differences

Mann-Whitney U tests reveal significant mean differences ($p < 0.05$) in 16 of 24 metrics. Large effect sizes ($|d| > 0.8$) are observed for:

- Shoulder/arm swing ROM ($d = -1.62$, control has much larger arm movements)
- Ankle ROM ($d = -1.51$)
- Elbow ROM ($d = -1.49$)
- Stride time CV ($d = -1.66$, control has much more stride variability)
- Hip SPARC ($d = +1.68$, runway is smoother)
- Hip ROM ($d = -1.35$)

These mean differences indicate that runway walks are not just lower-variance but occupy a distinct region of kinematic space: lower ROM (more controlled range), better smoothness, lower stride variability.

### Domain-Level Summary

| Domain | Median $R_\sigma$ | Significant / Total |
|--------|-------------------|-------------------|
| Composite (GDI + MQS) | 28.8$\times$ | 1/2 |
| Smoothness | 17.0$\times$ | 2/2 |
| Symmetry | 13.9$\times$ | 3/5 |
| Variability | 2.9$\times$ | 0/2 |
| Kinematics | 2.7$\times$ | 3/7 |
| Temporal | 1.7$\times$ | 1/3 |
| Upper Body | 1.6$\times$ | 0/1 |
| Coordination | 1.3$\times$ | 0/2 |

The variance reduction is most pronounced in **smoothness** (17$\times$) and **symmetry** (14$\times$) — precisely the domains that characterize professional movement quality. Kinematics shows moderate variance reduction (2.7$\times$), while coordination and temporal parameters show smaller differences.

### Multivariate Analysis

PCA on 21 standardized features (complete cases: 6 runway, 14 control) reveals:

- **3 components** explain 65.7% of total variance
- PC1 (35.4%) loads primarily on ROM and SPARC metrics
- PC2 (18.0%) loads on temporal parameters (stride time, cadence) and symmetry indices

**Distributional spread in PC space:**

| Metric | Runway | Control | Ratio |
|--------|--------|---------|-------|
| Trace (total variance) | 6.34 | 13.48 | 2.13$\times$ |
| Determinant (generalized variance) | — | — | 25.3$\times$ |
| Mean centroid distance | 2.20 | 3.34 | 1.52$\times$ |

The control distribution is **2.13$\times$ larger by trace** and **25.3$\times$ larger by generalized variance** (determinant ratio) in 3D PC space. Since ellipsoid volume scales as $\sqrt{\det(\mathbf{C})}$, this corresponds to approximately $\sqrt{25.3} \approx 5.0\times$ larger by volume. Both measures indicate that runway walks cluster tightly while control videos scatter across the kinematic feature space.

**Caveat:** With only 6 runway samples in 21-dimensional feature space ($p \gg n_R$), the runway covariance matrix is rank-deficient ($\text{rank} \leq 5$). The trace and determinant ratios are point estimates subject to high sampling uncertainty — the true spread difference could be larger or smaller than estimated, since 21-dimensional structure cannot be reliably estimated from 6 samples. The PCA projection to 3 components mitigates this partially, but the runway covariance in PC space remains estimated from only 6 points. Additionally, 3 PCs explain only 65.7% of total variance; the remaining 34.3% could contain additional group differences not captured here.

**LDA classification:**

| Metric | Accuracy |
|--------|----------|
| Resubstitution | 100% |
| LOO-CV | 70% |
| Majority-class baseline | 70% (14/20 = control) |

The LOO-CV accuracy of 70% equals the majority-class baseline (predicting "control" for all samples), indicating that the LDA classifier does not generalize reliably at this sample size. The 100% resubstitution accuracy indicates that the groups are linearly separable in the training set, but with $p = 21$ features and only $n_R = 6$ runway samples, this is expected even for random data ($p \gg n$ makes perfect separation trivial) and should not be interpreted as evidence of true group distinctiveness. The runway covariance matrix is rank-deficient ($\text{rank} \leq 5$), making LDA estimation unstable. Larger samples would reduce estimation instability, though whether LOO-CV accuracy would improve meaningfully remains an empirical question.

---

## Discussion

### Interpretation of Results

The results provide directionally consistent support for the hypothesis. Of the 10 metrics reaching Levene's significance, all 10 show higher variance in the control group. Three metrics (pelvis obliquity $R_\sigma = 0.33\times$, double support $0.71\times$, MQS $0.60\times$) show the reverse direction, though none reach significance.

The overall picture is one of partial support: 42% of metrics reach significance (below the pre-specified 50% threshold for "supported"), but the effect is entirely one-directional among significant results. Two factors contextualize this:

1. **Statistical power.** With $n = 17$--$22$ per group, the study is underpowered to detect moderate variance differences ($R_\sigma < 5\times$). Many non-significant metrics show the hypothesized direction but lack power.

2. **Domain concentration.** Significance clusters in smoothness (2/2 metrics) and symmetry (3/5 metrics) — the domains most directly relevant to movement quality — rather than distributing randomly across domains.

### Implications for Robot Learning

The 17$\times$ lower smoothness variance and 14$\times$ lower symmetry variance in runway walks directly address a key challenge in imitation learning: distribution shift from noisy training data. We hypothesize that a training distribution with narrower variance in these critical dimensions could:

- Yield more consistent learned policies (lower policy entropy)
- Accelerate convergence during training
- Improve generalization to controlled deployment environments
- Reduce the need for reward shaping to penalize jerky or asymmetric motion

These are motivating hypotheses, not demonstrated outcomes; testing them requires training and evaluating locomotion policies, which is beyond the scope of this study.

### Software Environment

Analysis performed with Python 3.12, MediaPipe 0.10.x (PoseLandmarker heavy model, float16), NumPy 1.26, SciPy 1.13, scikit-learn 1.5, pandas 2.2, matplotlib 3.9.

### Limitations

1. **Sample size.** The control group ($n = 17$) limits statistical power. The multivariate analysis is further constrained by listwise deletion to $n = 20$ complete cases.

2. **Video heterogeneity.** Control videos span diverse camera angles, environments, and subject demographics. While this heterogeneity is by design (representing "internet walking data"), it conflates multiple sources of variance.

3. **Monocular pose estimation.** MediaPipe provides 2D pose estimation from monocular video, which cannot capture out-of-plane joint angles or true 3D kinematics. Depth ambiguity particularly affects frontal-plane metrics.

4. **Selection bias.** Runway videos were professionally filmed with controlled conditions, while control videos were opportunistically sampled. Some variance difference may reflect filming conditions rather than movement quality.

5. **LDA generalization.** The LOO-CV accuracy (70%) equals the majority-class baseline, indicating that multivariate classification does not generalize at this sample size. The resubstitution accuracy (100%) is expected when $p \gg n$ and does not constitute evidence of true group separability.

6. **Non-independent metrics.** Several of the 24 metrics share underlying signal dependencies, inflating the apparent number of independent tests. The effective number of independent comparisons is likely lower than 24.

### Future Work

- Increase sample sizes to $n \geq 50$ per group for improved statistical power
- Add multi-camera or depth-sensor validation
- Test the causal link: train locomotion policies on runway vs. internet data and compare learned policy quality
- Extend to other structured movement domains (martial arts, dance, athletic training)

---

## Conclusion

This study provides quantitative evidence that professional runway model walks constitute a low-variance, high-quality kinematic distribution compared to general internet walking video. Across 24 biomechanical metrics spanning 8 movement quality domains, runway walks show **2.56$\times$ lower median variance**, with the strongest effects in movement smoothness ($17\times$) and bilateral symmetry ($14\times$). In multivariate kinematic feature space, runway walks occupy a region **$\sim$5$\times$ smaller by volume** ($25\times$ by generalized variance). All 10 statistically significant differences favor the hypothesized direction; 3 of 24 metrics show the reverse direction (higher runway variance), though none reach significance.

These findings support the use of runway walking data as a curated, low-entropy source that merits further evaluation for robotic movement learning systems.

---

## References

Baker, R., McGinley, J. L., Schwartz, M. H., et al. (2009). The Gait Profile Score and Movement Analysis Profile. *Gait & Posture*, 30(3), 265--269.

Balasubramanian, S., Melendez-Calderon, A., & Burdet, E. (2012). A robust and sensitive metric for quantifying movement smoothness. *IEEE Transactions on Biomedical Engineering*, 59(8), 2126--2136.

Brown, M. B., & Forsythe, A. B. (1974). Robust tests for the equality of variances. *Journal of the American Statistical Association*, 69(346), 364--367.

Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A., & Burdet, E. (2015). On the analysis of movement smoothness. *Journal of NeuroEngineering and Rehabilitation*, 12(1), 112.

Cooley, J. W., & Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. *Mathematics of Computation*, 19(90), 297--301.

Efron, B. (1979). Bootstrap methods: another look at the jackknife. *The Annals of Statistics*, 7(1), 1--26.

Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

Flash, T., & Hogan, N. (1985). The coordination of arm movements: an experimentally confirmed mathematical model. *Journal of Neuroscience*, 5(7), 1688--1703.

Fritsch, F. N., & Carlson, R. E. (1980). Monotone piecewise cubic interpolation. *SIAM Journal on Numerical Analysis*, 17(2), 238--246.

Hamill, J., van Emmerik, R. E. A., Heiderscheit, B. C., & Li, L. (1999). A dynamical systems approach to lower extremity running injuries. *Clinical Biomechanics*, 14(5), 297--308.

Hausdorff, J. M., et al. (2001). When human walking becomes random walking: fractal analysis and modeling of gait rhythm fluctuations. *Physica A*, 302, 138--147.

Kadaba, M. P., Ramakrishnan, H. K., & Wootten, M. E. (1989). Measurement of lower extremity kinematics during level walking. *Journal of Orthopaedic Research*, 7(6), 849--860.

Hof, A. L., Gazendam, M. G. J., & Sinke, W. E. (2005). The condition for dynamic stability. *Journal of Biomechanics*, 38(1), 1--8.

Levene, H. (1960). Robust tests for equality of variances. In *Contributions to Probability and Statistics*, 278--292. Stanford University Press.

Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50--60.

Perry, J., & Burnfield, J. M. (2010). *Gait Analysis: Normal and Pathological Function* (2nd ed.). SLACK Incorporated.

Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.

Robinson, R. O., Herzog, W., & Nigg, B. M. (1987). Use of force platform variables to quantify the effects of chiropractic manipulation on gait symmetry. *Journal of Manipulative and Physiological Therapeutics*, 10(4), 172--176.

Schwartz, M. H., & Rozumalski, A. (2008). The Gait Deviation Index: a new comprehensive index of gait pathology. *Gait & Posture*, 28(3), 351--357.

Schwartz, M. H., Trost, J. P., & Wervey, R. A. (2004). Measurement and management of errors in quantitative gait data. *Gait & Posture*, 20(2), 196--203.

Shorter, K. A., Polk, J. D., Rosengren, K. S., & Hsiao-Wecksler, E. T. (2008). A new approach to detecting asymmetries in gait. *Clinical Biomechanics*, 23(4), 459--467.

Teulings, H. L., Contreras-Vidal, J. L., Stelmach, G. E., & Adler, C. H. (1997). Parkinsonism reduces coordination of fingers, wrist, and arm in fine motor control. *Experimental Neurology*, 146(1), 159--170.

Winter, D. A. (2009). *Biomechanics and Motor Control of Human Movement* (4th ed.). John Wiley & Sons.

---

# Appendices: Complete Mathematical Derivations {-}

The following appendices provide complete mathematical derivations for every technique used in this study, starting from foundational calculus and building to the specific metrics and statistical tests. The target reader is assumed to have completed a first course in single-variable calculus (derivatives, integrals, chain rule) and basic algebra. All other mathematical machinery is derived from scratch.

---

## Appendix A: Vector Algebra and Geometric Foundations {-}

### A.1 Vectors in $\mathbb{R}^2$ and $\mathbb{R}^3$ {-}

A **vector** is an ordered list of numbers representing a quantity with both magnitude and direction. In our pose estimation pipeline, every anatomical landmark is a point in 2D or 3D space, and joint angles are computed from vectors between landmarks.

**Definition.** A vector in $\mathbb{R}^n$ is an ordered $n$-tuple:

$$\mathbf{v} = (v_1, v_2, \ldots, v_n)$$

For pose estimation, we work primarily in $\mathbb{R}^2$ (image pixel coordinates) and $\mathbb{R}^3$ (3D reconstructions):

$$\mathbf{v}_{2D} = (x, y), \qquad \mathbf{v}_{3D} = (x, y, z)$$

**Vector addition.** For two vectors $\mathbf{u} = (u_1, u_2)$ and $\mathbf{v} = (v_1, v_2)$:

$$\mathbf{u} + \mathbf{v} = (u_1 + v_1, \; u_2 + v_2)$$

**Scalar multiplication.** For a scalar $c \in \mathbb{R}$:

$$c\mathbf{v} = (cv_1, cv_2)$$

**Vector subtraction.** The vector from point $B$ to point $A$ is:

$$\mathbf{A} - \mathbf{B} = (A_x - B_x, \; A_y - B_y)$$

This is fundamental to our joint angle computation: we form vectors from the joint vertex $B$ to its proximal and distal landmarks $A$ and $C$.

### A.2 The Dot Product {-}

The **dot product** (also called inner product or scalar product) is the central operation for computing angles between vectors. It takes two vectors and returns a scalar.

**Algebraic definition.** For $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n$$

In $\mathbb{R}^2$: $\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2$

In $\mathbb{R}^3$: $\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 + u_3 v_3$

**Geometric definition.** The dot product equals the product of the magnitudes times the cosine of the angle between the vectors:

$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \cdot \|\mathbf{v}\| \cdot \cos\theta$$

where $\theta$ is the angle between $\mathbf{u}$ and $\mathbf{v}$, and $\|\cdot\|$ denotes the Euclidean norm (see A.3).

**Proof of equivalence (in $\mathbb{R}^2$).** Consider two vectors $\mathbf{u}$ and $\mathbf{v}$ with an angle $\theta$ between them. Using the law of cosines on the triangle formed by $\mathbf{u}$, $\mathbf{v}$, and $\mathbf{u} - \mathbf{v}$:

$$\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

Expanding the left side:

$$\|\mathbf{u} - \mathbf{v}\|^2 = (u_1 - v_1)^2 + (u_2 - v_2)^2 = u_1^2 - 2u_1 v_1 + v_1^2 + u_2^2 - 2u_2 v_2 + v_2^2$$

$$= (u_1^2 + u_2^2) + (v_1^2 + v_2^2) - 2(u_1 v_1 + u_2 v_2) = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(u_1 v_1 + u_2 v_2)$$

Comparing with the law of cosines:

$$\|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(u_1 v_1 + u_2 v_2) = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

Therefore: $u_1 v_1 + u_2 v_2 = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$, confirming algebraic $=$ geometric. $\square$

**Properties:**

1. Commutative: $\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$
2. Distributive: $\mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \cdot \mathbf{w}$
3. Scalar compatibility: $(c\mathbf{u}) \cdot \mathbf{v} = c(\mathbf{u} \cdot \mathbf{v})$
4. Self-dot: $\mathbf{v} \cdot \mathbf{v} = \|\mathbf{v}\|^2$

### A.3 Vector Norms and Distances {-}

The **Euclidean norm** (or $L^2$ norm, or magnitude) of a vector is its "length":

$$\|\mathbf{v}\| = \|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

This follows from the Pythagorean theorem. In $\mathbb{R}^2$:

$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2}$$

The **Euclidean distance** between two points $\mathbf{a}$ and $\mathbf{b}$ is:

$$d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}$$

We use this throughout the pipeline: for tracking persons across frames (pelvis centroid distance), computing arc lengths (SPARC), and measuring distributional spread in PCA space.

**A unit vector** is a vector with norm 1:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

### A.4 The Angle Between Two Vectors {-}

From the geometric definition of the dot product:

$$\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \cdot \|\mathbf{v}\|}$$

Solving for $\theta$:

$$\theta = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \cdot \|\mathbf{v}\|}\right)$$

**This is the fundamental formula for joint angle computation in our pipeline.** For three landmarks $A$, $B$, $C$ with $B$ at the joint vertex, we compute:

$$\mathbf{u} = \mathbf{A} - \mathbf{B}, \quad \mathbf{v} = \mathbf{C} - \mathbf{B}$$

$$\theta = \arccos\left(\frac{(\mathbf{A} - \mathbf{B}) \cdot (\mathbf{C} - \mathbf{B})}{\|\mathbf{A} - \mathbf{B}\| \cdot \|\mathbf{C} - \mathbf{B}\|}\right)$$

**Domain and range.** The arccos function is defined on $[-1, 1]$ and returns values in $[0, \pi]$ radians (or $[0°, 180°]$). This means:

- $\theta = 0°$: vectors point in the same direction (limb fully flexed — proximal and distal segments folded together)
- $\theta = 90°$: vectors are perpendicular (right-angle bend)
- $\theta = 180°$: vectors point in opposite directions (limb fully extended — segments form a straight line)

This is the **included angle**, not the flexion angle. The flexion angle convention used in the main text (Section 2.3) applies the transform $\theta_{\text{flex}} = 180° - \theta$, so that 0° = full extension and increasing values indicate flexion. See Appendix A.6 for the derivation.

**Numerical stability.** Due to floating-point arithmetic, the argument to arccos can occasionally fall slightly outside $[-1, 1]$. The implementation clamps the value: $\theta = \arccos(\text{clip}(\cos\theta, -1, 1))$.

### A.5 The $\arctan2$ Function {-}

While $\arccos$ gives the angle between two vectors, we sometimes need the **signed angle** of a vector relative to coordinate axes. The standard $\arctan$ function has a range of only $(-\pi/2, \pi/2)$ and cannot distinguish all four quadrants. The **two-argument arctangent** resolves this:

$$\arctan2(y, x) = \begin{cases}
\arctan(y/x) & x > 0 \\
\arctan(y/x) + \pi & x < 0, y \geq 0 \\
\arctan(y/x) - \pi & x < 0, y < 0 \\
+\pi/2 & x = 0, y > 0 \\
-\pi/2 & x = 0, y < 0 \\
\text{undefined} & x = 0, y = 0
\end{cases}$$

The range is $(-\pi, \pi]$ or equivalently $(-180°, 180°]$.

**Usage in our pipeline:**

- **Pelvis obliquity**: $\theta_{\text{pelvis}} = |\arctan2(y_R - y_L, x_R - x_L)|$ — the angle of the hip-to-hip line relative to horizontal
- **Trunk lean**: $\theta_{\text{trunk}} = \arctan2(x_{\text{shoulder}} - x_{\text{pelvis}}, y_{\text{shoulder}} - y_{\text{pelvis}})$ — the lateral deviation of the trunk from vertical

### A.6 The Flexion Angle Transform {-}

The arccos formula gives the **included angle** at a joint — the angle between the proximal and distal limb segments. But clinically, joint angles are reported as **flexion angles** where:

- $0°$ = full extension (limbs aligned)
- Increasing values = more flexion

Since the included angle at full extension is $180°$ (proximal and distal segments form a straight line):

$$\theta_{\text{flex}} = 180° - \theta_{\text{included}}$$

**Verification:** At full extension, $\theta_{\text{included}} = 180°$, so $\theta_{\text{flex}} = 0°$. At a 90° bend, $\theta_{\text{included}} = 90°$, so $\theta_{\text{flex}} = 90°$. This matches clinical convention.

**Special case — ankle dorsiflexion.** The anatomical neutral position of the ankle (foot perpendicular to shank) corresponds to an included angle of $90°$, not $180°$. Therefore:

$$\theta_{\text{ankle}} = 90° - \theta_{\text{included}}$$

This maps the neutral position to $0°$, with positive values indicating dorsiflexion and negative values indicating plantarflexion.

---

## Appendix B: Calculus Foundations {-}

### B.1 Derivatives and Rates of Change {-}

The **derivative** of a function $f(t)$ at a point $t$ is defined as the limit of the difference quotient:

$$f'(t) = \frac{df}{dt} = \lim_{h \to 0} \frac{f(t+h) - f(t)}{h}$$

The derivative measures the instantaneous rate of change of $f$ with respect to $t$.

In our context, if $\theta(t)$ is a joint angle as a function of time:

- $\dot{\theta}(t) = \frac{d\theta}{dt}$ is the **angular velocity** (rate of angle change)
- $\ddot{\theta}(t) = \frac{d^2\theta}{dt^2}$ is the **angular acceleration**
- $\dddot{\theta}(t) = \frac{d^3\theta}{dt^3}$ is the **angular jerk** (rate of acceleration change)

### B.2 The Chain Rule {-}

If $y = f(g(t))$ is a composition of functions, then:

$$\frac{dy}{dt} = f'(g(t)) \cdot g'(t)$$

or equivalently, if $y = f(u)$ and $u = g(t)$:

$$\frac{dy}{dt} = \frac{dy}{du} \cdot \frac{du}{dt}$$

The chain rule is fundamental to:
- **Backpropagation** in neural networks (Appendix R)
- **Computing normalized jerk** (cascaded derivatives)
- **Transfer functions** in digital filtering

### B.3 Higher-Order Derivatives {-}

The $n$-th derivative of $f$ is obtained by differentiating $n$ times:

$$f^{(n)}(t) = \frac{d^n f}{dt^n}$$

For joint angles:

| Order | Symbol | Physical meaning | Units |
|-------|--------|-----------------|-------|
| 0 | $\theta(t)$ | Position (angle) | degrees |
| 1 | $\dot{\theta}(t)$ | Velocity | deg/s |
| 2 | $\ddot{\theta}(t)$ | Acceleration | deg/s² |
| 3 | $\dddot{\theta}(t)$ | Jerk | deg/s³ |

**Jerk** is particularly important for movement quality: smooth movements minimize jerk, while jerky movements have large jerk values. This motivates the Normalized Jerk metric (Section 2.5.3 and Appendix M).

### B.4 Numerical Differentiation {-}

In practice, we have discrete samples $\theta_0, \theta_1, \ldots, \theta_N$ at times $t_k = k\Delta t$ rather than a continuous function. We approximate derivatives using **finite differences**.

**Forward difference (first-order):**

$$f'(t_k) \approx \frac{f(t_{k+1}) - f(t_k)}{\Delta t}$$

**Central difference (second-order, more accurate):**

$$f'(t_k) \approx \frac{f(t_{k+1}) - f(t_{k-1})}{2\Delta t}$$

**Why central differences are more accurate.** Using Taylor expansions:

$$f(t + \Delta t) = f(t) + f'(t)\Delta t + \frac{f''(t)}{2}\Delta t^2 + \frac{f'''(t)}{6}\Delta t^3 + \cdots$$

$$f(t - \Delta t) = f(t) - f'(t)\Delta t + \frac{f''(t)}{2}\Delta t^2 - \frac{f'''(t)}{6}\Delta t^3 + \cdots$$

Subtracting:

$$f(t + \Delta t) - f(t - \Delta t) = 2f'(t)\Delta t + \frac{f'''(t)}{3}\Delta t^3 + \cdots$$

$$f'(t) = \frac{f(t + \Delta t) - f(t - \Delta t)}{2\Delta t} - \frac{f'''(t)}{6}\Delta t^2 + \cdots$$

The error is $O(\Delta t^2)$ for central differences vs. $O(\Delta t)$ for forward differences.

**Second derivative (central difference):**

Adding the Taylor expansions instead of subtracting:

$$f(t + \Delta t) + f(t - \Delta t) = 2f(t) + f''(t)\Delta t^2 + \cdots$$

$$f''(t) \approx \frac{f(t + \Delta t) - 2f(t) + f(t - \Delta t)}{\Delta t^2}$$

**Third derivative** (for jerk): We compute the third derivative by applying the central difference formula three times in succession. NumPy's `gradient` function implements second-order central differences; applying it three times yields $\dddot{\theta}_t$.

### B.5 The Definite Integral {-}

The **definite integral** of $f$ from $a$ to $b$ is the signed area under the curve:

$$\int_a^b f(t) \, dt = \lim_{n \to \infty} \sum_{k=1}^{n} f(t_k^*) \Delta t$$

where $\Delta t = (b-a)/n$ and $t_k^*$ is a sample point in the $k$-th subinterval.

**The Fundamental Theorem of Calculus** connects derivatives and integrals: if $F'(t) = f(t)$, then:

$$\int_a^b f(t) \, dt = F(b) - F(a)$$

### B.6 Numerical Integration {-}

For discrete data, we approximate integrals numerically.

**Riemann sum (left endpoint):**

$$\int_a^b f(t) \, dt \approx \sum_{k=0}^{N-1} f(t_k) \Delta t$$

**Riemann sum (midpoint):**

$$\int_a^b f(t) \, dt \approx \sum_{k=0}^{N-1} f\left(\frac{t_k + t_{k+1}}{2}\right) \Delta t$$

**Trapezoidal rule (more accurate):**

$$\int_a^b f(t) \, dt \approx \sum_{k=0}^{N-1} \frac{f(t_k) + f(t_{k+1})}{2} \Delta t$$

We use Riemann sums in the Normalized Jerk computation:

$$\int_0^T \left(\frac{d^3\theta}{dt^3}\right)^2 dt \approx \Delta t \sum_{t=1}^{N} \left(\dddot{\theta}_t\right)^2$$

### B.7 Arc Length of a Curve {-}

The **arc length** of a curve $y = f(x)$ from $x = a$ to $x = b$ is:

$$L = \int_a^b \sqrt{1 + \left(\frac{dy}{dx}\right)^2} \, dx$$

**Derivation.** Consider a small segment of the curve between $(x, f(x))$ and $(x + dx, f(x + dx))$. By the Pythagorean theorem, the length of this infinitesimal segment is:

$$ds = \sqrt{dx^2 + dy^2} = \sqrt{dx^2 + \left(\frac{dy}{dx}\right)^2 dx^2} = \sqrt{1 + \left(\frac{dy}{dx}\right)^2} \, dx$$

Integrating over the entire curve gives the total arc length.

**Parametric form.** For a curve given parametrically as $(x(t), y(t))$:

$$L = \int_a^b \sqrt{\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2} \, dt$$

**Discrete approximation.** For a curve defined by points $(x_k, y_k)$, $k = 1, \ldots, K$:

$$L \approx \sum_{k=1}^{K-1} \sqrt{(x_{k+1} - x_k)^2 + (y_{k+1} - y_k)^2}$$

Each term is the Euclidean distance between consecutive points. This is exactly the formula used in SPARC (Appendix L), where $(x_k, y_k) = (f_k, \hat{V}_{\text{norm}}(f_k))$ — the frequency-amplitude curve.

---

## Appendix C: Probability and Statistics Foundations {-}

### C.1 Random Variables {-}

A **random variable** $X$ is a function that assigns a numerical value to each outcome of a random experiment. There are two types:

- **Discrete**: $X$ takes on a countable set of values (e.g., counts)
- **Continuous**: $X$ takes on values in an interval (e.g., joint angles)

The **probability density function** (pdf) of a continuous random variable satisfies:

$$P(a \leq X \leq b) = \int_a^b f_X(x) \, dx$$

with $f_X(x) \geq 0$ and $\int_{-\infty}^{\infty} f_X(x) \, dx = 1$.

The **cumulative distribution function** (cdf) is:

$$F_X(x) = P(X \leq x) = \int_{-\infty}^{x} f_X(t) \, dt$$

### C.2 Expected Value {-}

The **expected value** (or mean) of a continuous random variable is:

$$E[X] = \mu = \int_{-\infty}^{\infty} x \cdot f_X(x) \, dx$$

For a discrete random variable with values $x_1, x_2, \ldots$ and probabilities $p_1, p_2, \ldots$:

$$E[X] = \sum_i x_i \cdot p_i$$

**Properties of expected value:**

1. **Linearity**: $E[aX + b] = aE[X] + b$
2. **Additivity**: $E[X + Y] = E[X] + E[Y]$ (always, even if dependent)
3. **Constant**: $E[c] = c$

**Sample mean.** For observations $x_1, \ldots, x_n$, the sample mean is:

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

The sample mean is an unbiased estimator of $\mu$: $E[\bar{X}] = \mu$.

### C.3 Variance and Standard Deviation {-}

The **variance** measures the spread of a distribution around its mean:

$$\text{Var}(X) = \sigma^2 = E\left[(X - \mu)^2\right] = \int_{-\infty}^{\infty} (x - \mu)^2 f_X(x) \, dx$$

**Alternative formula (computational form).** Expanding the square:

$$\text{Var}(X) = E[X^2] - (E[X])^2 = E[X^2] - \mu^2$$

**Proof:**

$$E[(X - \mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - 2\mu^2 + \mu^2 = E[X^2] - \mu^2$$

**Standard deviation** is the square root of variance: $\sigma = \sqrt{\text{Var}(X)}$.

**Properties of variance:**

1. $\text{Var}(X) \geq 0$ always
2. $\text{Var}(aX + b) = a^2 \text{Var}(X)$ (scaling)
3. $\text{Var}(X) = 0$ if and only if $X$ is constant
4. If $X, Y$ are independent: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### C.4 Sample Variance and Bessel's Correction {-}

**Why divide by $n-1$ instead of $n$?** This is one of the most important corrections in applied statistics, and central to our variance comparison study.

The **population variance** is:

$$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2$$

But when estimating variance from a sample, we don't know $\mu$; we use $\bar{x}$ instead. The **naive sample variance** would be:

$$\hat{\sigma}_{\text{naive}}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

This estimator is **biased**: $E[\hat{\sigma}_{\text{naive}}^2] = \frac{n-1}{n}\sigma^2 < \sigma^2$.

**Proof of bias.** Start with:

$$\sum_{i=1}^{n}(x_i - \bar{x})^2 = \sum_{i=1}^{n}(x_i - \mu + \mu - \bar{x})^2 = \sum_{i=1}^{n}\left[(x_i - \mu) - (\bar{x} - \mu)\right]^2$$

Expanding:

$$= \sum_{i=1}^{n}(x_i - \mu)^2 - 2(\bar{x} - \mu)\sum_{i=1}^{n}(x_i - \mu) + n(\bar{x} - \mu)^2$$

Note that $\sum_{i=1}^{n}(x_i - \mu) = n(\bar{x} - \mu)$, so the middle term equals $-2n(\bar{x} - \mu)^2$:

$$= \sum_{i=1}^{n}(x_i - \mu)^2 - n(\bar{x} - \mu)^2$$

Taking expectations:

$$E\left[\sum_{i=1}^{n}(x_i - \bar{x})^2\right] = n\sigma^2 - n \cdot \text{Var}(\bar{X}) = n\sigma^2 - n \cdot \frac{\sigma^2}{n} = (n-1)\sigma^2$$

where we used the fact that $\text{Var}(\bar{X}) = \sigma^2/n$ (the variance of the sample mean).

Therefore:

$$E\left[\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2\right] = \frac{n-1}{n}\sigma^2 \neq \sigma^2$$

**Bessel's correction** divides by $n-1$ to obtain an unbiased estimator:

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

$$E[s^2] = \frac{1}{n-1} \cdot (n-1)\sigma^2 = \sigma^2 \quad \square$$

**Degrees of freedom interpretation.** The quantity $\sum(x_i - \bar{x})^2$ has only $n - 1$ degrees of freedom because the deviations $x_i - \bar{x}$ are constrained by $\sum(x_i - \bar{x}) = 0$. Knowing any $n - 1$ of the deviations determines the last one.

**In our study,** we use $s^2$ with Bessel's correction throughout: in computing variance ratios $R_\sigma = s^2_{\text{control}} / s^2_{\text{runway}}$, in Levene's test, and in the bootstrap.

### C.5 Covariance and Correlation {-}

**Covariance** measures the linear association between two random variables:

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

**Correlation coefficient** (Pearson's $r$):

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

where $\rho \in [-1, 1]$. Values near $\pm 1$ indicate strong linear association; $\rho = 0$ indicates no linear association.

**Sample covariance:**

$$s_{XY} = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$

**Sample correlation:**

$$r_{XY} = \frac{s_{XY}}{s_X s_Y}$$

**The covariance matrix** for $p$ variables $X_1, \ldots, X_p$ is the $p \times p$ matrix:

$$\mathbf{C} = \begin{pmatrix} s_{11} & s_{12} & \cdots & s_{1p} \\ s_{21} & s_{22} & \cdots & s_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ s_{p1} & s_{p2} & \cdots & s_{pp} \end{pmatrix}$$

where $s_{jk} = \text{Cov}(X_j, X_k)$ and $s_{jj} = \text{Var}(X_j)$. This matrix is symmetric ($s_{jk} = s_{kj}$) and positive semi-definite. The covariance matrix is central to PCA and LDA (Appendix P).

### C.6 The Normal Distribution {-}

The **normal** (Gaussian) distribution with mean $\mu$ and variance $\sigma^2$ has pdf:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

We write $X \sim N(\mu, \sigma^2)$.

The **standard normal** distribution has $\mu = 0$, $\sigma^2 = 1$:

$$\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$$

**Standardization.** If $X \sim N(\mu, \sigma^2)$, then $Z = \frac{X - \mu}{\sigma} \sim N(0, 1)$.

**Why the normal distribution matters:**

1. **Central Limit Theorem**: the sample mean of $n$ independent observations approaches a normal distribution as $n \to \infty$, regardless of the underlying distribution
2. Many test statistics (including the ones underlying Levene's test) are approximately normally distributed
3. Bootstrap distributions are often approximately normal

### C.7 The F-Distribution {-}

The $F$-distribution arises as the ratio of two independent chi-squared random variables, each divided by its degrees of freedom. If $U \sim \chi^2_{d_1}$ and $V \sim \chi^2_{d_2}$ are independent:

$$F = \frac{U/d_1}{V/d_2} \sim F(d_1, d_2)$$

The $F$-distribution is used in ANOVA-type tests, including **Levene's test**. Levene's $W$ statistic follows $F(k-1, N-k)$ under $H_0$, where $k$ is the number of groups and $N$ is the total sample size.

The **chi-squared distribution** $\chi^2_d$ is the sum of $d$ squared standard normal variables:

$$\chi^2_d = Z_1^2 + Z_2^2 + \cdots + Z_d^2, \quad Z_i \sim N(0,1) \text{ independent}$$

### C.8 The Coefficient of Variation {-}

The **coefficient of variation** (CV) expresses standard deviation as a percentage of the mean:

$$\text{CV} = \frac{\sigma}{\mu} \times 100\% = \frac{s}{\bar{x}} \times 100\%$$

CV is a **dimensionless** measure of relative variability. This is important because it allows comparison across metrics with different units or scales.

**Example:** A stride time of $1.2 \pm 0.05$ s (CV = 4.2%) represents moderate variability. A hip ROM of $40 \pm 5°$ (CV = 12.5%) represents higher relative variability.

In our study, we compute:
- **Stride time CV**: variability of stride intervals (normal: 1--3%)
- **Kinematic CV**: per-stride ROM variability (normal: 0--5%)

### C.9 Confidence Intervals {-}

A **confidence interval** (CI) is a range that contains the true parameter value with a specified probability.

For a sample mean with known variance: the 95% CI is:

$$\bar{x} \pm 1.96 \cdot \frac{\sigma}{\sqrt{n}}$$

**Interpretation:** If we repeated the experiment many times and constructed a CI each time, 95% of those intervals would contain the true parameter.

For our **bootstrap CIs on the variance ratio**, we use the percentile method (see Appendix H): the 2.5th and 97.5th percentiles of the bootstrap distribution.

---

## Appendix D: Hypothesis Testing Framework {-}

### D.1 The Logic of Hypothesis Testing {-}

A hypothesis test is a formalized procedure for making a yes/no decision about a population parameter based on sample data.

**Step 1: State hypotheses.**

- $H_0$ (null hypothesis): the "nothing interesting" claim. For our study: "runway and control groups have equal variance."
- $H_1$ (alternative hypothesis): the claim we seek evidence for. "The groups have unequal variance."

**Step 2: Choose a test statistic.** A function of the data that quantifies how much the data deviates from $H_0$.

**Step 3: Determine the null distribution.** The probability distribution of the test statistic assuming $H_0$ is true.

**Step 4: Compute the p-value.** The probability of observing a test statistic at least as extreme as the one computed, assuming $H_0$ is true.

**Step 5: Make a decision.** If $p < \alpha$ (significance level), reject $H_0$.

### D.2 P-Values {-}

The **p-value** is the probability of obtaining a result at least as extreme as the observed result, under the assumption that $H_0$ is true.

$$p = P(\text{Test statistic} \geq T_{\text{obs}} \mid H_0)$$

for a one-sided test, or

$$p = P(|T| \geq |T_{\text{obs}}| \mid H_0)$$

for a two-sided test.

**Common misconception:** The p-value is NOT the probability that $H_0$ is true. It is the probability of the data (or more extreme data) given $H_0$.

### D.3 Type I and Type II Errors {-}

| | $H_0$ true | $H_0$ false |
|---|---|---|
| **Reject $H_0$** | Type I error ($\alpha$) | Correct (power) |
| **Fail to reject $H_0$** | Correct | Type II error ($\beta$) |

- **Type I error rate** ($\alpha$): probability of rejecting $H_0$ when it's true (false positive). Typically set to 0.05.
- **Type II error rate** ($\beta$): probability of failing to reject $H_0$ when it's false (false negative).
- **Power** = $1 - \beta$: probability of correctly rejecting $H_0$ when it's false.

With $n = 17$--$22$ per group, our study has limited power to detect moderate variance differences.

### D.4 Multiple Comparison Correction {-}

When performing $m$ simultaneous **independent** hypothesis tests, the probability of at least one false positive increases:

$$P(\text{at least one Type I error}) = 1 - (1 - \alpha)^m$$

(This formula assumes independent tests. For correlated tests — as in our study where metrics share underlying signals — the true family-wise error rate is lower. The formula is presented here as a worst-case illustration.)

For $m = 24$ tests at $\alpha = 0.05$:

$$P(\text{at least one}) = 1 - (1 - 0.05)^{24} = 1 - 0.95^{24} \approx 1 - 0.292 = 0.708$$

This 70.8% chance of at least one false positive is unacceptably high for scientific claims.

**Bonferroni correction** sets the per-test significance level to:

$$\alpha_{\text{Bonf}} = \frac{\alpha}{m}$$

**Proof that this controls family-wise error rate (FWER).** By the union bound (Boole's inequality):

$$P\left(\bigcup_{i=1}^{m} A_i\right) \leq \sum_{i=1}^{m} P(A_i) = m \cdot \frac{\alpha}{m} = \alpha$$

where $A_i$ is the event of rejecting $H_{0,i}$ when it's true.

For our study: $\alpha_{\text{Bonf}} = 0.05 / 24 \approx 0.0021$.

**Note on conservatism.** The Boole-Bonferroni inequality holds regardless of dependence structure — it does not assume independence. When tests are positively correlated (as ours are — many metrics derive from the same underlying signals), Bonferroni is overly conservative because the union probability is smaller than the sum of individual probabilities. The effective number of independent tests $m_{\text{eff}} < m$ could be estimated from the eigenvalues of the inter-metric correlation matrix, but we use the conservative Bonferroni bound and report both corrected and uncorrected results.

---

## Appendix E: Levene's Test — Full Derivation {-}

### E.1 Motivation {-}

The classical **F-test** for equality of variances compares $s_1^2 / s_2^2$ against the $F$-distribution. However, the F-test is extremely sensitive to departures from normality — it has poor Type I error control when the underlying distributions are non-normal.

**Levene's test** (1960) provides a robust alternative by transforming the problem into an ANOVA on absolute deviations.

### E.2 The Transformation {-}

Given $k$ groups with $n_i$ observations in group $i$, define the **absolute deviation** from the group mean (or median):

$$Z_{ij} = |X_{ij} - \bar{X}_{i\cdot}|$$

where $X_{ij}$ is the $j$-th observation in group $i$ and $\bar{X}_{i\cdot}$ is the mean of group $i$.

**Key insight:** If group $i$ has large variance, the $Z_{ij}$ values will tend to be large. If group $i$ has small variance, the $Z_{ij}$ values will tend to be small. Testing whether the groups have equal variance is equivalent to testing whether the mean absolute deviations are equal.

### E.3 The W Statistic {-}

Define:
- $\bar{Z}_{i\cdot} = \frac{1}{n_i}\sum_{j=1}^{n_i} Z_{ij}$ — mean absolute deviation in group $i$
- $\bar{Z}_{\cdot\cdot} = \frac{1}{N}\sum_{i=1}^{k}\sum_{j=1}^{n_i} Z_{ij}$ — grand mean of absolute deviations
- $N = \sum_{i=1}^{k} n_i$ — total sample size

The Levene test statistic is:

$$W = \frac{(N - k) \sum_{i=1}^{k} n_i (\bar{Z}_{i\cdot} - \bar{Z}_{\cdot\cdot})^2}{(k - 1) \sum_{i=1}^{k}\sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_{i\cdot})^2}$$

This is an F-ratio: the numerator measures **between-group** variability of the absolute deviations, and the denominator measures **within-group** variability.

**Derivation from ANOVA.** Levene's test is simply a one-way ANOVA applied to the transformed data $Z_{ij}$. In standard ANOVA notation:

$$\text{SS}_{\text{between}} = \sum_{i=1}^{k} n_i (\bar{Z}_{i\cdot} - \bar{Z}_{\cdot\cdot})^2$$

$$\text{SS}_{\text{within}} = \sum_{i=1}^{k}\sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_{i\cdot})^2$$

$$\text{MS}_{\text{between}} = \frac{\text{SS}_{\text{between}}}{k - 1}, \quad \text{MS}_{\text{within}} = \frac{\text{SS}_{\text{within}}}{N - k}$$

$$W = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} = \frac{(N - k)}{(k - 1)} \cdot \frac{\text{SS}_{\text{between}}}{\text{SS}_{\text{within}}}$$

### E.4 Distribution Under $H_0$ {-}

Under $H_0: \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2$, the transformed $Z_{ij}$ values should have equal means across groups. The ANOVA F-test on these values approximately follows:

$$W \sim F(k-1, N-k)$$

For our two-group comparison ($k = 2$): $W \sim F(1, N - 2)$ under $H_0$.

The p-value is:

$$p = P(F \geq W_{\text{obs}}) = 1 - F_{\text{cdf}}(W_{\text{obs}}; \, k-1, \, N-k)$$

### E.5 The Brown-Forsythe Variant {-}

The **Brown-Forsythe** (1974) variant uses the group **median** instead of the mean in the transformation:

$$Z_{ij} = |X_{ij} - \tilde{X}_{i}|$$

where $\tilde{X}_i$ is the median of group $i$. This variant is even more robust to non-normality and is used in SciPy's `levene(center='median')`.

### E.6 Worked Example from Our Data {-}

For hip\_ROM: $n_R = 22$ runway samples with $s_R^2 = 376.0$, $n_C = 17$ control samples with $s_C^2 = 1413.6$.

Variance ratio: $R_\sigma = 1413.6 / 376.0 = 3.76$.

Levene's test yields $W = 12.33$, $p = 0.0008$, which is significant at $\alpha = 0.05$ and survives Bonferroni correction ($p < 0.0021$).

---

## Appendix F: Mann-Whitney U Test — Full Derivation {-}

### F.1 Motivation {-}

The **Mann-Whitney U test** (also called the Wilcoxon rank-sum test) compares the distributions of two groups without assuming normality. It tests whether one group tends to have larger values than the other.

$H_0$: The two groups come from the same distribution.
$H_1$: One group tends to produce larger values than the other.

### F.2 The Ranking Procedure {-}

1. Combine all $N = n_1 + n_2$ observations into a single set
2. Rank them from smallest (rank 1) to largest (rank $N$)
3. Assign average ranks for tied values
4. Compute the sum of ranks for group 1: $R_1 = \sum \text{ranks of group 1 observations}$

### F.3 The U Statistic {-}

$$U_1 = R_1 - \frac{n_1(n_1 + 1)}{2}$$

**Derivation.** The minimum possible rank sum for group 1 (if all its values are smallest) is $R_1^{\min} = 1 + 2 + \cdots + n_1 = \frac{n_1(n_1+1)}{2}$. The quantity $U_1 = R_1 - R_1^{\min}$ counts the number of pairs where a group-1 value exceeds a group-2 value.

Equivalently, $U_1$ equals the number of pairs $(x_{1i}, x_{2j})$ where $x_{1i} > x_{2j}$:

$$U_1 = \sum_{i=1}^{n_1}\sum_{j=1}^{n_2} \mathbb{1}[x_{1i} > x_{2j}]$$

The maximum possible value is $U_{\max} = n_1 n_2$ (when every group-1 value exceeds every group-2 value). Under $H_0$, $E[U] = n_1 n_2 / 2$.

### F.4 Normal Approximation {-}

For large samples, the distribution of $U$ is approximately normal:

$$Z = \frac{U - n_1 n_2 / 2}{\sqrt{n_1 n_2 (n_1 + n_2 + 1) / 12}}$$

### F.5 Effect Sizes {-}

**Cohen's d** (standardized mean difference):

$$d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}$$

where the pooled standard deviation uses the unweighted average of group variances:

$$s_{\text{pooled}} = \sqrt{\frac{s_1^2 + s_2^2}{2}}$$

This differs from the weighted pooled SD $s_p = \sqrt{((n_1-1)s_1^2 + (n_2-1)s_2^2)/(n_1+n_2-2)}$, which assumes equal population variances. We use the unweighted version because our hypothesis is precisely that the population variances differ (see Appendix AE.2 for discussion).

Interpretation: $|d| < 0.2$ small, $0.2$--$0.8$ medium, $>0.8$ large.

**Rank-biserial correlation** (non-parametric effect size):

$$r_{rb} = 1 - \frac{2U}{n_1 n_2}$$

This ranges from $-1$ to $+1$. Values near $\pm 1$ indicate complete separation of the groups.

---

## Appendix G: Permutation Testing — Full Derivation {-}

### G.1 The Exchangeability Assumption {-}

Under $H_0$, the group labels are irrelevant — any assignment of observations to groups is equally likely. This means we can assess the significance of the observed test statistic by comparing it to the distribution of statistics obtained under all possible permutations of the group labels.

### G.2 The Exact Permutation Distribution {-}

For a total of $N = n_1 + n_2$ observations, there are $\binom{N}{n_1}$ possible ways to assign $n_1$ observations to group 1. Each assignment defines a permuted test statistic.

For our sample sizes ($n_1 = 22$, $n_2 = 17$): $\binom{39}{22} = 51{,}021{,}117{,}810$ — far too many for exact enumeration. We use Monte Carlo approximation.

### G.3 Monte Carlo Approximation {-}

Instead of computing all permutations:

1. For $b = 1, \ldots, B$ (we use $B = 10{,}000$):
   - Randomly shuffle the $N$ observations
   - Assign the first $n_1$ to group 1, the rest to group 2
   - Compute the test statistic $T_b^*$ (in our case, the variance ratio)

2. The approximate p-value is:

$$p = \frac{|\{b : T_b^* \geq T_{\text{obs}}\}| + 1}{B + 1}$$

### G.4 The $+1$ Correction (Phipson & Smyth, 2010) {-}

Why add 1 to both numerator and denominator?

Without the correction, the minimum possible p-value is $0 / B = 0$. But a p-value of exactly 0 is philosophically problematic: it implies absolute certainty that $H_0$ is false, which a finite number of permutations cannot guarantee.

The correction:

$$p = \frac{|\{b : T_b^* \geq T_{\text{obs}}\}| + 1}{B + 1}$$

includes the observed statistic as one of the permutations (since the observed labeling is one valid permutation under $H_0$). This ensures:

- The minimum p-value is $1/(B+1)$ (not 0)
- The estimator is uniformly distributed on $\{1/(B+1), 2/(B+1), \ldots, 1\}$ under $H_0$
- The test has valid (conservative) Type I error control for Monte Carlo permutations; exact control holds only for full enumeration

---

## Appendix H: Bootstrap Methods — Full Derivation {-}

### H.1 The Bootstrap Principle {-}

The **bootstrap** (Efron, 1979) estimates the sampling distribution of a statistic by resampling from the observed data. The key insight: the empirical distribution of the sample is the best estimate of the population distribution.

### H.2 The Empirical Distribution Function {-}

Given observations $x_1, \ldots, x_n$, the **empirical distribution function** (EDF) is:

$$\hat{F}_n(x) = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[x_i \leq x]$$

This assigns equal probability $1/n$ to each observed value. The EDF converges to the true CDF by the Glivenko-Cantelli theorem.

### H.3 Resampling with Replacement {-}

A **bootstrap sample** is a random sample of size $n$ drawn with replacement from the original data:

$$x_1^*, x_2^*, \ldots, x_n^* \sim \hat{F}_n$$

Each observation has probability $1/n$ of being selected at each draw. Some observations will appear multiple times; some will not appear at all.

**Expected fraction of unique observations:** Each observation has probability $(1 - 1/n)^n \approx 1/e \approx 0.368$ of NOT appearing in a bootstrap sample. So about 63.2% of observations appear at least once.

### H.4 Bootstrap Confidence Intervals {-}

For our variance ratio statistic:

1. For $b = 1, \ldots, B$ ($B = 10{,}000$):
   - Draw $x_R^*$ (bootstrap sample from runway, size $n_R$, with replacement)
   - Draw $x_C^*$ (bootstrap sample from control, size $n_C$, with replacement)
   - Compute $R_b^* = s^{*2}_C / s^{*2}_R$ (bootstrap variance ratio)

2. **Percentile CI**: The $100(1-\alpha)\%$ confidence interval is:

$$\text{CI} = \left[\hat{R}^*_{(\alpha/2)}, \; \hat{R}^*_{(1-\alpha/2)}\right]$$

where $\hat{R}^*_{(q)}$ is the $q$-th quantile of the bootstrap distribution.

For a 95% CI: $[\hat{R}^*_{(0.025)}, \; \hat{R}^*_{(0.975)}]$

3. **Bootstrap proportion** (frequentist, not a Bayesian posterior) that the variance ratio exceeds 1:

$$\hat{P}(R > 1) = \frac{1}{B}\sum_{b=1}^{B} \mathbb{1}[R_b^* > 1]$$

### H.5 Why Percentile Bootstrap Works {-}

If the bootstrap distribution $\hat{G}^*$ is a good approximation to the true sampling distribution $G$, then:

$$P(R_{0.025}^* \leq R_{\text{true}} \leq R_{0.975}^*) \approx 0.95$$

This follows from the bootstrap consistency theorem: under mild regularity conditions, the bootstrap distribution converges to the true sampling distribution as $n \to \infty$.

---

## Appendix I: Fourier Analysis {-}

### I.1 Periodic Functions and Fourier Series {-}

Any periodic function $f(t)$ with period $T$ can be decomposed into a sum of sines and cosines (**Fourier's theorem**, 1807):

$$f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty}\left[a_n \cos\left(\frac{2\pi n t}{T}\right) + b_n \sin\left(\frac{2\pi n t}{T}\right)\right]$$

The coefficients are:

$$a_n = \frac{2}{T}\int_0^T f(t)\cos\left(\frac{2\pi n t}{T}\right) dt, \quad b_n = \frac{2}{T}\int_0^T f(t)\sin\left(\frac{2\pi n t}{T}\right) dt$$

**Intuition:** Any signal — no matter how complex — can be built from simple oscillations at different frequencies, amplitudes, and phases.

### I.2 From Fourier Series to Fourier Transform {-}

For non-periodic signals, we let the period $T \to \infty$. The discrete frequencies $n/T$ become a continuous frequency variable $f$, and the sum becomes an integral.

**The continuous Fourier transform:**

$$\hat{F}(f) = \int_{-\infty}^{\infty} f(t) \, e^{-j2\pi f t} \, dt$$

where $j = \sqrt{-1}$ is the imaginary unit. Using Euler's formula, $e^{-j2\pi ft} = \cos(2\pi ft) - j\sin(2\pi ft)$, so the Fourier transform decomposes a signal into its frequency components.

**The inverse Fourier transform:**

$$f(t) = \int_{-\infty}^{\infty} \hat{F}(f) \, e^{j2\pi f t} \, df$$

### I.3 The Discrete Fourier Transform (DFT) {-}

For $N$ discrete samples $x_0, x_1, \ldots, x_{N-1}$ taken at sampling rate $f_s$, the DFT is:

$$X_k = \sum_{n=0}^{N-1} x_n \, e^{-j2\pi kn/N}, \quad k = 0, 1, \ldots, N-1$$

The **frequency** corresponding to bin $k$ is:

$$f_k = \frac{k \cdot f_s}{N}$$

The **amplitude spectrum** is $|X_k|$ and the **power spectrum** is $|X_k|^2$.

**Inverse DFT:**

$$x_n = \frac{1}{N}\sum_{k=0}^{N-1} X_k \, e^{j2\pi kn/N}$$

### I.4 The Fast Fourier Transform (FFT) {-}

The DFT requires $O(N^2)$ multiplications. The **FFT** (Cooley & Tukey, 1965) reduces this to $O(N \log N)$ by exploiting the periodicity and symmetry of the complex exponentials.

**Key insight (divide and conquer).** The DFT of a length-$N$ sequence can be split into two DFTs of length $N/2$:

$$X_k = \underbrace{\sum_{m=0}^{N/2-1} x_{2m} \, e^{-j2\pi k(2m)/N}}_{\text{DFT of even-indexed samples}} + e^{-j2\pi k/N} \underbrace{\sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-j2\pi k(2m)/N}}_{\text{DFT of odd-indexed samples}}$$

This recursion, applied $\log_2 N$ times, gives the $O(N \log N)$ complexity.

In our SPARC computation, we apply FFT to the angular velocity signal $\dot{\theta}(t)$ to obtain the frequency spectrum.

### I.5 The Nyquist-Shannon Sampling Theorem {-}

**Theorem.** A continuous signal with maximum frequency $f_{\max}$ can be perfectly reconstructed from discrete samples if the sampling rate satisfies:

$$f_s > 2 f_{\max}$$

The frequency $f_{\text{Nyquist}} = f_s / 2$ is the **Nyquist frequency** — the highest frequency representable in the sampled signal.

For a 30 fps video: $f_{\text{Nyquist}} = 15$ Hz. This is sufficient for gait analysis since human walking involves frequencies primarily below 6 Hz.

**Aliasing.** If a signal contains frequencies above $f_{\text{Nyquist}}$, these fold back into the representable range, creating artifacts. True aliasing prevention requires anti-alias filtering *before* sampling; our Butterworth low-pass filter attenuates high-frequency noise in the already-sampled signal but cannot undo aliasing that occurred during digitization.

### I.6 The Amplitude Spectrum and Normalization {-}

The **amplitude spectrum** of a real signal is:

$$|X_k| = \sqrt{\text{Re}(X_k)^2 + \text{Im}(X_k)^2}$$

For SPARC, we normalize the amplitude spectrum to $[0, 1]$:

$$\hat{V}_{\text{norm}}(f_k) = \frac{|X_k|}{\max_k |X_k|}$$

This normalization ensures SPARC is invariant to signal amplitude (a faster movement and a slower movement with the same smoothness profile get the same SPARC value).

---

## Appendix J: Digital Filtering {-}

### J.1 Linear Time-Invariant Systems {-}

A **linear time-invariant** (LTI) system transforms an input signal $x(t)$ to an output $y(t)$. It is characterized by:

1. **Linearity**: $\mathcal{T}\{ax_1 + bx_2\} = a\mathcal{T}\{x_1\} + b\mathcal{T}\{x_2\}$
2. **Time-invariance**: if $\mathcal{T}\{x(t)\} = y(t)$, then $\mathcal{T}\{x(t - \tau)\} = y(t - \tau)$

Any LTI system is fully described by its **impulse response** $h(t)$, and the output is the **convolution**:

$$y(t) = (x * h)(t) = \int_{-\infty}^{\infty} x(\tau) h(t - \tau) \, d\tau$$

### J.2 Transfer Functions {-}

In the frequency domain (via Fourier transform), convolution becomes multiplication:

$$Y(f) = X(f) \cdot H(f)$$

where $H(f)$ is the **transfer function** — the Fourier transform of the impulse response. A low-pass filter has $|H(f)| \approx 1$ for $f < f_c$ (passband) and $|H(f)| \approx 0$ for $f > f_c$ (stopband).

### J.3 The Butterworth Filter {-}

The **Butterworth filter** (1930) is designed for maximally flat magnitude response in the passband. Its squared magnitude response is:

$$|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2n}}$$

where $n$ is the filter order and $\omega_c = 2\pi f_c$ is the cutoff angular frequency.

**Properties:**

- At the cutoff frequency ($\omega = \omega_c$): $|H|^2 = 1/2$, so $|H| = 1/\sqrt{2} \approx 0.707$ ($-3$ dB)
- The magnitude is **monotonically decreasing** — no ripples
- Higher order $n$ gives steeper roll-off (sharper cutoff)
- Our implementation uses $n = 2$ (second-order)

**Why Butterworth?** The maximally flat passband means the filter does not distort the signal at frequencies below the cutoff. This is important for joint angle data where we want to preserve the true movement dynamics while removing noise.

### J.4 Zero-Phase Filtering (filtfilt) {-}

A standard digital filter introduces a **phase delay** — the output is shifted in time relative to the input. For biomechanical analysis, phase delay would shift the timing of joint angle peaks, corrupting gait event detection.

**Solution: forward-backward filtering** (`filtfilt`).

1. Apply the filter forward: $y_{\text{fwd}} = h * x$
2. Reverse $y_{\text{fwd}}$
3. Apply the filter again: $y_{\text{rev}} = h * \text{reverse}(y_{\text{fwd}})$
4. Reverse the result to get $y_{\text{final}}$

The forward pass introduces phase $\phi(\omega)$; the backward pass introduces $-\phi(\omega)$. The net phase is zero.

The effective magnitude response is $|H_{\text{eff}}(\omega)| = |H(\omega)|^2$ (squared, not just the single-pass response). Equivalently, $|H_{\text{eff}}|^2 = |H|^4$. With our $n = 2$ Butterworth, this gives the rolloff equivalent of a 4th-order filter with zero phase distortion.

### J.5 Median Filtering {-}

A **median filter** replaces each sample with the median of its neighborhood:

$$\tilde{x}_t = \text{median}\{x_{t-w}, \ldots, x_{t-1}, x_t, x_{t+1}, \ldots, x_{t+w}\}$$

where the window size is $2w + 1$.

**Properties:**
- **Removes impulsive noise** (single-sample spikes) without distorting edges
- **Nonlinear** — unlike Butterworth, it does not have a transfer function
- **Preserves step edges** — important for preserving sharp transitions in joint angles

We use a 5-sample median filter ($w = 2$) in Stage 2 of the signal processing pipeline.

---

## Appendix K: Interpolation Theory {-}

### K.1 Linear Interpolation {-}

Given two known points $(x_0, y_0)$ and $(x_1, y_1)$, the linear interpolant at $x$ is:

$$y(x) = y_0 + \frac{y_1 - y_0}{x_1 - x_0}(x - x_0) = y_0 \cdot \frac{x_1 - x}{x_1 - x_0} + y_1 \cdot \frac{x - x_0}{x_1 - x_0}$$

Linear interpolation is $C^0$ continuous (continuous but not smooth — has corners at data points).

### K.2 Polynomial Interpolation {-}

Given $n+1$ points $(x_0, y_0), \ldots, (x_n, y_n)$, there exists a unique polynomial $P(x)$ of degree $\leq n$ passing through all points.

**Lagrange form:**

$$P(x) = \sum_{i=0}^{n} y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$$

**Runge's phenomenon.** For high-degree polynomials on equispaced nodes, the interpolant can oscillate wildly between data points, especially near the boundaries. This motivates piecewise approaches.

### K.3 Cubic Spline Interpolation {-}

A **cubic spline** is a piecewise cubic polynomial that is $C^2$ continuous (continuous up to the second derivative) at each data point.

On each interval $[x_k, x_{k+1}]$, the spline is a cubic polynomial $S_k(x)$. The conditions are:

1. **Interpolation**: $S_k(x_k) = y_k$ and $S_k(x_{k+1}) = y_{k+1}$
2. **$C^1$ continuity**: $S_k'(x_{k+1}) = S_{k+1}'(x_{k+1})$
3. **$C^2$ continuity**: $S_k''(x_{k+1}) = S_{k+1}''(x_{k+1})$

With $n$ intervals, there are $4n$ unknowns (4 coefficients per cubic), and $4n - 2$ conditions from (1)--(3). The remaining 2 conditions come from boundary conditions (natural spline: $S''(x_0) = S''(x_n) = 0$).

### K.4 Hermite Interpolation {-}

**Hermite interpolation** matches both function values AND derivative values at each node. Given values $f_k = f(x_k)$ and derivatives $d_k = f'(x_k)$, the cubic Hermite interpolant on $[x_k, x_{k+1}]$ is:

$$p(x) = f_k H_{00}(s) + h_k d_k H_{10}(s) + f_{k+1} H_{01}(s) + h_k d_{k+1} H_{11}(s)$$

where $s = (x - x_k)/h_k$, $h_k = x_{k+1} - x_k$, and the Hermite basis functions are:

$$H_{00}(s) = 2s^3 - 3s^2 + 1$$
$$H_{10}(s) = s^3 - 2s^2 + s$$
$$H_{01}(s) = -2s^3 + 3s^2$$
$$H_{11}(s) = s^3 - s^2$$

**Verification:**
- $H_{00}(0) = 1$, $H_{00}(1) = 0$, $H_{00}'(0) = 0$, $H_{00}'(1) = 0$
- $H_{10}(0) = 0$, $H_{10}(1) = 0$, $H_{10}'(0) = 1$, $H_{10}'(1) = 0$
- $H_{01}(0) = 0$, $H_{01}(1) = 1$, $H_{01}'(0) = 0$, $H_{01}'(1) = 0$
- $H_{11}(0) = 0$, $H_{11}(1) = 0$, $H_{11}'(0) = 0$, $H_{11}'(1) = 1$

### K.5 PCHIP — Monotone Piecewise Cubic Hermite Interpolation {-}

**PCHIP** (Fritsch & Carlson, 1980) is a Hermite interpolant where the derivatives $d_k$ are chosen to preserve **monotonicity** — if the data is locally increasing, the interpolant is also increasing.

**The Fritsch-Carlson algorithm:**

1. Compute slopes: $\delta_k = (f_{k+1} - f_k) / h_k$

2. If consecutive slopes have the same sign ($\delta_{k-1} \cdot \delta_k > 0$), set:

$$d_k = \frac{3(\delta_{k-1} + \delta_k)}{(\delta_{k-1} + 2\delta_k)/\delta_{k-1} + (2\delta_{k-1} + \delta_k)/\delta_k}$$

(harmonic mean-based formula)

3. If consecutive slopes have opposite signs ($\delta_{k-1} \cdot \delta_k \leq 0$), set $d_k = 0$ (local extremum).

**Why PCHIP over cubic splines?** Cubic splines can produce overshoots and oscillations near steep gradients. PCHIP sacrifices $C^2$ continuity (it is only $C^1$) to guarantee monotonicity preservation. For joint angle data with occasional sharp transitions, PCHIP avoids introducing artificial oscillations during gap-filling.

---

## Appendix L: Spectral Arc Length (SPARC) — Complete Derivation {-}

### L.1 Arc Length in the Plane {-}

From Appendix B.7, the arc length of a curve $(x(t), y(t))$ for $t \in [a, b]$ is:

$$L = \int_a^b \sqrt{\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2} \, dt$$

### L.2 The Frequency-Amplitude Curve {-}

For SPARC, the "curve" is not in physical space but in the **frequency-amplitude plane**. Given the normalized amplitude spectrum $\hat{V}_{\text{norm}}(f)$ of the angular velocity signal:

- The $x$-coordinate is frequency $f$
- The $y$-coordinate is the normalized amplitude $\hat{V}_{\text{norm}}(f)$

The "curve" consists of points $(f_k, \hat{V}_{\text{norm}}(f_k))$ for discrete frequency bins $f_k$.

### L.3 Discrete Arc Length Computation {-}

The arc length of this curve, computed discretely:

$$L = \sum_{k=1}^{K-1} \sqrt{(f_{k+1} - f_k)^2 + (\hat{V}_{\text{norm}}(f_{k+1}) - \hat{V}_{\text{norm}}(f_k))^2}$$

Each term is the Euclidean distance between consecutive points on the frequency-amplitude curve.

### L.4 SPARC Definition {-}

SPARC is defined as the **negative** arc length:

$$\text{SPARC} = -L = -\sum_{k=1}^{K-1} \sqrt{(f_{k+1} - f_k)^2 + (\hat{V}_{\text{norm}}(f_{k+1}) - \hat{V}_{\text{norm}}(f_k))^2}$$

**Why negative?** So that smoother movements get values closer to zero (less negative), providing an intuitive ordering: SPARC = $-1.2$ is smoother than SPARC = $-3.5$.

### L.5 Interpretation {-}

**Smooth movement:** The velocity signal is dominated by a single low-frequency component. The normalized amplitude spectrum is a compact, smooth curve. The arc length is short, so SPARC is close to zero.

**Jerky movement:** The velocity signal contains many frequency components (energy spread across frequencies). The normalized amplitude spectrum has many peaks and valleys. The arc length is long, so SPARC is a large negative number.

### L.6 Adaptive Frequency Cutoff {-}

We do not compute the arc length over the entire frequency range $[0, f_s/2]$. Instead:

$$f_c^{\text{adapt}} = \min\left(\max\{f : \hat{V}_{\text{norm}}(f) \geq 0.05\}, \; 10 \text{ Hz}\right)$$

This adaptive cutoff:
- Ignores frequencies where the amplitude is below 5% of the peak (noise floor)
- Caps at 10 Hz (well above gait-relevant frequencies)
- Ensures that SPARC is not inflated by noise in the high-frequency tail

### L.7 Comparison with Balasubramanian et al. (2012) {-}

The original formulation uses a continuous integral with frequency normalization:

$$\text{SPARC}_{\text{orig}} = -\int_0^{f_c} \sqrt{\left(\frac{1}{f_c}\right)^2 + \left(\frac{d\hat{V}_{\text{norm}}}{df}\right)^2} \, df$$

Our discrete implementation omits the $(1/f_c)^2$ normalization factor. Since $f_c$ is computed per-signal (adaptive cutoff), the normalization would make SPARC values difficult to compare across signals with different cutoff frequencies. As noted in the main text, this is an implementation variant that may alter ordering when adaptive cutoffs differ substantially between signals; within-study relative comparisons remain valid when groups share similar frequency content ranges.

---

## Appendix M: Normalized Jerk — Complete Derivation {-}

### M.1 Jerk as the Third Derivative {-}

**Jerk** is the rate of change of acceleration:

$$j(t) = \frac{d^3\theta}{dt^3} = \frac{d}{dt}\left(\frac{d^2\theta}{dt^2}\right)$$

In human movement, high jerk indicates abrupt changes in acceleration — "jerky" motion. Minimum-jerk trajectories are a hallmark of smooth, coordinated movement (Flash & Hogan, 1985).

### M.2 The Jerk Cost Functional {-}

The total "jerkiness" of a movement is quantified by the integrated squared jerk:

$$J = \int_0^T \left(\frac{d^3\theta}{dt^3}\right)^2 dt$$

This integral has units of (degrees/s³)² · s = degrees²/s⁵.

### M.3 Dimensionless Normalization {-}

To compare jerk across movements with different durations and amplitudes, Teulings et al. (1997) introduce the **normalized jerk**:

$$\text{NJ} = \sqrt{\frac{T^5}{2A^2} \cdot J} = \sqrt{\frac{T^5}{2A^2} \int_0^T \left(\frac{d^3\theta}{dt^3}\right)^2 dt}$$

where $T$ is the movement duration and $A = \max(\theta) - \min(\theta)$ is the peak-to-peak amplitude.

**Dimensional analysis:**

$$\left[\frac{T^5}{A^2} \cdot J\right] = \frac{s^5}{\text{deg}^2} \cdot \frac{\text{deg}^2}{s^5} = \text{dimensionless} \quad \square$$

**Derivation of the normalization factors:**

For a minimum-jerk trajectory of duration $T$ and amplitude $A$, the normalized jerk evaluates to $\text{NJ} = \sqrt{720/2} \approx 18.97$. Values above this indicate sub-optimal smoothness.

### M.4 Discrete Computation {-}

With $N$ samples at interval $\Delta t = 1/f_s$:

$$\text{NJ} \approx \sqrt{\frac{T^5}{2A^2} \cdot \Delta t \sum_{t=1}^{N} (\dddot{\theta}_t)^2}$$

The third derivative is computed by three successive applications of the central difference formula (NumPy's `gradient`):

1. $\dot{\theta}_t = \text{gradient}(\theta_t, \Delta t)$ — first derivative
2. $\ddot{\theta}_t = \text{gradient}(\dot{\theta}_t, \Delta t)$ — second derivative
3. $\dddot{\theta}_t = \text{gradient}(\ddot{\theta}_t, \Delta t)$ — third derivative

---

## Appendix N: Symmetry Metrics — Complete Derivations {-}

### N.1 The Robinson Symmetry Index {-}

For bilateral joint angle time series $\theta_R(t)$ (right) and $\theta_L(t)$ (left):

1. Compute the mean absolute amplitude of each side:

$$X_R = \frac{1}{T}\sum_{t=1}^{T} |\theta_R(t)|, \quad X_L = \frac{1}{T}\sum_{t=1}^{T} |\theta_L(t)|$$

2. The symmetry index is twice the absolute difference divided by the total:

$$\text{SI} = \frac{2|X_R - X_L|}{X_R + X_L} \times 100\%$$

**Derivation from first principles.** We want a metric that:
- Equals 0% when $X_R = X_L$ (perfect symmetry)
- Equals 200% when one side has zero amplitude ($X_R = 0$ or $X_L = 0$)
- Is dimensionless and scale-invariant

The formula $2|X_R - X_L|/(X_R + X_L)$ satisfies all three:
- If $X_R = X_L$: $2 \cdot 0 / 2X_R = 0$ $\square$
- If $X_L = 0$: $2X_R / X_R = 2 = 200\%$ $\square$
- Multiplying both $X_R$ and $X_L$ by $c > 0$: $2|cX_R - cX_L|/(cX_R + cX_L) = 2|X_R - X_L|/(X_R + X_L)$ $\square$

### N.2 Cross-Correlation {-}

The **cross-correlation** of two signals $f(t)$ and $g(t)$ is:

$$(f \star g)(\tau) = \int_{-\infty}^{\infty} f^*(t) \, g(t + \tau) \, dt$$

where $f^*$ is the complex conjugate of $f$ (for real signals, $f^* = f$). Cross-correlation measures the similarity of two signals as a function of a time lag $\tau$.

For discrete signals of length $T$:

$$(f \star g)(\tau) = \sum_{t=0}^{T-1} f(t) \, g(t + \tau)$$

### N.3 Normalized Cross-Correlation (NCC) {-}

The **normalized cross-correlation** removes the effect of signal mean and amplitude:

$$\text{NCC}(\theta_L, \theta_R) = \frac{\sum_{t=1}^{T} (\theta_L(t) - \bar{\theta}_L)(\theta_R(t) - \bar{\theta}_R)}{\|\theta_L - \bar{\theta}_L\|_2 \cdot \|\theta_R - \bar{\theta}_R\|_2}$$

**Derivation.** Define zero-mean signals:

$$\tilde{\theta}_L(t) = \theta_L(t) - \bar{\theta}_L, \quad \tilde{\theta}_R(t) = \theta_R(t) - \bar{\theta}_R$$

Then:

$$\text{NCC} = \frac{\sum_t \tilde{\theta}_L(t) \tilde{\theta}_R(t)}{\sqrt{\sum_t \tilde{\theta}_L(t)^2} \cdot \sqrt{\sum_t \tilde{\theta}_R(t)^2}} = \frac{\tilde{\boldsymbol{\theta}}_L \cdot \tilde{\boldsymbol{\theta}}_R}{\|\tilde{\boldsymbol{\theta}}_L\| \cdot \|\tilde{\boldsymbol{\theta}}_R\|}$$

This is the **cosine of the angle** between the two zero-mean signal vectors in $\mathbb{R}^T$. By the properties of the dot product (Appendix A):

$$\text{NCC} = \cos\alpha$$

where $\alpha$ is the angle between $\tilde{\boldsymbol{\theta}}_L$ and $\tilde{\boldsymbol{\theta}}_R$ in $T$-dimensional space.

**Properties:**
- $\text{NCC} \in [-1, 1]$
- NCC = 1: identical waveforms (same shape)
- NCC = $-1$: perfectly anti-phase waveforms
- NCC = 0: uncorrelated waveforms
- Invariant to amplitude scaling and DC offset

**Waveform Symmetry** = $|\text{NCC}| \times 100\%$. The absolute value is taken because anti-phase coupling ($\text{NCC} = -1$) in bilateral hip flexion is the expected pattern for symmetric walking — the right hip flexes while the left hip extends, and vice versa.

---

## Appendix O: The Hilbert Transform and Analytic Signals {-}

### O.1 Motivation {-}

To compute **continuous relative phase** between two oscillating signals, we need their **instantaneous phase** at each time point. The Hilbert transform provides this by constructing the **analytic signal** — a complex-valued signal whose phase at each instant corresponds to the phase of the oscillation.

### O.2 The Hilbert Transform {-}

The **Hilbert transform** of a real signal $x(t)$ is:

$$\mathcal{H}\{x(t)\} = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{x(\tau)}{t - \tau} \, d\tau$$

where P.V. denotes the Cauchy principal value (needed because the integrand has a singularity at $\tau = t$):

$$\text{P.V.} \int_{-\infty}^{\infty} \frac{x(\tau)}{t - \tau} d\tau = \lim_{\epsilon \to 0^+} \left[\int_{-\infty}^{t-\epsilon} \frac{x(\tau)}{t - \tau} d\tau + \int_{t+\epsilon}^{\infty} \frac{x(\tau)}{t - \tau} d\tau\right]$$

**Frequency domain interpretation.** The Hilbert transform shifts all positive frequencies by $-90°$ and all negative frequencies by $+90°$:

$$\mathcal{H}\{\cos(\omega t)\} = \sin(\omega t)$$
$$\mathcal{H}\{\sin(\omega t)\} = -\cos(\omega t)$$

In the frequency domain: $\hat{h}(f) = -j \cdot \text{sgn}(f)$, where $\text{sgn}$ is the sign function.

### O.3 The Analytic Signal {-}

The **analytic signal** is a complex signal formed by combining $x(t)$ with its Hilbert transform:

$$z(t) = x(t) + j\mathcal{H}\{x(t)\}$$

For a sinusoidal signal $x(t) = A\cos(\omega t + \phi)$:

$$\mathcal{H}\{x(t)\} = A\sin(\omega t + \phi)$$

$$z(t) = A\cos(\omega t + \phi) + jA\sin(\omega t + \phi) = Ae^{j(\omega t + \phi)}$$

### O.4 Instantaneous Phase {-}

The **instantaneous phase** is the argument (angle) of the analytic signal:

$$\phi(t) = \arg(z(t)) = \text{atan2}(\text{Im}(z(t)),\; \text{Re}(z(t))) = \text{atan2}(\mathcal{H}\{x(t)\},\; x(t))$$

The four-quadrant $\text{atan2}$ function is essential here; the two-argument form preserves quadrant information that single-argument $\arctan$ would lose.

For the sinusoidal example: $\phi(t) = \omega t + \phi_0$, which linearly increases with time. For more complex signals, $\phi(t)$ captures the local oscillatory phase.

**The instantaneous frequency** is the time derivative of phase:

$$f_{\text{inst}}(t) = \frac{1}{2\pi}\frac{d\phi}{dt}$$

### O.5 Continuous Relative Phase {-}

For two oscillating joint angle signals $\theta_a(t)$ and $\theta_b(t)$:

1. Center each signal: $\tilde{\theta}_a(t) = \theta_a(t) - \bar{\theta}_a$

2. Compute analytic signals: $z_a(t) = \tilde{\theta}_a(t) + j\mathcal{H}\{\tilde{\theta}_a(t)\}$

3. Extract phases: $\phi_a(t) = \arg(z_a(t))$, $\phi_b(t) = \arg(z_b(t))$

4. Compute relative phase:

$$\text{CRP}(t) = ((\phi_a(t) - \phi_b(t) + 180°) \bmod 360°) - 180°$$

This wrapping operation maps the result to $[-180°, 180°]$. Without the $+180°$ before the modulo, zero phase difference would map to $-180°$ rather than $0°$.

**Interpretation:**
- CRP $\approx 0°$: signals are **in-phase** (flexing together)
- CRP $\approx 180°$ or $-180°$: signals are **anti-phase** (one flexes while the other extends)
- For bilateral hip flexion in normal gait: CRP $\approx 180°$ (anti-phase, as expected)

### O.6 Circular Statistics {-}

CRP values are **circular data** — they wrap around at $\pm 180°$. Standard mean and standard deviation are inappropriate for circular data.

**Mean resultant length.** Treating each phase angle $\phi_t$ as a unit vector on the circle:

$$R = \sqrt{\left(\frac{1}{T}\sum_t \cos\phi_t\right)^2 + \left(\frac{1}{T}\sum_t \sin\phi_t\right)^2}$$

$R \in [0, 1]$:
- $R = 1$: all phases are identical (perfect consistency)
- $R = 0$: phases are uniformly distributed (no preferred phase)

**Circular standard deviation:**

$$\text{CSD} = \sqrt{-2\ln R}$$

**Derivation.** This formula arises from the von Mises distribution (the circular analogue of the normal distribution). For a von Mises distribution with concentration parameter $\kappa$:

$$E[R] \approx 1 - \frac{1}{2\kappa}$$

For large $\kappa$ (concentrated distribution), $-2\ln R \approx 1/\kappa$, which is analogous to $\sigma^2$ in the linear case (since $\kappa \approx 1/\sigma^2$ for concentrated distributions). Taking the square root gives a quantity with the same interpretation as linear standard deviation but appropriate for circular data.

The result is in radians; we convert to degrees by multiplying by $180°/\pi$.

---

## Appendix P: Multivariate Analysis {-}

### P.1 The Data Matrix {-}

Our multivariate analysis operates on a data matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ where:
- $n$ = number of observations (videos)
- $p$ = number of features (metrics)
- $X_{ij}$ = value of metric $j$ for video $i$

### P.2 Standardization {-}

Before PCA, we standardize each feature to zero mean and unit variance:

$$\tilde{X}_{ij} = \frac{X_{ij} - \bar{X}_j}{s_j}$$

where $\bar{X}_j$ and $s_j$ are the sample mean and standard deviation of feature $j$. This ensures all features contribute equally regardless of their original scale.

### P.3 Eigendecomposition {-}

An **eigenvalue** $\lambda$ and **eigenvector** $\mathbf{v}$ of a square matrix $\mathbf{A}$ satisfy:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

The eigenvector's direction is unchanged by the transformation $\mathbf{A}$; it is only scaled by the eigenvalue $\lambda$.

**Finding eigenvalues.** Rearranging: $(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$. For a non-trivial solution ($\mathbf{v} \neq \mathbf{0}$):

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

This is the **characteristic equation**, a polynomial of degree $p$ in $\lambda$.

**For symmetric matrices** (like covariance matrices):
- All eigenvalues are real
- Eigenvectors corresponding to distinct eigenvalues are orthogonal
- The matrix can be decomposed as $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$

### P.4 Principal Component Analysis (PCA) {-}

PCA finds the orthogonal directions (principal components) that capture the most variance in the data.

**Problem formulation.** Find the unit vector $\mathbf{w}$ that maximizes the variance of the projected data:

$$\max_{\mathbf{w}} \text{Var}(\tilde{\mathbf{X}}\mathbf{w}) = \max_{\mathbf{w}} \mathbf{w}^T \mathbf{C} \mathbf{w} \quad \text{subject to } \|\mathbf{w}\| = 1$$

where $\mathbf{C} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$ is the sample covariance matrix.

**Solution via Lagrange multipliers.** The constrained optimization problem with Lagrange multiplier $\lambda$:

$$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T\mathbf{C}\mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$$

Setting $\nabla_{\mathbf{w}}\mathcal{L} = 0$:

$$2\mathbf{C}\mathbf{w} - 2\lambda\mathbf{w} = 0 \implies \mathbf{C}\mathbf{w} = \lambda\mathbf{w}$$

This is the eigenvalue equation. The projection variance is:

$$\mathbf{w}^T\mathbf{C}\mathbf{w} = \mathbf{w}^T(\lambda\mathbf{w}) = \lambda\|\mathbf{w}\|^2 = \lambda$$

So the variance along direction $\mathbf{w}$ equals its eigenvalue $\lambda$. The first principal component is the eigenvector with the **largest** eigenvalue.

**Subsequent components.** The $k$-th principal component is the eigenvector with the $k$-th largest eigenvalue, subject to orthogonality with all previous components.

**Variance explained.** The proportion of total variance explained by the first $k$ components:

$$\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{p} \lambda_i}$$

In our study: PC1 explains 35.4%, PC2 explains 18.0%, PC3 explains 12.2%, for a cumulative 65.7%.

**Spread metrics in PC space.** Within each group $g$:

- **Trace** (total variance): $\text{tr}(\mathbf{C}_g) = \sum_i \lambda_{g,i}$ — proportional to the "total spread" of the group
- **Determinant** (generalized variance): $|\mathbf{C}_g| = \prod_i \lambda_{g,i}$ — ellipsoid volume scales as $\sqrt{|\mathbf{C}_g|}$, not $|\mathbf{C}_g|$ itself
- **Mean centroid distance**: $\bar{d}_g = \frac{1}{n_g}\sum_i \|\mathbf{z}_i - \bar{\mathbf{z}}_g\|$ — average distance from each point to the group center

### P.5 Linear Discriminant Analysis (LDA) {-}

LDA finds the linear projection that **maximizes class separability**.

**Fisher's criterion.** Find $\mathbf{w}$ maximizing:

$$J(\mathbf{w}) = \frac{\mathbf{w}^T\mathbf{S}_B\mathbf{w}}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}}$$

where:
- $\mathbf{S}_B = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T$ — **between-class scatter** (measures separation of class means)
- $\mathbf{S}_W = \sum_g \sum_{i \in g} (\mathbf{x}_i - \boldsymbol{\mu}_g)(\mathbf{x}_i - \boldsymbol{\mu}_g)^T$ — **within-class scatter** (measures spread within classes)

**Derivation of the solution.** Setting $\frac{\partial J}{\partial \mathbf{w}} = 0$:

By the quotient rule for matrix derivatives:

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{2\mathbf{S}_B\mathbf{w}(\mathbf{w}^T\mathbf{S}_W\mathbf{w}) - 2\mathbf{S}_W\mathbf{w}(\mathbf{w}^T\mathbf{S}_B\mathbf{w})}{(\mathbf{w}^T\mathbf{S}_W\mathbf{w})^2} = 0$$

Multiplying through by $(\mathbf{w}^T\mathbf{S}_W\mathbf{w})$:

$$\mathbf{S}_B\mathbf{w} - J(\mathbf{w})\mathbf{S}_W\mathbf{w} = 0$$

$$\mathbf{S}_B\mathbf{w} = J(\mathbf{w})\mathbf{S}_W\mathbf{w}$$

If $\mathbf{S}_W$ is invertible:

$$\mathbf{S}_W^{-1}\mathbf{S}_B\mathbf{w} = J(\mathbf{w})\mathbf{w}$$

This is a generalized eigenvalue problem. For the two-class case, $\mathbf{S}_B$ has rank 1 (it's an outer product), so there is exactly one non-zero eigenvalue. The solution simplifies to:

$$\mathbf{w}^* \propto \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$$

### P.6 Leave-One-Out Cross-Validation (LOO-CV) {-}

LOO-CV estimates generalization performance by:

1. For $i = 1, \ldots, n$:
   - Remove observation $i$ from the training set
   - Train the classifier on the remaining $n - 1$ observations
   - Predict the class of the held-out observation $i$
   - Record whether the prediction is correct

2. LOO-CV accuracy = $\frac{\text{number correct}}{n}$

LOO-CV is nearly unbiased (the training set has $n - 1$ observations, nearly as many as the full set) but has high variance (the $n$ training sets overlap extensively). For small samples like ours ($n = 20$), LOO-CV is the standard choice because $k$-fold CV would have even smaller training sets.

---

## Appendix Q: Detrended Fluctuation Analysis (DFA) {-}

### Q.1 Random Walks and Scaling {-}

A **random walk** is the cumulative sum of random increments:

$$Y(k) = \sum_{i=1}^{k} \epsilon_i$$

where $\epsilon_i$ are independent identically distributed random variables. For a simple random walk with $\epsilon_i \sim N(0, \sigma^2)$:

$$E[Y(k)] = 0, \quad \text{Var}(Y(k)) = k\sigma^2$$

The standard deviation (or RMS fluctuation) grows as $\sqrt{k}$:

$$F(k) \sim k^{0.5}$$

This $0.5$ is the **scaling exponent** for uncorrelated increments.

### Q.2 Self-Affinity and Long-Range Correlations {-}

If stride intervals have **long-range correlations** (the deviation from mean in one stride predicts deviations many strides later), the cumulative sum grows faster than $\sqrt{k}$:

$$F(n) \sim n^\alpha \quad \text{with } \alpha > 0.5$$

**Interpretation of $\alpha$:**
- $\alpha = 0.5$: uncorrelated increments (random walk) — **pathological gait**
- $\alpha \approx 0.75$: long-range correlations — **healthy gait**
- $\alpha = 1.0$: $1/f$ noise — **over-correlated**

### Q.3 The DFA Algorithm {-}

**Step 1: Compute the cumulative deviation series.**

Given stride intervals $s_1, s_2, \ldots, s_N$ with mean $\bar{s}$:

$$Y(k) = \sum_{i=1}^{k} (s_i - \bar{s}), \quad k = 1, \ldots, N$$

This is a "profile" of the stride interval fluctuations.

**Step 2: For each scale $n$, segment and detrend.**

Divide $Y$ into $\lfloor N/n \rfloor$ non-overlapping windows of length $n$. In each window $j$:

1. Fit a linear trend $\hat{Y}_n^{(j)}(k)$ by ordinary least squares:

$$\hat{Y}_n^{(j)}(k) = a_j + b_j k$$

where $b_j = \frac{\sum_{k=1}^{n}(k - \bar{k})(Y_{(j-1)n+k} - \overline{Y_j})}{\sum_{k=1}^{n}(k - \bar{k})^2}$ and $a_j = \overline{Y_j} - b_j \bar{k}$

Here $j = 1, 2, \ldots, \lfloor N/n \rfloor$ (one-based indexing).

2. Compute the root-mean-square residual:

$$F_j(n) = \sqrt{\frac{1}{n}\sum_{k=1}^{n}\left(Y((j-1)n + k) - \hat{Y}_n^{(j)}(k)\right)^2}$$

**Step 3: Average across segments.**

$$F(n) = \frac{1}{\lfloor N/n \rfloor}\sum_{j=1}^{\lfloor N/n \rfloor} F_j(n)$$

**Step 4: Estimate the scaling exponent.**

Plot $\log F(n)$ vs. $\log n$. The slope of the best-fit line is $\alpha$:

$$\alpha = \frac{\sum_i (\log n_i - \overline{\log n})(\log F(n_i) - \overline{\log F})}{\sum_i (\log n_i - \overline{\log n})^2}$$

This is ordinary least squares regression in log-log space.

### Q.4 Why Detrending? {-}

The "detrended" part is crucial. Without detrending, slow drifts (non-stationarity) in the stride interval would inflate $F(n)$ at large scales, falsely indicating long-range correlations. By removing the linear trend within each window, DFA isolates the fluctuations from trends.

### Q.5 Relationship to the Hurst Exponent {-}

The DFA scaling exponent $\alpha$ relates to the **Hurst exponent** $H$, but the relationship depends on the process type:

- **Fractional Gaussian noise (fGn):** $\alpha = H$ (stationary, long-range correlated)
- **Fractional Brownian motion (fBm):** $\alpha = H + 1$ (non-stationary, integrated process)

For stride interval sequences (which are stationary time series, not integrated processes), the fGn relationship $\alpha = H$ applies. The Hurst exponent was originally defined through rescaled range ($R/S$) analysis. DFA is preferred because it handles non-stationarity and short-range correlations better.

---

## Appendix R: Neural Network Mathematics for Pose Estimation {-}

### R.1 The Single Neuron (Perceptron) {-}

A **neuron** computes a weighted sum of its inputs, adds a bias, and applies an activation function:

$$y = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right) = \sigma(\mathbf{w}^T\mathbf{x} + b)$$

where:
- $\mathbf{x} = (x_1, \ldots, x_n)^T$ — input vector
- $\mathbf{w} = (w_1, \ldots, w_n)^T$ — weight vector
- $b$ — bias term
- $\sigma(\cdot)$ — activation function

### R.2 Activation Functions {-}

**Sigmoid:**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Range: $(0, 1)$. Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

**ReLU (Rectified Linear Unit):**

$$\text{ReLU}(z) = \max(0, z)$$

Derivative: $\text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}$ (undefined at $z = 0$, typically set to 0 or 1).

**Why ReLU is preferred:** It avoids the **vanishing gradient** problem. The sigmoid derivative has maximum value of 0.25 at $z = 0$. Chaining many sigmoid layers via backpropagation multiplies these small derivatives, causing gradients to approach zero. ReLU derivatives are either 0 or 1, preserving gradient magnitude.

### R.3 Multi-Layer Networks {-}

A **multi-layer perceptron** (MLP) chains multiple layers of neurons. For a network with $L$ layers:

$$\mathbf{h}_1 = \sigma_1(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \sigma_2(\mathbf{W}_2\mathbf{h}_1 + \mathbf{b}_2)$$
$$\vdots$$
$$\mathbf{y} = \sigma_L(\mathbf{W}_L\mathbf{h}_{L-1} + \mathbf{b}_L)$$

where $\mathbf{W}_\ell$ is the weight matrix and $\mathbf{b}_\ell$ is the bias vector for layer $\ell$.

### R.4 Loss Functions {-}

Training a neural network minimizes a **loss function** that measures prediction error.

**Mean Squared Error (MSE)** — for regression (e.g., keypoint coordinate prediction):

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N} \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|^2$$

**Cross-entropy** — for classification:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{ic}\log\hat{y}_{ic}$$

### R.5 Gradient Descent {-}

To minimize $\mathcal{L}(\mathbf{w})$ with respect to weights $\mathbf{w}$, we iteratively update:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}_t)$$

where $\eta > 0$ is the **learning rate** and $\nabla_{\mathbf{w}}\mathcal{L}$ is the gradient of the loss with respect to the weights.

**Why this works.** Taylor expansion of $\mathcal{L}$ around $\mathbf{w}_t$:

$$\mathcal{L}(\mathbf{w}_t + \Delta\mathbf{w}) \approx \mathcal{L}(\mathbf{w}_t) + \nabla\mathcal{L}^T \Delta\mathbf{w}$$

To decrease $\mathcal{L}$, we need $\nabla\mathcal{L}^T \Delta\mathbf{w} < 0$. Setting $\Delta\mathbf{w} = -\eta\nabla\mathcal{L}$:

$$\nabla\mathcal{L}^T(-\eta\nabla\mathcal{L}) = -\eta\|\nabla\mathcal{L}\|^2 < 0 \quad \square$$

The negative gradient direction gives the steepest descent.

### R.6 Backpropagation {-}

**Backpropagation** is an efficient algorithm for computing $\nabla_{\mathbf{w}}\mathcal{L}$ in multi-layer networks. It applies the chain rule layer by layer, from output to input.

**For a single hidden layer network:** $y = \sigma_2(\mathbf{W}_2\sigma_1(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2)$

Let $\mathbf{z}_1 = \mathbf{W}_1\mathbf{x} + \mathbf{b}_1$, $\mathbf{h}_1 = \sigma_1(\mathbf{z}_1)$, $\mathbf{z}_2 = \mathbf{W}_2\mathbf{h}_1 + \mathbf{b}_2$, $y = \sigma_2(\mathbf{z}_2)$.

**Forward pass:** compute $\mathbf{z}_1 \to \mathbf{h}_1 \to \mathbf{z}_2 \to y \to \mathcal{L}$

**Backward pass** (chain rule):

$$\frac{\partial\mathcal{L}}{\partial\mathbf{z}_2} = \frac{\partial\mathcal{L}}{\partial y} \cdot \sigma_2'(\mathbf{z}_2) \equiv \boldsymbol{\delta}_2$$

$$\frac{\partial\mathcal{L}}{\partial\mathbf{W}_2} = \boldsymbol{\delta}_2 \mathbf{h}_1^T$$

$$\frac{\partial\mathcal{L}}{\partial\mathbf{z}_1} = (\mathbf{W}_2^T\boldsymbol{\delta}_2) \odot \sigma_1'(\mathbf{z}_1) \equiv \boldsymbol{\delta}_1$$

$$\frac{\partial\mathcal{L}}{\partial\mathbf{W}_1} = \boldsymbol{\delta}_1 \mathbf{x}^T$$

where $\odot$ denotes element-wise multiplication. The key insight is that the "error signal" $\boldsymbol{\delta}$ is propagated backward through the network, hence "backpropagation."

### R.7 Convolutional Neural Networks (CNNs) {-}

CNNs are the backbone of modern pose estimation. They exploit **spatial structure** in images through three key operations.

**The convolution operation.** For a 2D input $I$ and a filter (kernel) $K$ of size $m \times m$:

$$(I * K)(i, j) = \sum_{p=0}^{m-1}\sum_{q=0}^{m-1} K(p, q) \cdot I(i+p, j+q)$$

The filter slides across the input, computing a dot product at each position. Different filters detect different features (edges, textures, shapes).

**Why convolution?**

1. **Parameter sharing**: the same filter is applied everywhere, dramatically reducing parameters compared to fully connected layers
2. **Translation equivariance**: if a feature moves in the input, its detection moves correspondingly in the output
3. **Spatial hierarchy**: stacking convolutional layers detects increasingly complex features (edges → parts → objects)

**Pooling.** Reduces spatial dimensions while retaining important features:

$$\text{MaxPool}_{2\times 2}(I)(i, j) = \max(I(2i, 2j), I(2i+1, 2j), I(2i, 2j+1), I(2i+1, 2j+1))$$

### R.8 Heatmap Regression for Keypoint Detection {-}

Modern pose estimation predicts a **heatmap** for each anatomical keypoint. For keypoint $k$, the network outputs a 2D heatmap $H_k \in \mathbb{R}^{h \times w}$ where each pixel value represents the probability that keypoint $k$ is located at that position.

**Ground truth heatmaps** are generated as 2D Gaussians centered at the annotated keypoint location:

$$H_k^{\text{gt}}(i, j) = \exp\left(-\frac{(i - i_k)^2 + (j - j_k)^2}{2\sigma^2}\right)$$

where $(i_k, j_k)$ is the ground truth location and $\sigma$ controls the spread.

**Training loss:** MSE between predicted and ground truth heatmaps:

$$\mathcal{L} = \sum_{k=1}^{K} \sum_{i,j} \left(H_k(i, j) - H_k^{\text{gt}}(i, j)\right)^2$$

**Keypoint extraction:** the predicted location is the argmax of the heatmap:

$$(i_k^*, j_k^*) = \arg\max_{(i,j)} H_k(i, j)$$

Sub-pixel refinement is achieved by fitting a parabola around the maximum.

### R.9 The MediaPipe Architecture {-}

MediaPipe PoseLandmarker uses a two-stage pipeline:

1. **Person detector**: A lightweight CNN locates bounding boxes around people in the image
2. **Pose estimator**: A regression CNN predicts 33 anatomical landmark coordinates within each bounding box

The "heavy" model variant uses a deeper backbone network for higher accuracy at the cost of inference speed. In VIDEO mode, temporal smoothing is applied across frames using a Kalman-like filter.

Each landmark prediction includes:
- $(x, y)$: normalized image coordinates in $[0, 1]$
- $z$: depth relative to the hip midpoint (unreliable from monocular video)
- $v$: visibility confidence in $[0, 1]$

---

## Appendix S: Gait Cycle Biomechanics {-}

### S.1 Phases of the Gait Cycle {-}

One complete **gait cycle** (stride) spans from one heel strike to the next heel strike of the same foot. It consists of:

- **Stance phase** (~60% of cycle): foot is in contact with the ground
  - Initial contact (heel strike)
  - Loading response
  - Midstance
  - Terminal stance
  - Pre-swing (toe-off)

- **Swing phase** (~40% of cycle): foot is off the ground
  - Initial swing
  - Mid-swing
  - Terminal swing

### S.2 Double Support {-}

**Double support** is the period when both feet are on the ground simultaneously. It occurs twice per gait cycle:

1. At **initial contact** (one foot strikes while the other is still in stance)
2. At **pre-swing** (one foot lifts while the other has already struck)

If $S$ is the stance fraction (proportion of the gait cycle spent in stance):

- Each leg spends $S$ in stance and $(1-S)$ in swing
- Both legs are in stance simultaneously for a total of $S + S - 1 = 2S - 1$ (since the two stance phases overlap)

$$\text{DS}\% = \max(0, \; 2S - 1) \times 100$$

**Verification:** At comfortable walking speed, $S \approx 0.60$:

$$\text{DS}\% = (2 \times 0.60 - 1) \times 100 = 20\% \quad \square$$

At running speed, $S < 0.50$: there is a **flight phase** (both feet off the ground) and $\text{DS}\% = 0$.

### S.3 Cadence {-}

Cadence is the number of steps per minute. Since our heel strike detector identifies **ipsilateral** heel strikes (same foot), each detected interval is a **stride** (two steps):

$$\text{cadence} = \frac{60}{\bar{\Delta t}_{\text{stride}}} \times 2 \quad \text{(steps/min)}$$

The $\times 2$ converts from strides/min to steps/min.

**Normal values:** 100--120 steps/min at comfortable walking speed (Perry & Burnfield, 2010).

### S.4 Heel Strike Detection {-}

We detect heel strikes as **peaks** in the low-pass filtered hip flexion signal. The biomechanical rationale: maximum hip flexion occurs at approximately the same time as heel strike (the leg is maximally forward).

**Peak detection criteria:**
- Local maximum of $\theta_{\text{hip}}^{\text{filt}}(t)$
- Prominence $> \max(0.15 \cdot \text{ROM}_{\text{hip}}, 1°)$

The prominence threshold scales with the signal's range of motion, preventing false detections from small fluctuations while remaining sensitive enough for low-amplitude signals.

**Refinement with heel Y-coordinates.** When pixel-space heel landmark positions are available, we refine the timing:

$$t_{\text{HS}}^* = \arg\max_{|t - t_{\text{HS}}| \leq 0.15f_s} y_{\text{heel}}(t)$$

The heel is at its lowest physical position (highest pixel Y, since Y increases downward in image coordinates) at the moment of heel strike.

---

## Appendix T: Composite Scoring — MQS Transfer Functions {-}

### T.1 Piecewise Linear Scoring {-}

Each biomechanical signal is mapped to a 0--100 score using a **piecewise linear transfer function** defined by four anchor points:

- $v_{\text{worst,lo}}$: worst-case low value (score = 0)
- $v_{\text{opt,lo}}$: optimal range low bound (score = 100)
- $v_{\text{opt,hi}}$: optimal range high bound (score = 100)
- $v_{\text{worst,hi}}$: worst-case high value (score = 0)

The transfer function:

$$S(v) = \begin{cases}
0 & v \leq v_{\text{worst,lo}} \\
100 \cdot \dfrac{v - v_{\text{worst,lo}}}{v_{\text{opt,lo}} - v_{\text{worst,lo}}} & v_{\text{worst,lo}} < v < v_{\text{opt,lo}} \\
100 & v_{\text{opt,lo}} \leq v \leq v_{\text{opt,hi}} \\
100 \cdot \dfrac{v_{\text{worst,hi}} - v}{v_{\text{worst,hi}} - v_{\text{opt,hi}}} & v_{\text{opt,hi}} < v < v_{\text{worst,hi}} \\
0 & v \geq v_{\text{worst,hi}}
\end{cases}$$

This creates a trapezoidal shape: zero at extremes, linearly increasing to 100 at the optimal range, constant at 100 within the optimal range, linearly decreasing back to zero.

### T.2 Domain Aggregation {-}

Each domain score is the mean of its constituent signal scores:

$$S_d = \frac{1}{|d|}\sum_{s \in d} S_s$$

The overall MQS is the weighted sum:

$$\text{MQS} = \sum_{d \in \mathcal{D}} w_d \cdot S_d$$

where the weights $w_d$ reflect the clinical importance of each domain (see Section 2.6 of the main paper).

### T.3 Confidence Weighting {-}

For video-derived analysis, measurement confidence varies with pose detection quality. The confidence factor:

$$c_f = f_{\text{obs}} \times \bar{v}_{\text{detected}} \times (1 - p_{\text{interp}})$$

Each component is in $[0, 1]$:
- $f_{\text{obs}}$: fraction of frames with detected pose
- $\bar{v}_{\text{detected}}$: mean landmark visibility confidence
- $1 - p_{\text{interp}}$: penalty for interpolated data

The interpolation penalty is clamped:

$$p_{\text{interp}} = \text{clip}(\lambda \cdot f_{\text{interp}}, \; 0, \; 0.5), \quad \lambda = 0.5$$

This ensures the confidence factor never drops below $0.5 \cdot f_{\text{obs}} \cdot \bar{v}_{\text{detected}}$ even for heavily interpolated signals, reflecting that interpolation (while less reliable than direct observation) still provides useful information.

---

## Appendix U: The Variance Ratio Distribution {-}

### U.1 Distribution of the Variance Ratio Under $H_0$ {-}

Under the null hypothesis $\sigma_1^2 = \sigma_2^2 = \sigma^2$, **assuming both samples are drawn independently from normal distributions**, the ratio of sample variances:

$$F = \frac{s_1^2}{s_2^2}$$

follows an **F-distribution** with $n_1 - 1$ and $n_2 - 1$ degrees of freedom.

**Derivation.** Each sample variance $s_i^2$ satisfies:

$$\frac{(n_i - 1)s_i^2}{\sigma^2} \sim \chi^2_{n_i - 1}$$

(this follows from the fact that $\sum(X_j - \bar{X})^2 / \sigma^2$ is a sum of squared standard normals with one constraint, giving $n_i - 1$ degrees of freedom).

Therefore:

$$F = \frac{s_1^2/\sigma^2}{s_2^2/\sigma^2} = \frac{s_1^2}{s_2^2} = \frac{\chi^2_{n_1-1}/(n_1-1)}{\chi^2_{n_2-1}/(n_2-1)} \sim F(n_1-1, n_2-1)$$

The $\sigma^2$ terms cancel under $H_0$.

### U.2 Why We Don't Use the F-Test Directly {-}

While the above derivation is elegant, it requires that the underlying data be normally distributed. The F-test for variance equality is **extremely sensitive** to non-normality — even mild departures can inflate the Type I error rate dramatically.

This is why we use:
1. **Levene's test** (robust to non-normality via the absolute deviation transformation)
2. **Permutation tests** (nonparametric under the exchangeability assumption)
3. **Bootstrap CIs** (nonparametric, though coverage guarantees require regularity conditions)

The F-distribution derivation remains useful for understanding the expected behavior of variance ratios and for calibrating our bootstrap results.

---

## Appendix V: Matrix Algebra from Scratch {-}

A Calculus 1 background does not include matrix algebra, yet matrices are essential for PCA (Appendix P), LDA (Appendix P.5), neural networks (Appendix R), and covariance computation. This appendix builds the necessary foundation.

### V.1 Matrices and Basic Operations {-}

A **matrix** is a rectangular array of numbers. An $m \times n$ matrix has $m$ rows and $n$ columns:

$$\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}$$

We write $\mathbf{A} \in \mathbb{R}^{m \times n}$. When $m = n$, the matrix is **square**.

**Matrix addition** (element-wise, same dimensions required):

$$(\mathbf{A} + \mathbf{B})_{ij} = a_{ij} + b_{ij}$$

**Scalar multiplication:**

$$(c\mathbf{A})_{ij} = c \cdot a_{ij}$$

**Transpose:** The transpose $\mathbf{A}^T$ swaps rows and columns: $(\mathbf{A}^T)_{ij} = a_{ji}$. If $\mathbf{A}$ is $m \times n$, then $\mathbf{A}^T$ is $n \times m$.

A matrix is **symmetric** if $\mathbf{A} = \mathbf{A}^T$ (equivalently, $a_{ij} = a_{ji}$ for all $i, j$). Covariance matrices are always symmetric.

### V.2 Matrix Multiplication {-}

For $\mathbf{A} \in \mathbb{R}^{m \times p}$ and $\mathbf{B} \in \mathbb{R}^{p \times n}$, the product $\mathbf{C} = \mathbf{AB}$ is an $m \times n$ matrix:

$$c_{ij} = \sum_{k=1}^{p} a_{ik} b_{kj}$$

Each entry is the dot product of the $i$-th row of $\mathbf{A}$ with the $j$-th column of $\mathbf{B}$.

**Example ($2 \times 2$):**

$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 1\cdot5+2\cdot7 & 1\cdot6+2\cdot8 \\ 3\cdot5+4\cdot7 & 3\cdot6+4\cdot8 \end{pmatrix} = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$$

**Matrix multiplication is NOT commutative:** $\mathbf{AB} \neq \mathbf{BA}$ in general.

**Matrix-vector multiplication.** A matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ multiplied by a vector $\mathbf{x} \in \mathbb{R}^n$ gives a vector $\mathbf{y} \in \mathbb{R}^m$:

$$y_i = \sum_{j=1}^{n} a_{ij} x_j$$

This is the operation performed at each layer of a neural network: $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$.

### V.3 The Identity Matrix and Inverse {-}

The **identity matrix** $\mathbf{I}_n$ is the $n \times n$ matrix with 1s on the diagonal and 0s elsewhere:

$$\mathbf{I}_n = \begin{pmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{pmatrix}$$

For any matrix $\mathbf{A}$: $\mathbf{AI} = \mathbf{IA} = \mathbf{A}$.

The **inverse** $\mathbf{A}^{-1}$ of a square matrix $\mathbf{A}$ satisfies:

$$\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$$

Not all matrices have inverses. A matrix is **invertible** (or non-singular) if and only if its determinant is non-zero.

**For $2 \times 2$ matrices:**

$$\mathbf{A} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, \quad \mathbf{A}^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

The inverse is used in LDA: $\mathbf{w}^* \propto \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$.

### V.4 The Determinant {-}

The **determinant** of a square matrix is a scalar that encodes several properties:

**For $2 \times 2$:**

$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

**For $3 \times 3$** (cofactor expansion along the first row):

$$\det\begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix} = a(ei - fh) - b(di - fg) + c(dh - eg)$$

**Geometric interpretation:** The absolute value of the determinant equals the volume of the parallelepiped spanned by the column vectors. For a covariance matrix, $\det(\mathbf{C})$ is the **generalized variance**. The volume of the confidence ellipsoid is proportional to $\sqrt{\det(\mathbf{C})}$, not $\det(\mathbf{C})$ itself (see Appendix AF.2). A determinant ratio of $25\times$ corresponds to a volume ratio of $\sqrt{25} = 5\times$.

### V.5 The Trace {-}

The **trace** of a square matrix is the sum of its diagonal elements:

$$\text{tr}(\mathbf{A}) = \sum_{i=1}^{n} a_{ii}$$

**Properties:**
- $\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})$
- $\text{tr}(c\mathbf{A}) = c \cdot \text{tr}(\mathbf{A})$
- $\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})$ (cyclic property)
- For a covariance matrix: $\text{tr}(\mathbf{C}) = \sum \lambda_i = \text{total variance}$

The trace of the covariance matrix equals the sum of all eigenvalues, which equals the total variance across all dimensions. This is why we use the trace ratio to compare total spread: $\rho = \text{tr}(\mathbf{C}_{\text{control}}) / \text{tr}(\mathbf{C}_{\text{runway}})$.

### V.6 Positive Semi-Definite Matrices {-}

A symmetric matrix $\mathbf{A}$ is **positive semi-definite** (PSD) if:

$$\mathbf{x}^T\mathbf{A}\mathbf{x} \geq 0 \quad \text{for all } \mathbf{x} \in \mathbb{R}^n$$

Equivalently, all eigenvalues are non-negative: $\lambda_i \geq 0$.

**Covariance matrices are always PSD.** Proof: for the sample covariance matrix $\mathbf{C} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$:

$$\mathbf{x}^T\mathbf{C}\mathbf{x} = \frac{1}{n-1}\mathbf{x}^T\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}\mathbf{x} = \frac{1}{n-1}\|\tilde{\mathbf{X}}\mathbf{x}\|^2 \geq 0$$

since the squared norm is always non-negative.

---

## Appendix W: Complex Numbers and Euler's Formula {-}

The Hilbert transform (Appendix O), Fourier analysis (Appendix I), and the DFT all use complex numbers and complex exponentials. A Calculus 1 background does not typically cover these topics.

### W.1 Complex Numbers {-}

A **complex number** has the form:

$$z = a + jb$$

where $a$ is the **real part** ($\text{Re}(z)$), $b$ is the **imaginary part** ($\text{Im}(z)$), and $j = \sqrt{-1}$ is the **imaginary unit** (engineers use $j$; mathematicians use $i$).

The key property of $j$: $j^2 = -1$.

**Arithmetic:**
- Addition: $(a + jb) + (c + jd) = (a+c) + j(b+d)$
- Multiplication: $(a + jb)(c + jd) = (ac - bd) + j(ad + bc)$

**The complex conjugate** of $z = a + jb$ is $z^* = a - jb$. Note: $z \cdot z^* = a^2 + b^2$.

**Modulus (magnitude):**

$$|z| = \sqrt{a^2 + b^2} = \sqrt{z \cdot z^*}$$

**Argument (phase angle):**

$$\arg(z) = \arctan2(b, a)$$

### W.2 The Complex Plane {-}

Complex numbers can be visualized as points in the **complex plane** (Argand diagram):
- The horizontal axis represents the real part
- The vertical axis represents the imaginary part
- The point $z = a + jb$ is at coordinates $(a, b)$

**Polar form:** A complex number can also be written as:

$$z = r(\cos\phi + j\sin\phi) = r \cdot \text{cis}(\phi)$$

where $r = |z|$ is the modulus and $\phi = \arg(z)$ is the argument.

### W.3 Euler's Formula {-}

**Euler's formula** connects the exponential function with trigonometry:

$$e^{j\phi} = \cos\phi + j\sin\phi$$

**Derivation from Taylor series.** The Taylor series for the exponential, cosine, and sine functions are:

$$e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \frac{x^4}{4!} + \cdots$$

$$\cos\phi = \sum_{n=0}^{\infty} \frac{(-1)^n \phi^{2n}}{(2n)!} = 1 - \frac{\phi^2}{2!} + \frac{\phi^4}{4!} - \cdots$$

$$\sin\phi = \sum_{n=0}^{\infty} \frac{(-1)^n \phi^{2n+1}}{(2n+1)!} = \phi - \frac{\phi^3}{3!} + \frac{\phi^5}{5!} - \cdots$$

Substituting $x = j\phi$ into the exponential series:

$$e^{j\phi} = 1 + j\phi + \frac{(j\phi)^2}{2!} + \frac{(j\phi)^3}{3!} + \frac{(j\phi)^4}{4!} + \frac{(j\phi)^5}{5!} + \cdots$$

Using powers of $j$: $j^2 = -1$, $j^3 = -j$, $j^4 = 1$, $j^5 = j$, etc.:

$$= 1 + j\phi - \frac{\phi^2}{2!} - j\frac{\phi^3}{3!} + \frac{\phi^4}{4!} + j\frac{\phi^5}{5!} - \cdots$$

Grouping real and imaginary parts:

$$= \underbrace{\left(1 - \frac{\phi^2}{2!} + \frac{\phi^4}{4!} - \cdots\right)}_{\cos\phi} + j\underbrace{\left(\phi - \frac{\phi^3}{3!} + \frac{\phi^5}{5!} - \cdots\right)}_{\sin\phi}$$

$$= \cos\phi + j\sin\phi \quad \square$$

**Key consequences:**

1. $|e^{j\phi}| = \sqrt{\cos^2\phi + \sin^2\phi} = 1$ — the unit circle in the complex plane
2. $e^{j\pi} = -1$ (**Euler's identity**, often called "the most beautiful equation in mathematics")
3. Any complex number in polar form: $z = re^{j\phi}$

### W.4 Complex Exponentials in the DFT {-}

The DFT uses the complex exponential $e^{-j2\pi kn/N}$ as the "basis function" for frequency analysis. By Euler's formula:

$$e^{-j2\pi kn/N} = \cos\left(\frac{2\pi kn}{N}\right) - j\sin\left(\frac{2\pi kn}{N}\right)$$

The DFT simultaneously extracts both the cosine and sine components at each frequency — the real part captures the cosine component and the imaginary part captures the sine component.

### W.5 Multiplication as Rotation {-}

Multiplying two complex numbers in polar form:

$$z_1 z_2 = r_1 e^{j\phi_1} \cdot r_2 e^{j\phi_2} = r_1 r_2 \, e^{j(\phi_1 + \phi_2)}$$

The magnitudes multiply and the angles add. Multiplication by $e^{j\phi}$ rotates a complex number by angle $\phi$ in the complex plane. The Hilbert transform uses this: it shifts positive frequencies by $-90°$ (multiplication by $-j$) and negative frequencies by $+90°$ (multiplication by $+j$). This frequency-sign-dependent phase shift is what creates the analytic signal (see Appendix O.2).

---

## Appendix X: Lagrange Multipliers and Constrained Optimization {-}

PCA (Appendix P.4) and LDA (Appendix P.5) are both constrained optimization problems. This appendix derives the method of Lagrange multipliers from first principles.

### X.1 The Problem {-}

**Unconstrained optimization:** find $\mathbf{x}$ that minimizes (or maximizes) $f(\mathbf{x})$. Solution: set $\nabla f = 0$.

**Constrained optimization:** find $\mathbf{x}$ that maximizes $f(\mathbf{x})$ subject to $g(\mathbf{x}) = 0$.

The constraint restricts the search to a surface (or curve) defined by $g(\mathbf{x}) = 0$.

### X.2 Geometric Intuition (Two Variables) {-}

Consider maximizing $f(x, y)$ subject to $g(x, y) = 0$.

The constraint $g(x, y) = 0$ defines a curve in the $(x, y)$ plane. As we move along this curve, $f$ increases and decreases. The maximum occurs where the **level curve** of $f$ is tangent to the constraint curve.

At a tangent point, the gradient of $f$ is parallel to the gradient of $g$ (both are perpendicular to the constraint curve):

$$\nabla f = \lambda \nabla g$$

for some scalar $\lambda$ (the **Lagrange multiplier**).

### X.3 The Method {-}

To maximize $f(\mathbf{x})$ subject to $g(\mathbf{x}) = 0$:

1. Form the **Lagrangian**: $\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda g(\mathbf{x})$

2. Set all partial derivatives to zero:

$$\frac{\partial\mathcal{L}}{\partial x_i} = 0 \quad \text{for all } i, \qquad \frac{\partial\mathcal{L}}{\partial\lambda} = 0$$

The condition $\frac{\partial\mathcal{L}}{\partial\lambda} = 0$ recovers the constraint $g(\mathbf{x}) = 0$.

### X.4 Application to PCA {-}

PCA maximizes $f(\mathbf{w}) = \mathbf{w}^T\mathbf{C}\mathbf{w}$ (projected variance) subject to $g(\mathbf{w}) = \mathbf{w}^T\mathbf{w} - 1 = 0$ (unit vector).

Lagrangian:

$$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T\mathbf{C}\mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$$

Setting $\frac{\partial\mathcal{L}}{\partial\mathbf{w}} = 0$. **Why $\frac{\partial}{\partial\mathbf{w}}(\mathbf{w}^T\mathbf{C}\mathbf{w}) = 2\mathbf{C}\mathbf{w}$:** In components, $\mathbf{w}^T\mathbf{C}\mathbf{w} = \sum_i \sum_j w_i C_{ij} w_j$. Taking $\partial/\partial w_k$: each term where $i = k$ contributes $C_{kj}w_j$ and each term where $j = k$ contributes $w_i C_{ik}$. Since $\mathbf{C}$ is symmetric ($C_{ij} = C_{ji}$), both sums equal $\sum_j C_{kj} w_j = (\mathbf{C}\mathbf{w})_k$, giving $2(\mathbf{C}\mathbf{w})_k$. Similarly, $\frac{\partial}{\partial\mathbf{w}}(\mathbf{w}^T\mathbf{w}) = 2\mathbf{w}$. Therefore:

$$2\mathbf{C}\mathbf{w} - 2\lambda\mathbf{w} = 0 \implies \mathbf{C}\mathbf{w} = \lambda\mathbf{w}$$

This is the eigenvalue equation. The optimal $\mathbf{w}$ is an eigenvector of $\mathbf{C}$, and the maximum variance equals the largest eigenvalue.

### X.5 Worked Example (Two Dimensions) {-}

Consider data with covariance matrix:

$$\mathbf{C} = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$$

**Finding eigenvalues.** Solve $\det(\mathbf{C} - \lambda\mathbf{I}) = 0$:

$$\det\begin{pmatrix} 4-\lambda & 2 \\ 2 & 3-\lambda \end{pmatrix} = (4-\lambda)(3-\lambda) - 4 = \lambda^2 - 7\lambda + 8 = 0$$

$$\lambda = \frac{7 \pm \sqrt{49 - 32}}{2} = \frac{7 \pm \sqrt{17}}{2}$$

$$\lambda_1 \approx 5.56, \quad \lambda_2 \approx 1.44$$

**Finding eigenvectors.** For $\lambda_1 \approx 5.56$:

$$(4 - 5.56)w_1 + 2w_2 = 0 \implies w_2 = 0.78 w_1$$

Normalizing: $\mathbf{v}_1 \approx (0.79, 0.62)^T$. This is the first principal component direction.

**Variance explained:** $\lambda_1 / (\lambda_1 + \lambda_2) = 5.56 / 7 = 79.4\%$ — the first PC captures 79.4% of the total variance.

---

## Appendix Y: The Central Limit Theorem {-}

### Y.1 Statement {-}

**Central Limit Theorem (CLT).** Let $X_1, X_2, \ldots, X_n$ be independent and identically distributed random variables with mean $\mu$ and finite variance $\sigma^2$. Then as $n \to \infty$:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1)$$

where $\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i$ is the sample mean.

Equivalently: $\bar{X}_n \approx N(\mu, \sigma^2/n)$ for large $n$.

### Y.2 Why the CLT Matters for This Study {-}

1. **Bootstrap**: Bootstrap consistency relies on the empirical distribution converging to the true distribution (Glivenko-Cantelli), not solely on CLT normality. However, the CLT provides intuition for why bootstrap distributions of means and standardized statistics are approximately normal
2. **P-values**: Many test statistics are approximately normally distributed under $H_0$, which allows computing p-values from normal tables
3. **Confidence intervals**: The $\pm 1.96\sigma/\sqrt{n}$ formula relies on the normality of $\bar{X}$

### Y.3 Intuition: Sum of Random Variables {-}

**Why does averaging tend to produce normality (given finite variance and independence)?** Consider summing $n$ independent random variables with finite variance. Each variable shifts the sum by a random amount. For each possible value of the first $n-1$ variables, the $n$-th variable "smears" the distribution a little. After enough smearing, the shape converges to the bell curve regardless of the original distribution (provided it has finite variance).

More precisely: the convolution of any distribution with itself, repeated enough times, approaches a Gaussian. This is because:

- Convolution in the time domain = multiplication in the frequency domain
- The characteristic function of a sum is the product of individual characteristic functions
- The Gaussian is the unique fixed point of this product operation (among distributions with finite variance)

### Y.4 Rate of Convergence {-}

The CLT is a limit theorem — it holds as $n \to \infty$. How large must $n$ be in practice?

**Rules of thumb:**
- Symmetric distributions (e.g., uniform): $n \geq 10$ usually sufficient
- Moderately skewed: $n \geq 30$
- Heavily skewed or heavy-tailed: $n \geq 100$ or more

Our study has $n = 17$--$22$ per group. This is marginal for the CLT, which is one reason we use:
- **Mann-Whitney U** (non-parametric, does not assume normality)
- **Permutation tests** (exact under exchangeability, no distributional assumption)
- **Bootstrap** (approximates the sampling distribution directly)

---

## Appendix Z: Worked Fourier and SPARC Example {-}

### Z.1 DFT of a Simple Signal {-}

Consider a 4-point signal: $x = [1, 0, -1, 0]$ sampled at $f_s = 4$ Hz.

The DFT is:

$$X_k = \sum_{n=0}^{3} x_n \, e^{-j2\pi kn/4}$$

**$k = 0$** (DC component):

$$X_0 = 1 \cdot e^0 + 0 \cdot e^0 + (-1) \cdot e^0 + 0 \cdot e^0 = 1 + 0 - 1 + 0 = 0$$

**$k = 1$** (frequency $f_1 = 1$ Hz):

$$X_1 = 1 \cdot e^0 + 0 \cdot e^{-j\pi/2} + (-1) \cdot e^{-j\pi} + 0 \cdot e^{-j3\pi/2}$$

Using Euler: $e^0 = 1$, $e^{-j\pi} = -1$:

$$X_1 = 1 + 0 + (-1)(-1) + 0 = 1 + 1 = 2$$

**$k = 2$** (frequency $f_2 = 2$ Hz = Nyquist):

$$X_2 = 1 \cdot e^0 + 0 \cdot e^{-j\pi} + (-1) \cdot e^{-j2\pi} + 0 \cdot e^{-j3\pi} = 1 + 0 - 1 + 0 = 0$$

**$k = 3$** (negative-frequency mirror of $k = 1$; for real signals, $X_{N-k} = X_k^*$):

$$X_3 = 1 + 0 + (-1)(e^{-j3\pi}) + 0 = 1 + 0 + 1 + 0 = 2$$

Result: $X = [0, 2, 0, 2]$. The amplitude spectrum $|X| = [0, 2, 0, 2]$ shows energy at $k = 1$ (1 Hz) and its negative-frequency mirror $k = 3$. For real signals, SPARC uses only the one-sided spectrum ($k = 0, \ldots, N/2$), so we would analyze $|X_0| = 0$, $|X_1| = 2$, $|X_2| = 0$. The signal is a pure 1 Hz oscillation, as expected.

### Z.2 SPARC Computation on Synthetic Data {-}

Consider a smooth hip velocity signal at 30 fps with energy concentrated at 1--2 Hz (typical gait). The normalized amplitude spectrum might look like:

| Frequency $f_k$ (Hz) | $\hat{V}_{\text{norm}}(f_k)$ |
|---|---|
| 0.0 | 0.10 |
| 0.5 | 0.30 |
| 1.0 | 1.00 |
| 1.5 | 0.85 |
| 2.0 | 0.40 |
| 2.5 | 0.15 |
| 3.0 | 0.03 |

The adaptive cutoff: $f_c^{\text{adapt}} = 2.5$ Hz (last frequency where $\hat{V}_{\text{norm}} \geq 0.05$).

Arc length computation (frequency spacing $\Delta f = 0.5$ Hz):

$$L_1 = \sqrt{(0.5-0)^2 + (0.30-0.10)^2} = \sqrt{0.25 + 0.04} = 0.539$$

$$L_2 = \sqrt{0.25 + (1.00-0.30)^2} = \sqrt{0.25 + 0.49} = 0.860$$

$$L_3 = \sqrt{0.25 + (0.85-1.00)^2} = \sqrt{0.25 + 0.0225} = 0.522$$

$$L_4 = \sqrt{0.25 + (0.40-0.85)^2} = \sqrt{0.25 + 0.2025} = 0.673$$

$$L_5 = \sqrt{0.25 + (0.15-0.40)^2} = \sqrt{0.25 + 0.0625} = 0.559$$

Total: $L = 0.539 + 0.860 + 0.522 + 0.673 + 0.559 = 3.153$

$$\text{SPARC} = -3.153$$

For comparison, a jerky signal might have SPARC $= -8.5$ (longer arc length due to spectral ripple).

---

## Appendix AA: Statistical Power Analysis {-}

### AA.1 Definition of Power {-}

The **power** of a statistical test is the probability of correctly rejecting $H_0$ when it is actually false:

$$\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ true})$$

Power depends on:
1. **Effect size** — how large the true difference is
2. **Sample size** ($n$) — more data increases power
3. **Significance level** ($\alpha$) — stricter $\alpha$ reduces power
4. **Variability** — more variable data reduces power

### AA.2 Power for Variance Ratio Tests {-}

For testing $H_0: \sigma_1^2 = \sigma_2^2$ vs. $H_1: \sigma_C^2 / \sigma_R^2 = R_\sigma$ using the F-test under normality:

Under $H_1$, the observed variance ratio follows a **scaled central F-distribution**:

$$\frac{s_C^2}{s_R^2} \sim R_\sigma \cdot F(n_C - 1, n_R - 1)$$

where $F(n_C - 1, n_R - 1)$ denotes a central F-distributed random variable. Let $F_q$ denote the $q$-th quantile of $F(n_C - 1, n_R - 1)$. The two-sided power is the probability that the observed ratio falls in the rejection region:

$$\text{Power}(R_\sigma) = P\left(R_\sigma \cdot F > F_{1-\alpha/2}\right) + P\left(R_\sigma \cdot F < F_{\alpha/2}\right) = P\left(F > \frac{F_{1-\alpha/2}}{R_\sigma}\right) + P\left(F < \frac{F_{\alpha/2}}{R_\sigma}\right)$$

### AA.3 Power Estimates for Our Study {-}

With $n_R = 22$, $n_C = 17$, $\alpha = 0.05$ (two-sided), approximate power:

| True variance ratio $R_\sigma$ | Power (two-sided) |
|---|---|
| 1.0 | 5% (= $\alpha$, by definition) |
| 2.0 | ~30% |
| 3.0 | ~55% |
| 5.0 | ~80% |
| 10.0 | ~97% |
| 20.0 | ~99.9% |

This explains our results: only metrics with $R_\sigma \geq 3.8$ reached Levene's significance. With our sample sizes, we simply lack power to detect variance ratios below ~$5\times$ reliably.

### AA.4 Sample Size Calculations {-}

To achieve 80% power for detecting $R_\sigma = 2.0$ at $\alpha = 0.05$ (two-sided), using the log-ratio asymptotic approximation $\text{Var}(\ln(s_C^2/s_R^2)) \approx 2/(n_C-1) + 2/(n_R-1) \approx 4/(n-1)$ for equal group sizes:

$$n \approx \frac{4(z_{\alpha/2} + z_\beta)^2}{(\ln R_\sigma)^2} + 1$$

For $R_\sigma = 2.0$: $n \approx \frac{4(1.96 + 0.84)^2}{(\ln 2)^2} + 1 \approx \frac{4 \times 7.84}{0.48} + 1 \approx 66$ per group.

This confirms the Discussion's recommendation to increase sample sizes substantially for future studies.

---

## Appendix AB: Numerical Stability and Floating-Point Arithmetic {-}

### AB.1 IEEE 754 Floating-Point Representation {-}

Computers represent real numbers using **floating-point** format. A 64-bit double-precision number stores:

- 1 bit for sign
- 11 bits for exponent
- 52 bits for mantissa (significand)

The represented value is: $(-1)^s \times 1.\text{mantissa} \times 2^{\text{exponent} - 1023}$

**Machine epsilon** is the smallest $\epsilon$ such that $1 + \epsilon \neq 1$ in floating-point arithmetic:

$$\epsilon_{\text{mach}} = 2^{-52} \approx 2.22 \times 10^{-16}$$

### AB.2 Why arccos Needs Clamping {-}

The arccos function is defined only for inputs in $[-1, 1]$. When computing:

$$\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \cdot \|\mathbf{v}\|}$$

floating-point rounding can produce values like $1.0000000000000002$ or $-1.0000000000000004$. Calling `arccos` on these values produces `NaN` (domain error).

**Solution:** clamp the argument: $\theta = \arccos(\max(-1, \min(1, \cos\theta)))$.

### AB.3 Catastrophic Cancellation in Variance {-}

The computational formula for variance:

$$\sigma^2 = E[X^2] - (E[X])^2$$

can suffer from **catastrophic cancellation** when $E[X]$ is large relative to $\sigma$. Example in single-precision floating point (7 significant digits):

$$X = [123456.0, 123457.0, 123458.0]$$

In exact arithmetic: $E[X] = 123457.0$, $E[X^2] = 15241628849.667$, so $\sigma^2 = 15241628849.667 - 15241628849.0 = 0.667$.

In single-precision: $E[X^2]$ rounds to $1.524163 \times 10^{10}$ and $(E[X])^2$ also rounds to $1.524163 \times 10^{10}$. The subtraction produces $0.0$ — complete loss of the variance information. The two-pass algorithm (compute $\bar{x}$ first, then $\sum(x_i - \bar{x})^2$) avoids this by working with small residuals. NumPy's `var` function uses this stable algorithm.

### AB.4 NaN Propagation in the Pipeline {-}

Our pipeline uses `NaN` (Not a Number) as a sentinel for missing or unreliable data. IEEE 754 defines:

- Any arithmetic operation with NaN produces NaN
- NaN $\neq$ NaN (this is the only value not equal to itself)
- NumPy functions like `nanmean`, `nanstd` ignore NaN values

This propagation ensures that unreliable measurements automatically exclude themselves from downstream computations without explicit bookkeeping.

---

## Appendix AC: Coordinate Systems and Camera Geometry {-}

### AC.1 Normalized Image Coordinates {-}

MediaPipe returns landmarks in **normalized image coordinates**:

$$x_{\text{norm}} \in [0, 1], \quad y_{\text{norm}} \in [0, 1]$$

where $(0, 0)$ is the top-left corner and $(1, 1)$ is the bottom-right. The pixel coordinates are:

$$x_{\text{px}} = x_{\text{norm}} \times W, \quad y_{\text{px}} = y_{\text{norm}} \times H$$

where $W$ and $H$ are image width and height in pixels.

**The y-axis points downward** in image coordinates. This means:
- Shoulder landmarks have *smaller* $y$ values than hip landmarks (shoulders are higher in the image but have lower $y$ coordinates)
- "Above" in the real world corresponds to "smaller $y$" in image coordinates

### AC.2 Left-Right Convention {-}

MediaPipe uses the subject's perspective: "left hip" is the subject's left hip, which appears on the *right* side of the image when the subject faces the camera. This is consistent with clinical convention but opposite to the viewer's perspective.

### AC.3 Sagittal vs. Frontal Plane Projection {-}

Our camera setup captures primarily frontal-plane motion (subject walks toward/away from camera). However, joint angles are computed from 2D projections of 3D anatomy:

**Sagittal-plane angles** (hip flexion, knee flexion, ankle dorsiflexion): These are best captured by a side-view camera. In our frontal camera setup, sagittal-plane motion projects onto the image plane with foreshortening that depends on the angle between the camera viewing direction and the sagittal plane. For a perfectly frontal view, sagittal-plane flexion would be invisible (zero projected angle). The exact projection relationship depends on limb segment lengths, camera distance, and lens model — no simple closed-form applies in general.

In practice, runway walks involve the model walking *toward* the camera at a slight angle, and MediaPipe's 3D landmark estimation partially compensates for projection effects. However, our angular measurements should be interpreted as **apparent 2D angles**, not true anatomical 3D angles.

**Frontal-plane angles** (pelvis obliquity, trunk lean): These are well-captured by our frontal camera setup, as the relevant motion occurs in the plane facing the camera.

### AC.4 Angle Wrapping {-}

The $\text{atan2}$ function returns values in $(-\pi, \pi]$ (or equivalently $(-180°, 180°]$). When computing angle differences or tracking angles over time, wrapping discontinuities can occur at $\pm 180°$. Our CRP computation explicitly handles this with the modular wrapping formula (Appendix O.5).

For joint angles computed via the dot product ($\arccos$), the result is always in $[0°, 180°]$, so wrapping is not an issue.

---

## Appendix AD: Complete Metric Inventory {-}

The following table lists all 24 metrics used in the variance analysis, with their domains, formulas, units, directionality, missing-data rules, and inclusion in each analysis.

| # | Metric | Domain | Formula | Units | Better$\to$ | Missing Rule | MQS | Var | PCA |
|---|--------|--------|---------|-------|-------------|-------------|-----|-----|-----|
| 1 | hip\_ROM | Kinematics | $\max\theta - \min\theta$ | degrees | $\to$ ref | NaN if <50% | Y | Y | Y |
| 2 | knee\_ROM | Kinematics | $\max\theta - \min\theta$ | degrees | $\to$ ref | NaN if <50% | Y | Y | Y |
| 3 | ankle\_ROM | Kinematics | $\max\theta - \min\theta$ | degrees | $\to$ ref | NaN if <50% | Y | Y | Y |
| 4 | pelvis\_obliquity | Kinematics | $|\text{atan2}(\Delta y, \Delta x)|$ | degrees | $\to 0$ | NaN if <50% | Y | Y | Y |
| 5 | trunk\_lean | Kinematics | $\text{atan2}(\Delta x, \Delta y)$ | degrees | $\to 0$ | NaN if <50% | Y | Y | Y |
| 6 | hip\_SPARC | Smoothness | $-\text{arc length}(\hat{V}_{\text{norm}})$ | Hz-dependent$^{\dagger}$ | $\to 0$ | NaN if FFT fails | Y | Y | Y |
| 7 | knee\_SPARC | Smoothness | $-\text{arc length}(\hat{V}_{\text{norm}})$ | Hz-dependent$^{\dagger}$ | $\to 0$ | NaN if FFT fails | Y | Y | Y |
| 8 | hip\_NJ | Smoothness | $\sqrt{T^5/(2A^2) \int \dddot{\theta}^2}$ | unitless | lower better | NaN if <10 frames | N | Y | Y |
| 9 | knee\_NJ | Smoothness | $\sqrt{T^5/(2A^2) \int \dddot{\theta}^2}$ | unitless | lower better | NaN if <10 frames | N | Y | Y |
| 10 | hip\_SI | Symmetry | $2|X_R - X_L|/(X_R + X_L)$ | % | $\to 0$ | NaN if either side missing | Y | Y | Y |
| 11 | knee\_SI | Symmetry | $2|X_R - X_L|/(X_R + X_L)$ | % | $\to 0$ | NaN if either side missing | Y | Y | Y |
| 12 | ankle\_SI | Symmetry | $2|X_R - X_L|/(X_R + X_L)$ | % | $\to 0$ | NaN if either side missing | Y | Y | Y |
| 13 | pelvis\_SI | Symmetry | $2|X_R - X_L|/(X_R + X_L)$ | % | $\to 0$ | NaN if either side missing | Y | Y | Y |
| 14 | hip\_waveform\_sym | Symmetry | $|\text{NCC}(\theta_L, \theta_R)| \times 100$ | % | $\to 100$ | NaN if <30 frames | Y | Y | Y |
| 15 | CRP\_bilateral\_hip | Coordination | CSD of bilateral hip CRP | degrees | $\to 0$ | NaN if Hilbert fails | Y | Y | Y |
| 16 | CRP\_ipsilateral | Coordination | CSD of hip-knee CRP | degrees | $\to 0$ | NaN if Hilbert fails | Y | Y | Y |
| 17 | stride\_time\_CV | Variability | $\sigma/\mu \times 100$ | % | $\to 0$ | NaN if <3 strides | Y | Y | Y |
| 18 | kinematic\_CV | Variability | mean per-joint ROM CV | % | $\to 0$ | NaN if <3 strides | Y | Y | Y |
| 19 | cadence | Temporal | $60/\bar{\Delta t} \times 2$ | steps/min | $\to$ ref | NaN if <2 strides | Y | Y | Y |
| 20 | stride\_time | Temporal | $\bar{\Delta t}_{\text{stride}}$ | seconds | $\to$ ref | NaN if <2 strides | Y | Y | Y |
| 21 | double\_support\_pct | Temporal | $\max(0, 2S-1) \times 100$ | % | $\to$ ref | NaN if no toe-off | N | Y | Y |
| 22 | arm\_swing\_ratio | Upper Body | $\text{ROM}_{\text{arm}}/25°$ | unitless | $\to 1$ | NaN if arms missing | N | Y | Y |
| 23 | GDI | Composite | $100 - \bar{d}/5° \times 10$ | score | $\to 100$ | NaN if <3 strides | N | Y | N |
| 24 | MQS | Composite | $\sum w_d S_d$ | 0--100 | $\to 100$ | NaN if <50% complete | N | Y | N |

**Column key:**
- "Better$\to$": direction of clinically better values ($\to$ ref = closer to reference range, $\to 0$ = closer to zero, $\to 100$ = closer to 100)
- "MQS": included in MQS composite score
- "Var": included in univariate variance analysis
- "PCA": included in multivariate PCA/LDA (composites excluded to avoid circularity)

$^{\dagger}$SPARC arc length sums terms $\sqrt{(\Delta f)^2 + (\Delta\hat{V})^2}$ where $\Delta f$ has units of Hz and $\Delta\hat{V}$ is dimensionless, making the result dependent on the frequency resolution. SPARC values are comparable only when computed with the same sampling rate and FFT parameters.

**Domain grouping for results presentation** uses 8 categories: Kinematics (5 metrics), Smoothness (4), Symmetry (5), Coordination (2), Variability (2), Temporal (3), Upper Body (1), Composite (2). The MQS composite uses 6 weighted domains that partially overlap with these categories.

**Note on metric set.** This table presents the canonical metric specification as designed. The computational pipeline's actual 24-metric set includes several metrics not listed here (e.g., `arm_swing_SI`, `shoulder_ROM`, `elbow_ROM`) and omits some listed here (e.g., `pelvis_SI` may be computed differently). Additionally, the Results domain summary shows different per-domain counts (Kinematics=7, Smoothness=2) than this table (Kinematics=5, Smoothness=4), reflecting that the pipeline groups metrics by different criteria. The total metric count (24) and the core mathematical definitions (SI, SPARC, NJ, CRP, etc.) are consistent; the specific metric set and domain assignments evolved between specification and implementation.

---

## Appendix AE: Statistical Test Directionality and Effect Size Conventions {-}

### AE.1 One-Sided vs. Two-Sided Tests {-}

**Levene's test** is inherently two-sided: it tests $H_0: \sigma_1^2 = \sigma_2^2$ against $H_1: \sigma_1^2 \neq \sigma_2^2$. The test statistic $W$ is always non-negative and compared to the upper tail of the $F$-distribution.

**Permutation test for variance ratio**: Our implementation is one-sided, testing $H_1: \sigma_C^2 > \sigma_R^2$ (control variance exceeds runway variance). We count only permutations where $R^* \geq R_{\text{obs}}$. For a two-sided test, one would use $\max(R^*, 1/R^*)$ as the test statistic.

**Mann-Whitney U test**: As implemented via SciPy's `mannwhitneyu`, this is a two-sided test of stochastic dominance: $H_0$: the two distributions are identical vs. $H_1$: one stochastically dominates the other.

### AE.2 Cohen's d Variants {-}

Multiple definitions of the pooled standard deviation exist:

**Weighted pooled SD** (Cohen's original, assumes equal population variances):

$$s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$$

**Unweighted average SD** (what we use):

$$s_{\text{avg}} = \sqrt{\frac{s_1^2 + s_2^2}{2}}$$

With $n_1 = 22$ and $n_2 = 17$, the difference is small but nonzero. We use the unweighted version because the core hypothesis is that the two groups have *different* variances, making the equal-variance assumption of the weighted version suspect. The unweighted version treats both groups' variability as equally informative.

### AE.3 Rank-Biserial Correlation Sign {-}

The rank-biserial correlation:

$$r_{rb} = 1 - \frac{2U}{n_1 n_2}$$

The sign depends on which group is designated as group 1 in the $U$ statistic. With group 1 = runway: positive $r_{rb}$ means runway values tend to be **smaller** (lower ranks, smaller $U_1$); negative means runway values tend to be **larger** (higher ranks, larger $U_1$).

SciPy's `mannwhitneyu(runway, control)` returns $U_1$, and $r_{rb} = 1 - 2U_1/(n_1 n_2)$. Verify the sign matches the intended interpretation for each metric.

---

## Appendix AF: PCA and LDA in the Small-Sample Regime {-}

### AF.1 Rank Deficiency {-}

A covariance matrix $\mathbf{C} \in \mathbb{R}^{p \times p}$ estimated from $n$ observations has rank at most $\min(n-1, p)$. In our study:

- Runway group: $n_R = 6$ complete cases, $p = 21$ features $\Rightarrow \text{rank}(\mathbf{C}_R) \leq 5$
- Control group: $n_C = 14$ complete cases, $p = 21$ features $\Rightarrow \text{rank}(\mathbf{C}_C) \leq 13$

The runway covariance matrix has at most 5 nonzero eigenvalues out of 21, meaning 16 dimensions have zero estimated variance. The distribution appears artificially concentrated in a 5-dimensional subspace.

### AF.2 Determinant vs. Volume {-}

The $k$-dimensional confidence ellipsoid of a distribution with covariance $\mathbf{C}$ has volume proportional to:

$$V \propto \sqrt{\det(\mathbf{C})} = \sqrt{\prod_{i=1}^{k} \lambda_i}$$

Therefore, the **volume ratio** between two distributions is:

$$\frac{V_C}{V_R} = \sqrt{\frac{\det(\mathbf{C}_C)}{\det(\mathbf{C}_R)}}$$

Our reported determinant ratio of $25.3\times$ corresponds to a volume ratio of $\sqrt{25.3} \approx 5.0\times$. We report both: the determinant ratio as generalized variance, and note the volume interpretation.

### AF.3 Complete-Case Deletion {-}

Our PCA/LDA analysis uses **listwise deletion**: any sample with any missing metric is excluded entirely. From 39 total samples, only 20 have complete data across all 21 features. This introduces potential bias if missingness correlates with group membership or metric values.

The complete-case group sizes ($n_R = 6$, $n_C = 14$) are substantially smaller than the full-data sizes ($n_R = 22$, $n_C = 17$), which:
- Reduces power for all multivariate tests
- Makes the runway covariance matrix severely rank-deficient
- Potentially over-represents "easy" videos (high pose detection quality)

### AF.4 Cross-Validation Leakage {-}

In our LOO-CV implementation, the feature standardization (centering and scaling) is applied to the entire dataset *before* the LOO splits. This means each test fold's standardization parameters are influenced by the test observation itself — a form of **data leakage**.

For honest LOO-CV:
1. Remove observation $i$
2. Compute mean and standard deviation from the remaining $n-1$ observations
3. Standardize all $n-1$ training observations and the held-out observation $i$ using these statistics
4. Fit LDA on the standardized training set
5. Predict the held-out observation

The leakage from our implementation is typically small (each observation contributes $1/n$ to the scaling parameters), but it systematically inflates accuracy estimates, especially with small $n$.

### AF.5 Resubstitution Accuracy in High Dimensions {-}

When $p \geq n$ (features exceed samples), a linear classifier can almost always achieve 100% resubstitution accuracy — in general position, $n$ points in $\mathbb{R}^p$ with $p \geq n$ are linearly separable regardless of their labels (provided no two points from different classes coincide). This is not evidence of meaningful class structure — it is a geometric near-inevitability.

With $p = 21$ features and $n = 20$ complete-case samples, we are in the $p \approx n$ regime. The 100% resubstitution accuracy should be interpreted as "the data is linearly separable" (expected in high dimensions), not as "the classifier works well."

The LOO-CV accuracy of 70% (equal to the majority-class baseline) is the more honest estimate and correctly indicates that the LDA classifier does not generalize.

---

## Appendix AG: Signal Processing Boundary Conditions {-}

### AG.1 Median Filter Edge Handling {-}

At signal boundaries, the median filter has insufficient context. For a window of half-width $w$ at position $t$:

- At the left edge ($t < w$): the window is truncated to $[0, t+w]$
- At the right edge ($t > N-1-w$): the window is truncated to $[t-w, N-1]$

SciPy's `medfilt` pads with zeros by default, which can introduce edge artifacts near signal boundaries. Our implementation uses `mode='reflect'` or operates only on interior points of contiguous non-NaN segments.

### AG.2 PCHIP Endpoint Derivatives {-}

PCHIP interpolation requires derivative estimates at each data point. At interior points, the Fritsch-Carlson algorithm uses a harmonic mean of neighboring slopes:

$$d_k = \begin{cases} 0 & \text{if } \delta_{k-1} \text{ and } \delta_k \text{ have different signs} \\ \dfrac{2}{\dfrac{1}{\delta_{k-1}} + \dfrac{1}{\delta_k}} & \text{otherwise} \end{cases}$$

where $\delta_k = (\theta_{k+1} - \theta_k) / h_k$ is the slope of segment $k$. This is the simple harmonic mean form for uniform spacing. Appendix K.5 presents the original Fritsch-Carlson (1980) weighted formula, which is algebraically different. Both enforce monotonicity through the same zero-derivative condition when slopes differ in sign. For non-uniform spacing, SciPy's `PchipInterpolator` uses a spacing-weighted variant of the harmonic mean; the details are implementation-specific and readers should consult the SciPy source for the exact formulation.

At **endpoints**, only one neighboring slope exists. SciPy's `PchipInterpolator` uses the Fritsch-Carlson one-sided formula:

$$d_0 = \frac{(2h_0 + h_1)\delta_0 - h_0\delta_1}{h_0 + h_1}$$

with a subsequent check to ensure monotonicity is preserved.

### AG.3 Butterworth Filter Design in Discrete Time {-}

The Butterworth filter is designed in the continuous-time (analog) domain and then converted to discrete time. SciPy's `butter(n, Wn)` function:

1. Designs an analog $n$-th order Butterworth prototype with cutoff at 1 rad/s
2. Applies the bilinear transform to convert to a digital filter:

$$s = \frac{2}{T}\frac{1 - z^{-1}}{1 + z^{-1}}$$

3. Pre-warps the cutoff frequency to compensate for the bilinear transform's frequency compression:

$$\omega_a = \frac{2}{T}\tan\left(\frac{\omega_d T}{2}\right)$$

where $\omega_d$ is the desired digital cutoff frequency and $\omega_a$ is the corresponding analog frequency.

The parameter `Wn` is the normalized cutoff frequency: $W_n = f_c / (f_s/2)$, where $f_c$ is the desired cutoff in Hz.

### AG.4 `filtfilt` Padding {-}

SciPy's `filtfilt` pads the signal at both ends to minimize startup transients. The default padding method (`padtype='odd'`) reflects the signal about the endpoint:

$$x_{\text{pad}}[-k] = 2x[0] - x[k], \quad k = 1, \ldots, 3n_{\text{filt}}$$

where $n_{\text{filt}}$ is the filter order. This ensures the padded signal has the same derivative at the boundary, reducing edge artifacts. The padded regions are removed after filtering.

### AG.5 Finite Difference Edge Behavior {-}

NumPy's `gradient` function uses:
- **Central differences** at interior points: $f'(x_i) = \frac{f(x_{i+1}) - f(x_{i-1})}{2h}$
- **Forward/backward differences** at edges: $f'(x_0) = \frac{f(x_1) - f(x_0)}{h}$

For our third derivative (jerk) computation via three cascaded `gradient` calls, the edge effects compound: the first call loses accuracy at 2 edge points, the second at 4, and the third at 6. With a 30 fps signal of 30 seconds (900 frames), this affects $<1\%$ of the data.

### AG.6 NaN Segment Handling {-}

When a joint angle time series contains NaN gaps (from occlusions or outlier rejection), filters cannot operate across gaps. Our pipeline:

1. **Identifies contiguous non-NaN segments** using `numpy.isnan` and run-length encoding
2. **Applies each processing stage independently** to each segment
3. **Segments shorter than the filter kernel** (e.g., <5 frames for the median filter, <6 frames for the Butterworth) are left unfiltered
4. **After PCHIP interpolation**, gaps are filled and subsequent stages operate on the complete signal

This segment-aware approach prevents filter artifacts from propagating across data gaps.

---

## Appendix AH: Gait Event Detection — Derivation and Validation {-}

### AH.1 Hip Flexion Peak Logic {-}

In normal gait, maximum hip flexion occurs at **initial contact** (heel strike). This is because the hip is maximally flexed as the swing leg reaches forward to contact the ground.

We detect heel strikes as peaks in the filtered hip flexion signal using SciPy's `find_peaks` with:

$$\text{prominence} \geq \max(0.15 \times \text{ROM}_{\text{hip}}, 1°)$$

The prominence threshold adapts to the signal amplitude:
- For large ROM (e.g., 40°): threshold = 6° — rejects minor fluctuations
- For small ROM (e.g., 5°): threshold = 1° — minimum absolute threshold to prevent silence

### AH.2 Heel-Y Refinement {-}

When heel landmark $y$-coordinates are available with sufficient confidence, heel strike timing is refined. The maximum pixel $y$-coordinate (lowest point in the image, accounting for the inverted y-axis) corresponds to the instant the heel contacts the ground.

Within $\pm 0.15$ seconds of each hip flexion peak:

$$t_{\text{HS}}^* = \arg\max_{|t - t_{\text{peak}}| \leq 0.15 f_s} y_{\text{heel}}(t)$$

The 0.15-second window (±4.5 frames at 30 fps) is chosen to span the typical timing offset between peak hip flexion and actual ground contact.

### AH.3 Toe-Off Estimation {-}

Toe-off timing (transition from stance to swing) is not directly detected from landmark motion. Instead, we estimate it from the stance phase fraction:

$$t_{\text{TO}} = t_{\text{HS}} + S \cdot \Delta t_{\text{stride}}$$

where $S \approx 0.6$ is the typical stance fraction at comfortable walking speed (Perry & Burnfield, 2010). This is an approximation; true toe-off detection would require foot contact sensors or more sophisticated kinematic analysis.

### AH.4 False Positive and False Negative Modes {-}

**False positives** (spurious detections):
- Arm swing or trunk oscillation coupling into the hip signal
- Noise spikes surviving the Butterworth filter
- Subject turning or stopping mid-walk

Mitigation: the prominence threshold and minimum inter-peak distance ($0.3 \times f_s$ frames) suppress most spurious detections.

**False negatives** (missed events):
- Low-amplitude gait (very small hip flexion ROM)
- Heavily occluded hip landmarks
- Non-standard gait patterns (shuffling, toe-walking)

### AH.5 Uncertainty Propagation {-}

Gait event timing uncertainty propagates to downstream metrics:

**Cadence**: If heel strike timing has uncertainty $\delta t$, cadence uncertainty is:

$$\delta\text{cadence} \approx \text{cadence}^2 \cdot \frac{\delta t}{60}$$

At 120 steps/min with $\delta t = 1/30$ s (one frame): $\delta\text{cadence} \approx 8$ steps/min.

**Stride time CV**: Both the mean and standard deviation of stride intervals are affected. With $\delta t = 1$ frame, the minimum detectable stride time difference is $2/f_s = 0.067$ s, placing a floor on CV measurements.

**Double support**: Depends on toe-off estimation accuracy ($S \approx 0.6$ assumption), which introduces systematic bias of unknown magnitude.

---

## Appendix AI: Simplified GDI vs. Clinical GDI {-}

### AI.1 The Clinical GDI (Schwartz & Rozumalski, 2008) {-}

The original GDI uses PCA on a reference database of healthy gait:

1. Collect 9 kinematic variables per side (hip flex/ext, hip ab/ad, hip rotation, knee flex/ext, ankle dorsi/plantar, pelvis tilt, obliquity, rotation, and foot progression) from a large normative database ($n > 100$ subjects). Left and right sides are scored separately.
2. Stride-normalize each waveform to 51 points (0--100% gait cycle at 2% intervals)
3. Concatenate all waveforms into a feature vector (for one side: $9 \times 51 = 459$ elements)
4. Perform PCA on the normative database, retaining components explaining 95% of variance
5. For a test subject, project their feature vector onto the PC space
6. Compute the Euclidean distance from the normative mean in PC space
7. Scale: $\text{GDI} = 100 - 10 \times d / \sigma_{\text{norm}}$, where $\sigma_{\text{norm}}$ is the SD of distances in the normative sample

**Note:** This is a simplified summary for exposition. The original Schwartz & Rozumalski (2008) definition involves additional details regarding the number of retained PCs, the distance metric in truncated PC space, and the specific normalization procedure. Readers should consult the original paper for the complete specification.

### AI.2 Our Simplified GDI {-}

Our version differs from the clinical GDI in several ways:

| Aspect | Clinical GDI | Our GDI |
|--------|-------------|---------|
| Reference | Large normative database ($n > 100$) | Synthetic reference waveforms |
| Features | 9 bilateral kinematic waveforms | Available joints (hip, knee, ankle) |
| Method | PCA-based distance | Direct RMS distance |
| Scaling | $\sigma_{\text{norm}}$ from database | Fixed $5°$ constant |
| Bilateral | Separate L/R GDI scores | Averaged L/R |
| Clipping | No floor | Clipped at 0 |

### AI.3 Calibration of the $5°$ Constant {-}

Our scoring formula $\text{GDI} = 100 - (\bar{d}/5°) \times 10$ maps each $5°$ of RMS deviation to a 10-point GDI reduction. The $5°$ value is chosen to approximate the within-subject variability of healthy gait:

- Kadaba et al. (1989): within-session variability of hip flexion RMS $\approx 3\text{--}5°$
- Schwartz et al. (2004): inter-session variability of knee flexion RMS $\approx 4\text{--}6°$

At $\bar{d} = 5°$: $\text{GDI} = 90$ (mild deviation, consistent with normal variability)
At $\bar{d} = 15°$: $\text{GDI} = 70$ (moderate deviation, clinically meaningful)
At $\bar{d} = 50°$: $\text{GDI} = 0$ (clipped at 0, severe deviation)

### AI.4 Bilateral Averaging {-}

The clinical GDI computes separate left and right scores. Our implementation averages:

$$\text{GDI} = \frac{\text{GDI}_L + \text{GDI}_R}{2}$$

This simplification loses information about unilateral pathology but is appropriate for our variance comparison, where we are interested in overall gait quality rather than side-specific deviations.

---

## Appendix AJ: Bootstrap and Permutation Implementation Details {-}

### AJ.1 Random Number Generation {-}

All resampling procedures use NumPy's `default_rng` with a fixed seed for reproducibility:

```
rng = numpy.random.default_rng(seed=42)
```

This ensures identical results across runs. The seed is arbitrary; changing it would produce slightly different p-values and confidence intervals, but the substantive conclusions should not change for well-powered comparisons.

### AJ.2 Degenerate Bootstrap Samples {-}

When resampling with replacement from a small group ($n = 17$ or $n = 22$), some bootstrap samples may have zero variance (all identical values drawn). The probability of this is negligible for continuous data but can occur with discrete or heavily tied data.

When $s^{*2}_R = 0$, the variance ratio $R^* = s^{*2}_C / s^{*2}_R$ is undefined ($\infty$). Our implementation:
- Skips degenerate resamples (does not include them in the count)
- This is equivalent to conditioning on non-degenerate samples

The number of skipped resamples is not logged in the current implementation; for full transparency, it should be reported.

### AJ.3 Percentile Interval Limitations {-}

The percentile bootstrap CI $[\hat{R}^*_{(0.025)}, \hat{R}^*_{(0.975)}]$ has known limitations:

1. **Bias**: If the bootstrap distribution is asymmetric (as variance ratios often are — they are bounded below by 0 and unbounded above), the percentile interval can be biased
2. **Coverage**: The nominal 95% coverage may be below 95% for skewed distributions
3. **Alternatives**: The BCa (bias-corrected and accelerated) interval or the basic bootstrap interval may have better coverage properties

For variance ratios, the distribution is typically right-skewed (long right tail), so the upper confidence limit may be underestimated by the percentile method.

### AJ.4 Bootstrap Proportion vs. Bayesian Posterior {-}

The quantity $\hat{P}(R > 1) = \frac{1}{B}\sum \mathbb{1}[R_b^* > 1]$ is a **frequentist bootstrap proportion**, not a Bayesian posterior probability. The distinction:

- **Bootstrap proportion**: fraction of resampled datasets where the variance ratio exceeds 1. This estimates the probability that a new dataset (drawn from the same population) would show $R > 1$.
- **Bayesian posterior**: $P(\sigma_C^2 > \sigma_R^2 \mid \text{data})$ under some prior. Would require specifying a prior distribution on variance parameters.

The bootstrap proportion is a reasonable proxy for directional confidence but should not be interpreted as a posterior probability without Bayesian justification.

---

## Appendix AK: Notation Index {-}

To reduce ambiguity from overloaded symbols, we provide a complete notation index. Where a symbol has multiple meanings, the context (section reference) disambiguates.

### Latin Symbols {-}

| Symbol | Meaning | Section |
|--------|---------|---------|
| $A$ | Peak-to-peak amplitude $\max\theta - \min\theta$ | Normalized Jerk (M) |
| $B$ | Number of bootstrap/permutation iterations | Permutation (G), Bootstrap (H) |
| $\mathbf{C}$ | Covariance matrix | PCA (P), Matrix Algebra (V) |
| $d$ | Cohen's d effect size | Mann-Whitney (F), Effect Sizes (AE) |
| $d_k$ | PCHIP derivative at knot $k$ | Interpolation (K) |
| $F$ | F-distribution test statistic | Appendix U |
| $F(n)$ | DFA fluctuation function at scale $n$ | DFA (Q) |
| $f_c$ | Butterworth cutoff frequency | Signal Processing Pipeline |
| $f_c^{\text{adapt}}$ | SPARC adaptive frequency cutoff | SPARC (L) |
| $f_s$ | Sampling frequency (frame rate) | Throughout |
| $H$ | Hurst exponent | DFA (Q) |
| $H_{ij}$ | Hermite basis functions | Interpolation (K) |
| $h_k$ | Interval length $t_{k+1} - t_k$ | Interpolation (K) |
| $j$ | Imaginary unit $\sqrt{-1}$ | Complex Numbers (W) |
| $K$ | Number of frequency bins | SPARC (L) |
| $k$ | Frequency bin index; also DFA segment index | Fourier (I), DFA (Q) |
| $N$ | Total sample size $n_1 + n_2$; also signal length | Statistics, DFA |
| $n$ | Sample size; also filter order; also DFA scale | Context-dependent |
| $n_R, n_C$ | Runway and control group sizes | Throughout |
| $p$ | Number of features (dimensions) | PCA (P) |
| $R$ | Mean resultant length (circular statistics) | CRP (O) |
| $R_\sigma$ | Variance ratio $s_C^2 / s_R^2$ | Variance analysis |
| $R_{\text{obs}}, R_b^*$ | Observed and permuted/bootstrap variance ratios | Permutation (G), Bootstrap (H) |
| $R_1$ | Rank sum for group 1 | Mann-Whitney (F) |
| $r_{rb}$ | Rank-biserial correlation | Mann-Whitney (F) |
| $S$ | Stance phase fraction | Gait Events |
| $\mathbf{S}_B, \mathbf{S}_W$ | Between/within-class scatter matrices | LDA (P.5) |
| $s^2$ | Sample variance | Throughout |
| $T$ | Signal duration; also number of time points | Context-dependent |
| $U$ | Mann-Whitney U statistic | Mann-Whitney (F) |
| $W$ | Levene's test statistic; also image width | Levene (E); Camera (AC) |
| $\mathbf{w}$ | Projection vector | PCA (P), LDA (P.5) |

### Greek Symbols {-}

| Symbol | Meaning | Section |
|--------|---------|---------|
| $\alpha$ | Significance level; also DFA scaling exponent | Hypothesis Testing (D); DFA (Q) |
| $\beta$ | Type II error probability | Power Analysis (AA) |
| $\lambda$ | Eigenvalue; also interpolation penalty factor | PCA (P); MQS |
| $\mu$ | Population/sample mean | Statistics (C) |
| $\phi$ | Phase angle; also Euler's formula angle | CRP (O); Complex Numbers (W) |
| $\rho$ | Spread ratio (trace ratio) | PCA Results |
| $\sigma^2$ | Population variance | Statistics (C) |
| $\theta$ | Joint angle | Throughout |
| $\tau$ | Integration variable (Hilbert transform) | Hilbert (O) |
| $\Delta t$ | Sampling interval $1/f_s$ | Throughout |
| $\delta_k$ | Slope of PCHIP segment $k$: $(\theta_{k+1}-\theta_k)/h_k$ | Interpolation (K), Boundaries (AG) |
| $\omega$ | Angular frequency $2\pi f$ | Fourier (I), Filtering (J) |

### Operators and Functions {-}

| Symbol | Meaning |
|--------|---------|
| $\|\cdot\|_2$ | Euclidean (L2) norm |
| $\mathcal{H}\{\cdot\}$ | Hilbert transform |
| $\text{atan2}(y, x)$ | Four-quadrant arctangent |
| $\text{tr}(\cdot)$ | Matrix trace |
| $\det(\cdot)$ | Matrix determinant |
| $\mathbb{1}[\cdot]$ | Indicator function (1 if true, 0 if false) |
| P.V. | Cauchy principal value |
| $E[\cdot]$ | Expected value |
| $\text{Var}(\cdot)$ | Variance |
| $\text{Cov}(\cdot, \cdot)$ | Covariance |
| $P(\cdot)$ | Probability |
| $\text{Re}(\cdot), \text{Im}(\cdot)$ | Real and imaginary parts |
| $\arg(\cdot)$ | Complex argument (phase angle) |
| $\nabla$ | Gradient operator |
| $\bmod$ | Modulo operation |
| $\text{clip}(x, a, b)$ | Clamp $x$ to $[a, b]$ |
| $\text{FFT}\{\cdot\}$ | Fast Fourier Transform |
| $Z_{ij}$ | Absolute deviation from group center (Levene's test) |
| $X_{ij}$ | Observation $j$ in group $i$ |
| $\mathcal{D}$ | Set of MQS domains |
| $w_d, S_d$ | Domain weight and score in MQS |
| $c_f$ | MQS confidence factor |
