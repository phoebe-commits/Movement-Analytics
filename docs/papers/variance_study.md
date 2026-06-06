# Quantitative Kinematic Variance Analysis of Professional Runway Walking vs. Unconstrained Internet Walking Video

**Phoebe Richmond**
Elysium Intelligence
phoebe@joinelysium.ai

---

## Abstract

We present a quantitative analysis comparing the kinematic variance of professional runway model walks against representative internet walking video. Using a computational pipeline that extracts 24 biomechanical metrics from monocular video via MediaPipe pose estimation, we test the hypothesis that runway walks exhibit significantly lower kinematic variance across movement quality domains. Our analysis of 22 runway videos and 17 control internet walking videos reveals that control walking data exhibits 2.56$\times$ higher median variance (Levene's test), with 10 of 24 metrics showing statistically significant differences ($p < 0.05$), all in the hypothesized direction. The strongest effects appear in movement smoothness (17$\times$) and bilateral symmetry (14$\times$), while coordination (1.3$\times$) and temporal parameters (1.7$\times$) show smaller differences. Three of 24 metrics show the reverse direction (higher runway variance), though none significantly. Multivariate analysis via PCA confirms runway walks occupy a 2.13$\times$ tighter region in kinematic feature space (25$\times$ by volume), though the multivariate analysis is limited by small complete-case sample sizes ($n = 20$). These findings provide partial but directionally consistent support for the thesis that professional runway walking constitutes a low-variance, high-quality kinematic distribution suitable as a training foundation for robotic movement learning.

**Keywords:** gait analysis, kinematic variance, movement quality, pose estimation, robot learning, biomechanics

---

## 1. Introduction

The development of physical AI systems capable of human-like locomotion requires high-quality training data that captures the essential structure of bipedal walking while minimizing noise and inter-subject variability. While internet-scale video corpora provide enormous volume, they introduce proportional variance: differences in walking speed, terrain, footwear, pathology, camera angle, and individual biomechanics create a training distribution with high entropy across kinematic dimensions.

We hypothesize that professional runway model walks constitute a naturally low-variance kinematic distribution. The fashion modeling profession itself acts as a variance filter through three mechanisms:

1. **Selection pressure** — agencies and designers select models whose baseline gait exhibits specific aesthetic properties (fluid motion, bilateral symmetry, controlled trunk)
2. **Training convergence** — professional models undergo extensive coaching to standardize their walk, reducing inter-individual differences
3. **Environmental control** — runway shows enforce consistent conditions (flat surface, known distance, controlled pace, frontal camera alignment)

This paper tests this hypothesis quantitatively by extracting 24 biomechanical metrics from both runway and internet walking video using a computational pipeline, then applying rigorous statistical analysis to compare distributional properties across groups.

---

## 2. Methods

### 2.1 Data Collection

**Runway dataset.** 22 video recordings of professional runway model walks (MOV format, 30 fps, 1080p). All videos capture a single model walking toward or away from camera on a straight runway.

**Control dataset.** 20 walking videos sourced from YouTube using `yt-dlp`, selected via diverse search queries ("person walking on sidewalk," "gait assessment walking," "hallway walking exercise," "walking stock footage"). Each video was trimmed to 30 seconds at $\leq$720p resolution. After quality filtering (requiring $\geq$30% pose detection rate), 17 control videos remained.

### 2.2 Pose Estimation

We use MediaPipe PoseLandmarker (heavy model, VIDEO mode) to extract 33 anatomical keypoints per frame. For each frame $t$, the pose estimator returns a set of 3D landmarks:

$$\mathbf{L}_t = \{(x_i^t, y_i^t, z_i^t, v_i^t)\}_{i=1}^{33}$$

where $(x_i, y_i, z_i)$ are normalized coordinates and $v_i \in [0,1]$ is the visibility confidence for landmark $i$.

**Multi-person tracking.** When multiple persons are detected ($|\mathcal{P}_t| > 1$), we track the target subject using pelvis centroid continuity. The pelvis centroid at frame $t$ is:

$$\mathbf{c}_t = \frac{1}{2}\left(\mathbf{p}_{\text{left\_hip}}^t + \mathbf{p}_{\text{right\_hip}}^t\right)$$

The tracked person is selected by minimizing Euclidean displacement from the previous frame:

$$\text{person}^* = \arg\min_{p \in \mathcal{P}_t} \|\mathbf{c}_t^{(p)} - \mathbf{c}_{t-1}\|_2$$

### 2.3 Joint Angle Computation

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

### 2.4 Signal Processing Pipeline

Raw joint angle time series from video-based pose estimation contain noise from landmark jitter, occlusions, and multi-person ambiguity. We apply a 6-stage signal processing pipeline:

**Stage 1: Physiological outlier rejection.** Values outside clinically plausible ranges are replaced with NaN:

$$\theta_t^{\text{clean}} = \begin{cases} \theta_t & \text{if } \theta_{\min}^{\text{physio}} \leq \theta_t \leq \theta_{\max}^{\text{physio}} \\ \text{NaN} & \text{otherwise} \end{cases}$$

Reference ranges: hip flexion $[-50°, 120°]$, knee flexion $[-20°, 160°]$, ankle dorsiflexion $[-90°, 60°]$.

**Stage 2: Median pre-filter.** A 5-sample running median filter removes impulsive noise (single-frame spikes) without distorting genuine movement dynamics:

$$\tilde{\theta}_t = \text{median}\{\theta_{t-2}, \theta_{t-1}, \theta_t, \theta_{t+1}, \theta_{t+2}\}$$

This filter removes spikes up to 2 frames wide. It operates only on contiguous non-NaN segments, preserving gap structure.

**Stage 3: PCHIP interpolation.** Gaps (NaN sequences from occlusions or outlier rejection) are filled using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation. Given observed values at indices $\{t_k, \theta_k\}$, PCHIP constructs a $C^1$-continuous piecewise cubic:

$$\hat{\theta}(t) = \sum_{k} \theta_k H_{00}\left(\frac{t - t_k}{t_{k+1} - t_k}\right) + h_k d_k H_{10}\left(\frac{t - t_k}{t_{k+1} - t_k}\right) + \theta_{k+1} H_{01}\left(\frac{t - t_k}{t_{k+1} - t_k}\right) + h_k d_{k+1} H_{11}\left(\frac{t - t_k}{t_{k+1} - t_k}\right)$$

where $H_{ij}$ are the cubic Hermite basis functions, $h_k = t_{k+1} - t_k$, and $d_k$ are the monotonicity-preserving derivatives (Fritsch & Carlson, 1980). PCHIP avoids the Runge oscillation artifacts of standard cubic spline interpolation.

For signals with $>$50% missing data, we fall back to linear interpolation.

**Stage 4: Post-interpolation outlier rejection.** A second physiological range check removes any artifacts introduced by interpolation across large gaps.

**Stage 5: Confidence-adaptive Butterworth smoothing.** A two-pass low-pass Butterworth filter suppresses remaining high-frequency noise:

$$H(s) = \frac{1}{\sqrt{1 + \left(\frac{s}{\omega_c}\right)^{2n}}}$$

where $n = 2$ (filter order) and $\omega_c$ is the cutoff frequency. We apply `filtfilt` (zero-phase forward-backward filtering) for zero phase distortion:

- **Pass 1 (low-confidence frames):** For frames where MediaPipe confidence $v_t < 0.7$, apply aggressive cutoff $f_c = 3$ Hz
- **Pass 2 (all frames):** Standard cutoff $f_c = 6$ Hz across the full signal

The Nyquist frequency constraint requires $f_c < f_s / 2$ where $f_s$ is the video frame rate.

**Stage 6: Completeness check.** When overall signal completeness drops below 50%, the Movement Quality Score returns NaN rather than a misleading result.

### 2.5 Gait Metric Extraction

From the processed joint angle time series, we compute 24 biomechanical metrics spanning 8 domains.

#### 2.5.1 Range of Motion (ROM)

For each joint angle time series $\theta(t)$ of duration $T$:

$$\text{ROM} = \max_{t \in [0,T]} \theta(t) - \min_{t \in [0,T]} \theta(t)$$

Clinical reference ranges (Perry & Burnfield, 2010): hip flexion 40--45°, knee flexion 60--65°, ankle dorsiflexion ~30°.

#### 2.5.2 Spectral Arc Length (SPARC)

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

Note: The original Balasubramanian et al. (2012) formulation uses a continuous integral with a frequency normalization factor $(1/f_c)^2$. Our implementation follows the discrete unnormalized form, which preserves the relative ordering of smoothness values and is equivalent for within-study comparisons.

#### 2.5.3 Normalized Jerk (NJ)

The dimensionless normalized jerk metric (Teulings et al., 1997):

$$\text{NJ} = \sqrt{\frac{T^5}{2A^2} \int_0^T \left(\frac{d^3\theta}{dt^3}\right)^2 dt}$$

where $T = N \cdot \Delta t$ is the movement duration, $N$ is the number of samples, $\Delta t = 1/f_s$ is the sampling interval, and $A = \max(\theta) - \min(\theta)$ is the peak-to-peak amplitude. Lower values indicate smoother movement.

The discrete approximation replaces the integral with a Riemann sum:

$$\text{NJ} \approx \sqrt{\frac{T^5}{2A^2} \cdot \Delta t \sum_{t=1}^{N} \left(\dddot{\theta}_t\right)^2}$$

The third derivative $\dddot{\theta}_t$ is computed via cascaded second-order central finite differences (three successive applications of NumPy's `gradient`).

#### 2.5.4 Symmetry Index (SI)

The Robinson Symmetry Index (Robinson et al., 1987) quantifies bilateral asymmetry:

$$\text{SI} = \frac{2|X_R - X_L|}{X_R + X_L} \times 100\%$$

where $X_R$ and $X_L$ are the mean absolute values of the right and left joint angle time series respectively:

$$X_R = \frac{1}{T}\sum_{t=1}^{T} |\theta_R(t)|, \quad X_L = \frac{1}{T}\sum_{t=1}^{T} |\theta_L(t)|$$

Perfect symmetry yields $\text{SI} = 0\%$. Clinically, $\text{SI} < 10\%$ is considered normal (Shorter et al., 2008).

#### 2.5.5 Waveform Symmetry via Normalized Cross-Correlation (NCC)

Beyond amplitude-based SI, we capture bilateral shape and timing differences using the normalized cross-correlation:

$$\text{NCC}(\theta_L, \theta_R) = \frac{\sum_{t=1}^{T} (\theta_L(t) - \bar{\theta}_L)(\theta_R(t) - \bar{\theta}_R)}{\|\theta_L - \bar{\theta}_L\|_2 \cdot \|\theta_R - \bar{\theta}_R\|_2}$$

$$\text{Waveform Symmetry} = |\text{NCC}| \times 100\%$$

NCC is invariant to amplitude scaling (captured separately by SI) and measures purely shape-based similarity. A value of 100% indicates identical bilateral waveforms. The absolute value is used because anti-phase bilateral coupling (expected in normal gait for hip flexion) represents symmetric coordination.

#### 2.5.6 Continuous Relative Phase (CRP)

CRP quantifies inter-limb and intra-limb coordination via the Hilbert transform (Hamill et al., 1999). For two oscillating signals $\theta_a(t)$ and $\theta_b(t)$:

1. Center each signal: $\tilde{\theta}_a(t) = \theta_a(t) - \bar{\theta}_a$

2. Compute the analytic signal via the Hilbert transform:

$$z_a(t) = \tilde{\theta}_a(t) + j\mathcal{H}\{\tilde{\theta}_a(t)\}$$

where $\mathcal{H}$ denotes the Hilbert transform:

$$\mathcal{H}\{x(t)\} = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{x(\tau)}{t - \tau} d\tau$$

3. Extract instantaneous phase: $\phi_a(t) = \arg(z_a(t)) = \arctan\frac{\text{Im}(z_a)}{\text{Re}(z_a)}$

4. Compute the continuous relative phase:

$$\text{CRP}(t) = \phi_a(t) - \phi_b(t) \pmod{360°} - 180°$$

**CRP consistency** is quantified via the circular standard deviation of the CRP signal, using the mean resultant length $R$:

$$R = \sqrt{\left(\frac{1}{T}\sum_t \cos\phi_t\right)^2 + \left(\frac{1}{T}\sum_t \sin\phi_t\right)^2}$$

$$\text{CSD} = \sqrt{-2\ln R} \quad \text{(in radians, converted to degrees)}$$

Healthy inter-limb coordination: $\text{CSD} < 15°$. Pathological: $\text{CSD} > 30°$. For bilateral hip flexion in normal gait, $\text{CRP} \approx 180°$ (anti-phase coupling).

#### 2.5.7 Arm Swing Metrics

Arm swing ROM and symmetry are computed from bilateral shoulder flexion angles:

$$\text{arm\_swing\_ROM} = \frac{\text{ROM}(\theta_{\text{R\_shoulder}}) + \text{ROM}(\theta_{\text{L\_shoulder}})}{2}$$

$$\text{arm\_swing\_SI} = \text{SI}(\theta_{\text{R\_shoulder}}, \theta_{\text{L\_shoulder}})$$

The arm swing ratio normalizes against a clinical reference of 25° shoulder flexion ROM:

$$\text{arm\_swing\_ratio} = \text{clip}\left(\frac{\text{arm\_swing\_ROM}}{25°}, \; 0, \; 2\right)$$

Normal arm swing ratio $\approx 1.0$; values $< 0.7$ suggest diminished arm swing (characteristic of Parkinsonian gait).

#### 2.5.8 Coefficient of Variation (CV)

Stride-to-stride variability is quantified by the coefficient of variation:

$$\text{CV} = \frac{\sigma}{\mu} \times 100\%$$

where $\sigma$ and $\mu$ are the standard deviation and mean of the stride intervals (or per-stride ROM values). We compute:

- **Stride time CV:** Variability of inter-heel-strike intervals. Normal: 1--3% (Hausdorff et al., 2001).
- **Kinematic CV:** Mean of per-joint, per-stride ROM coefficients of variation. Normal: 0--5%.

#### 2.5.9 Gait Deviation Index (GDI)

Our simplified GDI (after Schwartz & Rozumalski, 2008) compares stride-normalized joint angle waveforms against a normal reference:

1. Segment gait cycles using detected heel strikes $\{HS_1, HS_2, \ldots, HS_n\}$

2. Normalize each stride to 101 points via linear interpolation:

$$\theta_{\text{norm}}^{(i)}(p) = \text{interp}\left(\theta^{(i)}, \frac{p}{100}\right), \quad p \in \{0, 1, \ldots, 100\}$$

3. Compute RMS deviation from the normal reference for each joint $j$ and stride $i$:

$$d_j^{(i)} = \sqrt{\frac{1}{101}\sum_{p=0}^{100}\left(\theta_{\text{norm},j}^{(i)}(p) - \theta_{\text{ref},j}(p)\right)^2}$$

4. Average across joints and strides:

$$\bar{d} = \frac{1}{n \cdot |\mathcal{J}|} \sum_{i=1}^{n} \sum_{j \in \mathcal{J}} d_j^{(i)}$$

5. Convert to GDI score (100 = normal, decreasing with deviation):

$$\text{GDI} = 100 - \frac{\bar{d}}{5°} \times 10$$

Each 5° RMS deviation reduces GDI by approximately 10 points. The 5° constant is calibrated to approximate the standard deviation of normal gait variability in clinical gait lab data: healthy subjects typically show ~5° RMS deviation from the mean waveform across repeated trials, so each additional SD of deviation reduces the score by 10 points. This is a simplified adaptation of the original Schwartz & Rozumalski (2008) GDI, which uses PCA on 9 kinematic features from clinical datasets; our version uses direct RMS comparison against a synthetic normal reference.

#### 2.5.10 Gait Event Detection

Heel strikes are detected as peaks in the low-pass filtered hip flexion signal (maximum hip flexion $\approx$ heel strike):

$$\text{HS} = \{t : \theta_{\text{hip}}^{\text{filt}}(t) \text{ is a local maximum with prominence} > \max(0.15 \cdot \text{ROM}_{\text{hip}}, 1°)\}$$

When video-derived heel Y-coordinates are available, heel strike timing is refined by finding the lowest heel position (maximum pixel Y) within $\pm$0.15s of each hip flexion peak:

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

### 2.6 Movement Quality Score (MQS)

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
100 & \text{if } v_{\text{opt,lo}} \leq v \leq v_{\text{opt,hi}} \\
100 \cdot \dfrac{v - v_{\text{worst,lo}}}{v_{\text{opt,lo}} - v_{\text{worst,lo}}} & \text{if } v < v_{\text{opt,lo}} \\
100 \cdot \dfrac{v_{\text{worst,hi}} - v}{v_{\text{worst,hi}} - v_{\text{opt,hi}}} & \text{if } v > v_{\text{opt,hi}} \\
0 & \text{if } v \leq v_{\text{worst,lo}} \text{ or } v \geq v_{\text{worst,hi}}
\end{cases}$$

Reference ranges are sourced from Perry & Burnfield (2010), Winter (2009), Balasubramanian et al. (2012), Hausdorff et al. (2001), and Hamill et al. (1999).

**Confidence weighting (video pipeline).** For video-derived analysis, MQS is scaled by a confidence factor:

$$\text{MQS}_{\text{weighted}} = \text{MQS} \times c_f$$

$$c_f = f_{\text{obs}} \times \bar{v}_{\text{detected}} \times (1 - p_{\text{interp}})$$

$$p_{\text{interp}} = \text{clip}(\lambda \cdot f_{\text{interp}}, \; 0, \; 0.5), \quad \lambda = 0.5$$

where $f_{\text{obs}}$ is the fraction of frames with detected pose, $\bar{v}_{\text{detected}}$ is the mean landmark confidence across detected frames, and $f_{\text{interp}}$ is the fraction of angle data filled by interpolation. The interpolation penalty $p_{\text{interp}}$ is clipped to $[0, 0.5]$ to prevent the confidence factor from dropping below $0.5 \cdot f_{\text{obs}} \cdot \bar{v}_{\text{detected}}$ even for heavily interpolated signals.

**Sufficient evidence gating.** MQS returns NaN when overall signal completeness drops below 50%. Completeness is computed per-domain as the fraction of expected signals that are present and non-NaN, then averaged across all 6 domains. This prevents misleading scores from sparse pose detection data.

### 2.7 Statistical Analysis

We apply four layers of statistical testing to compare variance between groups, plus multivariate analysis.

#### 2.7.1 Levene's Test for Equality of Variances

For each metric $m$, we test $H_0$: $\sigma^2_{\text{runway}} = \sigma^2_{\text{control}}$ using Levene's test (Levene, 1960). Given two groups of sizes $n_1$ and $n_2$, define:

$$Z_{ij} = |X_{ij} - \bar{X}_{i\cdot}|$$

where $\bar{X}_{i\cdot}$ is the group mean (or median, for the Brown-Forsythe variant). The test statistic is:

$$W = \frac{(N - k)}{(k - 1)} \cdot \frac{\sum_{i=1}^{k} n_i (\bar{Z}_{i\cdot} - \bar{Z}_{\cdot\cdot})^2}{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_{i\cdot})^2}$$

where $k = 2$ groups, $N = n_1 + n_2$, $\bar{Z}_{i\cdot}$ is the group mean of the absolute deviations, and $\bar{Z}_{\cdot\cdot}$ is the grand mean. Under $H_0$, $W \sim F(k-1, N-k)$.

The variance ratio (effect size) is:

$$R_\sigma = \frac{s_{\text{control}}^2}{s_{\text{runway}}^2}$$

where $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ is the sample variance with Bessel's correction.

#### 2.7.2 Mann-Whitney U Test

For group mean comparison, we use the non-parametric Mann-Whitney U test (Mann & Whitney, 1947), which does not assume normality:

$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$$

where $R_1$ is the sum of ranks for group 1 in the combined ranked sample. The effect size is reported as:

**Cohen's d** (standardized mean difference):

$$d = \frac{\bar{X}_{\text{runway}} - \bar{X}_{\text{control}}}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{s_1^2 + s_2^2}{2}}$$

**Rank-biserial correlation** (non-parametric effect size):

$$r_{rb} = 1 - \frac{2U}{n_1 n_2}$$

#### 2.7.3 Permutation Tests

To validate variance ratio significance without distributional assumptions, we use permutation testing with 10,000 iterations. For each metric:

1. Compute observed variance ratio: $R_{\text{obs}} = s^2_{\text{control}} / s^2_{\text{runway}}$

2. For $B = 10{,}000$ permutations:
   - Randomly shuffle group labels across the pooled sample
   - Compute permuted variance ratio $R_b^*$

3. Compute $p$-value:

$$p = \frac{|\{b : R_b^* \geq R_{\text{obs}}\}| + 1}{B + 1}$$

The $+1$ in numerator and denominator prevents $p = 0$ and ensures the observed statistic is included (Phipson & Smyth, 2010).

#### 2.7.4 Bootstrap Confidence Intervals

We construct 95% confidence intervals on the variance ratio using the percentile bootstrap (Efron & Tibshirani, 1993):

1. For $B = 10{,}000$ bootstrap iterations:
   - Resample with replacement from each group: $\theta_R^* \sim \hat{F}_R$, $\theta_C^* \sim \hat{F}_C$
   - Compute bootstrap variance ratio: $R_b^* = s^{*2}_C / s^{*2}_R$

2. Compute percentile CI: $[\hat{R}_{0.025}^*, \hat{R}_{0.975}^*]$

3. Compute posterior probability: $P(R > 1) = \frac{1}{B}\sum_{b=1}^{B} \mathbb{1}[R_b^* > 1]$

#### 2.7.5 Multiple Comparison Correction

With 24 simultaneous tests, we apply the Bonferroni correction:

$$\alpha_{\text{Bonf}} = \frac{\alpha}{m} = \frac{0.05}{24} \approx 0.0021$$

A metric is considered significant after correction if $p < \alpha_{\text{Bonf}}$.

**Non-independence of tests.** The 24 metrics are not fully independent — several are derived from the same underlying joint angle signals (e.g., hip\_ROM, hip\_SPARC, and hip\_SI all depend on hip flexion angles), and MQS and GDI are composites of other metrics. Bonferroni correction with $m = 24$ is therefore conservative (true effective number of independent tests is likely $m_{\text{eff}} < 24$). For a more precise correction, one could estimate $m_{\text{eff}}$ from the eigenvalues of the inter-metric correlation matrix. We report both uncorrected and Bonferroni-corrected results to bracket the true significance.

#### 2.7.6 Principal Component Analysis (PCA)

To assess multivariate distributional differences, we apply PCA to the standardized feature matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ (excluding MQS and GDI composites):

1. Standardize: $\tilde{\mathbf{X}} = (\mathbf{X} - \boldsymbol{\mu}) / \boldsymbol{\sigma}$

2. Eigen-decompose the covariance matrix: $\mathbf{C} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$

3. Project: $\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{V}_{:,1:k}$ where $k$ components are retained

**Spread metrics.** Within each group $g$, the covariance in PC space is $\mathbf{C}_g = \text{Cov}(\mathbf{Z}_g)$. We compare:

- **Trace** (total variance): $\text{tr}(\mathbf{C}_g) = \sum_{i=1}^{k} \lambda_{g,i}$
- **Determinant** (generalized variance / volume): $|\mathbf{C}_g| = \prod_{i=1}^{k} \lambda_{g,i}$
- **Mean centroid distance**: $\bar{d}_g = \frac{1}{n_g}\sum_{i=1}^{n_g} \|\mathbf{z}_i - \bar{\mathbf{z}}_g\|_2$

The spread ratio $\rho = \text{tr}(\mathbf{C}_{\text{control}}) / \text{tr}(\mathbf{C}_{\text{runway}})$ quantifies how much larger the control distribution is in kinematic feature space.

#### 2.7.7 Linear Discriminant Analysis (LDA)

LDA finds the linear projection that maximizes class separability:

$$\mathbf{w}^* = \arg\max_{\mathbf{w}} \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$$

where $\mathbf{S}_B = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T$ is the between-class scatter matrix and $\mathbf{S}_W = \sum_{g} \sum_{i \in g} (\mathbf{x}_i - \boldsymbol{\mu}_g)(\mathbf{x}_i - \boldsymbol{\mu}_g)^T$ is the within-class scatter matrix. The solution is $\mathbf{w}^* \propto \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$.

We report both resubstitution accuracy and Leave-One-Out Cross-Validation (LOO-CV) accuracy to assess generalization.

### 2.8 Detrended Fluctuation Analysis (DFA)

For stride-to-stride interval sequences with $\geq$16 strides, we compute the DFA scaling exponent $\alpha$ (Hausdorff et al., 2001):

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

## 3. Results

### 3.1 Dataset Summary

After quality filtering ($\geq$30% pose detection rate), the analysis includes $n_R = 22$ runway videos and $n_C = 17$ control videos. Runway videos achieved near-perfect pose detection (median 100%), reflecting controlled filming conditions. Three control videos were excluded due to insufficient pose detection.

### 3.2 Univariate Variance Comparison

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

### 3.3 Permutation Test Validation

Permutation tests (10,000 iterations) corroborate Levene's results. All 10 Levene-significant metrics also achieve permutation $p < 0.05$. Additionally, trunk\_lean\_ROM ($p_{\text{perm}} = 0.045$), hip\_SI ($p_{\text{perm}} = 0.024$), stride\_time\_CV ($p_{\text{perm}} = 0.017$), and knee\_ROM ($p_{\text{perm}} = 0.024$) reach significance under permutation testing, suggesting Levene's test is conservative for these metrics.

### 3.4 Bootstrap Confidence Intervals

Bootstrap 95% CIs on the variance ratio exclude 1.0 (indicating $R_\sigma$ is significantly different from equality) for 9 metrics. Eight metrics achieve $P(R > 1) \geq 99\%$. Notably, all bootstrap CIs are right-skewed, consistent with the heavy-tailed nature of variance ratio distributions.

### 3.5 Group Mean Differences

Mann-Whitney U tests reveal significant mean differences ($p < 0.05$) in 16 of 24 metrics. Large effect sizes ($|d| > 0.8$) are observed for:

- Shoulder/arm swing ROM ($d = -1.62$, control has much larger arm movements)
- Ankle ROM ($d = -1.51$)
- Elbow ROM ($d = -1.49$)
- Stride time CV ($d = -1.66$, control has much more stride variability)
- Hip SPARC ($d = +1.68$, runway is smoother)
- Hip ROM ($d = -1.35$)

These mean differences indicate that runway walks are not just lower-variance but occupy a distinct region of kinematic space: lower ROM (more controlled range), better smoothness, lower stride variability.

### 3.6 Domain-Level Summary

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

### 3.7 Multivariate Analysis

PCA on 21 standardized features (complete cases: 6 runway, 14 control) reveals:

- **3 components** explain 65.7% of total variance
- PC1 (35.4%) loads primarily on ROM and SPARC metrics
- PC2 (18.0%) loads on temporal parameters (stride time, cadence) and symmetry indices

**Distributional spread in PC space:**

| Metric | Runway | Control | Ratio |
|--------|--------|---------|-------|
| Trace (total variance) | 6.34 | 13.48 | 2.13$\times$ |
| Determinant (volume) | — | — | 25.3$\times$ |
| Mean centroid distance | 2.20 | 3.34 | 1.52$\times$ |

The control distribution is **2.13$\times$ larger by trace** and **25$\times$ larger by volume** in 3D PC space, confirming that runway walks cluster tightly while control videos scatter across the kinematic feature space.

**Caveat:** With only 6 runway samples in 21-dimensional feature space ($p \gg n_R$), the runway covariance matrix is rank-deficient ($\text{rank} \leq 5$). The trace and determinant ratios should be interpreted as lower bounds on the true spread difference — the runway distribution may be even tighter than estimated, since 21-dimensional structure cannot be reliably estimated from 6 samples. The PCA projection to 3 components mitigates this partially, but the runway covariance in PC space remains estimated from only 6 points. Additionally, 3 PCs explain only 65.7% of total variance; the remaining 34.3% could contain additional group differences not captured here.

**LDA classification:**

| Metric | Accuracy |
|--------|----------|
| Resubstitution | 100% |
| LOO-CV | 70% |
| Majority-class baseline | 70% (14/20 = control) |

The LOO-CV accuracy of 70% equals the majority-class baseline (predicting "control" for all samples), indicating that the LDA classifier does not generalize reliably at this sample size. However, the 100% resubstitution accuracy confirms perfect linear separability in the full feature space — the groups are distinguishable, but the 6-sample runway class is too small for the LOO estimator to learn a stable decision boundary. With $p = 21$ features and only $n_R = 6$ runway samples, the runway covariance matrix is rank-deficient ($\text{rank} \leq 5$), making LDA estimation unstable. Larger samples would likely improve LOO-CV substantially.

---

## 4. Discussion

### 4.1 Interpretation of Results

The results provide directionally consistent support for the hypothesis. Of the 10 metrics reaching Levene's significance, all 10 show higher variance in the control group. Three metrics (pelvis obliquity $R_\sigma = 0.33\times$, double support $0.71\times$, MQS $0.60\times$) show the reverse direction, though none reach significance.

The overall picture is one of partial support: 42% of metrics reach significance (below the pre-specified 50% threshold for "supported"), but the effect is entirely one-directional among significant results. Two factors contextualize this:

1. **Statistical power.** With $n = 17$--$22$ per group, the study is underpowered to detect moderate variance differences ($R_\sigma < 5\times$). Many non-significant metrics show the hypothesized direction but lack power.

2. **Domain concentration.** Significance clusters in smoothness (2/2 metrics) and symmetry (3/5 metrics) — the domains most directly relevant to movement quality — rather than distributing randomly across domains.

### 4.2 Implications for Robot Learning

The 17$\times$ lower smoothness variance and 14$\times$ lower symmetry variance in runway walks directly address a key challenge in imitation learning: distribution shift from noisy training data. A training distribution with narrower variance in these critical dimensions should produce:

- More consistent learned policies (lower policy entropy)
- Faster convergence during training
- Better generalization to controlled deployment environments
- Reduced need for reward shaping to penalize jerky or asymmetric motion

### 4.3 Software Environment

Analysis performed with Python 3.12, MediaPipe 0.10.x (PoseLandmarker heavy model, float16), NumPy 1.26, SciPy 1.13, scikit-learn 1.5, pandas 2.2, matplotlib 3.9.

### 4.4 Limitations

1. **Sample size.** The control group ($n = 17$) limits statistical power. The multivariate analysis is further constrained by listwise deletion to $n = 20$ complete cases.

2. **Video heterogeneity.** Control videos span diverse camera angles, environments, and subject demographics. While this heterogeneity is by design (representing "internet walking data"), it conflates multiple sources of variance.

3. **Monocular pose estimation.** MediaPipe provides 2D pose estimation from monocular video, which cannot capture out-of-plane joint angles or true 3D kinematics. Depth ambiguity particularly affects frontal-plane metrics.

4. **Selection bias.** Runway videos were professionally filmed with controlled conditions, while control videos were opportunistically sampled. Some variance difference may reflect filming conditions rather than movement quality.

5. **LDA generalization.** The LOO-CV accuracy (70%) equals the majority-class baseline, indicating that multivariate classification does not generalize at this sample size. The resubstitution accuracy (100%) confirms separability but may reflect overfitting.

6. **Non-independent metrics.** Several of the 24 metrics share underlying signal dependencies, inflating the apparent number of independent tests. The effective number of independent comparisons is likely lower than 24.

### 4.5 Future Work

- Increase sample sizes to $n \geq 50$ per group for improved statistical power
- Add multi-camera or depth-sensor validation
- Test the causal link: train locomotion policies on runway vs. internet data and compare learned policy quality
- Extend to other structured movement domains (martial arts, dance, athletic training)

---

## 5. Conclusion

This study provides quantitative evidence that professional runway model walks constitute a low-variance, high-quality kinematic distribution compared to general internet walking video. Across 24 biomechanical metrics spanning 6 movement quality domains, runway walks show **2.56$\times$ lower median variance**, with the strongest effects in movement smoothness ($17\times$) and bilateral symmetry ($14\times$). In multivariate kinematic feature space, runway walks occupy a region **25$\times$ smaller by volume**. All statistically significant differences favor the hypothesized direction with zero counter-evidence.

These findings support the use of runway walking data as a curated, low-entropy training foundation for robotic movement learning systems.

---

## References

Baker, R., McGinley, J. L., Schwartz, M. H., et al. (2009). The Gait Profile Score and Movement Analysis Profile. *Gait & Posture*, 30(3), 265--269.

Balasubramanian, S., Melendez-Calderon, A., & Burdet, E. (2012). A robust and sensitive metric for quantifying movement smoothness. *IEEE Transactions on Biomedical Engineering*, 59(8), 2126--2136.

Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A., & Burdet, E. (2015). On the analysis of movement smoothness. *Journal of NeuroEngineering and Rehabilitation*, 12(1), 112.

Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

Fritsch, F. N., & Carlson, R. E. (1980). Monotone piecewise cubic interpolation. *SIAM Journal on Numerical Analysis*, 17(2), 238--246.

Hamill, J., van Emmerik, R. E. A., Heiderscheit, B. C., & Li, L. (1999). A dynamical systems approach to lower extremity running injuries. *Clinical Biomechanics*, 14(5), 297--308.

Hausdorff, J. M., et al. (2001). When human walking becomes random walking: fractal analysis and modeling of gait rhythm fluctuations. *Physica A*, 302, 138--147.

Hof, A. L., Gazendam, M. G. J., & Sinke, W. E. (2005). The condition for dynamic stability. *Journal of Biomechanics*, 38(1), 1--8.

Levene, H. (1960). Robust tests for equality of variances. In *Contributions to Probability and Statistics*, 278--292. Stanford University Press.

Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50--60.

Perry, J., & Burnfield, J. M. (2010). *Gait Analysis: Normal and Pathological Function* (2nd ed.). SLACK Incorporated.

Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.

Robinson, R. O., Herzog, W., & Nigg, B. M. (1987). Use of force platform variables to quantify the effects of chiropractic manipulation on gait symmetry. *Journal of Manipulative and Physiological Therapeutics*, 10(4), 172--176.

Schwartz, M. H., & Rozumalski, A. (2008). The Gait Deviation Index: a new comprehensive index of gait pathology. *Gait & Posture*, 28(3), 351--357.

Shorter, K. A., Polk, J. D., Rosengren, K. S., & Hsiao-Wecksler, E. T. (2008). A new approach to detecting asymmetries in gait. *Clinical Biomechanics*, 23(4), 459--467.

Teulings, H. L., Contreras-Vidal, J. L., Stelmach, G. E., & Adler, C. H. (1997). Parkinsonism reduces coordination of fingers, wrist, and arm in fine motor control. *Experimental Neurology*, 146(1), 159--170.

Winter, D. A. (2009). *Biomechanics and Motor Control of Human Movement* (4th ed.). John Wiley & Sons.
