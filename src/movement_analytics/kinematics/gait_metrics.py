"""Gait-level metrics computed from joint angle time series.

Implements clinically validated gait quality metrics including smoothness,
symmetry, variability, and temporal parameters. References:
  - Gait Deviation Index: Schwartz & Rozumalski, Gait & Posture, 2008
  - SPARC smoothness: Balasubramanian et al., IEEE TBME, 2012/2015
  - Symmetry Index: Robinson et al., Int J Rehabil Res, 1987
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert


def angular_velocity(angles: np.ndarray, fps: float) -> np.ndarray:
    """Compute angular velocity via central finite differences (degrees/s)."""
    dt = 1.0 / fps
    vel = np.gradient(angles, dt)
    return vel


def angular_acceleration(angles: np.ndarray, fps: float) -> np.ndarray:
    """Compute angular acceleration (degrees/s^2)."""
    dt = 1.0 / fps
    acc = np.gradient(np.gradient(angles, dt), dt)
    return acc


def jerk(angles: np.ndarray, fps: float) -> np.ndarray:
    """Compute jerk — third derivative of angle (degrees/s^3)."""
    dt = 1.0 / fps
    return np.gradient(angular_acceleration(angles, fps), dt)


def normalized_jerk(angles: np.ndarray, fps: float) -> float:
    """Normalized jerk metric (dimensionless). Lower = smoother.

    NJ = sqrt(0.5 * integral(jerk^2 dt) * T^5 / A^2)
    where T = movement duration, A = peak-to-peak amplitude.
    Teulings et al., Acta Psychologica, 1997.
    """
    dt = 1.0 / fps
    j = jerk(angles, fps)
    T = len(angles) * dt
    A = np.ptp(angles)
    if A < 1e-6:
        return 0.0
    nj = np.sqrt(0.5 * np.sum(j ** 2) * dt * T ** 5 / (A ** 2))
    return float(nj)


def sparc(velocity: np.ndarray, fps: float, fc: float = 10.0,
          amplitude_threshold: float = 0.05) -> float:
    """Spectral Arc Length (SPARC) smoothness metric. Closer to 0 = smoother.

    Balasubramanian, Melendez-Calderon & Burdet, IEEE TBME, 2012.
    Revised: Balasubramanian et al., Frontiers in Neurology, 2015.
    """
    N = len(velocity)
    freq = np.fft.rfftfreq(N, d=1.0 / fps)
    spectrum = np.abs(np.fft.rfft(velocity))
    spectrum /= (spectrum.max() + 1e-10)

    mask = freq <= fc
    freq = freq[mask]
    spectrum = spectrum[mask]

    above_threshold = spectrum >= amplitude_threshold
    if not np.any(above_threshold):
        return 0.0
    last_idx = np.max(np.where(above_threshold))
    freq = freq[:last_idx + 1]
    spectrum = spectrum[:last_idx + 1]

    if len(freq) < 2:
        return 0.0

    dfreq = np.diff(freq)
    dspec = np.diff(spectrum)
    arc_lengths = np.sqrt(dfreq ** 2 + dspec ** 2)
    sal = -float(np.sum(arc_lengths))
    return sal


def symmetry_index(left: np.ndarray, right: np.ndarray) -> float:
    """Symmetry Index (SI). 0 = perfect symmetry.

    SI = 2 * |mean(L) - mean(R)| / (mean(L) + mean(R)) * 100
    Robinson et al., Int J Rehabil Res, 1987.
    """
    ml = np.mean(np.abs(left))
    mr = np.mean(np.abs(right))
    denom = ml + mr
    if denom < 1e-6:
        return 0.0
    return float(2 * abs(ml - mr) / denom * 100)


def symmetry_ratio(left: np.ndarray, right: np.ndarray) -> float:
    """Symmetry Ratio. 1.0 = perfect symmetry."""
    ml = np.mean(np.abs(left))
    mr = np.mean(np.abs(right))
    if max(ml, mr) < 1e-6:
        return 1.0
    return float(min(ml, mr) / (max(ml, mr) + 1e-8))


def waveform_symmetry(left: np.ndarray, right: np.ndarray) -> float:
    """Waveform symmetry via normalized cross-correlation. 100 = identical shape.

    Captures shape and timing differences between bilateral signals.
    Insensitive to amplitude scaling (NCC is normalized). Use SI for
    amplitude-based asymmetry detection. |NCC| used: anti-phase = 100%.
    """
    l_centered = left - np.mean(left)
    r_centered = right - np.mean(right)
    l_norm = np.linalg.norm(l_centered)
    r_norm = np.linalg.norm(r_centered)
    if l_norm < 1e-6 or r_norm < 1e-6:
        return 100.0 if l_norm < 1e-6 and r_norm < 1e-6 else 0.0
    ncc = float(np.dot(l_centered, r_centered) / (l_norm * r_norm))
    return abs(ncc) * 100.0


def coefficient_of_variation(values: np.ndarray) -> float:
    """Coefficient of variation (CV) as percentage."""
    m = np.mean(values)
    if abs(m) < 1e-6:
        return 0.0
    return float(np.std(values) / abs(m) * 100)


def dfa_scaling_exponent(stride_intervals: np.ndarray) -> float:
    """Detrended Fluctuation Analysis scaling exponent (alpha).

    Measures long-range correlations in stride-to-stride fluctuations.
    Healthy gait: alpha ~ 0.75 (fractal dynamics).
    Pathological: alpha ~ 0.5 (random, e.g. Parkinson's) or ~1.0 (over-correlated).
    Hausdorff et al., J Appl Physiol, 2001.
    """
    N = len(stride_intervals)
    if N < 16:
        return float("nan")
    y = np.cumsum(stride_intervals - np.mean(stride_intervals))
    scales = np.unique(np.logspace(np.log10(4), np.log10(N // 4), 10).astype(int))
    scales = scales[scales >= 4]
    if len(scales) < 3:
        return float("nan")
    fluctuations = []
    for s in scales:
        n_segments = N // s
        if n_segments < 1:
            continue
        rms_vals = []
        for seg in range(n_segments):
            start = seg * s
            segment = y[start:start + s]
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms_vals.append(np.sqrt(np.mean((segment - trend) ** 2)))
        if rms_vals:
            fluctuations.append((s, np.mean(rms_vals)))
    if len(fluctuations) < 3:
        return float("nan")
    log_s = np.log(np.array([f[0] for f in fluctuations]))
    f_vals = np.array([f[1] for f in fluctuations])
    if np.all(f_vals < 1e-15):
        return float("nan")
    log_f = np.log(np.maximum(f_vals, 1e-15))
    valid = np.isfinite(log_s) & np.isfinite(log_f)
    if np.sum(valid) < 3:
        return float("nan")
    alpha = float(np.polyfit(log_s[valid], log_f[valid], 1)[0])
    return alpha


def continuous_relative_phase(signal_a: np.ndarray, signal_b: np.ndarray,
                              fps: float) -> np.ndarray:
    """Compute Continuous Relative Phase between two oscillating signals.

    Uses Hilbert transform to extract instantaneous phase of each signal,
    then computes the phase difference. Anti-phase coupling (~180°) is
    expected for bilateral hip flexion in healthy gait.

    Hamill J, van Emmerik REA, Heiderscheit BC, Li L. J Appl Biomech. 1999.
    """
    a_centered = signal_a - np.mean(signal_a)
    b_centered = signal_b - np.mean(signal_b)

    phase_a = np.angle(hilbert(a_centered))
    phase_b = np.angle(hilbert(b_centered))

    crp = np.degrees(phase_a - phase_b)
    crp = (crp + 180) % 360 - 180
    return crp


def crp_consistency(signal_a: np.ndarray, signal_b: np.ndarray,
                    fps: float) -> float:
    """CRP consistency: how stable is the inter-limb phase relationship.

    Returns the circular standard deviation of the CRP signal.
    Lower = more consistent coordination. Healthy gait: CSD < 15°.
    Pathological gait: CSD > 30°.

    Uses circular statistics (mean resultant length) to handle wraparound.
    """
    crp = continuous_relative_phase(signal_a, signal_b, fps)
    crp_rad = np.radians(crp)
    R = np.sqrt(np.mean(np.cos(crp_rad))**2 + np.mean(np.sin(crp_rad))**2)
    R = min(R, 1.0)
    circular_sd = np.degrees(np.sqrt(-2 * np.log(R))) if R > 1e-10 else 180.0
    return float(circular_sd)


def rom(angles: np.ndarray) -> float:
    """Range of motion: max - min of angle time series."""
    return float(np.ptp(angles))


def stride_pelvic_asymmetry(
    pelvis_obliquity: np.ndarray, heel_strikes: np.ndarray,
) -> float:
    """Pelvic drop asymmetry from stride-segmented obliquity.

    Splits each gait cycle (between consecutive ipsilateral heel strikes)
    into first-half (ipsilateral stance) and second-half (contralateral
    stance), then compares pelvic excursion. Works from a single global
    pelvis signal — no bilateral sensors needed.

    Returns SI-like percentage: 0 = symmetric, >10 = asymmetric.
    NaN if fewer than 2 complete strides.
    """
    if len(heel_strikes) < 3:
        return float("nan")

    first_half_exc = []
    second_half_exc = []
    for i in range(len(heel_strikes) - 1):
        start = heel_strikes[i]
        end = heel_strikes[i + 1]
        mid = (start + end) // 2
        seg_a = pelvis_obliquity[start:mid]
        seg_b = pelvis_obliquity[mid:end]
        if len(seg_a) > 2 and len(seg_b) > 2:
            first_half_exc.append(float(np.ptp(seg_a)))
            second_half_exc.append(float(np.ptp(seg_b)))

    if len(first_half_exc) < 2:
        return float("nan")

    mean_a = float(np.mean(first_half_exc))
    mean_b = float(np.mean(second_half_exc))
    denom = mean_a + mean_b
    if denom < 1e-6:
        return 0.0
    return float(2 * abs(mean_a - mean_b) / denom * 100)


_GDI_NORMAL_CACHE: dict = {}


def _get_normal_reference() -> dict[str, np.ndarray]:
    """Lazily generate and cache the normal gait reference waveforms (101 points per cycle).

    Uses gait event detection to segment strides at heel strikes, matching
    how test subjects are segmented in gait_deviation_index(). Each joint
    reference is the mean of detected strides, normalized to 101 points.
    """
    if _GDI_NORMAL_CACHE:
        return _GDI_NORMAL_CACHE
    from ..generators.gait_model import GaitParameters, generate_gait_cycle
    params = GaitParameters()
    fps = 30
    frames_per_cycle = max(fps * 60 // int(params.cadence * params.speed_factor) * 2, 20)
    n_cycles = 6
    n_frames = frames_per_cycle
    right = generate_gait_cycle(params, n_frames=n_frames, n_cycles=n_cycles, side="right")
    hip = right["hip_flexion"]
    knee = right["knee_flexion"]
    ankle = right.get("ankle_dorsiflexion")
    events = detect_gait_events(hip, knee, fps, ankle_dorsiflexion=ankle)
    hs = events["heel_strikes"]
    if len(hs) < 3:
        return _GDI_NORMAL_CACHE
    ref_joints: dict[str, np.ndarray] = {}
    for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]:
        if joint not in right:
            continue
        strides = []
        for i in range(len(hs) - 1):
            seg = right[joint][hs[i]:hs[i + 1]]
            if len(seg) >= 5:
                strides.append(
                    np.interp(np.linspace(0, 1, 101),
                              np.linspace(0, 1, len(seg)), seg)
                )
        if strides:
            ref_joints[joint] = np.mean(strides, axis=0)
    _GDI_NORMAL_CACHE.update(ref_joints)
    return _GDI_NORMAL_CACHE


def gait_deviation_index(
    angles: dict[str, np.ndarray],
    heel_strikes: np.ndarray,
) -> float:
    """Simplified Gait Deviation Index (Schwartz & Rozumalski 2008).

    Compares stride-normalized waveforms (hip, knee, ankle) against the
    normal gait reference. Returns ~100 for normal gait, decreasing with
    deviation. Each 1-SD deviation from normal reduces GDI by ~10 points.

    Requires ≥2 complete strides. Returns NaN if insufficient data.
    """
    if len(heel_strikes) < 3:
        return float("nan")

    ref = _get_normal_reference()
    joints = [j for j in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]
              if j in angles and j in ref]
    if not joints:
        return float("nan")

    distances = []
    for i in range(len(heel_strikes) - 1):
        start = heel_strikes[i]
        end = heel_strikes[i + 1]
        if end - start < 5:
            continue
        stride_dists = []
        for joint in joints:
            seg = angles[joint][start:end]
            normalized = np.interp(
                np.linspace(0, 1, 101), np.linspace(0, 1, len(seg)), seg
            )
            rms = np.sqrt(np.mean((normalized - ref[joint]) ** 2))
            stride_dists.append(rms)
        distances.append(np.mean(stride_dists))

    if not distances:
        return float("nan")

    mean_dist = float(np.mean(distances))
    # Scale: 10 points per ~5° RMS deviation (approximately 1 SD in clinical data)
    gdi = 100.0 - (mean_dist / 5.0) * 10.0
    return float(max(0.0, gdi))


def detect_gait_events(hip_flexion: np.ndarray, knee_flexion: np.ndarray,
                       fps: float,
                       ankle_dorsiflexion: np.ndarray | None = None,
                       heel_y: np.ndarray | None = None) -> dict:
    """Detect gait events (heel strikes, toe-offs) from joint angle signals.

    Primary: hip flexion peaks (max flexion ≈ heel strike).
    When *heel_y* is provided (pixel Y-coordinate of heel landmark),
    refines heel-strike timing using foot contact detection — the heel's
    lowest point (max Y in screen coords) within ±0.15s of each hip peak.
    Otherwise when *ankle_dorsiflexion* is not None, refines by finding
    the nearest knee-extension minimum within ±0.1s of each hip peak.
    """
    nyq = fps / 2
    if nyq <= 6.0 or len(hip_flexion) < 13:
        hip_filt = hip_flexion
    else:
        b, a = butter(4, 6.0 / nyq, btype="low")
        hip_filt = filtfilt(b, a, hip_flexion)

    hip_range = np.ptp(hip_filt)
    min_prominence = max(hip_range * 0.15, 1.0)
    hs_indices, _ = find_peaks(
        hip_filt, distance=int(fps * 0.3), prominence=min_prominence,
    )
    to_indices, _ = find_peaks(
        -hip_filt, distance=int(fps * 0.3), prominence=min_prominence,
    )

    if heel_y is not None and len(hs_indices) > 0 and not np.all(np.isnan(heel_y)):
        window = max(1, int(fps * 0.15))
        refined = []
        for hs in hs_indices:
            lo = max(0, hs - window)
            hi = min(len(heel_y), hs + window + 1)
            seg = heel_y[lo:hi]
            valid = ~np.isnan(seg)
            if np.any(valid):
                best = lo + int(np.where(valid, seg, -np.inf).argmax())
            else:
                best = hs
            refined.append(best)
        hs_indices = np.array(refined)
    elif ankle_dorsiflexion is not None and len(hs_indices) > 0:
        # Knee near full extension at heel strike; ankle gates data quality
        window = max(1, int(fps * 0.1))
        refined = []
        for hs in hs_indices:
            lo = max(0, hs - window)
            hi = min(len(knee_flexion), hs + window + 1)
            best = lo + int(np.argmin(knee_flexion[lo:hi]))
            refined.append(best)
        hs_indices = np.array(refined)

    stride_times = np.diff(hs_indices) / fps if len(hs_indices) > 1 else np.array([])
    cadence = 60.0 / np.mean(stride_times) * 2 if len(stride_times) > 0 else 0.0

    return {
        "heel_strikes": hs_indices,
        "toe_offs": to_indices,
        "stride_times": stride_times,
        "cadence_steps_per_min": cadence,
        "stride_time_mean": float(np.mean(stride_times)) if len(stride_times) > 0 else float("nan"),
        "stride_time_cv": (
            coefficient_of_variation(stride_times)
            if len(stride_times) > 2 else float("nan")
        ),
    }


def compute_gait_summary(angles_right: dict, angles_left: dict,
                         fps: float = 30.0,
                         pose_metadata: dict | None = None) -> dict:
    """Compute comprehensive gait quality metrics from bilateral angle data.

    When *pose_metadata* is provided (from ``process_video``), also writes
    ``pose_observed_fraction``, ``pose_mean_confidence``,
    ``pose_interpolation_fraction``, ``mqs_confidence_factor``, and
    ``movement_quality_score_weighted`` to the returned dict.

    Returns a flat dict of named metrics suitable for display and logging.
    """
    metrics = {}

    for name, side_label in [("right", "R"), ("left", "L")]:
        angles = angles_right if name == "right" else angles_left

        for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]:
            if joint not in angles:
                continue
            signal = angles[joint]
            vel = angular_velocity(signal, fps)

            prefix = f"{side_label}_{joint}"
            metrics[f"{prefix}_ROM"] = rom(signal)
            metrics[f"{prefix}_mean"] = float(np.mean(signal))
            metrics[f"{prefix}_peak"] = float(np.max(signal))
            metrics[f"{prefix}_min"] = float(np.min(signal))
            metrics[f"{prefix}_velocity_peak"] = float(np.max(np.abs(vel)))
            metrics[f"{prefix}_NJ"] = normalized_jerk(signal, fps)
            metrics[f"{prefix}_SPARC"] = sparc(vel, fps)

    for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion",
                   "pelvis_obliquity", "trunk_lateral_lean"]:
        if joint in angles_right and joint in angles_left:
            l_arr = angles_left[joint]
            r_arr = angles_right[joint]
            if l_arr is r_arr:
                continue
            metrics[f"{joint}_SI"] = symmetry_index(l_arr, r_arr)
            metrics[f"{joint}_SR"] = symmetry_ratio(l_arr, r_arr)
            metrics[f"{joint}_waveform_sym"] = waveform_symmetry(
                l_arr, r_arr
            )

    if "hip_flexion" in angles_right and "knee_flexion" in angles_right:
        ankle = angles_right.get("ankle_dorsiflexion")
        r_heel_y = angles_right.get("heel_y")
        events = detect_gait_events(
            angles_right["hip_flexion"], angles_right["knee_flexion"], fps,
            ankle_dorsiflexion=ankle, heel_y=r_heel_y,
        )
        metrics["cadence"] = events["cadence_steps_per_min"]
        metrics["stride_time_mean"] = events["stride_time_mean"]
        metrics["stride_time_CV"] = events["stride_time_cv"]
        metrics["n_strides"] = len(events["stride_times"])
        if len(events["stride_times"]) >= 16:
            metrics["stride_dfa_alpha"] = dfa_scaling_exponent(events["stride_times"])

        hs = events["heel_strikes"]
        to = events["toe_offs"]

        ds_pcts = []
        for i in range(len(hs) - 1):
            stride_start = hs[i]
            stride_end = hs[i + 1]
            stride_dur = stride_end - stride_start
            if stride_dur < 3:
                continue
            tos_in_stride = to[(to > stride_start) & (to < stride_end)]
            if len(tos_in_stride) == 0:
                continue
            stance_dur = tos_in_stride[0] - stride_start
            stance_pct = stance_dur / stride_dur
            ds_pct = max(0.0, 2 * stance_pct - 1) * 100
            ds_pcts.append(ds_pct)
        if ds_pcts:
            metrics["double_support_pct"] = float(np.mean(ds_pcts))
        else:
            metrics["double_support_pct"] = float("nan")
        if len(hs) >= 3:
            for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]:
                for side_l, ang in [("R", angles_right), ("L", angles_left)]:
                    if joint not in ang:
                        continue
                    stride_roms = []
                    for si in range(len(hs) - 1):
                        seg = ang[joint][hs[si]:hs[si + 1]]
                        if len(seg) > 2:
                            stride_roms.append(float(np.ptp(seg)))
                    if len(stride_roms) >= 2:
                        cv = coefficient_of_variation(np.array(stride_roms))
                        short = joint.replace("_flexion", "").replace(
                            "_dorsiflexion", ""
                        )
                        metrics[f"{side_l}_{short}_ROM_CV"] = cv

        rom_cvs = [v for k, v in metrics.items()
                   if k.endswith("_ROM_CV") and not np.isnan(v)]
        if rom_cvs:
            metrics["kinematic_CV_mean"] = float(np.mean(rom_cvs))

        pelvis = angles_right.get("pelvis_obliquity")
        if pelvis is not None and len(hs) >= 3:
            metrics["pelvic_drop_asymmetry"] = stride_pelvic_asymmetry(
                pelvis, hs,
            )
        trunk = angles_right.get("trunk_lateral_lean")
        if trunk is not None and len(hs) >= 3:
            metrics["trunk_lean_asymmetry"] = stride_pelvic_asymmetry(
                trunk, hs,
            )

        gdi_r = gait_deviation_index(angles_right, hs)
        if not np.isnan(gdi_r):
            metrics["R_GDI"] = gdi_r

        if "hip_flexion" in angles_left and "knee_flexion" in angles_left:
            l_ankle = angles_left.get("ankle_dorsiflexion")
            l_heel_y = angles_left.get("heel_y")
            l_events = detect_gait_events(
                angles_left["hip_flexion"], angles_left["knee_flexion"],
                fps, ankle_dorsiflexion=l_ankle, heel_y=l_heel_y,
            )
            l_hs = l_events["heel_strikes"]
            gdi_l = gait_deviation_index(angles_left, l_hs)
            if not np.isnan(gdi_l):
                metrics["L_GDI"] = gdi_l

        gdi_vals = [metrics.get("R_GDI"), metrics.get("L_GDI")]
        gdi_vals = [v for v in gdi_vals if v is not None]
        if gdi_vals:
            metrics["GDI"] = float(np.mean(gdi_vals))

    if "hip_flexion" in angles_right:
        metrics["R_hip_CV"] = coefficient_of_variation(
            np.abs(angles_right["hip_flexion"])
        )
    if "hip_flexion" in angles_left:
        metrics["L_hip_CV"] = coefficient_of_variation(
            np.abs(angles_left["hip_flexion"])
        )

    def _is_shared(key):
        return (key in angles_right and key in angles_left
                and angles_right[key] is angles_left[key])

    for side_label, angles in [("R", angles_right), ("L", angles_left)]:
        skip_frontal = side_label == "L"
        if "pelvis_obliquity" in angles:
            if not (skip_frontal and _is_shared("pelvis_obliquity")):
                metrics[f"{side_label}_pelvis_obliquity_ROM"] = rom(
                    angles["pelvis_obliquity"]
                )
        if "pelvis_obliquity_signed" in angles:
            if not (skip_frontal and _is_shared("pelvis_obliquity_signed")):
                signed = angles["pelvis_obliquity_signed"]
                metrics[f"{side_label}_pelvis_obliquity_mean_signed"] = float(
                    np.nanmean(signed)
                )
        if "trunk_lateral_lean" in angles:
            if not (skip_frontal and _is_shared("trunk_lateral_lean")):
                metrics[f"{side_label}_trunk_lean_ROM"] = rom(
                    angles["trunk_lateral_lean"]
                )

    for side_label, angles in [("R", angles_right), ("L", angles_left)]:
        if "shoulder_flexion" in angles:
            sf = angles["shoulder_flexion"]
            metrics[f"{side_label}_shoulder_ROM"] = rom(sf)
            metrics[f"{side_label}_elbow_ROM"] = (
                rom(angles["elbow_flexion"])
                if "elbow_flexion" in angles else float("nan")
            )
    if ("shoulder_flexion" in angles_right
            and "shoulder_flexion" in angles_left):
        r_sf = angles_right["shoulder_flexion"]
        l_sf = angles_left["shoulder_flexion"]
        if r_sf is not l_sf:
            metrics["arm_swing_SI"] = symmetry_index(r_sf, l_sf)
        arm_rom_mean = np.mean([rom(r_sf), rom(l_sf)])
        metrics["arm_swing_ROM_mean"] = float(arm_rom_mean)
        _NORMAL_ARM_SWING_ROM = 25.0
        metrics["arm_swing_ratio"] = float(
            np.clip(arm_rom_mean / _NORMAL_ARM_SWING_ROM, 0, 2)
        )

    if "hip_flexion" in angles_right and "hip_flexion" in angles_left:
        crp_mad = crp_consistency(
            angles_right["hip_flexion"], angles_left["hip_flexion"], fps
        )
        metrics["hip_CRP_MAD"] = crp_mad
    if "knee_flexion" in angles_right and "knee_flexion" in angles_left:
        metrics["knee_CRP_MAD"] = crp_consistency(
            angles_right["knee_flexion"], angles_left["knee_flexion"], fps
        )

    for side_label, angles in [("R", angles_right), ("L", angles_left)]:
        if "hip_flexion" in angles and "knee_flexion" in angles:
            metrics[f"{side_label}_hip_knee_CRP_MAD"] = crp_consistency(
                angles["hip_flexion"], angles["knee_flexion"], fps
            )

    mqs_breakdown = mqs_domain_scores(metrics)
    for domain, score in mqs_breakdown.items():
        metrics[f"mqs_{domain}"] = score

    completeness = mqs_signal_completeness(metrics)
    for domain, frac in completeness.items():
        metrics[f"mqs_{domain}_completeness"] = frac
    metrics["mqs_overall_completeness"] = float(np.mean(list(completeness.values())))

    sufficient = mqs_sufficient_evidence(metrics)
    metrics["mqs_sufficient_evidence"] = float(sufficient)

    if sufficient:
        metrics["movement_quality_score"] = movement_quality_score(metrics)
    else:
        metrics["movement_quality_score"] = float("nan")

    if pose_metadata is not None:
        obs = pose_metadata.get("observed_fraction", 1.0)
        conf = pose_metadata.get("mean_confidence", 1.0)
        metrics["pose_observed_fraction"] = obs
        metrics["pose_mean_confidence"] = conf
        interp = pose_metadata.get("interpolation_fractions", {})
        if interp:
            metrics["pose_interpolation_fraction"] = float(
                np.mean(list(interp.values()))
            )
        cf = mqs_confidence_factor(metrics)
        metrics["mqs_confidence_factor"] = cf
        metrics["movement_quality_score_weighted"] = (
            metrics["movement_quality_score"] * cf
        )

    return metrics


def _signal_score(value: float, optimal_low: float, optimal_high: float,
                  worst_low: float, worst_high: float) -> float:
    """Map a metric value to 0-100 based on clinical reference range.

    100 = within optimal range, linearly decreasing to 0 at worst bounds.
    """
    if np.isnan(value):
        return float("nan")
    if optimal_low <= value <= optimal_high:
        return 100.0
    if value < optimal_low:
        if worst_low >= optimal_low:
            return 0.0
        return max(0.0, 100.0 * (value - worst_low) / (optimal_low - worst_low))
    if worst_high <= optimal_high:
        return 0.0
    return max(0.0, 100.0 * (worst_high - value) / (worst_high - optimal_high))


# Clinical reference ranges: (optimal_low, optimal_high, worst_low, worst_high)
_SIGNAL_RANGES = {
    "hip_rom": (35.0, 50.0, 10.0, 70.0),
    "knee_rom": (50.0, 70.0, 15.0, 90.0),
    "ankle_rom": (20.0, 35.0, 5.0, 50.0),
    "pelvic_obliquity": (0.0, 7.0, 0.0, 20.0),
    "trunk_lean": (0.0, 5.0, 0.0, 15.0),
    "sparc_hip": (-2.0, -1.3, -6.0, -0.5),
    "sparc_knee": (-16.0, -12.0, -25.0, -8.0),
    "symmetry": (0.0, 10.0, 0.0, 50.0),
    "stride_cv": (0.0, 4.0, 0.0, 20.0),
    "kinematic_cv": (0.0, 5.0, 0.0, 30.0),
    "cadence": (90.0, 130.0, 40.0, 180.0),
    "stride_time": (0.8, 1.3, 0.3, 2.5),
    "crp_mad": (0.0, 15.0, 0.0, 60.0),
    "crp_hip_knee": (15.0, 35.0, 0.0, 60.0),
}

# Domain weights from research framework (Section 10)
_DOMAIN_WEIGHTS = {
    "kinematics": 0.25,
    "smoothness": 0.18,
    "symmetry": 0.18,
    "coordination": 0.14,
    "variability": 0.13,
    "temporal": 0.12,
}


def mqs_domain_scores(metrics: dict) -> dict[str, float]:
    """Compute per-domain scores (0-100) from gait summary metrics."""
    domains = {}

    kin_scores = []
    for side in ["R", "L"]:
        for joint, key in [("hip_flexion", "hip_rom"), ("knee_flexion", "knee_rom"),
                           ("ankle_dorsiflexion", "ankle_rom")]:
            val = metrics.get(f"{side}_{joint}_ROM")
            if _is_valid_metric(val):
                lo, hi, wlo, whi = _SIGNAL_RANGES[key]
                kin_scores.append(_signal_score(val, lo, hi, wlo, whi))
        for metric_suffix, range_key in [("pelvis_obliquity_ROM", "pelvic_obliquity"),
                                         ("trunk_lean_ROM", "trunk_lean")]:
            val = metrics.get(f"{side}_{metric_suffix}")
            if _is_valid_metric(val):
                lo, hi, wlo, whi = _SIGNAL_RANGES[range_key]
                kin_scores.append(_signal_score(val, lo, hi, wlo, whi))
    domains["kinematics"] = float(np.mean(kin_scores)) if kin_scores else 50.0

    hip_sparc_scores = []
    sm_scores = []
    for side in ["R", "L"]:
        for joint, range_key in [("hip_flexion", "sparc_hip"),
                                 ("knee_flexion", "sparc_knee")]:
            val = metrics.get(f"{side}_{joint}_SPARC")
            if _is_valid_metric(val):
                lo, hi, wlo, whi = _SIGNAL_RANGES[range_key]
                score = _signal_score(val, lo, hi, wlo, whi)
                sm_scores.append(score)
                if joint == "hip_flexion":
                    hip_sparc_scores.append(score)
    if sm_scores:
        overall = float(np.mean(sm_scores))
        hip_floor = float(np.mean(hip_sparc_scores)) if hip_sparc_scores else overall
        domains["smoothness"] = min(overall, hip_floor)
    else:
        domains["smoothness"] = 50.0

    si_scores = []
    for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion",
                   "pelvis_obliquity"]:
        val = metrics.get(f"{joint}_SI")
        if _is_valid_metric(val):
            lo, hi, wlo, whi = _SIGNAL_RANGES["symmetry"]
            si_scores.append(_signal_score(val, lo, hi, wlo, whi))
    si_mean = float(np.mean(si_scores)) if si_scores else 50.0

    hip_ws = metrics.get("hip_flexion_waveform_sym")
    if si_scores and _is_valid_metric(hip_ws):
        domains["symmetry"] = min(si_mean, hip_ws)
    elif si_scores:
        domains["symmetry"] = si_mean
    else:
        domains["symmetry"] = 50.0

    coord_scores = []
    hip_crp = metrics.get("hip_CRP_MAD")
    if _is_valid_metric(hip_crp):
        lo, hi, wlo, whi = _SIGNAL_RANGES["crp_mad"]
        coord_scores.append(_signal_score(hip_crp, lo, hi, wlo, whi))
    for side in ["R", "L"]:
        hk_crp = metrics.get(f"{side}_hip_knee_CRP_MAD")
        if _is_valid_metric(hk_crp):
            lo, hi, wlo, whi = _SIGNAL_RANGES["crp_hip_knee"]
            coord_scores.append(_signal_score(hk_crp, lo, hi, wlo, whi))
    domains["coordination"] = float(np.mean(coord_scores)) if coord_scores else 50.0

    var_scores = []
    cv_val = metrics.get("stride_time_CV", float("nan"))
    if not np.isnan(cv_val):
        lo, hi, wlo, whi = _SIGNAL_RANGES["stride_cv"]
        var_scores.append(_signal_score(cv_val, lo, hi, wlo, whi))
    kin_cv = metrics.get("kinematic_CV_mean", float("nan"))
    if not np.isnan(kin_cv):
        lo, hi, wlo, whi = _SIGNAL_RANGES["kinematic_cv"]
        var_scores.append(_signal_score(kin_cv, lo, hi, wlo, whi))
    domains["variability"] = float(np.mean(var_scores)) if var_scores else 50.0

    t_scores = []
    cad = metrics.get("cadence", float("nan"))
    if not np.isnan(cad):
        lo, hi, wlo, whi = _SIGNAL_RANGES["cadence"]
        t_scores.append(_signal_score(cad, lo, hi, wlo, whi))
    st = metrics.get("stride_time_mean", float("nan"))
    if not np.isnan(st):
        lo, hi, wlo, whi = _SIGNAL_RANGES["stride_time"]
        t_scores.append(_signal_score(st, lo, hi, wlo, whi))
    domains["temporal"] = float(np.mean(t_scores)) if t_scores else 50.0

    return domains


def _is_valid_metric(value) -> bool:
    """Check if a metric value is present and not NaN."""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return True


def mqs_signal_completeness(metrics: dict) -> dict[str, float]:
    """Compute signal completeness per domain (0-1).

    Reports what fraction of expected signals were present and non-NaN.
    A completeness below 1.0 means the domain score is based on partial data.
    """
    completeness = {}

    kin_expected = 0
    kin_present = 0
    for side in ["R", "L"]:
        for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]:
            kin_expected += 1
            if _is_valid_metric(metrics.get(f"{side}_{joint}_ROM")):
                kin_present += 1
        for suffix in ["pelvis_obliquity_ROM", "trunk_lean_ROM"]:
            kin_expected += 1
            if _is_valid_metric(metrics.get(f"{side}_{suffix}")):
                kin_present += 1
    completeness["kinematics"] = kin_present / kin_expected if kin_expected else 0.0

    sm_present = 0
    for s in ["R", "L"]:
        for joint in ["hip_flexion", "knee_flexion"]:
            if _is_valid_metric(metrics.get(f"{s}_{joint}_SPARC")):
                sm_present += 1
    completeness["smoothness"] = sm_present / 4.0

    sy_present = sum(
        1 for j in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion",
                     "pelvis_obliquity"]
        if _is_valid_metric(metrics.get(f"{j}_SI"))
    )
    completeness["symmetry"] = sy_present / 4.0

    coord_present = 0
    coord_expected = 3
    if _is_valid_metric(metrics.get("hip_CRP_MAD")):
        coord_present += 1
    for side in ["R", "L"]:
        if _is_valid_metric(metrics.get(f"{side}_hip_knee_CRP_MAD")):
            coord_present += 1
    completeness["coordination"] = coord_present / coord_expected

    var_present = 0
    if _is_valid_metric(metrics.get("stride_time_CV")):
        var_present += 1
    if _is_valid_metric(metrics.get("kinematic_CV_mean")):
        var_present += 1
    completeness["variability"] = var_present / 2.0

    t_count = 0
    cad = metrics.get("cadence", float("nan"))
    if not np.isnan(cad):
        t_count += 1
    st = metrics.get("stride_time_mean", float("nan"))
    if not np.isnan(st):
        t_count += 1
    completeness["temporal"] = t_count / 2.0

    return completeness


def mqs_sufficient_evidence(metrics: dict, min_completeness: float = 0.5) -> bool:
    """Check whether enough signals are present for a reliable MQS.

    Returns False when overall signal completeness falls below the
    threshold, indicating the score should be treated as unreliable.
    """
    completeness = mqs_signal_completeness(metrics)
    overall = float(np.mean(list(completeness.values())))
    return overall >= min_completeness


def mqs_confidence_factor(metrics: dict) -> float:
    """Compute a 0-1 confidence factor for MQS based on data quality.

    When pose_observed_fraction or pose_mean_confidence are present
    (injected by video pipeline), returns a factor < 1.0 if data
    quality is insufficient. Synthetic data (no pose keys) returns 1.0.

    Factor = observed_fraction * mean_confidence * (1 - interp_penalty).
    The interpolation penalty scales with the fraction of angle data
    that was linearly interpolated (missing frames filled in).
    """
    obs = metrics.get("pose_observed_fraction")
    conf = metrics.get("pose_mean_confidence")
    if obs is None and conf is None:
        return 1.0
    obs = np.clip(obs if obs is not None else 1.0, 0, 1)
    conf = np.clip(conf if conf is not None else 1.0, 0, 1)
    interp = metrics.get("pose_interpolation_fraction", 0.0)
    interp_penalty = np.clip(interp * 0.5, 0, 0.5)
    return float(obs * conf * (1.0 - interp_penalty))


def movement_quality_score(metrics: dict) -> float:
    """Compute composite Movement Quality Score (0-100).

    Weighted combination across 6 biomechanical domains:
    kinematics (25%), smoothness (18%), symmetry (18%),
    coordination (14%), variability (13%), temporal (12%).
    """
    domains = mqs_domain_scores(metrics)
    mqs = sum(domains[d] * _DOMAIN_WEIGHTS[d] for d in _DOMAIN_WEIGHTS)
    return float(np.clip(mqs, 0, 100))
