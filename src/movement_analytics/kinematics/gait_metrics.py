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
    """Waveform symmetry via normalized cross-correlation. 100 = identical waveforms.

    Captures shape, timing, and amplitude differences simultaneously.
    More sensitive than mean-based SI to phase-specific asymmetries.
    Returns 0-100 scale for consistency with other metrics.
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


def detect_gait_events(hip_flexion: np.ndarray, knee_flexion: np.ndarray,
                       fps: float,
                       ankle_dorsiflexion: np.ndarray | None = None) -> dict:
    """Detect gait events (heel strikes, toe-offs) from joint angle signals.

    Primary: hip flexion peaks (max flexion ≈ heel strike).
    When ankle_dorsiflexion is provided, refines heel-strike timing by
    finding the nearest knee-extension minimum within ±0.1s of each
    hip peak — closer to true initial contact (Perry & Burnfield 2010).
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

    if ankle_dorsiflexion is not None and len(hs_indices) > 0:
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
                         fps: float = 30.0) -> dict:
    """Compute comprehensive gait quality metrics from bilateral angle data.

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
            metrics[f"{joint}_SI"] = symmetry_index(
                angles_left[joint], angles_right[joint]
            )
            metrics[f"{joint}_SR"] = symmetry_ratio(
                angles_left[joint], angles_right[joint]
            )
            metrics[f"{joint}_waveform_sym"] = waveform_symmetry(
                angles_left[joint], angles_right[joint]
            )

    if "hip_flexion" in angles_right and "knee_flexion" in angles_right:
        ankle = angles_right.get("ankle_dorsiflexion")
        events = detect_gait_events(
            angles_right["hip_flexion"], angles_right["knee_flexion"], fps,
            ankle_dorsiflexion=ankle,
        )
        metrics["cadence"] = events["cadence_steps_per_min"]
        metrics["stride_time_mean"] = events["stride_time_mean"]
        metrics["stride_time_CV"] = events["stride_time_cv"]
        metrics["n_strides"] = len(events["stride_times"])

        hs = events["heel_strikes"]
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

    if "hip_flexion" in angles_right:
        metrics["R_hip_CV"] = coefficient_of_variation(
            np.abs(angles_right["hip_flexion"])
        )
    if "hip_flexion" in angles_left:
        metrics["L_hip_CV"] = coefficient_of_variation(
            np.abs(angles_left["hip_flexion"])
        )

    for side_label, angles in [("R", angles_right), ("L", angles_left)]:
        if "pelvis_obliquity" in angles:
            metrics[f"{side_label}_pelvis_obliquity_ROM"] = rom(angles["pelvis_obliquity"])
        if "trunk_lateral_lean" in angles:
            metrics[f"{side_label}_trunk_lean_ROM"] = rom(angles["trunk_lateral_lean"])

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

    metrics["movement_quality_score"] = movement_quality_score(metrics)
    mqs_breakdown = mqs_domain_scores(metrics)
    for domain, score in mqs_breakdown.items():
        metrics[f"mqs_{domain}"] = score

    completeness = mqs_signal_completeness(metrics)
    for domain, frac in completeness.items():
        metrics[f"mqs_{domain}_completeness"] = frac
    metrics["mqs_overall_completeness"] = float(np.mean(list(completeness.values())))
    metrics["mqs_sufficient_evidence"] = float(
        mqs_sufficient_evidence(metrics)
    )

    return metrics


def _signal_score(value: float, optimal_low: float, optimal_high: float,
                  worst_low: float, worst_high: float) -> float:
    """Map a metric value to 0-100 based on clinical reference range.

    100 = within optimal range, linearly decreasing to 0 at worst bounds.
    """
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
    "sparc": (-2.0, -1.3, -6.0, -0.5),
    "symmetry": (0.0, 10.0, 0.0, 50.0),
    "stride_cv": (0.0, 4.0, 0.0, 20.0),
    "cadence": (90.0, 130.0, 40.0, 180.0),
    "stride_time": (0.8, 1.3, 0.3, 2.5),
    "crp_mad": (0.0, 15.0, 0.0, 60.0),
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
            if val is not None:
                lo, hi, wlo, whi = _SIGNAL_RANGES[key]
                kin_scores.append(_signal_score(val, lo, hi, wlo, whi))
        for metric_suffix, range_key in [("pelvis_obliquity_ROM", "pelvic_obliquity"),
                                         ("trunk_lean_ROM", "trunk_lean")]:
            val = metrics.get(f"{side}_{metric_suffix}")
            if val is not None:
                lo, hi, wlo, whi = _SIGNAL_RANGES[range_key]
                kin_scores.append(_signal_score(val, lo, hi, wlo, whi))
    domains["kinematics"] = float(np.mean(kin_scores)) if kin_scores else 50.0

    sm_scores = []
    for side in ["R", "L"]:
        val = metrics.get(f"{side}_hip_flexion_SPARC")
        if val is not None:
            lo, hi, wlo, whi = _SIGNAL_RANGES["sparc"]
            sm_scores.append(_signal_score(val, lo, hi, wlo, whi))
    domains["smoothness"] = float(np.mean(sm_scores)) if sm_scores else 50.0

    si_scores = []
    for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion",
                   "pelvis_obliquity"]:
        val = metrics.get(f"{joint}_SI")
        if val is not None:
            lo, hi, wlo, whi = _SIGNAL_RANGES["symmetry"]
            si_scores.append(_signal_score(val, lo, hi, wlo, whi))
    si_mean = float(np.mean(si_scores)) if si_scores else 50.0

    hip_ws = metrics.get("hip_flexion_waveform_sym")
    if si_scores and hip_ws is not None:
        domains["symmetry"] = min(si_mean, hip_ws)
    elif si_scores:
        domains["symmetry"] = si_mean
    else:
        domains["symmetry"] = 50.0

    hip_crp = metrics.get("hip_CRP_MAD")
    if hip_crp is not None:
        lo, hi, wlo, whi = _SIGNAL_RANGES["crp_mad"]
        domains["coordination"] = _signal_score(hip_crp, lo, hi, wlo, whi)
    else:
        domains["coordination"] = 50.0

    cv_val = metrics.get("stride_time_CV", float("nan"))
    lo, hi, wlo, whi = _SIGNAL_RANGES["stride_cv"]
    if np.isnan(cv_val):
        domains["variability"] = 50.0
    else:
        domains["variability"] = _signal_score(cv_val, lo, hi, wlo, whi)

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


def mqs_signal_completeness(metrics: dict) -> dict[str, float]:
    """Compute signal completeness per domain (0-1).

    Reports what fraction of expected signals were present. A completeness
    below 1.0 means the domain score is based on partial data.
    """
    completeness = {}

    kin_expected = 0
    kin_present = 0
    for side in ["R", "L"]:
        for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]:
            kin_expected += 1
            if metrics.get(f"{side}_{joint}_ROM") is not None:
                kin_present += 1
        for suffix in ["pelvis_obliquity_ROM", "trunk_lean_ROM"]:
            kin_expected += 1
            if metrics.get(f"{side}_{suffix}") is not None:
                kin_present += 1
    completeness["kinematics"] = kin_present / kin_expected if kin_expected else 0.0

    sm_present = sum(1 for s in ["R", "L"] if metrics.get(f"{s}_hip_flexion_SPARC") is not None)
    completeness["smoothness"] = sm_present / 2.0

    sy_present = sum(
        1 for j in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion",
                     "pelvis_obliquity"]
        if metrics.get(f"{j}_SI") is not None
    )
    completeness["symmetry"] = sy_present / 4.0

    completeness["coordination"] = 1.0 if metrics.get("hip_CRP_MAD") is not None else 0.0

    cv_val = metrics.get("stride_time_CV", float("nan"))
    completeness["variability"] = 0.0 if np.isnan(cv_val) else 1.0

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

    Factor = observed_fraction * mean_confidence (both clamped to [0,1]).
    Below 50% observed or 0.3 confidence, factor drops sharply.
    """
    obs = metrics.get("pose_observed_fraction")
    conf = metrics.get("pose_mean_confidence")
    if obs is None and conf is None:
        return 1.0
    obs = np.clip(obs if obs is not None else 1.0, 0, 1)
    conf = np.clip(conf if conf is not None else 1.0, 0, 1)
    return float(obs * conf)


def movement_quality_score(metrics: dict) -> float:
    """Compute composite Movement Quality Score (0-100).

    Weighted combination across 6 biomechanical domains:
    kinematics (25%), smoothness (18%), symmetry (18%),
    coordination (14%), variability (13%), temporal (12%).
    """
    domains = mqs_domain_scores(metrics)
    mqs = sum(domains[d] * _DOMAIN_WEIGHTS[d] for d in _DOMAIN_WEIGHTS)
    return float(np.clip(mqs, 0, 100))
