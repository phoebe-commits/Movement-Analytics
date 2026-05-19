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
                       fps: float) -> dict:
    """Detect gait events (heel strikes, toe-offs) from joint angle signals.

    Uses hip flexion peaks (max flexion = ~heel strike) and
    knee flexion peaks in swing as proxy gait events.
    """
    nyq = fps / 2
    if nyq <= 6.0 or len(hip_flexion) < 13:
        hip_filt = hip_flexion
    else:
        b, a = butter(4, 6.0 / nyq, btype="low")
        hip_filt = filtfilt(b, a, hip_flexion)

    hs_indices, _ = find_peaks(hip_filt, distance=int(fps * 0.3))
    to_indices, _ = find_peaks(-hip_filt, distance=int(fps * 0.3))

    stride_times = np.diff(hs_indices) / fps if len(hs_indices) > 1 else np.array([])
    cadence = 60.0 / np.mean(stride_times) * 2 if len(stride_times) > 0 else 0.0

    return {
        "heel_strikes": hs_indices,
        "toe_offs": to_indices,
        "stride_times": stride_times,
        "cadence_steps_per_min": cadence,
        "stride_time_mean": float(np.mean(stride_times)) if len(stride_times) > 0 else float("nan"),
        "stride_time_cv": coefficient_of_variation(stride_times) if len(stride_times) > 2 else float("nan"),
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

    for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]:
        if joint in angles_right and joint in angles_left:
            metrics[f"{joint}_SI"] = symmetry_index(
                angles_left[joint], angles_right[joint]
            )
            metrics[f"{joint}_SR"] = symmetry_ratio(
                angles_left[joint], angles_right[joint]
            )

    if "hip_flexion" in angles_right and "knee_flexion" in angles_right:
        events = detect_gait_events(
            angles_right["hip_flexion"], angles_right["knee_flexion"], fps
        )
        metrics["cadence"] = events["cadence_steps_per_min"]
        metrics["stride_time_mean"] = events["stride_time_mean"]
        metrics["stride_time_CV"] = events["stride_time_cv"]
        metrics["n_strides"] = len(events["stride_times"])

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

    metrics["movement_quality_score"] = movement_quality_score(metrics)
    mqs_breakdown = mqs_domain_scores(metrics)
    for domain, score in mqs_breakdown.items():
        metrics[f"mqs_{domain}"] = score

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
            val = metrics.get(f"{side}_{joint}_ROM", 0)
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
        val = metrics.get(f"{side}_hip_flexion_SPARC", -3.0)
        lo, hi, wlo, whi = _SIGNAL_RANGES["sparc"]
        sm_scores.append(_signal_score(val, lo, hi, wlo, whi))
    domains["smoothness"] = float(np.mean(sm_scores)) if sm_scores else 50.0

    sy_scores = []
    for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]:
        val = metrics.get(f"{joint}_SI", 0)
        lo, hi, wlo, whi = _SIGNAL_RANGES["symmetry"]
        sy_scores.append(_signal_score(val, lo, hi, wlo, whi))
    domains["symmetry"] = float(np.mean(sy_scores)) if sy_scores else 50.0

    coord_scores = []
    for crp_key in ["hip_CRP_MAD", "knee_CRP_MAD"]:
        val = metrics.get(crp_key)
        if val is not None:
            lo, hi, wlo, whi = _SIGNAL_RANGES["crp_mad"]
            coord_scores.append(_signal_score(val, lo, hi, wlo, whi))
    domains["coordination"] = float(np.mean(coord_scores)) if coord_scores else 50.0

    cv_val = metrics.get("stride_time_CV", float("nan"))
    lo, hi, wlo, whi = _SIGNAL_RANGES["stride_cv"]
    if np.isnan(cv_val):
        domains["variability"] = 50.0
    else:
        domains["variability"] = _signal_score(cv_val, lo, hi, wlo, whi)

    t_scores = []
    cad = metrics.get("cadence", 0)
    if not np.isnan(cad):
        lo, hi, wlo, whi = _SIGNAL_RANGES["cadence"]
        t_scores.append(_signal_score(cad, lo, hi, wlo, whi))
    st = metrics.get("stride_time_mean", float("nan"))
    if not np.isnan(st):
        lo, hi, wlo, whi = _SIGNAL_RANGES["stride_time"]
        t_scores.append(_signal_score(st, lo, hi, wlo, whi))
    domains["temporal"] = float(np.mean(t_scores)) if t_scores else 50.0

    return domains


def movement_quality_score(metrics: dict) -> float:
    """Compute composite Movement Quality Score (0-100).

    Weighted combination across 5 biomechanical domains:
    kinematics (30%), smoothness (20%), symmetry (20%),
    variability (15%), temporal (15%).
    """
    domains = mqs_domain_scores(metrics)
    mqs = sum(domains[d] * _DOMAIN_WEIGHTS[d] for d in _DOMAIN_WEIGHTS)
    return float(np.clip(mqs, 0, 100))
