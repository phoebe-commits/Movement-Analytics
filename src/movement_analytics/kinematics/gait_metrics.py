"""Gait-level metrics computed from joint angle time series.

Implements clinically validated gait quality metrics including smoothness,
symmetry, variability, and temporal parameters. References:
  - Gait Deviation Index: Schwartz & Rozumalski, Gait & Posture, 2008
  - SPARC smoothness: Balasubramanian et al., IEEE TBME, 2012/2015
  - Symmetry Index: Robinson et al., Int J Rehabil Res, 1987
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


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


def rom(angles: np.ndarray) -> float:
    """Range of motion: max - min of angle time series."""
    return float(np.ptp(angles))


def detect_gait_events(hip_flexion: np.ndarray, knee_flexion: np.ndarray,
                       fps: float) -> dict:
    """Detect gait events (heel strikes, toe-offs) from joint angle signals.

    Uses hip flexion peaks (max flexion = ~heel strike) and
    knee flexion peaks in swing as proxy gait events.
    """
    b, a = butter(4, 6.0 / (fps / 2), btype="low")
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
        "stride_time_mean": float(np.mean(stride_times)) if len(stride_times) > 0 else 0.0,
        "stride_time_cv": coefficient_of_variation(stride_times) if len(stride_times) > 1 else 0.0,
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

    return metrics
