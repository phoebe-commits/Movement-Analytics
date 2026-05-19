"""Movement Analytics — gait synthesis, pose estimation, and movement quality scoring."""

__version__ = "0.7.0"

from .generators.gait_model import GAIT_PROFILES, GaitParameters
from .generators.stick_figure import generate_frames
from .kinematics.gait_metrics import (
    compute_gait_summary,
    continuous_relative_phase,
    crp_consistency,
    dfa_scaling_exponent,
    gait_deviation_index,
    movement_quality_score,
    mqs_confidence_factor,
    mqs_domain_scores,
    mqs_signal_completeness,
    mqs_sufficient_evidence,
    rom,
    sparc,
    stride_pelvic_asymmetry,
    symmetry_index,
    waveform_symmetry,
)


def analyze_video(video_path: str, fps: float = None) -> dict:
    """Analyze a walking video and return movement quality metrics.

    High-level API: video in → MQS result out. Runs the full pipeline:
    MediaPipe pose estimation → joint angles → gait metrics → MQS scoring.

    Returns a dict containing movement_quality_score, domain scores,
    per-joint metrics, confidence factor, and pose quality metadata.
    """
    from .pose.estimator import process_video

    frames, angles_right, angles_left, actual_fps, meta = process_video(
        video_path, fps=fps,
    )
    summary = compute_gait_summary(
        angles_right, angles_left, fps=actual_fps, pose_metadata=meta,
    )
    summary["n_frames"] = len(frames)
    summary["fps"] = actual_fps
    return summary


__all__ = [
    "analyze_video",
    "compute_gait_summary",
    "movement_quality_score",
    "mqs_domain_scores",
    "mqs_signal_completeness",
    "sparc",
    "symmetry_index",
    "rom",
    "continuous_relative_phase",
    "crp_consistency",
    "GaitParameters",
    "GAIT_PROFILES",
    "generate_frames",
    "waveform_symmetry",
    "mqs_confidence_factor",
    "mqs_sufficient_evidence",
    "stride_pelvic_asymmetry",
    "gait_deviation_index",
    "dfa_scaling_exponent",
]
