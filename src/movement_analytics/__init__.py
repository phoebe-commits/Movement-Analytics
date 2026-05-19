"""Movement Analytics — gait synthesis, pose estimation, and movement quality scoring."""

__version__ = "0.8.0"

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


def analyze_video(video_path: str, fps: float | None = None) -> dict:
    """Analyze a walking video and return movement quality metrics.

    High-level API: video in → MQS result out. Runs the full pipeline:
    MediaPipe pose estimation → joint angles → gait metrics → MQS scoring.

    Returns a dict containing movement_quality_score, domain scores,
    per-joint metrics, confidence factor, and pose quality metadata.
    """
    from .pose.estimator import process_video

    _, angles_right, angles_left, actual_fps, meta = process_video(
        video_path, fps=fps, store_frames=False,
    )
    summary = compute_gait_summary(
        angles_right, angles_left, fps=actual_fps, pose_metadata=meta,
    )
    summary["n_frames"] = meta.get("n_frames", 0)
    summary["fps"] = actual_fps
    return summary


_SAGITTAL_KEYS = {
    "hip_flexion", "knee_flexion", "ankle_dorsiflexion", "elbow_flexion",
}
_FRONTAL_KEYS = {
    "pelvis_obliquity", "pelvis_obliquity_signed", "trunk_lateral_lean",
    "pelvic_obliquity",
}


def analyze_multi_view(
    video_paths: list[str],
    view_labels: list[str] | None = None,
    fps: float | None = None,
) -> dict:
    """Analyze walking from multiple camera angles and merge into unified MQS.

    Merges sagittal-plane signals (hip/knee/ankle ROM) from the sagittal view
    with frontal-plane signals (pelvis obliquity, trunk lean) from the frontal
    view. When view_labels are not provided, the best-confidence view is used
    for each signal domain.

    Returns the same dict format as analyze_video, plus per-view metadata.
    """
    import numpy as np

    from .pose.estimator import process_video

    if not video_paths:
        raise ValueError("At least one video path required")

    if view_labels and len(view_labels) != len(video_paths):
        raise ValueError("view_labels must match video_paths length")

    views = []
    for i, path in enumerate(video_paths):
        _, ar, al, actual_fps, meta = process_video(
            path, fps=fps, store_frames=False,
        )
        label = view_labels[i] if view_labels else f"view_{i}"
        views.append({
            "label": label,
            "angles_right": ar,
            "angles_left": al,
            "fps": actual_fps,
            "meta": meta,
        })

    best_fps = views[0]["fps"]

    if view_labels:
        sag_idx = next(
            (i for i, label in enumerate(view_labels)
             if "sag" in label.lower() or "side" in label.lower()),
            0,
        )
        front_idx = next(
            (i for i, label in enumerate(view_labels)
             if "front" in label.lower() or "coronal" in label.lower()),
            sag_idx,
        )
    else:
        sag_idx = max(
            range(len(views)),
            key=lambda i: views[i]["meta"].get("mean_detected_confidence", 0),
        )
        front_idx = sag_idx

    merged_right: dict[str, np.ndarray] = {}
    merged_left: dict[str, np.ndarray] = {}

    sag = views[sag_idx]
    for key, arr in sag["angles_right"].items():
        merged_right[key] = arr
    for key, arr in sag["angles_left"].items():
        merged_left[key] = arr

    if front_idx != sag_idx:
        front = views[front_idx]
        for key in list(front["angles_right"].keys()):
            base = key.replace("right_", "").replace("left_", "")
            if any(fk in base for fk in _FRONTAL_KEYS):
                if key in front["angles_right"]:
                    merged_right[key] = front["angles_right"][key]
                if key in front["angles_left"]:
                    merged_left[key] = front["angles_left"][key]

    merged_meta = dict(sag["meta"])
    if front_idx != sag_idx:
        front_meta = views[front_idx]["meta"]
        merged_meta["frontal_observed_fraction"] = front_meta.get(
            "observed_fraction", 0,
        )
        merged_meta["frontal_mean_detected_confidence"] = front_meta.get(
            "mean_detected_confidence", 0,
        )

    summary = compute_gait_summary(
        merged_right, merged_left, fps=best_fps,
        pose_metadata=merged_meta,
    )
    summary["n_views"] = len(views)
    summary["view_labels"] = [v["label"] for v in views]
    summary["sagittal_view"] = views[sag_idx]["label"]
    if front_idx != sag_idx:
        summary["frontal_view"] = views[front_idx]["label"]
    for i, v in enumerate(views):
        summary[f"view_{i}_observed_fraction"] = v["meta"].get(
            "observed_fraction", 0,
        )
        summary[f"view_{i}_confidence"] = v["meta"].get(
            "mean_detected_confidence", 0,
        )

    return summary


__all__ = [
    "analyze_multi_view",
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
