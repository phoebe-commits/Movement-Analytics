"""Movement Analytics — gait synthesis, pose estimation, and movement quality scoring."""

__version__ = "0.7.0"

from .generators.gait_model import GAIT_PROFILES, GaitParameters
from .generators.stick_figure import generate_frames
from .kinematics.gait_metrics import (
    compute_gait_summary,
    continuous_relative_phase,
    crp_consistency,
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

__all__ = [
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
]
