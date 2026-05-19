"""Movement Analytics — gait synthesis, pose estimation, kinematic analysis, and movement quality scoring."""

__version__ = "0.2.0"

from .kinematics.gait_metrics import (
    compute_gait_summary,
    movement_quality_score,
    mqs_domain_scores,
    sparc,
    symmetry_index,
    rom,
    continuous_relative_phase,
    crp_consistency,
)
from .generators.gait_model import GaitParameters, GAIT_PROFILES
from .generators.stick_figure import generate_frames

__all__ = [
    "compute_gait_summary",
    "movement_quality_score",
    "mqs_domain_scores",
    "sparc",
    "symmetry_index",
    "rom",
    "continuous_relative_phase",
    "crp_consistency",
    "GaitParameters",
    "GAIT_PROFILES",
    "generate_frames",
]
