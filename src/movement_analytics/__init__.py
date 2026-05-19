"""Movement Analytics — gait synthesis, pose estimation, and movement quality scoring."""

__version__ = "0.2.0"

from .generators.gait_model import GAIT_PROFILES, GaitParameters
from .generators.stick_figure import generate_frames
from .kinematics.gait_metrics import (
    compute_gait_summary,
    continuous_relative_phase,
    crp_consistency,
    movement_quality_score,
    mqs_domain_scores,
    rom,
    sparc,
    symmetry_index,
)

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
