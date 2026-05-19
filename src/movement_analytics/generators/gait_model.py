"""Procedural biomechanical gait model.

Generates realistic 2D sagittal-plane joint angle trajectories for a full gait cycle
using sinusoidal approximations of clinical gait analysis norms (Perry & Burnfield, 2010;
Winter, 2009). All angles in degrees, following ISB convention (flexion positive).
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class GaitParameters:
    """Configurable gait parameters controlling the walking pattern."""

    cadence: float = 110.0           # steps/min (normal adult ~100-120)
    stride_length: float = 1.4       # meters (normal adult ~1.2-1.6)
    hip_rom: float = 40.0            # hip flexion/extension range of motion (degrees)
    knee_rom: float = 60.0           # knee flexion ROM in swing (degrees)
    ankle_rom: float = 30.0          # ankle plantarflexion/dorsiflexion ROM (degrees)
    pelvic_tilt: float = 4.0         # pelvic anterior tilt oscillation (degrees)
    pelvic_obliquity: float = 5.0    # pelvic drop amplitude (degrees)
    trunk_sway: float = 3.0          # lateral trunk sway (degrees)
    arm_swing: float = 25.0          # shoulder flexion/extension ROM (degrees)
    stance_ratio: float = 0.60       # fraction of cycle in stance (normal ~0.60)
    speed_factor: float = 1.0        # multiplier on overall speed
    asymmetry: float = 0.0           # left-right asymmetry factor (0 = symmetric, 0.3 = noticeable limp)
    noise_level: float = 0.0         # motor noise / variability (degrees std)


@dataclass
class GaitProfile:
    """Named gait profile with descriptive metadata."""

    name: str
    params: GaitParameters
    description: str = ""


GAIT_PROFILES = {
    "normal": GaitProfile(
        name="Normal Adult Walk",
        params=GaitParameters(),
        description="Healthy adult gait at self-selected comfortable speed (~1.2 m/s)",
    ),
    "slow": GaitProfile(
        name="Slow Cautious Walk",
        params=GaitParameters(
            cadence=80, stride_length=0.9, hip_rom=30, knee_rom=45,
            ankle_rom=20, arm_swing=15, speed_factor=0.65,
        ),
        description="Slow, cautious gait pattern typical of elderly or post-injury",
    ),
    "fast": GaitProfile(
        name="Fast Walk",
        params=GaitParameters(
            cadence=135, stride_length=1.7, hip_rom=50, knee_rom=70,
            ankle_rom=35, arm_swing=35, speed_factor=1.4,
        ),
        description="Fast-paced walking approaching jog transition",
    ),
    "limp": GaitProfile(
        name="Asymmetric / Limping Gait",
        params=GaitParameters(
            cadence=90, stride_length=1.0, hip_rom=35, knee_rom=50,
            ankle_rom=25, arm_swing=18, asymmetry=0.35, speed_factor=0.75,
        ),
        description="Asymmetric gait with reduced stance on affected side",
    ),
    "stiff_knee": GaitProfile(
        name="Stiff Knee Gait",
        params=GaitParameters(
            cadence=95, stride_length=1.0, hip_rom=30, knee_rom=25,
            ankle_rom=20, arm_swing=15, speed_factor=0.7,
        ),
        description="Reduced knee flexion in swing, common in spastic gait",
    ),
    "trendelenburg": GaitProfile(
        name="Trendelenburg Gait",
        params=GaitParameters(
            cadence=95, stride_length=1.0, hip_rom=35, knee_rom=55,
            pelvic_obliquity=12, trunk_sway=8, arm_swing=20, speed_factor=0.8,
        ),
        description="Excessive pelvic drop and trunk lean, weak hip abductors",
    ),
    "model_runway": GaitProfile(
        name="Fashion Model Runway Walk",
        params=GaitParameters(
            cadence=105, stride_length=1.5, hip_rom=45, knee_rom=55,
            ankle_rom=32, pelvic_tilt=6, pelvic_obliquity=7, trunk_sway=2,
            arm_swing=20, speed_factor=0.95, noise_level=0.5,
        ),
        description="Trained runway gait: exaggerated pelvic motion, controlled trunk, precise foot placement",
    ),
    "noisy": GaitProfile(
        name="High-Variability Walk",
        params=GaitParameters(noise_level=4.0),
        description="Normal gait with high motor noise / variability added",
    ),
    "parkinsonian": GaitProfile(
        name="Parkinsonian Gait",
        params=GaitParameters(
            cadence=100, stride_length=0.7, hip_rom=20, knee_rom=30,
            ankle_rom=15, pelvic_tilt=2, pelvic_obliquity=3,
            trunk_sway=1.5, arm_swing=8, speed_factor=0.55,
            asymmetry=0.15, noise_level=1.5,
        ),
        description="Shuffling gait: reduced ROM, diminished arm swing, shortened stride, mild asymmetry (Plotnik et al., 2005)",
    ),
}


def generate_gait_cycle(params: GaitParameters, n_frames: int = 120,
                        n_cycles: int = 3, side: str = "right") -> dict[str, np.ndarray]:
    """Generate joint angle trajectories over multiple gait cycles.

    Returns dict mapping joint names to arrays of shape (n_frames * n_cycles,).
    Angles in degrees, following biomechanical sign conventions:
      - Hip: flexion (+), extension (-)
      - Knee: flexion (+), extension (0)
      - Ankle: dorsiflexion (+), plantarflexion (-)
      - Pelvis tilt: anterior (+)
      - Pelvis obliquity: contralateral drop (+)
      - Shoulder: flexion (+), extension (-)
    """
    total_frames = n_frames * n_cycles
    t = np.linspace(0, n_cycles, total_frames, endpoint=False)
    phase = 2 * np.pi * t

    asym = params.asymmetry if side == "right" else -params.asymmetry

    hip_flex_amp = params.hip_rom / 2
    hip_offset = -5  # slight extension bias at midstance
    hip = hip_offset + hip_flex_amp * np.sin(phase - 0.3) * (1 + asym * 0.3)

    stance_frac = params.stance_ratio
    knee_stance_flex = 15  # loading response knee flexion
    knee = np.zeros(total_frames)
    cycle_phase = t % 1.0
    for i in range(total_frames):
        cp = cycle_phase[i]
        if cp < stance_frac * 0.3:
            # loading response: 0 -> knee_stance_flex
            knee[i] = knee_stance_flex * np.sin(np.pi * cp / (stance_frac * 0.3))
        elif cp < stance_frac:
            # mid/terminal stance: knee_stance_flex -> 0
            progress = (cp - stance_frac * 0.3) / (stance_frac * 0.7)
            knee[i] = knee_stance_flex * (1 - progress) * 0.5
        else:
            # swing: large flexion then extension
            swing_progress = (cp - stance_frac) / (1 - stance_frac)
            knee[i] = params.knee_rom * np.sin(np.pi * swing_progress)
    knee *= (1 + asym * 0.2)

    ankle = np.zeros(total_frames)
    for i in range(total_frames):
        cp = cycle_phase[i]
        if cp < stance_frac * 0.15:
            # initial contact: slight plantarflexion
            ankle[i] = -5 * np.sin(np.pi * cp / (stance_frac * 0.15))
        elif cp < stance_frac * 0.5:
            # loading to midstance: dorsiflexion
            progress = (cp - stance_frac * 0.15) / (stance_frac * 0.35)
            ankle[i] = -5 + 15 * progress
        elif cp < stance_frac:
            # terminal stance: plantarflexion (push-off)
            progress = (cp - stance_frac * 0.5) / (stance_frac * 0.5)
            ankle[i] = 10 - (10 + params.ankle_rom * 0.5) * progress
        else:
            # swing: dorsiflexion for clearance
            swing_progress = (cp - stance_frac) / (1 - stance_frac)
            ankle[i] = -params.ankle_rom * 0.3 + 10 * np.sin(np.pi * swing_progress)

    pelvis_tilt = params.pelvic_tilt * np.sin(2 * phase)
    pelvis_obliq = params.pelvic_obliquity * np.sin(phase)
    shoulder = -params.arm_swing / 2 * np.sin(phase - 0.1)
    elbow = 10 + 15 * np.abs(np.sin(phase))
    trunk_lateral = params.trunk_sway * np.sin(phase)

    if params.noise_level > 0:
        rng = np.random.default_rng(42)
        for arr in [hip, knee, ankle, pelvis_tilt, pelvis_obliq, shoulder, elbow, trunk_lateral]:
            arr += rng.normal(0, params.noise_level, total_frames)

    knee = np.clip(knee, 0, None)

    return {
        "hip_flexion": hip,
        "knee_flexion": knee,
        "ankle_dorsiflexion": ankle,
        "pelvis_tilt": pelvis_tilt,
        "pelvis_obliquity": pelvis_obliq,
        "shoulder_flexion": shoulder,
        "elbow_flexion": elbow,
        "trunk_lateral_lean": trunk_lateral,
        "cycle_phase": cycle_phase,
        "time_normalized": t,
    }
