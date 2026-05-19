"""Stick-figure renderer for synthetic gait visualization.

Converts joint angle trajectories from the gait model into a 2D sagittal-plane
stick figure animation rendered with OpenCV. The figure walks across the screen
with biomechanically accurate segment proportions (Winter, 2009).
"""

import cv2
import numpy as np
from .gait_model import GaitParameters, generate_gait_cycle

# Segment lengths as fraction of total height (Winter, 2009 anthropometric data)
SEGMENT_RATIOS = {
    "head_radius": 0.065,
    "neck": 0.05,
    "trunk": 0.30,
    "upper_arm": 0.145,
    "forearm": 0.120,
    "thigh": 0.245,
    "shank": 0.246,
    "foot": 0.072,
}


def _joint_positions(angles_right: dict, angles_left: dict, frame_idx: int,
                     figure_height: float, hip_x: float, ground_y: float) -> dict:
    """Compute 2D joint positions from angle data at a single frame."""
    def deg2rad(d):
        return np.radians(d)

    pelvis_y = ground_y - (SEGMENT_RATIOS["thigh"] + SEGMENT_RATIOS["shank"] +
                           SEGMENT_RATIOS["foot"] * 0.5) * figure_height
    pelvis_tilt = angles_right["pelvis_tilt"][frame_idx]
    trunk_len = SEGMENT_RATIOS["trunk"] * figure_height
    neck_len = SEGMENT_RATIOS["neck"] * figure_height
    head_r = SEGMENT_RATIOS["head_radius"] * figure_height

    pelvis = np.array([hip_x, pelvis_y])

    trunk_angle = deg2rad(90 + pelvis_tilt)
    shoulder = pelvis + trunk_len * np.array([-np.cos(trunk_angle), -np.sin(trunk_angle)])
    neck = shoulder + neck_len * np.array([0, -1])
    head_center = neck + head_r * np.array([0, -1])

    positions = {"pelvis": pelvis, "shoulder": shoulder, "neck": neck, "head": head_center}

    for side, angles in [("right", angles_right), ("left", angles_left)]:
        thigh_len = SEGMENT_RATIOS["thigh"] * figure_height
        shank_len = SEGMENT_RATIOS["shank"] * figure_height
        foot_len = SEGMENT_RATIOS["foot"] * figure_height
        upper_arm_len = SEGMENT_RATIOS["upper_arm"] * figure_height
        forearm_len = SEGMENT_RATIOS["forearm"] * figure_height

        hip_ang = deg2rad(-angles["hip_flexion"][frame_idx])
        knee_ang = deg2rad(angles["knee_flexion"][frame_idx])
        ankle_ang = deg2rad(angles["ankle_dorsiflexion"][frame_idx])
        shoulder_ang = deg2rad(-angles["shoulder_flexion"][frame_idx])
        elbow_ang = deg2rad(angles["elbow_flexion"][frame_idx])

        abs_thigh = -np.pi / 2 + hip_ang
        knee_pos = pelvis + thigh_len * np.array([np.cos(abs_thigh), -np.sin(abs_thigh)])

        abs_shank = abs_thigh + knee_ang
        ankle_pos = knee_pos + shank_len * np.array([np.cos(abs_shank), -np.sin(abs_shank)])

        abs_foot = ankle_ang * 0.5
        toe_pos = ankle_pos + foot_len * np.array([np.cos(abs_foot), np.sin(abs_foot) * 0.3])

        abs_upper_arm = -np.pi / 2 + shoulder_ang
        elbow_pos = shoulder + upper_arm_len * np.array(
            [np.cos(abs_upper_arm), -np.sin(abs_upper_arm)]
        )

        abs_forearm = abs_upper_arm + elbow_ang * 0.5
        wrist_pos = elbow_pos + forearm_len * np.array(
            [np.cos(abs_forearm), -np.sin(abs_forearm)]
        )

        positions[f"{side}_hip"] = pelvis.copy()
        positions[f"{side}_knee"] = knee_pos
        positions[f"{side}_ankle"] = ankle_pos
        positions[f"{side}_toe"] = toe_pos
        positions[f"{side}_shoulder"] = shoulder.copy()
        positions[f"{side}_elbow"] = elbow_pos
        positions[f"{side}_wrist"] = wrist_pos

    return positions


def render_walking_video(params: GaitParameters, output_path: str | None = None,
                         width: int = 1280, height: int = 720, fps: int = 30,
                         n_cycles: int = 4, figure_height_px: float = 400,
                         show_joint_markers: bool = True,
                         background_color: tuple = (20, 20, 25)) -> str | None:
    """Render a synthetic walking stick-figure video.

    If output_path is provided, writes an MP4 file and returns the path.
    If output_path is None, returns None (use generate_frames() for real-time).
    """
    frames_per_cycle = fps * 60 // int(params.cadence * params.speed_factor) * 2
    frames_per_cycle = max(frames_per_cycle, 20)

    angles_right = generate_gait_cycle(params, n_frames=frames_per_cycle,
                                       n_cycles=n_cycles, side="right")
    angles_left = generate_gait_cycle(params, n_frames=frames_per_cycle,
                                      n_cycles=n_cycles, side="left")

    # Left side is half-cycle offset
    total = frames_per_cycle * n_cycles
    half_cycle = frames_per_cycle // 2
    for key in angles_left:
        if key not in ("cycle_phase", "time_normalized"):
            angles_left[key] = np.roll(angles_left[key], half_cycle)

    ground_y = int(height * 0.82)
    stride_px = params.stride_length * figure_height_px / 1.75
    speed_px_per_frame = stride_px * params.cadence * params.speed_factor / (60 * fps)

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        writer = None

    frames = []
    for i in range(total):
        frame = np.full((height, width, 3), background_color, dtype=np.uint8)

        cv2.line(frame, (0, ground_y), (width, ground_y), (60, 60, 65), 2)
        for gx in range(0, width, 80):
            cv2.line(frame, (gx, ground_y), (gx, ground_y + 5), (50, 50, 55), 1)

        hip_x = width * 0.35 + speed_px_per_frame * (i % frames_per_cycle)
        positions = _joint_positions(angles_right, angles_left, i,
                                     figure_height_px, hip_x, ground_y)

        # Draw segments
        bone_color_back = (100, 120, 140)
        bone_color_front = (180, 200, 220)
        joint_color = (0, 180, 255)
        head_color = (180, 200, 220)

        # Back leg first (dimmer)
        back_side = "left" if angles_right["hip_flexion"][i] > angles_left["hip_flexion"][i] else "right"
        front_side = "right" if back_side == "left" else "left"

        for side, color, thickness in [(back_side, bone_color_back, 3), (front_side, bone_color_front, 4)]:
            pts = [positions[f"{side}_hip"], positions[f"{side}_knee"],
                   positions[f"{side}_ankle"], positions[f"{side}_toe"]]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness)

            arm_pts = [positions[f"{side}_shoulder"], positions[f"{side}_elbow"],
                       positions[f"{side}_wrist"]]
            for a, b in zip(arm_pts[:-1], arm_pts[1:]):
                cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness)

        # Trunk
        cv2.line(frame, tuple(positions["pelvis"].astype(int)),
                 tuple(positions["shoulder"].astype(int)), bone_color_front, 4)
        # Neck
        cv2.line(frame, tuple(positions["shoulder"].astype(int)),
                 tuple(positions["neck"].astype(int)), bone_color_front, 3)
        # Head
        head_r = int(SEGMENT_RATIOS["head_radius"] * figure_height_px)
        cv2.circle(frame, tuple(positions["head"].astype(int)), head_r, head_color, 2)

        if show_joint_markers:
            for key, pos in positions.items():
                if key in ("head", "neck"):
                    continue
                cv2.circle(frame, tuple(pos.astype(int)), 5, joint_color, -1)

        # Frame info overlay
        cycle_num = i // frames_per_cycle + 1
        cycle_pct = (i % frames_per_cycle) / frames_per_cycle * 100
        phase = "Stance" if angles_right["cycle_phase"][i] < params.stance_ratio else "Swing"
        info_text = f"Cycle {cycle_num} | {cycle_pct:.0f}% | {phase} | Frame {i+1}/{total}"
        cv2.putText(frame, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (150, 150, 160), 1, cv2.LINE_AA)

        if writer:
            writer.write(frame)
        frames.append(frame)

    if writer:
        writer.release()

    return output_path


def generate_frames(params: GaitParameters, width: int = 1280, height: int = 720,
                    fps: int = 30, n_cycles: int = 4,
                    figure_height_px: float = 400) -> tuple[list[np.ndarray], dict, dict, GaitParameters]:
    """Generate frames and return (frames, angles_right, angles_left, params).

    This is the primary interface for the analysis pipeline — it returns
    both the rendered frames and the ground-truth joint angles.
    """
    frames_per_cycle = fps * 60 // int(params.cadence * params.speed_factor) * 2
    frames_per_cycle = max(frames_per_cycle, 20)

    angles_right = generate_gait_cycle(params, n_frames=frames_per_cycle,
                                       n_cycles=n_cycles, side="right")
    angles_left = generate_gait_cycle(params, n_frames=frames_per_cycle,
                                      n_cycles=n_cycles, side="left")

    total = frames_per_cycle * n_cycles
    half_cycle = frames_per_cycle // 2
    for key in angles_left:
        if key not in ("cycle_phase", "time_normalized"):
            angles_left[key] = np.roll(angles_left[key], half_cycle)

    ground_y = int(height * 0.82)
    stride_px = params.stride_length * figure_height_px / 1.75
    speed_px_per_frame = stride_px * params.cadence * params.speed_factor / (60 * fps)

    # Normal ROM ranges for color-coding joints
    _NORMAL_RANGES = {
        "hip_flexion": (35, 50),
        "knee_flexion": (50, 70),
        "ankle_dorsiflexion": (20, 35),
    }

    frames = []
    for i in range(total):
        frame = np.full((height, width, 3), (20, 20, 25), dtype=np.uint8)

        # Ground with subtle gradient
        cv2.line(frame, (0, ground_y), (width, ground_y), (60, 60, 65), 2)
        for gx in range(0, width, 80):
            cv2.line(frame, (gx, ground_y), (gx, ground_y + 5), (50, 50, 55), 1)
        for gy_off in range(1, 40):
            alpha = max(0, 25 - gy_off)
            cv2.line(frame, (0, ground_y + gy_off), (width, ground_y + gy_off),
                     (20 + alpha // 3, 20 + alpha // 3, 25 + alpha // 4), 1)

        hip_x = width * 0.35 + speed_px_per_frame * (i % frames_per_cycle)
        positions = _joint_positions(angles_right, angles_left, i,
                                     figure_height_px, hip_x, ground_y)

        bone_color_back = (100, 120, 140)
        bone_color_front = (180, 200, 220)

        back_side = "left" if angles_right["hip_flexion"][i] > angles_left["hip_flexion"][i] else "right"
        front_side = "right" if back_side == "left" else "left"

        for side, color, thickness in [(back_side, bone_color_back, 3), (front_side, bone_color_front, 4)]:
            pts = [positions[f"{side}_hip"], positions[f"{side}_knee"],
                   positions[f"{side}_ankle"], positions[f"{side}_toe"]]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness)
            arm_pts = [positions[f"{side}_shoulder"], positions[f"{side}_elbow"],
                       positions[f"{side}_wrist"]]
            for a, b in zip(arm_pts[:-1], arm_pts[1:]):
                cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness)

        cv2.line(frame, tuple(positions["pelvis"].astype(int)),
                 tuple(positions["shoulder"].astype(int)), bone_color_front, 4)
        cv2.line(frame, tuple(positions["shoulder"].astype(int)),
                 tuple(positions["neck"].astype(int)), bone_color_front, 3)
        head_r = int(SEGMENT_RATIOS["head_radius"] * figure_height_px)
        cv2.circle(frame, tuple(positions["head"].astype(int)), head_r, (180, 200, 220), 2)

        # Color-coded joint markers with real-time angle labels
        front_angles = angles_right if front_side == "right" else angles_left
        joint_angle_map = {
            f"{front_side}_hip": ("hip_flexion", front_angles["hip_flexion"][i]),
            f"{front_side}_knee": ("knee_flexion", front_angles["knee_flexion"][i]),
            f"{front_side}_ankle": ("ankle_dorsiflexion", front_angles["ankle_dorsiflexion"][i]),
        }

        for key, pos in positions.items():
            if key in ("head", "neck"):
                continue
            pt = tuple(pos.astype(int))

            if key in joint_angle_map:
                joint_name, angle_val = joint_angle_map[key]
                lo, hi = _NORMAL_RANGES.get(joint_name, (0, 180))
                abs_angle = abs(angle_val)
                if lo <= abs_angle <= hi:
                    jcolor = (80, 200, 80)
                else:
                    jcolor = (50, 200, 255) if abs(abs_angle - lo) < 10 or abs(abs_angle - hi) < 10 else (60, 60, 255)
                cv2.circle(frame, pt, 7, jcolor, -1)
                cv2.circle(frame, pt, 7, (255, 255, 255), 1)

                label_text = f"{angle_val:.0f}"
                cv2.putText(frame, label_text, (pt[0] + 10, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, jcolor, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 5, (0, 180, 255), -1)

        frames.append(frame)

    return frames, angles_right, angles_left, params
