"""Joint angle computation from 2D keypoint positions.

Computes sagittal-plane joint angles using vector geometry.
Compatible with both ground-truth (synthetic) and estimated (pose) keypoints.
"""

import numpy as np


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two 2D vectors in degrees."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def compute_joint_angle(p_proximal: np.ndarray, p_joint: np.ndarray,
                        p_distal: np.ndarray) -> float:
    """Compute the included angle at a joint from three 2D points.

    Returns the angle in degrees (0 = fully flexed/folded, 180 = fully extended).
    """
    v1 = p_proximal - p_joint
    v2 = p_distal - p_joint
    return angle_between_vectors(v1, v2)


def compute_flexion_angle(p_proximal: np.ndarray, p_joint: np.ndarray,
                          p_distal: np.ndarray) -> float:
    """Compute flexion angle: 180 - included angle.

    Returns degrees of flexion (0 = extended, higher = more flexed).
    """
    return 180.0 - compute_joint_angle(p_proximal, p_joint, p_distal)


def segment_angle_to_vertical(p_proximal: np.ndarray, p_distal: np.ndarray) -> float:
    """Angle of a segment relative to vertical (downward = 0 degrees).

    Positive = anterior/forward lean, Negative = posterior.
    """
    seg = p_distal - p_proximal
    vertical = np.array([0, 1])  # downward in image coords
    angle = np.degrees(np.arctan2(seg[0], seg[1]))
    return angle


def compute_all_angles(positions: dict) -> dict[str, float]:
    """Compute all joint angles from a positions dict (as produced by stick_figure).

    Returns dict of angle names to values in degrees.
    """
    angles = {}

    for side in ["right", "left"]:
        hip = positions.get(f"{side}_hip")
        knee = positions.get(f"{side}_knee")
        ankle = positions.get(f"{side}_ankle")
        toe = positions.get(f"{side}_toe")
        shoulder = positions.get(f"{side}_shoulder", positions.get("shoulder"))
        elbow = positions.get(f"{side}_elbow")
        wrist = positions.get(f"{side}_wrist")

        if hip is not None and knee is not None and ankle is not None:
            angles[f"{side}_knee_flexion"] = compute_flexion_angle(hip, knee, ankle)

        if positions.get("shoulder") is not None and hip is not None and knee is not None:
            angles[f"{side}_hip_flexion"] = compute_flexion_angle(
                positions["shoulder"], hip, knee
            )

        if knee is not None and ankle is not None and toe is not None:
            angles[f"{side}_ankle_angle"] = compute_joint_angle(knee, ankle, toe)

        if shoulder is not None and elbow is not None and wrist is not None:
            angles[f"{side}_elbow_flexion"] = compute_flexion_angle(shoulder, elbow, wrist)

    if positions.get("pelvis") is not None and positions.get("shoulder") is not None:
        angles["trunk_lean"] = segment_angle_to_vertical(
            positions["pelvis"], positions["shoulder"]
        )

    return angles
