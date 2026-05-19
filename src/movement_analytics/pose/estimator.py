"""Pose estimation from video using MediaPipe PoseLandmarker (Tasks API).

Extracts 33 keypoints per frame and maps them to the joint position format
expected by the kinematics engine. Supports both file input and frame-by-frame
processing for real-time analysis.
"""

import os
import cv2
import numpy as np
import mediapipe as mp

_BaseOptions = mp.tasks.BaseOptions
_PoseLandmarker = mp.tasks.vision.PoseLandmarker
_PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode
_PoseLandmark = mp.tasks.vision.PoseLandmark

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
_DEFAULT_MODEL = os.path.join(_MODEL_DIR, "pose_landmarker_heavy.task")

# Landmark indices
LM = _PoseLandmark
LANDMARK_MAP = {
    "nose": LM.NOSE,
    "left_shoulder": LM.LEFT_SHOULDER,
    "right_shoulder": LM.RIGHT_SHOULDER,
    "left_elbow": LM.LEFT_ELBOW,
    "right_elbow": LM.RIGHT_ELBOW,
    "left_wrist": LM.LEFT_WRIST,
    "right_wrist": LM.RIGHT_WRIST,
    "left_hip": LM.LEFT_HIP,
    "right_hip": LM.RIGHT_HIP,
    "left_knee": LM.LEFT_KNEE,
    "right_knee": LM.RIGHT_KNEE,
    "left_ankle": LM.LEFT_ANKLE,
    "right_ankle": LM.RIGHT_ANKLE,
    "left_heel": LM.LEFT_HEEL,
    "right_heel": LM.RIGHT_HEEL,
    "left_foot_index": LM.LEFT_FOOT_INDEX,
    "right_foot_index": LM.RIGHT_FOOT_INDEX,
}


def _download_model(model_path: str):
    """Download the pose landmarker model if not present."""
    if os.path.exists(model_path):
        return
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/latest/"
        "pose_landmarker_heavy.task"
    )
    print(f"Downloading pose landmarker model ({url})...")
    import urllib.request
    urllib.request.urlretrieve(url, model_path)
    print(f"Model saved to {model_path}")


class PoseEstimator:
    """MediaPipe PoseLandmarker wrapper for gait analysis."""

    def __init__(self, model_path: str | None = None, num_poses: int = 1):
        path = model_path or _DEFAULT_MODEL
        _download_model(path)

        options = _PoseLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=path),
            running_mode=_RunningMode.IMAGE,
            num_poses=num_poses,
        )
        self.landmarker = _PoseLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray,
                      min_visibility: float = 0.5) -> tuple[dict | None, float]:
        """Extract keypoints from a single BGR frame.

        Returns (positions, confidence) where positions is a dict compatible
        with kinematics.joint_angles.compute_all_angles() or None if no
        person detected. Confidence is the mean visibility of key landmarks
        (0-1). Landmarks below min_visibility are excluded.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None, 0.0

        h, w = frame.shape[:2]
        landmarks = result.pose_landmarks[0]

        def lm_visibility(name: str) -> float:
            idx = LANDMARK_MAP[name].value
            return landmarks[idx].visibility

        def lm_to_px(name: str) -> np.ndarray | None:
            idx = LANDMARK_MAP[name].value
            lm = landmarks[idx]
            if lm.visibility < min_visibility:
                return None
            return np.array([lm.x * w, lm.y * h])

        def midpoint(name_a: str, name_b: str) -> np.ndarray | None:
            a = lm_to_px(name_a)
            b = lm_to_px(name_b)
            if a is None or b is None:
                return None
            return (a + b) / 2

        key_landmarks = ["left_hip", "right_hip", "left_knee", "right_knee",
                         "left_ankle", "right_ankle", "left_shoulder", "right_shoulder"]
        visibilities = [lm_visibility(n) for n in key_landmarks]
        confidence = float(np.mean(visibilities))

        pelvis = midpoint("left_hip", "right_hip")
        shoulder_center = midpoint("left_shoulder", "right_shoulder")
        head = lm_to_px("nose")

        if pelvis is None or shoulder_center is None:
            return None, confidence

        positions = {
            "pelvis": pelvis,
            "shoulder": shoulder_center,
            "neck": shoulder_center.copy(),
        }
        if head is not None:
            positions["head"] = head

        for side in ["left", "right"]:
            for joint, lm_name in [("hip", f"{side}_hip"), ("knee", f"{side}_knee"),
                                   ("ankle", f"{side}_ankle"), ("toe", f"{side}_foot_index"),
                                   ("shoulder", f"{side}_shoulder"), ("elbow", f"{side}_elbow"),
                                   ("wrist", f"{side}_wrist")]:
                pt = lm_to_px(lm_name)
                if pt is not None:
                    positions[f"{side}_{joint}"] = pt

        return positions, confidence

    def draw_landmarks(self, frame: np.ndarray, positions: dict,
                       skeleton_color: tuple = (0, 255, 200),
                       joint_color: tuple = (0, 180, 255)) -> np.ndarray:
        """Draw detected skeleton overlay on frame."""
        out = frame.copy()

        skeleton_links = [
            ("pelvis", "shoulder"), ("shoulder", "neck"), ("neck", "head"),
        ]
        for side in ["left", "right"]:
            skeleton_links.extend([
                (f"{side}_hip", f"{side}_knee"),
                (f"{side}_knee", f"{side}_ankle"),
                (f"{side}_ankle", f"{side}_toe"),
                (f"{side}_shoulder", f"{side}_elbow"),
                (f"{side}_elbow", f"{side}_wrist"),
                ("pelvis", f"{side}_hip"),
                ("shoulder", f"{side}_shoulder"),
            ])

        for a, b in skeleton_links:
            if a in positions and b in positions:
                pt_a = tuple(positions[a].astype(int))
                pt_b = tuple(positions[b].astype(int))
                cv2.line(out, pt_a, pt_b, skeleton_color, 2, cv2.LINE_AA)

        for name, pos in positions.items():
            pt = tuple(pos.astype(int))
            cv2.circle(out, pt, 4, joint_color, -1)

        return out

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def process_video(video_path: str, fps: float = None) -> tuple[list[np.ndarray], dict, dict, float]:
    """Process a video file and extract bilateral joint angle time series.

    Returns (frames, angles_right, angles_left, actual_fps) in the same format
    as generators.stick_figure.generate_frames() for pipeline compatibility.
    """
    from ..kinematics.joint_angles import compute_all_angles

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    actual_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    all_angles = []

    confidences = []

    with PoseEstimator() as estimator:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            positions, confidence = estimator.process_frame(frame)
            confidences.append(confidence)
            if positions is not None:
                annotated = estimator.draw_landmarks(frame, positions)
                angles = compute_all_angles(positions)
                frames.append(annotated)
                all_angles.append(angles)
            else:
                frames.append(frame)
                all_angles.append({})

    cap.release()

    if not all_angles:
        return frames, {}, {}, actual_fps

    angle_keys_right = set()
    angle_keys_left = set()
    for a in all_angles:
        for k in a:
            if k.startswith("right_"):
                angle_keys_right.add(k)
            elif k.startswith("left_"):
                angle_keys_left.add(k)

    n = len(all_angles)
    angles_right = {}
    angles_left = {}

    key_mapping = {
        "right_hip_flexion": "hip_flexion",
        "right_knee_flexion": "knee_flexion",
        "right_ankle_angle": "ankle_dorsiflexion",
        "right_elbow_flexion": "elbow_flexion",
        "left_hip_flexion": "hip_flexion",
        "left_knee_flexion": "knee_flexion",
        "left_ankle_angle": "ankle_dorsiflexion",
        "left_elbow_flexion": "elbow_flexion",
    }

    for orig_key in angle_keys_right:
        mapped = key_mapping.get(orig_key, orig_key.replace("right_", ""))
        arr = np.zeros(n)
        for i, a in enumerate(all_angles):
            arr[i] = a.get(orig_key, 0.0)
        angles_right[mapped] = arr

    for orig_key in angle_keys_left:
        mapped = key_mapping.get(orig_key, orig_key.replace("left_", ""))
        arr = np.zeros(n)
        for i, a in enumerate(all_angles):
            arr[i] = a.get(orig_key, 0.0)
        angles_left[mapped] = arr

    if "trunk_lean" in all_angles[0]:
        trunk = np.array([a.get("trunk_lean", 0.0) for a in all_angles])
        angles_right["pelvis_tilt"] = trunk
        angles_left["pelvis_tilt"] = trunk

    if "hip_flexion" in angles_right:
        from scipy.signal import butter, filtfilt
        hip = angles_right["hip_flexion"]
        try:
            b, a = butter(4, 6.0 / (actual_fps / 2), btype="low")
            hip_filt = filtfilt(b, a, hip)
        except Exception:
            hip_filt = hip
        hip_range = np.ptp(hip_filt)
        if hip_range > 1e-6:
            hip_norm = (hip_filt - hip_filt.min()) / hip_range
        else:
            hip_norm = np.zeros_like(hip_filt)
        angles_right["cycle_phase"] = hip_norm
        angles_left["cycle_phase"] = hip_norm

    return frames, angles_right, angles_left, actual_fps
