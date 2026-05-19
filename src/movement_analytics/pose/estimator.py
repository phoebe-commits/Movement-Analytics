"""Pose estimation from video using MediaPipe PoseLandmarker (Tasks API).

Extracts 33 keypoints per frame and maps them to the joint position format
expected by the kinematics engine. Supports both file input and frame-by-frame
processing for real-time analysis.
"""

import os

import cv2
import mediapipe as mp
import numpy as np

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

    def __init__(self, model_path: str | None = None, num_poses: int = 1,
                 video_mode: bool = False, fps: float = 30.0):
        path = model_path or _DEFAULT_MODEL
        _download_model(path)

        mode = _RunningMode.VIDEO if video_mode else _RunningMode.IMAGE
        options = _PoseLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=path),
            running_mode=mode,
            num_poses=num_poses,
        )
        self.landmarker = _PoseLandmarker.create_from_options(options)
        self._video_mode = video_mode
        self._frame_count = 0
        self._ms_per_frame = 1000.0 / fps

    def process_frame(self, frame: np.ndarray,
                      min_visibility: float = 0.5) -> tuple[dict | None, float]:
        """Extract keypoints from a single BGR frame.

        Returns (positions, confidence) where positions is a dict compatible
        with kinematics.joint_angles.compute_all_angles() or None if no
        person detected. Confidence is the mean visibility of key landmarks
        (0-1). Landmarks below min_visibility are excluded.

        In VIDEO mode, timestamps are tracked automatically. Frames must
        be passed in chronological order.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video_mode:
            self._frame_count += 1
            timestamp_ms = int(self._frame_count * self._ms_per_frame)
            result = self.landmarker.detect_for_video(
                mp_image, timestamp_ms
            )
        else:
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
                                   ("ankle", f"{side}_ankle"), ("heel", f"{side}_heel"),
                                   ("toe", f"{side}_foot_index"),
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
                (f"{side}_ankle", f"{side}_heel"),
                (f"{side}_heel", f"{side}_toe"),
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


def process_video(
    video_path: str, fps: float = None,
) -> tuple[list[np.ndarray], dict, dict, float, dict]:
    """Process a video file and extract bilateral joint angle time series.

    Returns (frames, angles_right, angles_left, actual_fps, metadata) where
    metadata contains pose quality stats: mean_confidence, observed_fraction,
    interpolation_fractions (per-key dict), and per-frame confidences.
    """
    from ..kinematics.joint_angles import compute_all_angles

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    actual_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    all_angles = []

    confidences = []

    with PoseEstimator(video_mode=True, fps=actual_fps) as estimator:
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
        return frames, {}, {}, actual_fps, {
            "mean_confidence": 0.0, "observed_fraction": 0.0,
            "interpolation_fractions": {}, "confidences": [],
        }

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
        "right_ankle_dorsiflexion": "ankle_dorsiflexion",
        "right_elbow_flexion": "elbow_flexion",
        "left_hip_flexion": "hip_flexion",
        "left_knee_flexion": "knee_flexion",
        "left_ankle_dorsiflexion": "ankle_dorsiflexion",
        "left_elbow_flexion": "elbow_flexion",
    }

    for orig_key in angle_keys_right:
        mapped = key_mapping.get(orig_key, orig_key.replace("right_", ""))
        arr = np.full(n, np.nan)
        for i, a in enumerate(all_angles):
            if orig_key in a:
                arr[i] = a[orig_key]
        angles_right[mapped] = arr

    for orig_key in angle_keys_left:
        mapped = key_mapping.get(orig_key, orig_key.replace("left_", ""))
        arr = np.full(n, np.nan)
        for i, a in enumerate(all_angles):
            if orig_key in a:
                arr[i] = a[orig_key]
        angles_left[mapped] = arr

    if any("trunk_lean" in a for a in all_angles):
        trunk = np.array([a.get("trunk_lean", np.nan) for a in all_angles])
        angles_right["trunk_lateral_lean"] = trunk
        angles_left["trunk_lateral_lean"] = trunk

    if any("pelvic_obliquity" in a for a in all_angles):
        obliq = np.array([a.get("pelvic_obliquity", np.nan) for a in all_angles])
        angles_right["pelvis_obliquity"] = obliq
        angles_left["pelvis_obliquity"] = obliq
        obliq_signed = np.array(
            [a.get("pelvic_obliquity_signed", np.nan) for a in all_angles]
        )
        angles_right["pelvis_obliquity_signed"] = obliq_signed
        angles_left["pelvis_obliquity_signed"] = obliq_signed

    interpolation_fractions = {}
    for side, d in [("R", angles_right), ("L", angles_left)]:
        for key in d:
            arr = d[key]
            nan_mask = np.isnan(arr)
            frac_key = f"{side}_{key}"
            if np.any(nan_mask):
                interpolation_fractions[frac_key] = float(np.mean(nan_mask))
                valid = ~nan_mask
                if np.any(valid):
                    indices = np.arange(len(arr))
                    arr[~valid] = np.interp(
                        indices[~valid], indices[valid], arr[valid]
                    )
                    d[key] = arr
            else:
                interpolation_fractions[frac_key] = 0.0

    if "hip_flexion" in angles_right:
        from ..kinematics.gait_metrics import detect_gait_events
        knee = angles_right.get("knee_flexion", np.zeros(n))
        ankle = angles_right.get("ankle_dorsiflexion")
        r_heel_y = angles_right.get("heel_y")
        events = detect_gait_events(
            angles_right["hip_flexion"], knee, actual_fps,
            ankle_dorsiflexion=ankle, heel_y=r_heel_y,
        )
        hs = events["heel_strikes"]
        phase = np.zeros(n)
        if len(hs) >= 2:
            for j in range(len(hs) - 1):
                start, end = hs[j], hs[j + 1]
                phase[start:end] = np.linspace(0, 1, end - start, endpoint=False)
            phase[hs[-1]:] = np.linspace(
                0, (n - hs[-1]) / max(1, hs[-1] - hs[-2] if len(hs) >= 2 else 1),
                n - hs[-1], endpoint=False,
            ) % 1.0
        angles_right["cycle_phase"] = phase
        angles_left["cycle_phase"] = phase

    detected_frames = sum(1 for a in all_angles if a)
    metadata = {
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "observed_fraction": detected_frames / len(all_angles) if all_angles else 0.0,
        "interpolation_fractions": interpolation_fractions,
        "confidences": confidences,
    }

    return frames, angles_right, angles_left, actual_fps, metadata
