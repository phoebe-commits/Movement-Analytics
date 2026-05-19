"""Pose estimation from video using MediaPipe PoseLandmarker (Tasks API).

Extracts 33 keypoints per frame and maps them to the joint position format
expected by the kinematics engine. Supports both file input and frame-by-frame
processing for real-time analysis.
"""

import os

import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt

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

_PHYSIO_RANGES = {
    "hip_flexion": (-50.0, 120.0),
    "knee_flexion": (-20.0, 160.0),
    "ankle_dorsiflexion": (-90.0, 60.0),
    "elbow_flexion": (-20.0, 170.0),
    "shoulder_flexion": (-60.0, 180.0),
    "pelvis_obliquity": (-40.0, 40.0),
    "pelvic_obliquity": (-40.0, 40.0),
    "trunk_lean": (-45.0, 45.0),
    "trunk_lateral_lean": (-45.0, 45.0),
}


def _reject_outliers(arr: np.ndarray, key: str) -> np.ndarray:
    """Replace values outside physiological range with NaN."""
    base = key.replace("right_", "").replace("left_", "")
    for canon, (lo, hi) in _PHYSIO_RANGES.items():
        if canon in base:
            out = arr.copy()
            mask = (out < lo) | (out > hi)
            out[mask] = np.nan
            return out
    return arr


def _lowpass_smooth(arr: np.ndarray, fps: float, cutoff: float = 6.0) -> np.ndarray:
    """Apply Butterworth low-pass filter for temporal smoothing."""
    valid = ~np.isnan(arr)
    if np.sum(valid) < 13:
        return arr
    nyq = fps / 2
    if nyq <= cutoff:
        return arr
    b, a = butter(2, cutoff / nyq, btype="low")
    smoothed = arr.copy()
    smoothed[valid] = filtfilt(b, a, arr[valid])
    return smoothed


def _adaptive_smooth(
    arr: np.ndarray,
    fps: float,
    frame_confidences: np.ndarray,
    base_cutoff: float = 6.0,
    low_conf_cutoff: float = 3.0,
    conf_threshold: float = 0.7,
) -> np.ndarray:
    """Two-pass smoothing: aggressive on low-confidence regions, gentle elsewhere.

    Frames with confidence below conf_threshold get a tighter low-pass filter
    (low_conf_cutoff Hz) before the standard pass, reducing MediaPipe jitter
    in uncertain frames without over-smoothing confident regions.
    """
    valid = ~np.isnan(arr)
    n_valid = np.sum(valid)
    if n_valid < 13:
        return arr
    nyq = fps / 2
    if nyq <= low_conf_cutoff:
        return arr

    out = arr.copy()

    low_conf = frame_confidences < conf_threshold
    low_conf_valid = low_conf & valid
    if np.sum(low_conf_valid) >= 4:
        indices = np.where(low_conf_valid)[0]
        runs = np.split(indices, np.where(np.diff(indices) > 2)[0] + 1)
        b, a = butter(2, low_conf_cutoff / nyq, btype="low")
        padlen = 3 * max(len(b), len(a))
        for run in runs:
            if len(run) > padlen:
                out[run] = filtfilt(b, a, out[run])

    return _lowpass_smooth(out, fps, cutoff=base_cutoff)


_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)
_MODEL_SHA256 = "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b"


def _download_model(model_path: str):
    """Download the pose landmarker model if not present, with SHA256 verification."""
    if os.path.exists(model_path):
        return
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Downloading pose landmarker model ({_MODEL_URL})...")
    import hashlib
    import urllib.request
    tmp_path = model_path + ".tmp"
    urllib.request.urlretrieve(_MODEL_URL, tmp_path)
    sha = hashlib.sha256(open(tmp_path, "rb").read()).hexdigest()
    if sha != _MODEL_SHA256:
        os.remove(tmp_path)
        raise RuntimeError(
            f"Model checksum mismatch: expected {_MODEL_SHA256[:16]}..., "
            f"got {sha[:16]}... — download may be corrupted or model version changed"
        )
    os.replace(tmp_path, model_path)
    print(f"Model saved to {model_path} (SHA256 verified)")


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
            timestamp_ms = int(self._frame_count * self._ms_per_frame)
            self._frame_count += 1
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
    video_path: str, fps: float | None = None, store_frames: bool = True,
) -> tuple[list[np.ndarray], dict, dict, float, dict]:
    """Process a video file and extract bilateral joint angle time series.

    Returns (frames, angles_right, angles_left, actual_fps, metadata) where
    metadata contains pose quality stats: mean_confidence, observed_fraction,
    interpolation_fractions (per-key dict), and per-frame confidences.

    When store_frames=False, frames list is empty (saves memory for headless
    analysis). Angles and metadata are still computed.
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

            positions, confidence = estimator.process_frame(
                frame, min_visibility=0.3,
            )
            confidences.append(confidence)
            if positions is not None:
                angles = compute_all_angles(positions)
                if store_frames:
                    frames.append(estimator.draw_landmarks(frame, positions))
                all_angles.append(angles)
            else:
                if store_frames:
                    frames.append(frame)
                all_angles.append({})

    cap.release()

    if not all_angles:
        return frames, {}, {}, actual_fps, {
            "mean_confidence": 0.0, "observed_fraction": 0.0,
            "interpolation_fractions": {}, "confidences": [],
            "n_frames": len(frames),
        }

    per_frame_confidences = confidences

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
        "right_shoulder_flexion": "shoulder_flexion",
        "left_hip_flexion": "hip_flexion",
        "left_knee_flexion": "knee_flexion",
        "left_ankle_dorsiflexion": "ankle_dorsiflexion",
        "left_elbow_flexion": "elbow_flexion",
        "left_shoulder_flexion": "shoulder_flexion",
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

    _shared_keys = {
        "pelvis_obliquity", "pelvis_obliquity_signed",
        "trunk_lateral_lean", "cycle_phase",
    }
    processed_shared: dict[str, np.ndarray] = {}

    interpolation_fractions = {}
    for side, d in [("R", angles_right), ("L", angles_left)]:
        for key in list(d.keys()):
            if key in _shared_keys and key in processed_shared:
                d[key] = processed_shared[key]
                interpolation_fractions[f"{side}_{key}"] = interpolation_fractions.get(
                    f"R_{key}", 0.0
                )
                continue
            arr = _reject_outliers(d[key], key)
            nan_mask = np.isnan(arr)
            frac_key = f"{side}_{key}"
            if np.any(nan_mask):
                missing_frac = float(np.mean(nan_mask))
                interpolation_fractions[frac_key] = missing_frac
                valid = ~nan_mask
                if np.any(valid):
                    indices = np.arange(len(arr))
                    n_valid = int(np.sum(valid))
                    if n_valid >= 4 and missing_frac < 0.5:
                        interp_fn = PchipInterpolator(
                            indices[valid], arr[valid], extrapolate=False,
                        )
                        pchip_vals = interp_fn(indices[~valid])
                        still_nan = np.isnan(pchip_vals)
                        arr[~valid] = pchip_vals
                        if np.any(still_nan):
                            edge_nans = np.where(~valid)[0][still_nan]
                            arr[edge_nans] = np.interp(
                                edge_nans, indices[valid], arr[valid],
                            )
                    else:
                        arr[~valid] = np.interp(
                            indices[~valid], indices[valid], arr[valid]
                        )
                    arr = _reject_outliers(arr, key)
            else:
                interpolation_fractions[frac_key] = 0.0
            conf_arr = np.array(per_frame_confidences[:len(arr)])
            smoothed = _adaptive_smooth(arr, actual_fps, conf_arr)
            d[key] = smoothed
            if key in _shared_keys:
                processed_shared[key] = smoothed

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
    detected_confidences = [c for c, a in zip(confidences, all_angles) if a]

    joint_detection_rates = {}
    all_joint_keys = angle_keys_right | angle_keys_left
    for jk in all_joint_keys:
        present = sum(1 for a in all_angles if jk in a)
        joint_detection_rates[jk] = present / n if n > 0 else 0.0

    metadata = {
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "mean_detected_confidence": (
            float(np.mean(detected_confidences)) if detected_confidences else 0.0
        ),
        "observed_fraction": detected_frames / len(all_angles) if all_angles else 0.0,
        "interpolation_fractions": interpolation_fractions,
        "joint_detection_rates": joint_detection_rates,
        "confidences": confidences,
        "n_frames": len(all_angles) or len(frames),
    }

    return frames, angles_right, angles_left, actual_fps, metadata
