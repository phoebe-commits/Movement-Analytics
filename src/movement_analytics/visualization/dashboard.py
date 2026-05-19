"""Real-time visualization dashboard for movement analysis.

Renders a composite view: walking animation on the left, real-time joint angle
plots and metric gauges on the right. All kinematic quantities update frame-by-frame.
"""

import cv2
import numpy as np
from collections import deque


# Color palette (BGR)
COLORS = {
    "bg": (20, 20, 25),
    "panel_bg": (30, 30, 35),
    "text": (200, 210, 220),
    "text_dim": (120, 130, 140),
    "accent": (0, 180, 255),       # orange-amber
    "hip": (80, 180, 255),          # warm orange
    "knee": (255, 160, 50),         # blue
    "ankle": (100, 255, 100),       # green
    "grid": (45, 45, 50),
    "good": (80, 200, 80),
    "warn": (50, 200, 255),
    "bad": (60, 60, 255),
    "separator": (55, 55, 60),
}

JOINT_COLORS = {
    "hip_flexion": COLORS["hip"],
    "knee_flexion": COLORS["knee"],
    "ankle_dorsiflexion": COLORS["ankle"],
}


class RealTimeDashboard:
    """Manages real-time display state for the analysis dashboard."""

    def __init__(self, history_length: int = 150, panel_width: int = 520):
        self.history_length = history_length
        self.panel_width = panel_width
        self.histories: dict[str, deque] = {}
        self.frame_count = 0

    def _ensure_history(self, key: str):
        if key not in self.histories:
            self.histories[key] = deque(maxlen=self.history_length)

    def update(self, angles: dict[str, float], metrics: dict[str, float]):
        """Push new frame data into history buffers."""
        self.frame_count += 1
        for key, val in angles.items():
            self._ensure_history(key)
            self.histories[key].append(val)

    def _draw_time_series(self, canvas: np.ndarray, x: int, y: int,
                          w: int, h: int, key: str, label: str,
                          color: tuple, y_range: tuple = (-30, 80)):
        """Draw a mini time-series plot for a joint angle."""
        # Background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLORS["panel_bg"], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLORS["separator"], 1)

        # Grid lines
        y_min, y_max = y_range
        for grid_val in range(int(y_min), int(y_max) + 1, 20):
            py = int(y + h - (grid_val - y_min) / (y_max - y_min) * h)
            if y <= py <= y + h:
                cv2.line(canvas, (x, py), (x + w, py), COLORS["grid"], 1)
                cv2.putText(canvas, f"{grid_val}", (x + 2, py - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["text_dim"], 1)

        # Zero line
        zero_y = int(y + h - (0 - y_min) / (y_max - y_min) * h)
        if y <= zero_y <= y + h:
            cv2.line(canvas, (x, zero_y), (x + w, zero_y), (70, 70, 75), 1)

        # Data
        if key in self.histories and len(self.histories[key]) > 1:
            data = list(self.histories[key])
            n = len(data)
            points = []
            for i, val in enumerate(data):
                px = int(x + i * w / self.history_length)
                py = int(y + h - (val - y_min) / (y_max - y_min) * h)
                py = np.clip(py, y, y + h)
                points.append((px, py))

            for i in range(len(points) - 1):
                alpha = 0.3 + 0.7 * (i / len(points))
                c = tuple(int(ch * alpha) for ch in color)
                cv2.line(canvas, points[i], points[i + 1], c, 2)

            # Current value
            current = data[-1]
            cv2.putText(canvas, f"{current:.1f} deg", (x + w - 90, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Label
        cv2.putText(canvas, label, (x + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _draw_gauge(self, canvas: np.ndarray, x: int, y: int,
                    label: str, value: float, unit: str,
                    normal_range: tuple = None, width: int = 240):
        """Draw a horizontal metric gauge."""
        h = 32
        cv2.rectangle(canvas, (x, y), (x + width, y + h), COLORS["panel_bg"], -1)

        cv2.putText(canvas, label, (x + 5, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["text_dim"], 1, cv2.LINE_AA)

        val_text = f"{value:.1f} {unit}"
        color = COLORS["text"]
        if normal_range:
            lo, hi = normal_range
            if lo <= value <= hi:
                color = COLORS["good"]
            elif abs(value - lo) < (hi - lo) * 0.3 or abs(value - hi) < (hi - lo) * 0.3:
                color = COLORS["warn"]
            else:
                color = COLORS["bad"]

        cv2.putText(canvas, val_text, (x + width - 100, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        if normal_range:
            bar_x = x + 5
            bar_y = y + 20
            bar_w = width - 10
            bar_h = 6
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          COLORS["grid"], -1)

            lo, hi = normal_range
            total_range = hi - lo
            if total_range > 0:
                fill = np.clip((value - lo) / total_range, 0, 1)
                fill_w = int(fill * bar_w)
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                              color, -1)

    def _draw_section_header(self, canvas: np.ndarray, x: int, y: int,
                             text: str, width: int):
        """Draw a section header with underline."""
        cv2.putText(canvas, text, (x + 5, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["accent"], 1, cv2.LINE_AA)
        cv2.line(canvas, (x, y + 20), (x + width, y + 20), COLORS["separator"], 1)

    def render_panel(self, canvas: np.ndarray, x_offset: int,
                     angles: dict[str, float], metrics: dict[str, float],
                     gait_phase: str, cycle_pct: float):
        """Render the full analytics panel onto the canvas."""
        pw = self.panel_width
        h = canvas.shape[0]

        # Panel background
        cv2.rectangle(canvas, (x_offset, 0), (x_offset + pw, h),
                      COLORS["panel_bg"], -1)
        cv2.line(canvas, (x_offset, 0), (x_offset, h), COLORS["separator"], 2)

        x = x_offset + 10
        y = 10

        # Title
        cv2.putText(canvas, "MOVEMENT ANALYTICS", (x, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS["accent"], 2, cv2.LINE_AA)
        y += 35

        # Gait phase indicator
        phase_color = COLORS["good"] if gait_phase == "Stance" else COLORS["hip"]
        cv2.putText(canvas, f"Phase: {gait_phase}", (x, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, phase_color, 1, cv2.LINE_AA)
        # Cycle progress bar
        bar_x = x + 130
        bar_w = pw - 155
        cv2.rectangle(canvas, (bar_x, y + 4), (bar_x + bar_w, y + 14), COLORS["grid"], -1)
        fill = int(cycle_pct / 100 * bar_w)
        cv2.rectangle(canvas, (bar_x, y + 4), (bar_x + fill, y + 14), phase_color, -1)
        cv2.putText(canvas, f"{cycle_pct:.0f}%", (bar_x + bar_w + 5, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_dim"], 1, cv2.LINE_AA)
        y += 30

        # Joint angle time series
        self._draw_section_header(canvas, x, y, "JOINT ANGLES (Right)", pw - 20)
        y += 28

        plot_h = 75
        plot_w = pw - 30

        for joint, label, y_range in [
            ("right_hip_flexion", "Hip Flex/Ext", (-30, 50)),
            ("right_knee_flexion", "Knee Flexion", (-5, 75)),
            ("right_ankle_angle", "Ankle Angle", (60, 130)),
        ]:
            color = JOINT_COLORS.get(joint.replace("right_", ""), COLORS["text"])
            self._draw_time_series(canvas, x, y, plot_w, plot_h, joint, label,
                                   color, y_range)
            y += plot_h + 5

        y += 5

        # Metrics section
        self._draw_section_header(canvas, x, y, "GAIT METRICS", pw - 20)
        y += 28

        gauge_w = pw - 30
        gauge_pairs = [
            ("Cadence", metrics.get("cadence", 0), "spm", (90, 130)),
            ("Stride Time", metrics.get("stride_time_mean", 0), "s", (0.8, 1.3)),
            ("Stride CV", metrics.get("stride_time_CV", 0), "%", (0, 8)),
            ("Hip ROM (R)", metrics.get("R_hip_flexion_ROM", 0), "deg", (30, 50)),
            ("Knee ROM (R)", metrics.get("R_knee_flexion_ROM", 0), "deg", (50, 70)),
            ("Ankle ROM (R)", metrics.get("R_ankle_dorsiflexion_ROM", 0), "deg", (20, 35)),
        ]

        for label, val, unit, nrange in gauge_pairs:
            self._draw_gauge(canvas, x, y, label, val, unit, nrange, gauge_w)
            y += 36

        y += 5

        # Symmetry section
        self._draw_section_header(canvas, x, y, "SYMMETRY", pw - 20)
        y += 28

        sym_pairs = [
            ("Hip SI", metrics.get("hip_flexion_SI", 0), "%", (0, 15)),
            ("Knee SI", metrics.get("knee_flexion_SI", 0), "%", (0, 15)),
            ("Ankle SI", metrics.get("ankle_dorsiflexion_SI", 0), "%", (0, 15)),
        ]
        for label, val, unit, nrange in sym_pairs:
            self._draw_gauge(canvas, x, y, label, val, unit, nrange, gauge_w)
            y += 36

        y += 5

        # Smoothness section
        self._draw_section_header(canvas, x, y, "SMOOTHNESS", pw - 20)
        y += 28

        smooth_pairs = [
            ("Hip SPARC (R)", metrics.get("R_hip_flexion_SPARC", 0), "", (-5, -1)),
            ("Knee SPARC (R)", metrics.get("R_knee_flexion_SPARC", 0), "", (-5, -1)),
            ("Hip NJ (R)", metrics.get("R_hip_flexion_NJ", 0), "", (0, 100)),
        ]
        for label, val, unit, nrange in smooth_pairs:
            self._draw_gauge(canvas, x, y, label, val, unit, nrange, gauge_w)
            y += 36


def create_dashboard_frame(video_frame: np.ndarray, angles: dict[str, float],
                           metrics: dict[str, float], dashboard: RealTimeDashboard,
                           gait_phase: str = "Stance", cycle_pct: float = 0.0,
                           total_width: int = 1800) -> np.ndarray:
    """Compose a single dashboard frame: video on left, analytics on right."""
    vh, vw = video_frame.shape[:2]
    panel_w = dashboard.panel_width
    canvas_w = vw + panel_w

    canvas = np.full((vh, canvas_w, 3), COLORS["bg"], dtype=np.uint8)

    # Place video frame
    canvas[:vh, :vw] = video_frame

    # Update and render analytics panel
    dashboard.update(angles, metrics)
    dashboard.render_panel(canvas, vw, angles, metrics, gait_phase, cycle_pct)

    return canvas
