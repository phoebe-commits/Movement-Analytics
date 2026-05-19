"""Real-time visualization dashboard for movement analysis.

Renders a composite view: walking animation on the left, real-time joint angle
plots and metric gauges on the right. All kinematic quantities update frame-by-frame.
Includes a prominent Movement Quality Score and bilateral joint overlays.
"""

import cv2
import numpy as np
from collections import deque


COLORS = {
    "bg": (20, 20, 25),
    "panel_bg": (30, 30, 35),
    "text": (200, 210, 220),
    "text_dim": (120, 130, 140),
    "accent": (0, 180, 255),
    "hip": (80, 180, 255),
    "knee": (255, 160, 50),
    "ankle": (100, 255, 100),
    "grid": (45, 45, 50),
    "good": (80, 200, 80),
    "warn": (50, 200, 255),
    "bad": (60, 60, 255),
    "separator": (55, 55, 60),
    "mqs_ring_bg": (50, 50, 55),
    "left_side": (255, 140, 100),
    "right_side": (100, 180, 255),
    "domain_kin": (80, 200, 220),
    "domain_smooth": (200, 160, 80),
    "domain_sym": (180, 100, 255),
    "domain_var": (100, 255, 180),
    "domain_temp": (220, 120, 160),
    "domain_coord": (140, 200, 100),
}

JOINT_COLORS = {
    "hip_flexion": COLORS["hip"],
    "knee_flexion": COLORS["knee"],
    "ankle_dorsiflexion": COLORS["ankle"],
}

DOMAIN_COLORS = {
    "kinematics": COLORS["domain_kin"],
    "smoothness": COLORS["domain_smooth"],
    "symmetry": COLORS["domain_sym"],
    "variability": COLORS["domain_var"],
    "temporal": COLORS["domain_temp"],
    "coordination": COLORS["domain_coord"],
}


def _score_color(score: float) -> tuple:
    """Return BGR color interpolated from red (0) through yellow (50) to green (100)."""
    if score <= 50:
        t = score / 50.0
        return (int(60 + 50 * t), int(60 + 180 * t), int(255 - 55 * t))
    t = (score - 50) / 50.0
    return (int(50 + 30 * t), int(200 + 55 * t), int(200 - 120 * t))


class RealTimeDashboard:
    """Manages real-time display state for the analysis dashboard."""

    def __init__(self, history_length: int = 150, panel_width: int = 560):
        self.history_length = history_length
        self.panel_width = panel_width
        self.histories: dict[str, deque] = {}
        self.frame_count = 0

    def _ensure_history(self, key: str):
        if key not in self.histories:
            self.histories[key] = deque(maxlen=self.history_length)

    def update(self, angles: dict[str, float], metrics: dict[str, float]):
        self.frame_count += 1
        for key, val in angles.items():
            self._ensure_history(key)
            self.histories[key].append(val)

    def _draw_mqs_card(self, canvas: np.ndarray, x: int, y: int,
                       w: int, metrics: dict[str, float]) -> int:
        """Draw the Movement Quality Score card with arc gauge and domain breakdown."""
        mqs = metrics.get("movement_quality_score", 0)
        card_h = 110

        cv2.rectangle(canvas, (x, y), (x + w, y + card_h), (35, 35, 40), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + card_h), COLORS["separator"], 1)

        score_color = _score_color(mqs)

        cx, cy = x + 60, y + 60
        radius = 40
        cv2.ellipse(canvas, (cx, cy), (radius, radius), 0, 135, 405,
                     COLORS["mqs_ring_bg"], 6)
        end_angle = 135 + int(mqs / 100.0 * 270)
        cv2.ellipse(canvas, (cx, cy), (radius, radius), 0, 135, end_angle,
                     score_color, 6)

        score_text = f"{mqs:.0f}"
        (tw, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(canvas, score_text, (cx - tw // 2, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, "MQS", (cx - 14, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_dim"], 1, cv2.LINE_AA)

        label_x = x + 125
        cv2.putText(canvas, "MOVEMENT QUALITY", (label_x, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["accent"], 1, cv2.LINE_AA)

        domains = [
            ("Kinematics", "mqs_kinematics", "domain_kin"),
            ("Smoothness", "mqs_smoothness", "domain_smooth"),
            ("Symmetry", "mqs_symmetry", "domain_sym"),
            ("Coordination", "mqs_coordination", "domain_coord"),
            ("Variability", "mqs_variability", "domain_var"),
            ("Temporal", "mqs_temporal", "domain_temp"),
        ]

        bar_y_start = y + 30
        bar_w = w - 145
        for i, (label, key, color_key) in enumerate(domains):
            dy = bar_y_start + i * 16
            val = metrics.get(key, 50)
            dc = COLORS[color_key]

            cv2.putText(canvas, label[:4], (label_x, dy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLORS["text_dim"], 1, cv2.LINE_AA)

            bx = label_x + 40
            bw = bar_w - 80
            cv2.rectangle(canvas, (bx, dy + 3), (bx + bw, dy + 9), COLORS["grid"], -1)
            fill = int(np.clip(val / 100.0, 0, 1) * bw)
            cv2.rectangle(canvas, (bx, dy + 3), (bx + fill, dy + 9), dc, -1)

            cv2.putText(canvas, f"{val:.0f}", (bx + bw + 4, dy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, dc, 1, cv2.LINE_AA)

        return card_h + 5

    def _draw_bilateral_plot(self, canvas: np.ndarray, x: int, y: int,
                             w: int, h: int, joint: str, label: str,
                             color: tuple, y_range: tuple):
        """Draw time-series with left/right overlay and phase shading."""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLORS["panel_bg"], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLORS["separator"], 1)

        y_min, y_max = y_range
        for grid_val in range(int(y_min), int(y_max) + 1, 20):
            py = int(y + h - (grid_val - y_min) / (y_max - y_min) * h)
            if y <= py <= y + h:
                cv2.line(canvas, (x, py), (x + w, py), COLORS["grid"], 1)
                cv2.putText(canvas, f"{grid_val}", (x + 2, py - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, COLORS["text_dim"], 1)

        zero_y = int(y + h - (0 - y_min) / (y_max - y_min) * h)
        if y <= zero_y <= y + h:
            cv2.line(canvas, (x, zero_y), (x + w, zero_y), (70, 70, 75), 1)

        right_key = f"right_{joint}"
        left_key = f"left_{joint}"

        for side_key, side_color, thickness in [
            (left_key, COLORS["left_side"], 1),
            (right_key, COLORS["right_side"], 2),
        ]:
            if side_key in self.histories and len(self.histories[side_key]) > 1:
                data = list(self.histories[side_key])
                points = []
                for i, val in enumerate(data):
                    px = int(x + i * w / self.history_length)
                    py = int(y + h - (val - y_min) / (y_max - y_min) * h)
                    py = np.clip(py, y, y + h)
                    points.append((px, py))

                for i in range(len(points) - 1):
                    alpha = 0.3 + 0.7 * (i / len(points))
                    c = tuple(int(ch * alpha) for ch in side_color)
                    cv2.line(canvas, points[i], points[i + 1], c, thickness)

        if right_key in self.histories and self.histories[right_key]:
            current = list(self.histories[right_key])[-1]
            cv2.putText(canvas, f"R:{current:.1f}", (x + w - 80, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLORS["right_side"], 1, cv2.LINE_AA)
        if left_key in self.histories and self.histories[left_key]:
            current = list(self.histories[left_key])[-1]
            cv2.putText(canvas, f"L:{current:.1f}", (x + w - 80, y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLORS["left_side"], 1, cv2.LINE_AA)

        cv2.putText(canvas, label, (x + 5, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    def _draw_gauge(self, canvas: np.ndarray, x: int, y: int,
                    label: str, value: float, unit: str,
                    normal_range: tuple = None, width: int = 240):
        h = 30
        cv2.rectangle(canvas, (x, y), (x + width, y + h), COLORS["panel_bg"], -1)

        cv2.putText(canvas, label, (x + 5, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_dim"], 1, cv2.LINE_AA)

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

        cv2.putText(canvas, val_text, (x + width - 95, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        if normal_range:
            bar_x = x + 5
            bar_y = y + 19
            bar_w = width - 10
            bar_h = 5
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
        cv2.putText(canvas, text, (x + 5, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["accent"], 1, cv2.LINE_AA)
        cv2.line(canvas, (x, y + 17), (x + width, y + 17), COLORS["separator"], 1)

    def render_panel(self, canvas: np.ndarray, x_offset: int,
                     angles: dict[str, float], metrics: dict[str, float],
                     gait_phase: str, cycle_pct: float,
                     profile_name: str = ""):
        """Render the full analytics panel onto the canvas."""
        pw = self.panel_width
        h = canvas.shape[0]

        cv2.rectangle(canvas, (x_offset, 0), (x_offset + pw, h),
                      COLORS["panel_bg"], -1)
        cv2.line(canvas, (x_offset, 0), (x_offset, h), COLORS["separator"], 2)

        x = x_offset + 8
        y = 6

        mqs_h = self._draw_mqs_card(canvas, x, y, pw - 16, metrics)
        y += mqs_h

        phase_color = COLORS["good"] if gait_phase == "Stance" else COLORS["hip"]
        cv2.putText(canvas, f"Phase: {gait_phase}", (x + 2, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, phase_color, 1, cv2.LINE_AA)

        if profile_name:
            cv2.putText(canvas, profile_name.upper(), (x + 120, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_dim"], 1, cv2.LINE_AA)

        bar_x = x + 240
        bar_w = pw - 260
        cv2.rectangle(canvas, (bar_x, y + 4), (bar_x + bar_w, y + 12), COLORS["grid"], -1)
        fill = int(cycle_pct / 100 * bar_w)
        cv2.rectangle(canvas, (bar_x, y + 4), (bar_x + fill, y + 12), phase_color, -1)
        cv2.putText(canvas, f"{cycle_pct:.0f}%", (bar_x + bar_w + 4, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, COLORS["text_dim"], 1, cv2.LINE_AA)
        y += 22

        self._draw_section_header(canvas, x, y, "JOINT ANGLES  R / L", pw - 16)
        y += 22

        plot_h = 65
        plot_w = pw - 20

        for joint, label, y_range in [
            ("hip_flexion", "Hip Flex/Ext", (-30, 50)),
            ("knee_flexion", "Knee Flexion", (-5, 75)),
            ("ankle_dorsiflexion", "Ankle DF/PF", (-25, 20)),
        ]:
            color = JOINT_COLORS.get(joint, COLORS["text"])
            self._draw_bilateral_plot(canvas, x, y, plot_w, plot_h, joint,
                                      label, color, y_range)
            y += plot_h + 3

        y += 3

        gauge_w = (pw - 24) // 2

        self._draw_section_header(canvas, x, y, "GAIT METRICS", pw - 16)
        y += 22

        left_gauges = [
            ("Cadence", metrics.get("cadence", 0), "spm", (90, 130)),
            ("Stride Time", metrics.get("stride_time_mean", 0), "s", (0.8, 1.3)),
            ("Stride CV", metrics.get("stride_time_CV", 0), "%", (0, 8)),
        ]
        right_gauges = [
            ("Hip ROM R", metrics.get("R_hip_flexion_ROM", 0), "deg", (30, 50)),
            ("Knee ROM R", metrics.get("R_knee_flexion_ROM", 0), "deg", (50, 70)),
            ("Ankle ROM R", metrics.get("R_ankle_dorsiflexion_ROM", 0), "deg", (20, 35)),
        ]

        for i in range(3):
            l, v, u, nr = left_gauges[i]
            self._draw_gauge(canvas, x, y, l, v, u, nr, gauge_w)
            l2, v2, u2, nr2 = right_gauges[i]
            self._draw_gauge(canvas, x + gauge_w + 8, y, l2, v2, u2, nr2, gauge_w)
            y += 32

        y += 3

        self._draw_section_header(canvas, x, y, "SYMMETRY", (pw - 24) // 2 + 4)

        sx2 = x + gauge_w + 8
        self._draw_section_header(canvas, sx2, y, "SMOOTHNESS", (pw - 24) // 2)
        y += 22

        sym_pairs = [
            ("Hip SI", metrics.get("hip_flexion_SI", 0), "%", (0, 15)),
            ("Knee SI", metrics.get("knee_flexion_SI", 0), "%", (0, 15)),
            ("Ankle SI", metrics.get("ankle_dorsiflexion_SI", 0), "%", (0, 15)),
        ]
        smooth_pairs = [
            ("Hip SPARC", metrics.get("R_hip_flexion_SPARC", 0), "", (-5, -1)),
            ("Knee SPARC", metrics.get("R_knee_flexion_SPARC", 0), "", (-5, -1)),
            ("Hip NJ", metrics.get("R_hip_flexion_NJ", 0), "", (0, 100)),
        ]

        for i in range(3):
            l, v, u, nr = sym_pairs[i]
            self._draw_gauge(canvas, x, y, l, v, u, nr, gauge_w)
            l2, v2, u2, nr2 = smooth_pairs[i]
            self._draw_gauge(canvas, sx2, y, l2, v2, u2, nr2, gauge_w)
            y += 32


def create_dashboard_frame(video_frame: np.ndarray, angles: dict[str, float],
                           metrics: dict[str, float], dashboard: RealTimeDashboard,
                           gait_phase: str = "Stance", cycle_pct: float = 0.0,
                           profile_name: str = "") -> np.ndarray:
    """Compose a single dashboard frame: video on left, analytics on right."""
    vh, vw = video_frame.shape[:2]
    panel_w = dashboard.panel_width
    canvas_w = vw + panel_w

    canvas = np.full((vh, canvas_w, 3), COLORS["bg"], dtype=np.uint8)
    canvas[:vh, :vw] = video_frame

    dashboard.update(angles, metrics)
    dashboard.render_panel(canvas, vw, angles, metrics, gait_phase, cycle_pct,
                           profile_name)

    return canvas
