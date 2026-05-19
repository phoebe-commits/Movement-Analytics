"""CLI entry point for Movement Analytics.

Usage:
    python -m movement_analytics [--profile PROFILE] [--output PATH] [--no-display]

Generates a synthetic walking animation, runs real-time kinematic analysis,
and displays the composite dashboard.
"""

import argparse
import sys
import time

import cv2
import numpy as np

from .generators.gait_model import GaitParameters, GAIT_PROFILES
from .generators.stick_figure import generate_frames
from .kinematics.joint_angles import compute_all_angles
from .kinematics.gait_metrics import compute_gait_summary
from .visualization.dashboard import RealTimeDashboard, create_dashboard_frame


def run_analysis(profile_name: str = "normal", output_path: str | None = None,
                 display: bool = True, fps: int = 30, n_cycles: int = 6):
    """Run the full analysis pipeline: generate, analyze, visualize."""

    if profile_name in GAIT_PROFILES:
        profile = GAIT_PROFILES[profile_name]
        params = profile.params
        print(f"Gait profile: {profile.name}")
        print(f"  {profile.description}")
    else:
        print(f"Unknown profile '{profile_name}'. Available: {list(GAIT_PROFILES.keys())}")
        sys.exit(1)

    print(f"\nGenerating {n_cycles}-cycle walking animation at {fps} fps...")
    frames, angles_right, angles_left, _ = generate_frames(
        params, fps=fps, n_cycles=n_cycles
    )
    print(f"  Generated {len(frames)} frames")

    print("Computing gait summary metrics...")
    summary = compute_gait_summary(angles_right, angles_left, fps=fps)

    print("\n--- Gait Summary ---")
    for key, val in sorted(summary.items()):
        print(f"  {key}: {val:.2f}")
    print()

    dashboard = RealTimeDashboard(history_length=150, panel_width=560)

    writer = None
    if output_path:
        sample = create_dashboard_frame(
            frames[0], {}, summary, dashboard, "Stance", 0, profile_name
        )
        h, w = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        print(f"Writing output to {output_path} ({w}x{h})")

    # Reset dashboard for clean run
    dashboard = RealTimeDashboard(history_length=150, panel_width=560)

    print("Running real-time analysis...")
    frame_time = 1.0 / fps

    for i, frame in enumerate(frames):
        t_start = time.perf_counter()

        # Extract current angles from ground truth
        current_angles = {}
        for side, angles_dict in [("right", angles_right), ("left", angles_left)]:
            for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]:
                if joint in angles_dict:
                    current_angles[f"{side}_{joint}"] = float(angles_dict[joint][i])
            if "shoulder_flexion" in angles_dict:
                current_angles[f"{side}_shoulder_flexion"] = float(angles_dict["shoulder_flexion"][i])
            if "elbow_flexion" in angles_dict:
                current_angles[f"{side}_elbow_flexion"] = float(angles_dict["elbow_flexion"][i])

        if "pelvis_tilt" in angles_right:
            current_angles["pelvis_tilt"] = float(angles_right["pelvis_tilt"][i])
        if "pelvis_obliquity" in angles_right:
            current_angles["pelvis_obliquity"] = float(angles_right["pelvis_obliquity"][i])

        # Map to dashboard-expected keys
        display_angles = {}
        for key, val in current_angles.items():
            display_angles[key] = val
        if "right_ankle_dorsiflexion" in current_angles:
            display_angles["right_ankle_angle"] = 90 + current_angles["right_ankle_dorsiflexion"]
        if "left_ankle_dorsiflexion" in current_angles:
            display_angles["left_ankle_angle"] = 90 + current_angles["left_ankle_dorsiflexion"]

        cycle_phase = angles_right.get("cycle_phase", np.zeros(len(frames)))
        cp = cycle_phase[i] if i < len(cycle_phase) else 0
        gait_phase = "Stance" if cp < params.stance_ratio else "Swing"
        cycle_pct = cp * 100

        composite = create_dashboard_frame(
            frame, display_angles, summary, dashboard, gait_phase, cycle_pct,
            profile_name
        )

        if writer:
            writer.write(composite)

        if display:
            cv2.imshow("Movement Analytics Dashboard", composite)
            elapsed = time.perf_counter() - t_start
            wait_ms = max(1, int((frame_time - elapsed) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q") or key == 27:
                break

    if writer:
        writer.release()
        print(f"\nOutput saved to {output_path}")

    if display:
        cv2.destroyAllWindows()

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Movement Analytics — synthetic gait analysis dashboard"
    )
    parser.add_argument(
        "--profile", "-p", default="normal",
        choices=list(GAIT_PROFILES.keys()),
        help="Gait profile to simulate"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output video file path (MP4)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Skip live display (useful for headless rendering)"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second"
    )
    parser.add_argument(
        "--cycles", "-c", type=int, default=6,
        help="Number of gait cycles"
    )
    parser.add_argument(
        "--all-profiles", action="store_true",
        help="Generate output for all gait profiles"
    )
    args = parser.parse_args()

    if args.all_profiles:
        for name in GAIT_PROFILES:
            out = f"output_{name}.mp4" if not args.output else f"{args.output}_{name}.mp4"
            print(f"\n{'='*60}")
            print(f"Profile: {name}")
            print(f"{'='*60}")
            run_analysis(name, out, display=False, fps=args.fps, n_cycles=args.cycles)
    else:
        run_analysis(args.profile, args.output, display=not args.no_display,
                     fps=args.fps, n_cycles=args.cycles)


if __name__ == "__main__":
    main()
