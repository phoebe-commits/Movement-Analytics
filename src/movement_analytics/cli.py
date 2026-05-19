"""CLI entry point for Movement Analytics.

Usage:
    python -m movement_analytics [--profile PROFILE] [--output PATH] [--no-display]
    python -m movement_analytics --video path/to/walking.mp4 [--output PATH]

Generates a synthetic walking animation or analyzes real video, runs real-time
kinematic analysis, and displays the composite dashboard.
"""

import argparse
import sys
import time

import cv2
import numpy as np

from .generators.gait_model import GAIT_PROFILES
from .generators.stick_figure import generate_frames
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
                val = float(angles_dict["shoulder_flexion"][i])
                current_angles[f"{side}_shoulder_flexion"] = val
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


def run_video_analysis(video_path: str, output_path: str | None = None,
                       display: bool = True, target_fps: int | None = None):
    """Run analysis on a real video file using MediaPipe pose estimation."""
    from .pose.estimator import process_video

    print(f"Processing video: {video_path}")
    print("Running pose estimation (MediaPipe BlazePose)...")

    frames, angles_right, angles_left, fps, meta = process_video(video_path, fps=target_fps)

    if not frames:
        print("Error: No frames extracted from video.")
        sys.exit(1)

    print(f"  Extracted {len(frames)} frames at {fps:.1f} fps")
    if meta:
        obs = meta.get("observed_fraction", 0)
        conf = meta.get("mean_confidence", 0)
        print(f"  Pose detected: {obs:.1%} of frames, mean confidence: {conf:.2f}")

    if not angles_right:
        print("Warning: No pose detected in any frame.")
        return

    print("Computing gait summary metrics...")
    summary = compute_gait_summary(angles_right, angles_left, fps=fps)
    if meta:
        summary["pose_observed_fraction"] = meta.get("observed_fraction", 0.0)
        summary["pose_mean_confidence"] = meta.get("mean_confidence", 0.0)
        interp = meta.get("interpolation_fractions", {})
        if interp:
            summary["pose_interpolation_fraction"] = float(
                np.mean(list(interp.values()))
            )
        from .kinematics.gait_metrics import mqs_confidence_factor
        cf = mqs_confidence_factor(summary)
        summary["mqs_confidence_factor"] = cf
        if cf < 1.0:
            raw = summary["movement_quality_score"]
            summary["mqs_raw"] = raw
            summary["movement_quality_score"] = round(raw * cf, 1)

    print("\n--- Gait Summary ---")
    for key, val in sorted(summary.items()):
        print(f"  {key}: {val:.2f}")
    print()

    dashboard = RealTimeDashboard(history_length=150, panel_width=560)

    writer = None
    if output_path:
        sample = create_dashboard_frame(
            frames[0], {}, summary, dashboard, "—", 0, "Video Analysis"
        )
        h, w = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        print(f"Writing output to {output_path} ({w}x{h})")

    dashboard = RealTimeDashboard(history_length=150, panel_width=560)

    print("Running real-time analysis...")
    frame_time = 1.0 / fps

    for i, frame in enumerate(frames):
        t_start = time.perf_counter()

        display_angles = {}
        for side, angles_dict in [("right", angles_right), ("left", angles_left)]:
            for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]:
                if joint in angles_dict:
                    display_angles[f"{side}_{joint}"] = float(angles_dict[joint][i])
            if "elbow_flexion" in angles_dict:
                display_angles[f"{side}_elbow_flexion"] = float(angles_dict["elbow_flexion"][i])

        if "pelvis_tilt" in angles_right:
            display_angles["pelvis_tilt"] = float(angles_right["pelvis_tilt"][i])

        cycle_phase = angles_right.get("cycle_phase", np.zeros(len(frames)))
        cp = cycle_phase[i] if i < len(cycle_phase) else 0
        gait_phase = "Stance" if cp < 0.6 else "Swing"
        cycle_pct = cp * 100

        composite = create_dashboard_frame(
            frame, display_angles, summary, dashboard, gait_phase, cycle_pct,
            "Video Analysis"
        )

        if writer:
            writer.write(composite)

        if display:
            cv2.imshow("Movement Analytics — Video Analysis", composite)
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


def generate_comparison_report(output_path: str, fps: int = 30, n_cycles: int = 4):
    """Generate a visual comparison report of MQS across all gait profiles."""

    print("Generating profile comparison report...")
    results = []
    for name, profile in GAIT_PROFILES.items():
        _, ar, al, _ = generate_frames(profile.params, fps=fps, n_cycles=n_cycles)
        summary = compute_gait_summary(ar, al, fps=fps)
        results.append((name, profile, summary))

    results.sort(key=lambda x: x[2]["movement_quality_score"], reverse=True)

    w, h = 1200, 800
    img = np.full((h, w, 3), (20, 20, 25), dtype=np.uint8)

    cv2.putText(img, "MOVEMENT QUALITY SCORE — PROFILE COMPARISON", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 255), 2, cv2.LINE_AA)
    cv2.line(img, (30, 55), (w - 30, 55), (55, 55, 60), 2)

    domain_colors = {
        "kinematics": (80, 200, 220),
        "smoothness": (200, 160, 80),
        "symmetry": (180, 100, 255),
        "coordination": (140, 200, 100),
        "variability": (100, 255, 180),
        "temporal": (220, 120, 160),
    }

    headers = ["Profile", "MQS", "Kin", "Smo", "Sym", "Crd", "Var", "Tmp"]
    header_x = [30, 200, 310, 420, 530, 640, 750, 860]
    for hx, ht in zip(header_x, headers):
        cv2.putText(img, ht, (hx, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (150, 160, 170), 1, cv2.LINE_AA)

    y = 105
    bar_max_w = 100

    for name, profile, summary in results:
        mqs = summary["movement_quality_score"]
        domains = {
            "kinematics": summary.get("mqs_kinematics", 0),
            "smoothness": summary.get("mqs_smoothness", 0),
            "symmetry": summary.get("mqs_symmetry", 0),
            "coordination": summary.get("mqs_coordination", 0),
            "variability": summary.get("mqs_variability", 0),
            "temporal": summary.get("mqs_temporal", 0),
        }

        if mqs >= 95:
            score_color = (80, 200, 80)
        elif mqs >= 80:
            score_color = (50, 200, 255)
        else:
            score_color = (60, 60, 255)

        cv2.putText(img, name, (30, y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (200, 210, 220), 1, cv2.LINE_AA)

        cv2.putText(img, f"{mqs:.0f}", (200, y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, score_color, 2, cv2.LINE_AA)

        mqs_bar_w = int(mqs / 100 * bar_max_w)
        cv2.rectangle(img, (240, y + 5), (240 + bar_max_w, y + 20), (45, 45, 50), -1)
        cv2.rectangle(img, (240, y + 5), (240 + mqs_bar_w, y + 20), score_color, -1)

        domain_keys = [
            "kinematics", "smoothness", "symmetry",
            "coordination", "variability", "temporal",
        ]
        domain_x_offsets = [310, 420, 530, 640, 750, 860]

        for dk, dx in zip(domain_keys, domain_x_offsets):
            val = domains[dk]
            dc = domain_colors[dk]
            cv2.putText(img, f"{val:.0f}", (dx, y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, dc, 1, cv2.LINE_AA)
            bw = int(val / 100 * 80)
            cv2.rectangle(img, (dx + 35, y + 7), (dx + 35 + 80, y + 17), (45, 45, 50), -1)
            cv2.rectangle(img, (dx + 35, y + 7), (dx + 35 + bw, y + 17), dc, -1)

        y += 35
        cv2.line(img, (30, y - 10), (w - 30, y - 10), (40, 40, 45), 1)

    # Legend
    y += 15
    cv2.putText(img, "Domains:", (30, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (120, 130, 140), 1, cv2.LINE_AA)
    legend_items = [("Kin=Kinematics 25%", "kinematics"), ("Smo=Smoothness 18%", "smoothness"),
                    ("Sym=Symmetry 18%", "symmetry"), ("Crd=Coordination 14%", "coordination"),
                    ("Var=Variability 13%", "variability"), ("Tmp=Temporal 12%", "temporal")]
    lx = 110
    for label, dk in legend_items:
        cv2.circle(img, (lx, y + 10), 4, domain_colors[dk], -1)
        cv2.putText(img, label, (lx + 10, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.30, (120, 130, 140), 1, cv2.LINE_AA)
        lx += 165

    # Key findings
    y += 35
    key_text = (
        "Key: ROM 35-50 hip, 50-70 knee | SPARC -2.0 to -1.3 | SI <10%"
    )
    cv2.putText(img, key_text, (30, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 110, 120), 1, cv2.LINE_AA)
    y += 20
    cv2.putText(img, "Cadence 90-130 spm | Stride CV <4% | Stride time 0.8-1.3s",
                (30, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 110, 120), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print(f"Comparison report saved to {output_path}")


def run_benchmark(output_path: str | None = None, fps: int = 30, n_cycles: int = 6):
    """Run MQS benchmark across all profiles, output JSON for reproducibility."""
    import json

    results = {}
    for name, profile in GAIT_PROFILES.items():
        _, ar, al, _ = generate_frames(profile.params, fps=fps, n_cycles=n_cycles)
        summary = compute_gait_summary(ar, al, fps=fps)
        entry = {
            "mqs": round(summary["movement_quality_score"], 1),
            "domains": {},
            "completeness": {},
            "key_metrics": {},
        }
        for k, v in summary.items():
            if k.startswith("mqs_") and k.endswith("_completeness"):
                ckey = k.replace("mqs_", "").replace("_completeness", "")
                entry["completeness"][ckey] = round(v, 2)
            elif (k.startswith("mqs_")
                  and k not in ("mqs_overall_completeness",
                                "mqs_sufficient_evidence")):
                entry["domains"][k.replace("mqs_", "")] = round(v, 1)
            elif k in ("cadence", "stride_time_mean", "stride_time_CV", "n_strides"):
                is_nan = isinstance(v, float) and np.isnan(v)
                entry["key_metrics"][k] = None if is_nan else round(v, 2)
            elif (k.endswith("_ROM") or k.endswith("_SI")
                  or k.endswith("_SPARC") or k.endswith("_CRP_MAD")
                  or k.endswith("_waveform_sym")):
                entry["key_metrics"][k] = round(v, 2)
        results[name] = entry

    output = {
        "benchmark": "movement_quality_score",
        "version": "1.3.0",
        "n_domains": 6,
        "fps": fps,
        "n_cycles": n_cycles,
        "profiles": results,
    }

    text = json.dumps(output, indent=2)
    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(text)
        print(f"Benchmark saved to {output_path}")
    else:
        print(text)


def generate_sensitivity_report(output_path: str, fps: int = 30, n_cycles: int = 4):
    """Generate sensitivity analysis plots showing MQS vs parameter variation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .generators.gait_model import GaitParameters

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("MQS Sensitivity Analysis", fontsize=14, fontweight="bold")

    sweeps = [
        ("Knee ROM", "knee_rom", np.linspace(15, 70, 12)),
        ("Noise Level", "noise_level", np.linspace(0, 6, 13)),
        ("Asymmetry", "asymmetry", np.linspace(0, 0.5, 11)),
    ]

    for ax, (label, param, values) in zip(axes, sweeps):
        mqs_vals = []
        domain_data = {d: [] for d in ["kinematics", "smoothness", "symmetry",
                                        "coordination", "variability", "temporal"]}
        for v in values:
            params = GaitParameters(**{param: v})
            _, ar, al, _ = generate_frames(params, fps=fps, n_cycles=n_cycles)
            summary = compute_gait_summary(ar, al, fps=fps)
            mqs_vals.append(summary["movement_quality_score"])
            for d in domain_data:
                domain_data[d].append(summary.get(f"mqs_{d}", 50))

        ax.plot(values, mqs_vals, "k-", linewidth=2.5, label="MQS")
        colors = {"kinematics": "#52C8DC", "smoothness": "#C8A050",
                  "symmetry": "#B464FF", "coordination": "#8CC864",
                  "variability": "#64FFB4", "temporal": "#DC78A0"}
        for d, color in colors.items():
            ax.plot(values, domain_data[d], "--", color=color, alpha=0.7, label=d)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Score (0–100)", fontsize=11)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sensitivity report saved to {output_path}")


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
    parser.add_argument(
        "--compare", action="store_true",
        help="Generate MQS comparison report across all profiles"
    )
    parser.add_argument(
        "--video", "-v", default=None,
        help="Input video file for pose estimation analysis"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run MQS benchmark across all profiles (outputs JSON)"
    )
    parser.add_argument(
        "--sensitivity", action="store_true",
        help="Generate MQS sensitivity analysis plots (PNG)"
    )
    args = parser.parse_args()

    if args.sensitivity:
        out = args.output or "output/mqs_sensitivity.png"
        generate_sensitivity_report(out, fps=args.fps, n_cycles=args.cycles)
        return

    if args.benchmark:
        run_benchmark(args.output, fps=args.fps, n_cycles=args.cycles)
        return

    if args.compare:
        out = args.output or "output/mqs_comparison.png"
        generate_comparison_report(out, fps=args.fps, n_cycles=args.cycles)
        return

    if args.video:
        run_video_analysis(
            args.video, args.output,
            display=not args.no_display,
            target_fps=args.fps if args.fps != 30 else None,
        )
        return

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
