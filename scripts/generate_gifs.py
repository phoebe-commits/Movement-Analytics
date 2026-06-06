#!/usr/bin/env python3
"""Generate animated GIFs for the variance study results.

Reads processed results from data/results/ and creates publication-quality
animated GIFs showing statistical significance visually.

Usage:
    python scripts/generate_gifs.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIGURES_DIR = PROJECT_ROOT / "data" / "figures"
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"

METRIC_DOMAINS = {
    "hip_ROM": "Kinematics", "knee_ROM": "Kinematics",
    "ankle_ROM": "Kinematics", "pelvis_obliquity": "Kinematics",
    "trunk_lean_ROM": "Kinematics", "shoulder_ROM": "Kinematics",
    "elbow_ROM": "Kinematics",
    "hip_SPARC": "Smoothness", "knee_SPARC": "Smoothness",
    "hip_SI": "Symmetry", "knee_SI": "Symmetry",
    "ankle_SI": "Symmetry", "hip_waveform_sym": "Symmetry",
    "arm_swing_SI": "Symmetry",
    "cadence": "Temporal", "stride_time": "Temporal",
    "stride_time_CV": "Variability", "kinematic_CV": "Variability",
    "hip_CRP_MAD": "Coordination", "hip_knee_CRP": "Coordination",
    "arm_swing_ROM": "Upper Body", "double_support": "Temporal",
    "MQS": "Composite", "GDI": "Composite",
}


def load_data():
    df = pd.read_csv(RESULTS_DIR / "all_metrics.csv")
    var_df = pd.read_csv(RESULTS_DIR / "variance_tests.csv")
    boot_df = pd.read_csv(RESULTS_DIR / "bootstrap_results.csv")
    return df, var_df, boot_df


def gif_forest_plot(var_df, boot_df):
    """Animated forest plot: bars grow from 1.0 line outward."""
    merged = var_df.merge(boot_df, on="metric", how="left")
    merged = merged.sort_values("var_ratio", ascending=True).reset_index(drop=True)

    n = len(merged)
    n_frames = 60

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.4)))
    fig.set_facecolor("white")

    def animate(frame):
        ax.clear()
        progress = min(1.0, frame / (n_frames * 0.7))
        ease = 1 - (1 - progress) ** 3

        y_pos = range(n)
        current_ratios = 1.0 + (merged.var_ratio.values - 1.0) * ease
        colors = ["#D32F2F" if p < 0.05 else "#9E9E9E" for p in merged.levene_p]

        ax.barh(y_pos, current_ratios, color=colors, alpha=0.7, height=0.6)

        if frame > n_frames * 0.7 and "ci_lo" in merged.columns:
            ci_progress = min(1.0, (frame - n_frames * 0.7) / (n_frames * 0.3))
            ci_ease = 1 - (1 - ci_progress) ** 2
            for i, (_, row) in enumerate(merged.iterrows()):
                if not np.isnan(row.get("ci_lo", np.nan)):
                    lo = 1.0 + (row.ci_lo - 1.0) * ci_ease
                    hi = 1.0 + (row.ci_hi - 1.0) * ci_ease
                    ax.plot([lo, hi], [i, i], color="black", linewidth=1.5, zorder=5)
                    ax.plot([lo, lo], [i - 0.15, i + 0.15], color="black", linewidth=1.5, zorder=5)
                    ax.plot([hi, hi], [i - 0.15, i + 0.15], color="black", linewidth=1.5, zorder=5)

        ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(merged.metric, fontsize=9)
        ax.set_xlabel("Variance Ratio (Control / Runway)", fontsize=11)
        ax.set_title(
            "Variance Ratio Forest Plot\n"
            "Red = significant (p<0.05), bars right of 1.0 = control has MORE variance",
            fontsize=11, fontweight="bold",
        )

        if frame >= n_frames - 1:
            ax.text(
                0.98, 0.02,
                f"Median ratio: {merged.var_ratio.median():.2f}×",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )

        ax.set_xlim(0, max(merged.var_ratio.max() * 1.1, 10))
        plt.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=50, repeat=True)
    out_path = ASSETS_DIR / "variance_forest_animated.gif"
    anim.save(str(out_path), writer="pillow", fps=20, dpi=100)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def gif_pca_scatter(df):
    """Animated PCA scatter: points appear, then ellipses draw."""
    metrics = [m for m in METRIC_DOMAINS if m not in ("MQS", "GDI")]
    usable = [m for m in metrics if df[m].notna().sum() >= len(df) * 0.5]
    df_mv = df[["group"] + usable].dropna()
    if len(df_mv) < 8:
        print("  -> SKIPPED pca_scatter (insufficient data)")
        return

    X = df_mv[usable].values
    y = (df_mv["group"] == "runway").astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(5, X_scaled.shape[1], X_scaled.shape[0] - 1))
    X_pca = pca.fit_transform(X_scaled)

    r_mask = y == 1
    c_mask = y == 0
    n_points = len(X_pca)
    n_frames = n_points + 30

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.set_facecolor("white")

    def confidence_ellipse(x_data, y_data, ax_ref, n_std=2.0, **kwargs):
        if len(x_data) < 3:
            return None
        cov = np.cov(x_data, y_data)
        pearson = cov[0, 1] / (np.sqrt(cov[0, 0] * cov[1, 1]) + 1e-12)
        ell_rx = np.sqrt(1 + pearson)
        ell_ry = np.sqrt(1 - pearson)
        ell = Ellipse((0, 0), width=ell_rx * 2, height=ell_ry * 2, **kwargs)
        sx = np.sqrt(cov[0, 0]) * n_std
        sy = np.sqrt(cov[1, 1]) * n_std
        mx, my = np.mean(x_data), np.mean(y_data)
        transf = (
            transforms.Affine2D().rotate_deg(45).scale(sx, sy).translate(mx, my)
        )
        ell.set_transform(transf + ax_ref.transData)
        ax_ref.add_patch(ell)
        return ell

    all_x = X_pca[:, 0]
    all_y = X_pca[:, 1]
    x_margin = (all_x.max() - all_x.min()) * 0.15
    y_margin = (all_y.max() - all_y.min()) * 0.15

    evr = pca.explained_variance_ratio_

    def animate(frame):
        ax.clear()

        n_show = min(frame + 1, n_points)

        shown_r = r_mask[:n_show]
        shown_c = c_mask[:n_show]

        if shown_r.any():
            ax.scatter(
                X_pca[:n_show][shown_r, 0], X_pca[:n_show][shown_r, 1],
                c="#2196F3", s=80, alpha=0.7, label="Runway",
                edgecolors="black", linewidth=0.5, zorder=3,
            )
        if shown_c.any():
            ax.scatter(
                X_pca[:n_show][shown_c, 0], X_pca[:n_show][shown_c, 1],
                c="#FF5722", s=80, alpha=0.7, label="Control",
                edgecolors="black", linewidth=0.5, zorder=3,
            )

        if frame >= n_points:
            ell_progress = min(1.0, (frame - n_points) / 20)
            ell_alpha = 0.15 * ell_progress

            if r_mask.sum() >= 3:
                confidence_ellipse(
                    X_pca[r_mask, 0], X_pca[r_mask, 1], ax,
                    n_std=2.0 * ell_progress,
                    edgecolor="#1565C0", linewidth=2, linestyle="--",
                    facecolor="#2196F3", alpha=ell_alpha,
                )
            if c_mask.sum() >= 3:
                confidence_ellipse(
                    X_pca[c_mask, 0], X_pca[c_mask, 1], ax,
                    n_std=2.0 * ell_progress,
                    edgecolor="#BF360C", linewidth=2, linestyle="--",
                    facecolor="#FF5722", alpha=ell_alpha,
                )

        if frame >= n_frames - 5 and r_mask.sum() >= 3 and c_mask.sum() >= 3:
            r_spread = np.trace(np.cov(X_pca[r_mask, :2].T))
            c_spread = np.trace(np.cov(X_pca[c_mask, :2].T))
            ax.text(
                0.02, 0.98,
                f"Spread (trace): Runway={r_spread:.2f}, Control={c_spread:.2f}\n"
                f"Ratio: {c_spread / r_spread:.1f}×",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )

        ax.set_xlabel(f"PC1 ({evr[0]:.1%} variance)", fontsize=12)
        ax.set_ylabel(f"PC2 ({evr[1]:.1%} variance)", fontsize=12)
        ax.set_title(
            "PCA: Kinematic Feature Space\n"
            "Ellipses = 95% confidence regions (smaller = lower variance)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
        ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)
        plt.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=80, repeat=True)
    out_path = ASSETS_DIR / "pca_scatter_animated.gif"
    anim.save(str(out_path), writer="pillow", fps=15, dpi=100)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def gif_domain_summary(var_df):
    """Animated domain summary: bars fill in with magnitude."""
    domains = var_df.groupby("domain").agg(
        median_ratio=("var_ratio", "median"),
        n_metrics=("metric", "count"),
        n_sig=("levene_p", lambda x: (x < 0.05).sum()),
    ).reset_index()
    domains = domains.sort_values("median_ratio", ascending=True).reset_index(drop=True)

    n = len(domains)
    n_frames = 50

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor("white")

    def animate(frame):
        ax.clear()
        progress = min(1.0, frame / (n_frames * 0.6))
        ease = 1 - (1 - progress) ** 3

        current_ratios = ease * domains.median_ratio.values
        colors = ["#D32F2F" if r > 1.5 else "#FF9800" if r > 1 else "#4CAF50"
                  for r in domains.median_ratio]

        bars = ax.barh(
            domains.domain, current_ratios,
            color=colors, alpha=0.8, edgecolor="black", linewidth=0.5,
        )

        if frame > n_frames * 0.6:
            label_progress = min(1.0, (frame - n_frames * 0.6) / (n_frames * 0.3))
            if label_progress > 0.5:
                for bar, (_, row) in zip(bars, domains.iterrows()):
                    ax.text(
                        bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                        f"{row.median_ratio:.1f}× ({int(row.n_sig)}/{int(row.n_metrics)} sig)",
                        va="center", fontsize=10, alpha=min(1.0, (label_progress - 0.5) * 2),
                    )

        ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Median Variance Ratio (Control / Runway)", fontsize=12)
        ax.set_title(
            "Variance Ratio by Domain\n"
            "Red = strong support, orange = moderate, green = no difference",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlim(0, max(domains.median_ratio.max() * 1.15, 5))
        plt.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=60, repeat=True)
    out_path = ASSETS_DIR / "domain_summary_animated.gif"
    anim.save(str(out_path), writer="pillow", fps=15, dpi=100)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def gif_bootstrap_distribution(df, var_df):
    """Animated bootstrap: histogram fills in as resamples accumulate."""
    target_metric = "GDI"
    runway = df[df.group == "runway"][target_metric].dropna().values
    control = df[df.group == "control"][target_metric].dropna().values

    if len(runway) < 3 or len(control) < 3:
        print("  -> SKIPPED bootstrap_distribution (insufficient GDI data)")
        return

    rng = np.random.default_rng(42)
    n_boot = 2000
    all_ratios = []
    for _ in range(n_boot):
        rb = rng.choice(runway, size=len(runway), replace=True)
        cb = rng.choice(control, size=len(control), replace=True)
        rv = rb.var(ddof=1)
        if rv > 1e-12:
            all_ratios.append(cb.var(ddof=1) / rv)
    all_ratios = np.array(all_ratios)

    n_frames = 60
    steps = np.linspace(10, len(all_ratios), n_frames).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor("white")

    x_max = min(np.percentile(all_ratios, 99), 300)
    x_min = max(0, np.percentile(all_ratios, 1))
    bins = np.linspace(x_min, x_max, 50)

    def animate(frame):
        ax.clear()
        n_show = steps[min(frame, len(steps) - 1)]
        subset = all_ratios[:n_show]

        ax.hist(subset, bins=bins, color="#2196F3", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="No difference (ratio=1)")

        median_val = np.median(subset)
        ax.axvline(x=median_val, color="#FF9800", linestyle="-", linewidth=2,
                   label=f"Median: {median_val:.1f}×")

        ci_lo, ci_hi = np.percentile(subset, [2.5, 97.5])
        ax.axvspan(ci_lo, ci_hi, alpha=0.15, color="#4CAF50", label=f"95% CI: [{ci_lo:.1f}, {ci_hi:.1f}]")

        above_1 = (subset > 1).mean()
        ax.set_title(
            f"Bootstrap Variance Ratio Distribution ({target_metric})\n"
            f"n={n_show:,} resamples | P(ratio > 1) = {above_1:.1%}",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Variance Ratio (Control / Runway)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlim(x_min, x_max)
        plt.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=60, repeat=True)
    out_path = ASSETS_DIR / "bootstrap_distribution_animated.gif"
    anim.save(str(out_path), writer="pillow", fps=15, dpi=100)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def gif_effect_size_waterfall(var_df):
    """Animated effect size waterfall: Cohen's d bars drop in one by one."""
    mean_df = pd.read_csv(RESULTS_DIR / "mean_comparison.csv")
    merged = mean_df.sort_values("cohens_d", key=abs, ascending=False).reset_index(drop=True)

    n = len(merged)
    n_frames = n + 20

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_facecolor("white")

    def animate(frame):
        ax.clear()
        n_show = min(frame + 1, n)

        subset = merged.iloc[:n_show]
        colors = ["#D32F2F" if p < 0.05 else "#9E9E9E" for p in subset.u_p]
        x_pos = range(n_show)

        ax.bar(x_pos, subset.cohens_d.values, color=colors, alpha=0.8,
               edgecolor="black", linewidth=0.3, width=0.7)

        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.axhline(y=0.8, color="#4CAF50", linestyle=":", alpha=0.5, linewidth=1)
        ax.axhline(y=-0.8, color="#4CAF50", linestyle=":", alpha=0.5, linewidth=1)

        if n_show <= 12:
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(subset.metric, rotation=45, ha="right", fontsize=8)
        else:
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(subset.metric, rotation=60, ha="right", fontsize=7)

        ax.set_ylabel("Cohen's d (effect size)", fontsize=11)
        ax.set_title(
            f"Effect Size Waterfall ({n_show}/{n} metrics)\n"
            "Red = significant (p<0.05) | Dashed = large effect threshold (|d|=0.8)",
            fontsize=11, fontweight="bold",
        )

        y_max = max(abs(merged.cohens_d.max()), abs(merged.cohens_d.min())) * 1.2
        ax.set_ylim(-y_max, y_max)
        ax.set_xlim(-0.5, n - 0.5)
        plt.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=80, repeat=True)
    out_path = ASSETS_DIR / "effect_size_waterfall.gif"
    anim.save(str(out_path), writer="pillow", fps=12, dpi=100)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, var_df, boot_df = load_data()
    print(f"  {len(df)} samples, {len(var_df)} metrics tested")

    print("\nGenerating animated GIFs:")

    print("  [1/5] Forest plot...")
    gif_forest_plot(var_df, boot_df)

    print("  [2/5] PCA scatter...")
    gif_pca_scatter(df)

    print("  [3/5] Domain summary...")
    gif_domain_summary(var_df)

    print("  [4/5] Bootstrap distribution...")
    gif_bootstrap_distribution(df, var_df)

    print("  [5/5] Effect size waterfall...")
    gif_effect_size_waterfall(var_df)

    print(f"\nAll GIFs saved to {ASSETS_DIR}/")


if __name__ == "__main__":
    main()
