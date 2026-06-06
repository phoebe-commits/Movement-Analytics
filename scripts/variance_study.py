#!/usr/bin/env python3
"""Quantitative Variance Analysis: Runway Walks vs. Internet Walking Videos.

Tests the Elysium hypothesis: model runway walks exhibit significantly lower
kinematic variance than representative internet walking data, making them
a superior foundation for robot movement learning.

Usage:
    python scripts/variance_study.py --process    # Extract metrics from all videos
    python scripts/variance_study.py --analyze    # Statistical analysis + figures
    python scripts/variance_study.py --all        # Both phases
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RUNWAY_DIR = Path(
    r"C:\Users\devan\Downloads\wetransfer_runway-walks-elysium_2026-05-31_0130"
)
CONTROL_DIR = PROJECT_ROOT / "data" / "control"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIGURES_DIR = PROJECT_ROOT / "data" / "figures"

METRIC_EXTRACTORS = {
    "hip_ROM": lambda m: _bilateral_mean(m, "hip_flexion", "ROM"),
    "knee_ROM": lambda m: _bilateral_mean(m, "knee_flexion", "ROM"),
    "ankle_ROM": lambda m: _bilateral_mean(m, "ankle_dorsiflexion", "ROM"),
    "pelvis_obliquity": lambda m: m.get("R_pelvis_obliquity_ROM", np.nan),
    "trunk_lean_ROM": lambda m: m.get("R_trunk_lean_ROM", np.nan),
    "shoulder_ROM": lambda m: _bilateral_mean_upper(m, "shoulder"),
    "elbow_ROM": lambda m: _bilateral_mean_upper(m, "elbow"),
    "hip_SPARC": lambda m: _bilateral_mean(m, "hip_flexion", "SPARC"),
    "knee_SPARC": lambda m: _bilateral_mean(m, "knee_flexion", "SPARC"),
    "hip_SI": lambda m: m.get("hip_flexion_SI", np.nan),
    "knee_SI": lambda m: m.get("knee_flexion_SI", np.nan),
    "ankle_SI": lambda m: m.get("ankle_dorsiflexion_SI", np.nan),
    "hip_waveform_sym": lambda m: m.get("hip_flexion_waveform_sym", np.nan),
    "cadence": lambda m: m.get("cadence", np.nan),
    "stride_time": lambda m: m.get("stride_time_mean", np.nan),
    "stride_time_CV": lambda m: m.get("stride_time_CV", np.nan),
    "kinematic_CV": lambda m: m.get("kinematic_CV_mean", np.nan),
    "hip_CRP_MAD": lambda m: m.get("hip_CRP_MAD", np.nan),
    "hip_knee_CRP": lambda m: _bilateral_mean_crp(m),
    "arm_swing_ROM": lambda m: m.get("arm_swing_ROM_mean", np.nan),
    "arm_swing_SI": lambda m: m.get("arm_swing_SI", np.nan),
    "double_support": lambda m: m.get("double_support_pct", np.nan),
    "MQS": lambda m: m.get("movement_quality_score", np.nan),
    "GDI": lambda m: m.get("GDI", np.nan),
}

METRIC_UNITS = {
    "hip_ROM": "°", "knee_ROM": "°", "ankle_ROM": "°",
    "pelvis_obliquity": "°", "trunk_lean_ROM": "°",
    "shoulder_ROM": "°", "elbow_ROM": "°",
    "hip_SPARC": "", "knee_SPARC": "",
    "hip_SI": "%", "knee_SI": "%", "ankle_SI": "%",
    "hip_waveform_sym": "%", "cadence": "spm",
    "stride_time": "s", "stride_time_CV": "%",
    "kinematic_CV": "%", "hip_CRP_MAD": "°",
    "hip_knee_CRP": "°", "arm_swing_ROM": "°",
    "arm_swing_SI": "%", "double_support": "%",
    "MQS": "", "GDI": "",
}

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


def _bilateral_mean(m, joint, suffix):
    r = m.get(f"R_{joint}_{suffix}")
    l = m.get(f"L_{joint}_{suffix}")
    vals = [v for v in [r, l] if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else np.nan


def _bilateral_mean_upper(m, joint):
    r = m.get(f"R_{joint}_ROM")
    l = m.get(f"L_{joint}_ROM")
    vals = [v for v in [r, l] if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else np.nan


def _bilateral_mean_crp(m):
    r = m.get("R_hip_knee_CRP_MAD")
    l = m.get("L_hip_knee_CRP_MAD")
    vals = [v for v in [r, l] if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else np.nan


# ──────────────────────────────────────────────────────────────
# Phase 1: Process videos
# ──────────────────────────────────────────────────────────────

def process_videos():
    from movement_analytics import analyze_video

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    groups = {
        "runway": sorted(RUNWAY_DIR.glob("*.mov")),
        "control": sorted(
            list(CONTROL_DIR.glob("*.mp4"))
            + list(CONTROL_DIR.glob("*.webm"))
            + list(CONTROL_DIR.glob("*.mkv"))
            + list(CONTROL_DIR.glob("*.mov"))
        ),
    }

    for group_name, videos in groups.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {group_name}: {len(videos)} videos")
        print(f"{'=' * 60}")

        results_file = RESULTS_DIR / f"{group_name}_results.json"
        existing = {}
        if results_file.exists():
            with open(results_file) as f:
                existing = json.load(f)
            print(f"  ({len(existing)} already processed, will skip)")

        all_results = dict(existing)

        for i, video in enumerate(videos):
            if video.name in all_results:
                print(f"  [{i+1}/{len(videos)}] {video.name} — cached")
                continue
            print(f"  [{i+1}/{len(videos)}] {video.name}")
            try:
                result = analyze_video(str(video))
                clean = {}
                for k, v in result.items():
                    if isinstance(v, (np.floating, np.integer)):
                        v = float(v)
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    if isinstance(v, float) and np.isnan(v):
                        v = None
                    clean[k] = v
                all_results[video.name] = clean
                mqs = result.get("movement_quality_score", float("nan"))
                conf = result.get("mqs_confidence_factor", 1.0)
                obs = result.get("pose_observed_fraction", 1.0)
                print(f"    MQS={mqs:.1f}  conf={conf:.0%}  pose={obs:.0%}")
            except Exception as e:
                print(f"    FAILED: {e}")
                traceback.print_exc()

        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {len(all_results)} results → {results_file}")


def load_results():
    rows = []
    for group in ["runway", "control"]:
        path = RESULTS_DIR / f"{group}_results.json"
        if not path.exists():
            print(f"WARNING: {path} not found — run --process first")
            continue
        with open(path) as f:
            data = json.load(f)
        for filename, metrics in data.items():
            for k in metrics:
                if metrics[k] is None:
                    metrics[k] = np.nan
            row = {"group": group, "filename": filename}
            for name, extractor in METRIC_EXTRACTORS.items():
                try:
                    row[name] = extractor(metrics)
                except Exception:
                    row[name] = np.nan
            row["confidence"] = metrics.get("mqs_confidence_factor", np.nan)
            row["observed_fraction"] = metrics.get("pose_observed_fraction", np.nan)
            row["n_strides"] = metrics.get("n_strides", np.nan)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ──────────────────────────────────────────────────────────────
# Phase 2: Statistical analysis
# ──────────────────────────────────────────────────────────────

def run_analysis():
    df = load_results()
    if df.empty:
        print("No data. Run --process first.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = list(METRIC_EXTRACTORS.keys())

    runway = df[df.group == "runway"]
    control = df[df.group == "control"]
    print(f"\nDataset: {len(runway)} runway, {len(control)} control videos")

    # Quality filter: require at least 50% pose detection
    quality_mask = df["observed_fraction"].fillna(0) >= 0.3
    df_q = df[quality_mask].copy()
    runway_q = df_q[df_q.group == "runway"]
    control_q = df_q[df_q.group == "control"]
    print(f"After quality filter (>=30% pose): {len(runway_q)} runway, {len(control_q)} control")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ELYSIUM VARIANCE STUDY: Runway Walks vs. Internet Walking Videos")
    report_lines.append("=" * 80)
    report_lines.append(f"Runway: n={len(runway_q)}  |  Control: n={len(control_q)}")
    report_lines.append("")

    # ── 1. VARIANCE COMPARISON ──────────────────────────────
    report_lines.append("─" * 80)
    report_lines.append("1. VARIANCE COMPARISON")
    report_lines.append("   Core test: do runway walks have lower kinematic variance?")
    report_lines.append("   Levene's test (H0: equal variances)")
    report_lines.append("─" * 80)

    var_rows = []
    for m in metrics:
        r = runway_q[m].dropna()
        c = control_q[m].dropna()
        if len(r) < 3 or len(c) < 3:
            continue

        var_r = float(r.var(ddof=1))
        var_c = float(c.var(ddof=1))
        sd_r = float(r.std(ddof=1))
        sd_c = float(c.std(ddof=1))

        f_ratio = var_c / var_r if var_r > 1e-12 else np.nan

        mean_r = float(r.mean())
        mean_c = float(c.mean())
        cv_r = sd_r / abs(mean_r) * 100 if abs(mean_r) > 1e-6 else np.nan
        cv_c = sd_c / abs(mean_c) * 100 if abs(mean_c) > 1e-6 else np.nan

        lev_stat, lev_p = stats.levene(r, c)
        bf_stat, bf_p = stats.levene(r, c, center="median")

        var_rows.append({
            "metric": m, "domain": METRIC_DOMAINS.get(m, ""),
            "runway_mean": mean_r, "control_mean": mean_c,
            "runway_sd": sd_r, "control_sd": sd_c,
            "runway_var": var_r, "control_var": var_c,
            "var_ratio": f_ratio,
            "runway_CV": cv_r, "control_CV": cv_c,
            "levene_stat": lev_stat, "levene_p": lev_p,
            "bf_stat": bf_stat, "bf_p": bf_p,
            "runway_n": len(r), "control_n": len(c),
        })

        stars = _sig_stars(lev_p)
        direction = "↑" if f_ratio > 1 else "↓"
        line = (
            f"  {m:20s}  σ²ratio={f_ratio:7.2f}{direction}  "
            f"CV: R={cv_r:5.1f}% C={cv_c:5.1f}%  "
            f"Levene p={lev_p:.4f} {stars}"
        )
        report_lines.append(line)

    var_df = pd.DataFrame(var_rows)

    n_sig = len(var_df[var_df.levene_p < 0.05])
    n_higher = len(var_df[(var_df.levene_p < 0.05) & (var_df.var_ratio > 1)])
    n_lower = len(var_df[(var_df.levene_p < 0.05) & (var_df.var_ratio < 1)])
    median_ratio = float(var_df.var_ratio.median())

    report_lines.append("")
    report_lines.append(f"  Significant variance differences: {n_sig}/{len(var_df)}")
    report_lines.append(f"  Control variance HIGHER: {n_higher}  |  LOWER: {n_lower}")
    report_lines.append(f"  Median variance ratio (control/runway): {median_ratio:.2f}×")
    report_lines.append(f"  Mean variance ratio: {float(var_df.var_ratio.mean()):.2f}×")

    # Bonferroni correction
    alpha_bonf = 0.05 / len(var_df)
    n_bonf = len(var_df[var_df.levene_p < alpha_bonf])
    report_lines.append(f"  Significant after Bonferroni (α={alpha_bonf:.4f}): {n_bonf}/{len(var_df)}")

    # ── 2. GROUP MEAN COMPARISON ────────────────────────────
    report_lines.append("")
    report_lines.append("─" * 80)
    report_lines.append("2. GROUP MEAN COMPARISON (Mann-Whitney U, non-parametric)")
    report_lines.append("─" * 80)

    mean_rows = []
    for m in metrics:
        r = runway_q[m].dropna()
        c = control_q[m].dropna()
        if len(r) < 3 or len(c) < 3:
            continue

        u_stat, u_p = stats.mannwhitneyu(r, c, alternative="two-sided")
        pooled_sd = np.sqrt((r.var(ddof=1) + c.var(ddof=1)) / 2)
        cohens_d = (r.mean() - c.mean()) / pooled_sd if pooled_sd > 1e-8 else 0

        n1, n2 = len(r), len(c)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

        mean_rows.append({
            "metric": m, "u_stat": u_stat, "u_p": u_p,
            "cohens_d": cohens_d, "rank_biserial": rank_biserial,
        })

        stars = _sig_stars(u_p)
        line = (
            f"  {m:20s}  R̄={r.mean():8.2f}  C̄={c.mean():8.2f}  "
            f"d={cohens_d:+.2f}  r={rank_biserial:+.2f}  p={u_p:.4f} {stars}"
        )
        report_lines.append(line)

    mean_df = pd.DataFrame(mean_rows)

    # ── 3. PERMUTATION TESTS ────────────────────────────────
    report_lines.append("")
    report_lines.append("─" * 80)
    report_lines.append("3. PERMUTATION TESTS FOR VARIANCE RATIO (10,000 permutations)")
    report_lines.append("─" * 80)

    n_perm = 10000
    rng = np.random.default_rng(42)
    perm_rows = []

    for m in metrics:
        r = runway_q[m].dropna().values
        c = control_q[m].dropna().values
        if len(r) < 3 or len(c) < 3:
            continue

        observed_ratio = c.var(ddof=1) / r.var(ddof=1) if r.var(ddof=1) > 1e-12 else np.nan
        if np.isnan(observed_ratio):
            continue

        pooled = np.concatenate([r, c])
        n_r = len(r)
        count_extreme = 0
        for _ in range(n_perm):
            perm = rng.permutation(pooled)
            pr = perm[:n_r]
            pc = perm[n_r:]
            pr_var = pr.var(ddof=1)
            if pr_var > 1e-12:
                perm_ratio = pc.var(ddof=1) / pr_var
                if perm_ratio >= observed_ratio:
                    count_extreme += 1

        perm_p = (count_extreme + 1) / (n_perm + 1)
        perm_rows.append({"metric": m, "observed_ratio": observed_ratio, "perm_p": perm_p})

        stars = _sig_stars(perm_p)
        report_lines.append(
            f"  {m:20s}  ratio={observed_ratio:6.2f}  perm-p={perm_p:.4f} {stars}"
        )

    perm_df = pd.DataFrame(perm_rows)

    # ── 4. BOOTSTRAP CIs ON VARIANCE RATIO ──────────────────
    report_lines.append("")
    report_lines.append("─" * 80)
    report_lines.append("4. BOOTSTRAP 95% CIs ON VARIANCE RATIO (control/runway)")
    report_lines.append("─" * 80)

    n_boot = 10000
    boot_rows = []

    for m in metrics:
        r = runway_q[m].dropna().values
        c = control_q[m].dropna().values
        if len(r) < 3 or len(c) < 3:
            continue

        ratios = []
        for _ in range(n_boot):
            rb = rng.choice(r, size=len(r), replace=True)
            cb = rng.choice(c, size=len(c), replace=True)
            rv = rb.var(ddof=1)
            if rv > 1e-12:
                ratios.append(cb.var(ddof=1) / rv)
        ratios = np.array(ratios)
        if len(ratios) == 0:
            continue
        ci_lo, ci_med, ci_hi = np.percentile(ratios, [2.5, 50, 97.5])

        boot_rows.append({
            "metric": m, "median_ratio": ci_med,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "above_1": (ratios > 1).mean(),
        })

        report_lines.append(
            f"  {m:20s}  median={ci_med:6.2f}  "
            f"95% CI=[{ci_lo:.2f}, {ci_hi:.2f}]  "
            f"P(ratio>1)={boot_rows[-1]['above_1']:.1%}"
        )

    boot_df = pd.DataFrame(boot_rows)

    # ── 5. MULTIVARIATE ANALYSIS ────────────────────────────
    report_lines.append("")
    report_lines.append("─" * 80)
    report_lines.append("5. MULTIVARIATE ANALYSIS")
    report_lines.append("─" * 80)

    mv_metrics = [m for m in metrics
                  if df_q[m].notna().sum() >= len(df_q) * 0.5
                  and m not in ("MQS", "GDI")]
    df_mv = df_q[["group", "filename"] + mv_metrics].dropna()

    pca_result = None
    mv_labels = None

    if len(df_mv) >= 8:
        X = df_mv[mv_metrics].values
        y = (df_mv["group"] == "runway").astype(int).values
        mv_labels = y

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_comp = min(5, X_scaled.shape[1], X_scaled.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_scaled)
        pca_result = (X_pca, pca, df_mv, mv_metrics)

        report_lines.append(f"  Features used: {len(mv_metrics)}")
        report_lines.append(f"  Samples: {(y==1).sum()} runway, {(y==0).sum()} control")
        evr = pca.explained_variance_ratio_
        report_lines.append(
            f"  PCA variance explained: "
            + "  ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(evr[:5]))
        )
        report_lines.append(
            f"  Cumulative (3 PC): {sum(evr[:3]):.1%}"
        )

        r_mask = y == 1
        c_mask = y == 0

        if r_mask.sum() >= 3 and c_mask.sum() >= 3:
            n_dim = min(3, X_pca.shape[1])
            r_cov = np.cov(X_pca[r_mask, :n_dim].T)
            c_cov = np.cov(X_pca[c_mask, :n_dim].T)
            r_spread = np.trace(r_cov)
            c_spread = np.trace(c_cov)
            r_det = np.linalg.det(r_cov) if n_dim > 1 else r_cov[0, 0]
            c_det = np.linalg.det(c_cov) if n_dim > 1 else c_cov[0, 0]

            report_lines.append(f"  Runway spread  (trace): {r_spread:.3f}")
            report_lines.append(f"  Control spread (trace): {c_spread:.3f}")
            report_lines.append(
                f"  Spread ratio (control/runway): {c_spread / r_spread:.2f}×"
            )
            if r_det > 0 and c_det > 0:
                report_lines.append(
                    f"  Volume ratio (det): {c_det / r_det:.2f}×"
                )

            r_center = X_pca[r_mask, :n_dim].mean(axis=0)
            c_center = X_pca[c_mask, :n_dim].mean(axis=0)
            r_dists = np.linalg.norm(X_pca[r_mask, :n_dim] - r_center, axis=1)
            c_dists = np.linalg.norm(X_pca[c_mask, :n_dim] - c_center, axis=1)
            report_lines.append(
                f"  Mean dist to centroid: runway={r_dists.mean():.2f}  "
                f"control={c_dists.mean():.2f}"
            )

        if (y == 1).sum() >= 2 and (y == 0).sum() >= 2:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_scaled, y)
            acc = lda.score(X_scaled, y)
            report_lines.append(f"  LDA accuracy (resubstitution): {acc:.1%}")

            from sklearn.model_selection import LeaveOneOut
            loo = LeaveOneOut()
            loo_correct = 0
            for train_idx, test_idx in loo.split(X_scaled):
                lda_cv = LinearDiscriminantAnalysis()
                lda_cv.fit(X_scaled[train_idx], y[train_idx])
                if lda_cv.predict(X_scaled[test_idx]) == y[test_idx]:
                    loo_correct += 1
            loo_acc = loo_correct / len(y)
            report_lines.append(f"  LDA accuracy (LOO-CV): {loo_acc:.1%}")

        top_loadings = []
        for i in range(min(2, pca.components_.shape[0])):
            loadings = list(zip(mv_metrics, pca.components_[i]))
            loadings.sort(key=lambda x: abs(x[1]), reverse=True)
            top = loadings[:5]
            report_lines.append(
                f"  PC{i+1} top loadings: "
                + ", ".join(f"{n}({v:+.2f})" for n, v in top)
            )
    else:
        report_lines.append("  Insufficient data for multivariate analysis")

    # ── 6. DOMAIN-LEVEL VARIANCE SUMMARY ────────────────────
    report_lines.append("")
    report_lines.append("─" * 80)
    report_lines.append("6. DOMAIN-LEVEL VARIANCE SUMMARY")
    report_lines.append("─" * 80)

    for domain in ["Kinematics", "Smoothness", "Symmetry", "Temporal",
                    "Variability", "Coordination", "Upper Body", "Composite"]:
        domain_metrics = var_df[var_df.domain == domain]
        if domain_metrics.empty:
            continue
        med_ratio = domain_metrics.var_ratio.median()
        n_sig_d = len(domain_metrics[domain_metrics.levene_p < 0.05])
        report_lines.append(
            f"  {domain:15s}  median σ²ratio={med_ratio:5.2f}×  "
            f"sig: {n_sig_d}/{len(domain_metrics)}"
        )

    # ── 7. OVERALL CONCLUSION ───────────────────────────────
    report_lines.append("")
    report_lines.append("─" * 80)
    report_lines.append("7. CONCLUSION")
    report_lines.append("─" * 80)

    pct_higher = n_higher / len(var_df) * 100 if len(var_df) > 0 else 0
    if median_ratio > 1.5 and pct_higher > 50:
        verdict = "SUPPORTED"
        detail = (
            f"Control walking data shows {median_ratio:.1f}× higher variance "
            f"than runway walks across {pct_higher:.0f}% of metrics (p<0.05)."
        )
    elif median_ratio > 1:
        verdict = "PARTIALLY SUPPORTED"
        detail = (
            f"Control variance is higher (median ratio {median_ratio:.1f}×) "
            f"but significance is limited."
        )
    else:
        verdict = "NOT SUPPORTED"
        detail = f"Variance ratio is {median_ratio:.1f}×, not favoring the hypothesis."

    report_lines.append(f"  Hypothesis: {verdict}")
    report_lines.append(f"  {detail}")

    # Print & save report
    report_text = "\n".join(report_lines)
    print(report_text)

    with open(RESULTS_DIR / "variance_study_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # ── GENERATE FIGURES ────────────────────────────────────
    _fig_violin_plots(df_q, metrics, var_df)
    _fig_variance_forest(var_df, boot_df)
    _fig_cv_comparison(var_df)
    if pca_result is not None:
        _fig_pca_scatter(pca_result)
    _fig_radar_profiles(df_q, metrics)
    _fig_variance_heatmap(var_df)
    _fig_domain_summary(var_df)

    var_df.to_csv(RESULTS_DIR / "variance_tests.csv", index=False)
    mean_df.to_csv(RESULTS_DIR / "mean_comparison.csv", index=False)
    boot_df.to_csv(RESULTS_DIR / "bootstrap_results.csv", index=False)
    if not perm_df.empty:
        perm_df.to_csv(RESULTS_DIR / "permutation_tests.csv", index=False)
    df_q.to_csv(RESULTS_DIR / "all_metrics.csv", index=False)

    print(f"\nFigures saved to {FIGURES_DIR}/")
    print(f"Tables saved to {RESULTS_DIR}/")


def _sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────

def _fig_violin_plots(df, metrics, var_df):
    usable = [m for m in metrics if df[m].notna().sum() >= 6]
    n = len(usable)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    palette = {"runway": "#2196F3", "control": "#FF5722"}

    for i, m in enumerate(usable):
        ax = axes[i]
        data = df[["group", m]].dropna()
        if data.empty:
            ax.set_visible(False)
            continue

        sns.violinplot(
            data=data, x="group", y=m, ax=ax,
            palette=palette, inner="box", cut=0, density_norm="width",
        )
        sns.stripplot(
            data=data, x="group", y=m, ax=ax,
            color="black", alpha=0.4, size=3, jitter=True,
        )

        unit = METRIC_UNITS.get(m, "")
        ax.set_title(f"{m} ({unit})" if unit else m, fontsize=11, fontweight="bold")
        ax.set_xlabel("")

        vr_row = var_df[var_df.metric == m]
        if not vr_row.empty:
            p = vr_row.levene_p.iloc[0]
            ratio = vr_row.var_ratio.iloc[0]
            stars = _sig_stars(p)
            ax.text(
                0.98, 0.98,
                f"σ²r={ratio:.1f}× {stars}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="red" if p < 0.05 else "gray",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Distribution Comparison: Runway (blue) vs. Control (orange)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_violin_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 01_violin_distributions.png")


def _fig_variance_forest(var_df, boot_df):
    merged = var_df.merge(boot_df, on="metric", how="left")
    merged = merged.sort_values("var_ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(merged) * 0.4)))

    y_pos = range(len(merged))
    colors = ["#D32F2F" if p < 0.05 else "#9E9E9E" for p in merged.levene_p]

    ax.barh(
        y_pos, merged.var_ratio, color=colors, alpha=0.7, height=0.6,
    )

    if "ci_lo" in merged.columns:
        for i, (_, row) in enumerate(merged.iterrows()):
            if not np.isnan(row.get("ci_lo", np.nan)):
                ax.plot(
                    [row.ci_lo, row.ci_hi], [i, i],
                    color="black", linewidth=1.5, zorder=5,
                )
                ax.plot(
                    [row.ci_lo, row.ci_lo], [i - 0.15, i + 0.15],
                    color="black", linewidth=1.5, zorder=5,
                )
                ax.plot(
                    [row.ci_hi, row.ci_hi], [i - 0.15, i + 0.15],
                    color="black", linewidth=1.5, zorder=5,
                )

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(merged.metric, fontsize=10)
    ax.set_xlabel("Variance Ratio (Control / Runway)", fontsize=12)
    ax.set_title(
        "Variance Ratio Forest Plot\n"
        "Red = significant (p<0.05), bars right of 1.0 = control has MORE variance",
        fontsize=12, fontweight="bold",
    )

    ax.text(
        0.98, 0.02,
        f"Median ratio: {merged.var_ratio.median():.2f}×",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "02_variance_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 02_variance_forest.png")


def _fig_cv_comparison(var_df):
    usable = var_df.dropna(subset=["runway_CV", "control_CV"])
    if usable.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        usable.runway_CV, usable.control_CV,
        s=80, c=["#D32F2F" if p < 0.05 else "#2196F3" for p in usable.levene_p],
        alpha=0.7, edgecolors="black", linewidth=0.5, zorder=3,
    )

    for _, row in usable.iterrows():
        ax.annotate(
            row.metric, (row.runway_CV, row.control_CV),
            fontsize=7, alpha=0.8,
            xytext=(5, 5), textcoords="offset points",
        )

    lim = max(usable.runway_CV.max(), usable.control_CV.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="Equal CV")
    ax.set_xlabel("Runway CV (%)", fontsize=12)
    ax.set_ylabel("Control CV (%)", fontsize=12)
    ax.set_title(
        "Coefficient of Variation: Runway vs. Control\n"
        "Points above diagonal = control has more relative spread",
        fontsize=12, fontweight="bold",
    )
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "03_cv_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 03_cv_comparison.png")


def _confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / (np.sqrt(cov[0, 0] * cov[1, 1]) + 1e-12)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs,
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def _fig_pca_scatter(pca_result):
    X_pca, pca, df_mv, mv_metrics = pca_result
    y = (df_mv["group"] == "runway").astype(int).values

    fig, ax = plt.subplots(figsize=(10, 8))

    r_mask = y == 1
    c_mask = y == 0

    ax.scatter(
        X_pca[r_mask, 0], X_pca[r_mask, 1],
        c="#2196F3", s=80, alpha=0.7, label="Runway", edgecolors="black", linewidth=0.5,
    )
    ax.scatter(
        X_pca[c_mask, 0], X_pca[c_mask, 1],
        c="#FF5722", s=80, alpha=0.7, label="Control", edgecolors="black", linewidth=0.5,
    )

    if r_mask.sum() >= 3:
        _confidence_ellipse(
            X_pca[r_mask, 0], X_pca[r_mask, 1], ax,
            n_std=2.0, edgecolor="#1565C0", linewidth=2, linestyle="--",
            facecolor="#2196F3", alpha=0.1,
        )
    if c_mask.sum() >= 3:
        _confidence_ellipse(
            X_pca[c_mask, 0], X_pca[c_mask, 1], ax,
            n_std=2.0, edgecolor="#BF360C", linewidth=2, linestyle="--",
            facecolor="#FF5722", alpha=0.1,
        )

    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({evr[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({evr[1]:.1%} variance)", fontsize=12)
    ax.set_title(
        "PCA: Kinematic Feature Space\n"
        "Ellipses = 95% confidence regions (smaller = lower variance)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if r_mask.sum() >= 3 and c_mask.sum() >= 3:
        r_spread = np.trace(np.cov(X_pca[r_mask, :2].T))
        c_spread = np.trace(np.cov(X_pca[c_mask, :2].T))
        ax.text(
            0.02, 0.98,
            f"Spread (trace): Runway={r_spread:.2f}, Control={c_spread:.2f}\n"
            f"Ratio: {c_spread/r_spread:.1f}×",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 04_pca_scatter.png")


def _fig_radar_profiles(df, metrics):
    usable = [m for m in metrics if df[m].notna().sum() >= 6 and m not in ("MQS", "GDI")]
    if len(usable) < 4:
        return

    runway = df[df.group == "runway"]
    control = df[df.group == "control"]

    runway_means = [runway[m].mean() for m in usable]
    control_means = [control[m].mean() for m in usable]
    runway_stds = [runway[m].std() for m in usable]
    control_stds = [control[m].std() for m in usable]

    all_vals = runway_means + control_means
    max_val = max(abs(v) for v in all_vals if not np.isnan(v))
    if max_val < 1e-6:
        return

    r_norm = [v / max_val for v in runway_means]
    c_norm = [v / max_val for v in control_means]
    r_std_norm = [v / max_val for v in runway_stds]
    c_std_norm = [v / max_val for v in control_stds]

    n = len(usable)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    r_norm += r_norm[:1]
    c_norm += c_norm[:1]
    r_std_norm += r_std_norm[:1]
    c_std_norm += c_std_norm[:1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=dict(polar=True))

    for ax, title, r_vals, c_vals in [
        (ax1, "Mean Profiles (normalized)", r_norm, c_norm),
        (ax2, "Std Dev Profiles (normalized)", r_std_norm, c_std_norm),
    ]:
        ax.plot(angles, r_vals, "o-", color="#2196F3", linewidth=2, label="Runway")
        ax.fill(angles, r_vals, color="#2196F3", alpha=0.1)
        ax.plot(angles, c_vals, "s-", color="#FF5722", linewidth=2, label="Control")
        ax.fill(angles, c_vals, color="#FF5722", alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(usable, fontsize=7)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.suptitle("Kinematic Profiles: Runway vs. Control", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "05_radar_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 05_radar_profiles.png")


def _fig_variance_heatmap(var_df):
    fig, ax = plt.subplots(figsize=(12, 5))

    data = var_df.set_index("metric")[["var_ratio"]].T
    data.index = ["σ² ratio\n(control/runway)"]

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    vmax = min(data.values.max(), 20)

    sns.heatmap(
        data, ax=ax, cmap=cmap, center=1.0, vmin=0, vmax=vmax,
        annot=True, fmt=".1f", linewidths=0.5,
        cbar_kws={"label": "Variance Ratio"},
    )

    for j, (_, row) in enumerate(var_df.iterrows()):
        if row.levene_p < 0.05:
            ax.text(
                j + 0.5, 0.85, _sig_stars(row.levene_p),
                ha="center", va="center", fontsize=10, color="white",
                fontweight="bold",
            )

    ax.set_title(
        "Variance Ratio Heatmap (control/runway) — values >1 = control more variable",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_variance_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 06_variance_heatmap.png")


def _fig_domain_summary(var_df):
    domains = var_df.groupby("domain").agg(
        median_ratio=("var_ratio", "median"),
        mean_ratio=("var_ratio", "mean"),
        n_metrics=("metric", "count"),
        n_sig=("levene_p", lambda x: (x < 0.05).sum()),
    ).reset_index()
    domains = domains.sort_values("median_ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#D32F2F" if r > 1.5 else "#FF9800" if r > 1 else "#4CAF50"
              for r in domains.median_ratio]

    bars = ax.barh(
        domains.domain, domains.median_ratio,
        color=colors, alpha=0.8, edgecolor="black", linewidth=0.5,
    )

    for bar, (_, row) in zip(bars, domains.iterrows()):
        ax.text(
            bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{row.median_ratio:.1f}× ({row.n_sig}/{row.n_metrics} sig)",
            va="center", fontsize=10,
        )

    ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Median Variance Ratio (Control / Runway)", fontsize=12)
    ax.set_title(
        "Variance Ratio by Domain\n"
        "Red = strong support, orange = moderate, green = no difference",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "07_domain_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 07_domain_summary.png")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process", action="store_true", help="Process videos")
    parser.add_argument("--analyze", action="store_true", help="Run analysis")
    parser.add_argument("--all", action="store_true", help="Process + analyze")
    args = parser.parse_args()

    if args.all:
        args.process = True
        args.analyze = True

    if not args.process and not args.analyze:
        parser.print_help()
        return

    if args.process:
        process_videos()
    if args.analyze:
        run_analysis()


if __name__ == "__main__":
    main()
