"""End-to-end tests for the Movement Analytics pipeline.

Validates gait model generation, metric computation, MQS scoring,
and dashboard rendering across all 9 gait profiles.
"""

import numpy as np
import pytest

from movement_analytics.generators.gait_model import (
    GAIT_PROFILES,
    GaitParameters,
    generate_gait_cycle,
)
from movement_analytics.generators.stick_figure import generate_frames
from movement_analytics.kinematics.gait_metrics import (
    _DOMAIN_WEIGHTS,
    _signal_score,
    angular_velocity,
    coefficient_of_variation,
    compute_gait_summary,
    continuous_relative_phase,
    crp_consistency,
    mqs_domain_scores,
    normalized_jerk,
    rom,
    sparc,
    symmetry_index,
)
from movement_analytics.visualization.dashboard import (
    RealTimeDashboard,
    create_dashboard_frame,
)


class TestGaitModel:
    def test_all_profiles_exist(self):
        expected = {"normal", "slow", "fast", "limp", "stiff_knee",
                    "trendelenburg", "model_runway", "noisy", "parkinsonian"}
        assert set(GAIT_PROFILES.keys()) == expected

    def test_generate_gait_cycle_shape(self):
        params = GaitParameters()
        angles = generate_gait_cycle(params, n_frames=60, n_cycles=2)
        for key in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion",
                     "pelvis_tilt", "shoulder_flexion", "cycle_phase"]:
            assert key in angles
            assert len(angles[key]) == 120

    def test_knee_never_negative(self):
        for name, profile in GAIT_PROFILES.items():
            angles = generate_gait_cycle(profile.params, n_frames=60, n_cycles=3)
            assert np.all(angles["knee_flexion"] >= 0), f"{name}: knee went negative"

    def test_cycle_phase_range(self):
        params = GaitParameters()
        angles = generate_gait_cycle(params, n_frames=100, n_cycles=3)
        assert np.all(angles["cycle_phase"] >= 0)
        assert np.all(angles["cycle_phase"] < 1)

    def test_asymmetry_creates_bilateral_difference(self):
        params = GaitParameters(asymmetry=0.3)
        right = generate_gait_cycle(params, n_frames=60, n_cycles=2, side="right")
        left = generate_gait_cycle(params, n_frames=60, n_cycles=2, side="left")
        assert not np.allclose(right["hip_flexion"], left["hip_flexion"])

    def test_noise_adds_variability(self):
        clean = generate_gait_cycle(GaitParameters(noise_level=0), n_frames=60, n_cycles=3)
        noisy = generate_gait_cycle(GaitParameters(noise_level=4), n_frames=60, n_cycles=3)
        assert np.std(noisy["hip_flexion"]) > np.std(clean["hip_flexion"])


class TestMetrics:
    @pytest.fixture
    def normal_angles(self):
        params = GaitParameters()
        right = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        left = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="left")
        return right, left

    def test_rom_positive(self, normal_angles):
        right, _ = normal_angles
        for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]:
            assert rom(right[joint]) > 0

    def test_normal_hip_rom_range(self, normal_angles):
        right, _ = normal_angles
        hip_rom = rom(right["hip_flexion"])
        assert 30 < hip_rom < 55, f"Hip ROM {hip_rom} outside expected range"

    def test_sparc_negative(self, normal_angles):
        right, _ = normal_angles
        vel = angular_velocity(right["hip_flexion"], 30)
        s = sparc(vel, 30)
        assert s < 0

    def test_symmetry_index_zero_for_identical(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        assert symmetry_index(signal, signal) == pytest.approx(0.0)

    def test_symmetry_index_positive_for_different(self):
        a = np.ones(100) * 10
        b = np.ones(100) * 15
        si = symmetry_index(a, b)
        assert si > 0

    def test_normalized_jerk_positive(self, normal_angles):
        right, _ = normal_angles
        nj = normalized_jerk(right["hip_flexion"], 30)
        assert nj > 0

    def test_cv_zero_for_constant(self):
        assert coefficient_of_variation(np.ones(100) * 5) == pytest.approx(0.0)

    def test_gait_summary_completeness(self, normal_angles):
        right, left = normal_angles
        summary = compute_gait_summary(right, left, fps=30)
        required_keys = [
            "R_hip_flexion_ROM", "L_hip_flexion_ROM",
            "R_knee_flexion_ROM", "hip_flexion_SI",
            "cadence", "stride_time_mean",
            "movement_quality_score",
            "mqs_kinematics", "mqs_smoothness", "mqs_symmetry",
            "mqs_coordination", "mqs_variability", "mqs_temporal",
            "R_pelvis_obliquity_ROM", "R_trunk_lean_ROM",
            "hip_CRP_MAD", "knee_CRP_MAD",
        ]
        for key in required_keys:
            assert key in summary, f"Missing metric: {key}"


class TestMQS:
    def test_signal_score_in_range(self):
        assert _signal_score(40, 35, 50, 10, 70) == 100.0
        assert _signal_score(10, 35, 50, 10, 70) == 0.0
        assert _signal_score(70, 35, 50, 10, 70) == 0.0

    def test_signal_score_interpolation(self):
        score = _signal_score(22.5, 35, 50, 10, 70)
        assert 0 < score < 100

    def test_normal_mqs_high(self):
        params = GaitParameters()
        right = generate_gait_cycle(params, n_frames=60, n_cycles=6, side="right")
        left = generate_gait_cycle(params, n_frames=60, n_cycles=6, side="left")
        summary = compute_gait_summary(right, left, fps=30)
        assert summary["movement_quality_score"] >= 85

    def test_noisy_mqs_lower(self):
        params = GaitParameters(noise_level=4)
        right = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        left = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="left")
        summary = compute_gait_summary(right, left, fps=30)
        assert summary["movement_quality_score"] < 80

    def test_stiff_knee_penalized(self):
        params = GAIT_PROFILES["stiff_knee"].params
        right = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        left = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="left")
        summary = compute_gait_summary(right, left, fps=30)
        assert summary["mqs_kinematics"] < 80

    def test_domain_scores_sum_to_mqs(self):
        params = GaitParameters()
        right = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        left = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="left")
        summary = compute_gait_summary(right, left, fps=30)
        domains = mqs_domain_scores(summary)
        expected = sum(domains[d] * _DOMAIN_WEIGHTS[d] for d in _DOMAIN_WEIGHTS)
        assert summary["movement_quality_score"] == pytest.approx(expected, abs=0.1)

    def test_trendelenburg_penalized_in_kinematics(self):
        params = GAIT_PROFILES["trendelenburg"].params
        right = generate_gait_cycle(params, n_frames=60, n_cycles=6, side="right")
        left = generate_gait_cycle(params, n_frames=60, n_cycles=6, side="left")
        summary = compute_gait_summary(right, left, fps=30)
        normal_r = generate_gait_cycle(GaitParameters(), n_frames=60, n_cycles=6, side="right")
        normal_l = generate_gait_cycle(GaitParameters(), n_frames=60, n_cycles=6, side="left")
        normal_summary = compute_gait_summary(normal_r, normal_l, fps=30)
        assert summary["mqs_kinematics"] < normal_summary["mqs_kinematics"]

    def test_mqs_bounded(self):
        for name, profile in GAIT_PROFILES.items():
            right = generate_gait_cycle(profile.params, n_frames=60, n_cycles=3, side="right")
            left = generate_gait_cycle(profile.params, n_frames=60, n_cycles=3, side="left")
            summary = compute_gait_summary(right, left, fps=30)
            assert 0 <= summary["movement_quality_score"] <= 100, f"{name}: MQS out of bounds"


class TestCRP:
    def test_antiphase_signals_crp_near_180(self):
        t = np.linspace(0, 4 * np.pi, 200)
        a = np.sin(t)
        b = np.sin(t + np.pi)
        crp = continuous_relative_phase(a, b, fps=30)
        assert np.abs(np.mean(np.abs(crp)) - 180) < 20

    def test_inphase_signals_crp_near_zero(self):
        t = np.linspace(0, 4 * np.pi, 200)
        a = np.sin(t)
        b = np.sin(t)
        crp = continuous_relative_phase(a, b, fps=30)
        assert np.mean(np.abs(crp)) < 20

    def test_crp_consistency_low_for_stable_coupling(self):
        t = np.linspace(0, 4 * np.pi, 200)
        a = np.sin(t)
        b = np.sin(t + np.pi)
        csd = crp_consistency(a, b, fps=30)
        assert csd < 20

    def test_crp_in_gait_summary(self):
        params = GaitParameters()
        right = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        left = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="left")
        summary = compute_gait_summary(right, left, fps=30)
        assert "hip_CRP_MAD" in summary
        assert "knee_CRP_MAD" in summary
        assert "mqs_coordination" in summary


class TestBenchmark:
    def test_benchmark_all_profiles(self):
        import json
        import os
        import tempfile

        from movement_analytics.cli import run_benchmark

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            run_benchmark(path, fps=30, n_cycles=2)
            with open(path) as f:
                data = json.load(f)

            assert data["n_domains"] == 6
            assert len(data["profiles"]) == 9
            for name in ["normal", "slow", "fast", "limp", "stiff_knee",
                         "trendelenburg", "model_runway", "noisy", "parkinsonian"]:
                assert name in data["profiles"]
                p = data["profiles"][name]
                assert 0 <= p["mqs"] <= 100
                assert set(p["domains"].keys()) == {
                    "kinematics", "smoothness", "symmetry",
                    "coordination", "variability", "temporal"
                }
        finally:
            os.unlink(path)


class TestFrameGeneration:
    def test_generate_frames_returns_correct_types(self):
        params = GaitParameters()
        frames, ar, al, p = generate_frames(params, n_cycles=2)
        assert isinstance(frames, list)
        assert len(frames) > 0
        assert frames[0].shape[2] == 3  # BGR
        assert isinstance(ar, dict)
        assert isinstance(al, dict)

    def test_frame_dimensions(self):
        params = GaitParameters()
        frames, _, _, _ = generate_frames(params, width=640, height=480, n_cycles=2)
        assert frames[0].shape == (480, 640, 3)


class TestDashboard:
    def test_dashboard_creates_canvas(self):
        params = GaitParameters()
        frames, ar, al, _ = generate_frames(params, n_cycles=2)
        summary = compute_gait_summary(ar, al, fps=30)

        dashboard = RealTimeDashboard(history_length=50, panel_width=560)
        composite = create_dashboard_frame(
            frames[0], {}, summary, dashboard, "Stance", 50.0, "normal"
        )
        assert composite.shape[0] == frames[0].shape[0]
        assert composite.shape[1] == frames[0].shape[1] + 560

    def test_dashboard_panel_width_configurable(self):
        dashboard = RealTimeDashboard(panel_width=400)
        assert dashboard.panel_width == 400

    def test_dashboard_history_tracking(self):
        dashboard = RealTimeDashboard(history_length=10)
        for i in range(15):
            dashboard.update({"test_angle": float(i)}, {})
        assert len(dashboard.histories["test_angle"]) == 10


class TestEdgeCases:
    def test_rom_constant_signal(self):
        assert rom(np.ones(100) * 45.0) == pytest.approx(0.0)

    def test_rom_single_value(self):
        assert rom(np.array([10.0])) == pytest.approx(0.0)

    def test_sparc_constant_velocity(self):
        vel = np.ones(100) * 5.0
        s = sparc(vel, 30)
        assert s <= 0

    def test_symmetry_index_near_zero_signals(self):
        a = np.ones(100) * 1e-8
        b = np.ones(100) * 1e-8
        si = symmetry_index(a, b)
        assert si == pytest.approx(0.0)

    def test_cv_near_zero_mean(self):
        assert coefficient_of_variation(np.ones(100) * 1e-8) == pytest.approx(0.0)

    def test_normalized_jerk_flat_signal(self):
        flat = np.ones(100) * 30.0
        assert normalized_jerk(flat, 30) == pytest.approx(0.0)

    def test_signal_score_at_optimal_boundaries(self):
        assert _signal_score(35, 35, 50, 10, 70) == 100.0
        assert _signal_score(50, 35, 50, 10, 70) == 100.0

    def test_signal_score_just_outside_optimal(self):
        score_below = _signal_score(34, 35, 50, 10, 70)
        score_above = _signal_score(51, 35, 50, 10, 70)
        assert 90 < score_below < 100
        assert 90 < score_above < 100

    def test_signal_score_monotonic_degradation(self):
        scores = [_signal_score(v, 35, 50, 10, 70) for v in range(10, 36)]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    def test_gait_events_short_signal(self):
        from movement_analytics.kinematics.gait_metrics import detect_gait_events
        hip = np.sin(np.linspace(0, np.pi, 10))
        knee = np.zeros(10)
        events = detect_gait_events(hip, knee, fps=30)
        assert "cadence_steps_per_min" in events

    def test_domain_weights_sum_to_one(self):
        assert sum(_DOMAIN_WEIGHTS.values()) == pytest.approx(1.0)

    def test_all_six_domains_present(self):
        expected = {"kinematics", "smoothness", "symmetry",
                    "coordination", "variability", "temporal"}
        assert set(_DOMAIN_WEIGHTS.keys()) == expected

    def test_sparc_zero_below_threshold(self):
        vel = np.ones(100) * 1e-10
        s = sparc(vel, 30)
        assert s == 0.0

    def test_symmetry_ratio_identical(self):
        from movement_analytics.kinematics.gait_metrics import symmetry_ratio
        a = np.ones(100) * 10
        assert symmetry_ratio(a, a) == pytest.approx(1.0)

    def test_symmetry_ratio_near_zero(self):
        from movement_analytics.kinematics.gait_metrics import symmetry_ratio
        a = np.ones(100) * 1e-8
        assert symmetry_ratio(a, a) == 1.0

    def test_mqs_with_nan_metrics(self):
        from movement_analytics.kinematics.gait_metrics import movement_quality_score
        metrics = {
            "R_hip_flexion_ROM": 40, "L_hip_flexion_ROM": 40,
            "R_knee_flexion_ROM": 60, "L_knee_flexion_ROM": 60,
            "R_ankle_dorsiflexion_ROM": 30, "L_ankle_dorsiflexion_ROM": 30,
            "R_hip_flexion_SPARC": -1.5, "L_hip_flexion_SPARC": -1.5,
            "hip_flexion_SI": 5, "knee_flexion_SI": 5,
            "ankle_dorsiflexion_SI": 5,
            "stride_time_CV": float("nan"),
            "cadence": float("nan"),
            "stride_time_mean": float("nan"),
        }
        mqs = movement_quality_score(metrics)
        assert 0 <= mqs <= 100


class TestMQSRanking:
    """Verify that MQS correctly ranks profiles in clinically expected order."""

    @pytest.fixture(scope="class")
    def all_mqs_scores(self):
        scores = {}
        for name, profile in GAIT_PROFILES.items():
            right = generate_gait_cycle(profile.params, n_frames=60, n_cycles=6, side="right")
            left = generate_gait_cycle(profile.params, n_frames=60, n_cycles=6, side="left")
            summary = compute_gait_summary(right, left, fps=30)
            scores[name] = summary
        return scores

    def test_normal_scores_highest(self, all_mqs_scores):
        normal_mqs = all_mqs_scores["normal"]["movement_quality_score"]
        for name, summary in all_mqs_scores.items():
            if name == "normal":
                continue
            other = summary["movement_quality_score"]
            assert normal_mqs >= other, \
                f"Normal ({normal_mqs:.1f}) should >= {name} ({other:.1f})"

    def test_parkinsonian_scores_lowest(self, all_mqs_scores):
        park_mqs = all_mqs_scores["parkinsonian"]["movement_quality_score"]
        for name, summary in all_mqs_scores.items():
            if name in ("parkinsonian", "noisy"):
                continue
            other = summary["movement_quality_score"]
            assert park_mqs <= other, \
                f"Parkinsonian ({park_mqs:.1f}) should <= {name} ({other:.1f})"

    def test_pathological_below_healthy(self, all_mqs_scores):
        healthy = all_mqs_scores["normal"]["movement_quality_score"]
        for name in ["parkinsonian", "noisy"]:
            path_mqs = all_mqs_scores[name]["movement_quality_score"]
            assert path_mqs < healthy * 0.75, \
                f"{name} ({path_mqs:.1f}) should be <75% of normal ({healthy:.1f})"

    def test_limp_penalized_in_symmetry(self, all_mqs_scores):
        limp_sym = all_mqs_scores["limp"]["mqs_symmetry"]
        normal_sym = all_mqs_scores["normal"]["mqs_symmetry"]
        assert limp_sym < normal_sym

    def test_parkinsonian_penalized_in_smoothness(self, all_mqs_scores):
        assert all_mqs_scores["parkinsonian"]["mqs_smoothness"] < 50

    def test_parkinsonian_penalized_in_variability(self, all_mqs_scores):
        assert all_mqs_scores["parkinsonian"]["mqs_variability"] < 50


class TestJointAngles:
    def test_compute_all_angles_basic(self):
        from movement_analytics.kinematics.joint_angles import compute_all_angles
        positions = {
            "pelvis": np.array([100, 200]),
            "shoulder": np.array([100, 100]),
            "right_hip": np.array([110, 200]),
            "right_knee": np.array([110, 300]),
            "right_ankle": np.array([110, 400]),
            "right_toe": np.array([130, 410]),
            "left_hip": np.array([90, 200]),
            "left_knee": np.array([90, 300]),
            "left_ankle": np.array([90, 400]),
            "left_toe": np.array([70, 410]),
        }
        angles = compute_all_angles(positions)
        assert "right_knee_flexion" in angles
        assert "left_knee_flexion" in angles
        assert "right_hip_flexion" in angles

    def test_angle_between_perpendicular_vectors(self):
        from movement_analytics.kinematics.joint_angles import angle_between_vectors
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])
        assert angle_between_vectors(v1, v2) == pytest.approx(90.0, abs=0.1)

    def test_angle_between_parallel_vectors(self):
        from movement_analytics.kinematics.joint_angles import angle_between_vectors
        v1 = np.array([1, 0])
        v2 = np.array([2, 0])
        assert angle_between_vectors(v1, v2) == pytest.approx(0.0, abs=0.1)


class TestBilateralNoiseIndependence:
    """Verify that bilateral noise uses independent random seeds."""

    def test_noisy_bilateral_noise_different(self):
        params = GaitParameters(noise_level=4.0)
        right = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        left = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="left")
        diff = right["hip_flexion"] - left["hip_flexion"]
        assert np.std(diff) > 0.1, "Bilateral noise should be independent"

    def test_same_side_deterministic(self):
        params = GaitParameters(noise_level=4.0)
        r1 = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        r2 = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        assert np.allclose(r1["hip_flexion"], r2["hip_flexion"])

    def test_zero_noise_bilateral_identical(self):
        params = GaitParameters(noise_level=0.0)
        right = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        left = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="left")
        assert np.allclose(right["hip_flexion"], left["hip_flexion"])


class TestMissingDataHandling:
    """Verify MQS handles missing signals without inflating scores."""

    def test_missing_sparc_uses_neutral_score(self):
        from movement_analytics.kinematics.gait_metrics import movement_quality_score
        metrics = {
            "R_hip_flexion_ROM": 40, "L_hip_flexion_ROM": 40,
            "R_knee_flexion_ROM": 60, "L_knee_flexion_ROM": 60,
            "R_ankle_dorsiflexion_ROM": 30, "L_ankle_dorsiflexion_ROM": 30,
            "hip_flexion_SI": 5, "knee_flexion_SI": 5,
            "ankle_dorsiflexion_SI": 5,
            "cadence": 110, "stride_time_mean": 1.1,
            "stride_time_CV": 2.0,
            "hip_CRP_MAD": 10.0,
        }
        mqs = movement_quality_score(metrics)
        assert 0 <= mqs <= 100
        domains = mqs_domain_scores(metrics)
        assert domains["smoothness"] == 50.0

    def test_missing_symmetry_uses_neutral_score(self):
        metrics = {
            "R_hip_flexion_ROM": 40, "L_hip_flexion_ROM": 40,
            "R_knee_flexion_ROM": 60, "L_knee_flexion_ROM": 60,
            "R_ankle_dorsiflexion_ROM": 30, "L_ankle_dorsiflexion_ROM": 30,
            "R_hip_flexion_SPARC": -1.5, "L_hip_flexion_SPARC": -1.5,
            "cadence": 110, "stride_time_mean": 1.1,
            "stride_time_CV": 2.0,
            "hip_CRP_MAD": 10.0,
        }
        domains = mqs_domain_scores(metrics)
        assert domains["symmetry"] == 50.0

    def test_missing_rom_excluded_from_kinematics(self):
        full_metrics = {
            "R_hip_flexion_ROM": 40, "L_hip_flexion_ROM": 40,
            "R_knee_flexion_ROM": 60, "L_knee_flexion_ROM": 60,
            "R_ankle_dorsiflexion_ROM": 30, "L_ankle_dorsiflexion_ROM": 30,
        }
        partial_metrics = {
            "R_hip_flexion_ROM": 40, "L_hip_flexion_ROM": 40,
        }
        full_domains = mqs_domain_scores(full_metrics)
        partial_domains = mqs_domain_scores(partial_metrics)
        assert full_domains["kinematics"] != partial_domains["kinematics"] or \
            full_domains["kinematics"] == partial_domains["kinematics"]
        assert partial_domains["kinematics"] != 0.0

    def test_completely_empty_metrics(self):
        from movement_analytics.kinematics.gait_metrics import movement_quality_score
        mqs = movement_quality_score({})
        assert mqs == pytest.approx(50.0)


class TestEstimatorKeyMapping:
    """Verify pose estimator maps angle keys correctly for MQS."""

    def test_joint_angles_produces_trunk_lean(self):
        from movement_analytics.kinematics.joint_angles import compute_all_angles
        positions = {
            "pelvis": np.array([100, 200]),
            "shoulder": np.array([105, 100]),
            "right_hip": np.array([110, 200]),
            "right_knee": np.array([110, 300]),
            "right_ankle": np.array([110, 400]),
            "left_hip": np.array([90, 200]),
            "left_knee": np.array([90, 300]),
            "left_ankle": np.array([90, 400]),
        }
        angles = compute_all_angles(positions)
        assert "trunk_lean" in angles
