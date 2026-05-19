"""End-to-end tests for the Movement Analytics pipeline.

Validates gait model generation, metric computation, MQS scoring,
and dashboard rendering across all 8 gait profiles.
"""

import numpy as np
import pytest

from movement_analytics.generators.gait_model import (
    GaitParameters, GAIT_PROFILES, generate_gait_cycle,
)
from movement_analytics.generators.stick_figure import generate_frames
from movement_analytics.kinematics.gait_metrics import (
    angular_velocity, sparc, symmetry_index, rom,
    normalized_jerk, coefficient_of_variation,
    continuous_relative_phase, crp_consistency,
    compute_gait_summary, movement_quality_score, mqs_domain_scores,
    _signal_score, _DOMAIN_WEIGHTS,
)
from movement_analytics.visualization.dashboard import (
    RealTimeDashboard, create_dashboard_frame,
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
        r = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        l = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="left")
        return r, l

    def test_rom_positive(self, normal_angles):
        r, _ = normal_angles
        for joint in ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]:
            assert rom(r[joint]) > 0

    def test_normal_hip_rom_range(self, normal_angles):
        r, _ = normal_angles
        hip_rom = rom(r["hip_flexion"])
        assert 30 < hip_rom < 55, f"Hip ROM {hip_rom} outside expected range"

    def test_sparc_negative(self, normal_angles):
        r, _ = normal_angles
        vel = angular_velocity(r["hip_flexion"], 30)
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
        r, _ = normal_angles
        nj = normalized_jerk(r["hip_flexion"], 30)
        assert nj > 0

    def test_cv_zero_for_constant(self):
        assert coefficient_of_variation(np.ones(100) * 5) == pytest.approx(0.0)

    def test_gait_summary_completeness(self, normal_angles):
        r, l = normal_angles
        summary = compute_gait_summary(r, l, fps=30)
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
        r = generate_gait_cycle(params, n_frames=60, n_cycles=6, side="right")
        l = generate_gait_cycle(params, n_frames=60, n_cycles=6, side="left")
        summary = compute_gait_summary(r, l, fps=30)
        assert summary["movement_quality_score"] >= 85

    def test_noisy_mqs_lower(self):
        params = GaitParameters(noise_level=4)
        r = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        l = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="left")
        summary = compute_gait_summary(r, l, fps=30)
        assert summary["movement_quality_score"] < 80

    def test_stiff_knee_penalized(self):
        params = GAIT_PROFILES["stiff_knee"].params
        r = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        l = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="left")
        summary = compute_gait_summary(r, l, fps=30)
        assert summary["mqs_kinematics"] < 80

    def test_domain_scores_sum_to_mqs(self):
        params = GaitParameters()
        r = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        l = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="left")
        summary = compute_gait_summary(r, l, fps=30)
        domains = mqs_domain_scores(summary)
        expected = sum(domains[d] * _DOMAIN_WEIGHTS[d] for d in _DOMAIN_WEIGHTS)
        assert summary["movement_quality_score"] == pytest.approx(expected, abs=0.1)

    def test_trendelenburg_penalized_in_kinematics(self):
        params = GAIT_PROFILES["trendelenburg"].params
        r = generate_gait_cycle(params, n_frames=60, n_cycles=6, side="right")
        l = generate_gait_cycle(params, n_frames=60, n_cycles=6, side="left")
        summary = compute_gait_summary(r, l, fps=30)
        normal_r = generate_gait_cycle(GaitParameters(), n_frames=60, n_cycles=6, side="right")
        normal_l = generate_gait_cycle(GaitParameters(), n_frames=60, n_cycles=6, side="left")
        normal_summary = compute_gait_summary(normal_r, normal_l, fps=30)
        assert summary["mqs_kinematics"] < normal_summary["mqs_kinematics"]

    def test_mqs_bounded(self):
        for name, profile in GAIT_PROFILES.items():
            r = generate_gait_cycle(profile.params, n_frames=60, n_cycles=3, side="right")
            l = generate_gait_cycle(profile.params, n_frames=60, n_cycles=3, side="left")
            summary = compute_gait_summary(r, l, fps=30)
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
        r = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        l = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="left")
        summary = compute_gait_summary(r, l, fps=30)
        assert "hip_CRP_MAD" in summary
        assert "knee_CRP_MAD" in summary
        assert "mqs_coordination" in summary


class TestBenchmark:
    def test_benchmark_all_profiles(self):
        from movement_analytics.cli import run_benchmark
        import json
        import tempfile
        import os

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
