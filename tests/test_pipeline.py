"""End-to-end tests for the Movement Analytics pipeline.

Validates gait model generation, metric computation, MQS scoring,
and dashboard rendering across all 9 gait profiles.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
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
            "pelvis_obliquity_SI", "trunk_lateral_lean_SI",
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

    def test_gait_events_ankle_refinement(self):
        from movement_analytics.kinematics.gait_metrics import detect_gait_events
        t = np.linspace(0, 6 * np.pi, 180)
        hip = 20 * np.sin(t)
        knee = 30 * (1 - np.cos(t)) / 2
        ankle = 10 * np.sin(t - 0.2)
        events_no_ankle = detect_gait_events(hip, knee, fps=30)
        events_with_ankle = detect_gait_events(
            hip, knee, fps=30, ankle_dorsiflexion=ankle,
        )
        assert len(events_with_ankle["heel_strikes"]) == len(
            events_no_ankle["heel_strikes"]
        )
        assert events_with_ankle["cadence_steps_per_min"] > 0

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
            _, right, left, _ = generate_frames(profile.params, fps=30, n_cycles=6)
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

    def test_noisy_penalized_in_variability(self, all_mqs_scores):
        assert all_mqs_scores["noisy"]["mqs_variability"] < 50


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


class TestWaveformSymmetry:
    """Verify waveform symmetry metric detects shape-based asymmetry."""

    def test_identical_signals(self):
        from movement_analytics.kinematics.gait_metrics import waveform_symmetry
        a = np.sin(np.linspace(0, 4 * np.pi, 200))
        assert waveform_symmetry(a, a) == pytest.approx(100.0)

    def test_antiphase_signals(self):
        from movement_analytics.kinematics.gait_metrics import waveform_symmetry
        t = np.linspace(0, 4 * np.pi, 200)
        a = np.sin(t)
        b = np.sin(t + np.pi)
        assert waveform_symmetry(a, b) == pytest.approx(100.0, abs=1.0)

    def test_noisy_signals_lower(self):
        from movement_analytics.kinematics.gait_metrics import waveform_symmetry
        t = np.linspace(0, 4 * np.pi, 200)
        a = np.sin(t)
        rng = np.random.default_rng(42)
        b = np.sin(t) + rng.normal(0, 0.5, 200)
        ws = waveform_symmetry(a, b)
        assert ws < 100.0
        assert ws > 50.0

    def test_waveform_sym_in_summary(self):
        params = GaitParameters()
        right = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="right")
        left = generate_gait_cycle(params, n_frames=60, n_cycles=3, side="left")
        summary = compute_gait_summary(right, left, fps=30)
        assert "hip_flexion_waveform_sym" in summary


class TestStrideROMVariability:
    """Verify stride-level ROM coefficient of variation computation."""

    def test_normal_gait_low_rom_cv(self):
        params = GaitParameters()
        _, right, left, _ = generate_frames(params, fps=30, n_cycles=6)
        summary = compute_gait_summary(right, left, fps=30)
        hip_cv = summary.get("R_hip_ROM_CV", None)
        assert hip_cv is not None
        assert hip_cv < 1.0

    def test_noisy_gait_high_rom_cv(self):
        params = GaitParameters(noise_level=4.0)
        _, right, left, _ = generate_frames(params, fps=30, n_cycles=6)
        summary = compute_gait_summary(right, left, fps=30)
        hip_cv = summary.get("R_hip_ROM_CV", None)
        assert hip_cv is not None
        assert hip_cv > 5.0


class TestFrontalPlaneSymmetry:
    """Verify frontal-plane bilateral symmetry detects asymmetric gait."""

    def test_limp_has_pelvis_obliquity_asymmetry(self):
        params = GAIT_PROFILES["limp"].params
        _, ar, al, _ = generate_frames(params, fps=30, n_cycles=6)
        summary = compute_gait_summary(ar, al, fps=30)
        assert summary["pelvis_obliquity_SI"] > 10, (
            f"Limp should show frontal asymmetry, got SI={summary['pelvis_obliquity_SI']:.1f}"
        )

    def test_normal_has_symmetric_pelvis(self):
        params = GaitParameters()
        _, ar, al, _ = generate_frames(params, fps=30, n_cycles=6)
        summary = compute_gait_summary(ar, al, fps=30)
        assert summary["pelvis_obliquity_SI"] < 5

    def test_asymmetry_increases_pelvis_si(self):
        si_values = []
        for asym in [0.0, 0.15, 0.35]:
            params = GaitParameters(asymmetry=asym)
            _, ar, al, _ = generate_frames(params, fps=30, n_cycles=6)
            s = compute_gait_summary(ar, al, fps=30)
            si_values.append(s["pelvis_obliquity_SI"])
        for i in range(len(si_values) - 1):
            assert si_values[i] < si_values[i + 1], (
                f"Pelvis SI should increase with asymmetry: {si_values}"
            )

    def test_frontal_symmetry_in_mqs_domain(self):
        params = GAIT_PROFILES["limp"].params
        _, ar, al, _ = generate_frames(params, fps=30, n_cycles=6)
        summary = compute_gait_summary(ar, al, fps=30)
        normal_params = GaitParameters()
        _, nr, nl, _ = generate_frames(normal_params, fps=30, n_cycles=6)
        normal_summary = compute_gait_summary(nr, nl, fps=30)
        assert summary["mqs_symmetry"] < normal_summary["mqs_symmetry"]


class TestStridePelvicAsymmetry:
    """Verify stride-phase pelvic drop asymmetry metric."""

    def test_symmetric_signal_near_zero(self):
        from movement_analytics.kinematics.gait_metrics import stride_pelvic_asymmetry
        t = np.linspace(0, 6 * np.pi, 180)
        signal = 5.0 * np.sin(t)
        hs = np.array([0, 30, 60, 90, 120, 150])
        asym = stride_pelvic_asymmetry(signal, hs)
        assert abs(asym) < 2.0

    def test_asymmetric_signal_detected(self):
        from movement_analytics.kinematics.gait_metrics import stride_pelvic_asymmetry
        n = 180
        hs = np.array([0, 30, 60, 90, 120, 150])
        signal = np.zeros(n)
        for i in range(len(hs) - 1):
            start, end = hs[i], hs[i + 1]
            mid = (start + end) // 2
            signal[start:mid] = 3.0 * np.sin(
                np.pi * np.linspace(0, 1, mid - start)
            )
            signal[mid:end] = 8.0 * np.sin(
                np.pi * np.linspace(0, 1, end - mid)
            )
        asym = stride_pelvic_asymmetry(signal, hs)
        assert asym > 20, f"Should detect half-cycle asymmetry, got {asym:.1f}"

    def test_too_few_strides_returns_nan(self):
        from movement_analytics.kinematics.gait_metrics import stride_pelvic_asymmetry
        signal = np.sin(np.linspace(0, 2 * np.pi, 60))
        hs = np.array([0, 30])
        assert np.isnan(stride_pelvic_asymmetry(signal, hs))

    def test_metric_present_in_summary(self):
        params = GaitParameters()
        _, ar, al, _ = generate_frames(params, fps=30, n_cycles=6)
        summary = compute_gait_summary(ar, al, fps=30)
        assert "pelvic_drop_asymmetry" in summary
        assert "trunk_lean_asymmetry" in summary


class TestDoubleSupportTime:
    """Verify double support time estimation from gait events."""

    def test_metric_present_in_summary(self):
        params = GaitParameters()
        _, ar, al, _ = generate_frames(params, fps=30, n_cycles=6)
        summary = compute_gait_summary(ar, al, fps=30)
        assert "double_support_pct" in summary

    def test_value_non_negative(self):
        for name in ["normal", "slow", "parkinsonian"]:
            p = GAIT_PROFILES[name]
            _, ar, al, _ = generate_frames(p.params, fps=30, n_cycles=6)
            s = compute_gait_summary(ar, al, fps=30)
            ds = s["double_support_pct"]
            if not np.isnan(ds):
                assert ds >= 0.0, f"{name}: double_support_pct should be >= 0, got {ds}"

    def test_value_below_100(self):
        for name in ["normal", "fast", "limp"]:
            p = GAIT_PROFILES[name]
            _, ar, al, _ = generate_frames(p.params, fps=30, n_cycles=6)
            s = compute_gait_summary(ar, al, fps=30)
            ds = s["double_support_pct"]
            if not np.isnan(ds):
                assert ds < 100.0, f"{name}: double_support_pct should be < 100, got {ds}"

    def test_not_scored_in_mqs(self):
        """DS is computed but not integrated into MQS (synthetic model limitation)."""
        from movement_analytics.kinematics.gait_metrics import mqs_domain_scores
        params = GaitParameters()
        _, ar, al, _ = generate_frames(params, fps=30, n_cycles=6)
        summary = compute_gait_summary(ar, al, fps=30)
        assert "double_support_pct" in summary
        domains_with = mqs_domain_scores(summary)
        summary_without = {k: v for k, v in summary.items() if k != "double_support_pct"}
        domains_without = mqs_domain_scores(summary_without)
        assert domains_with["temporal"] == domains_without["temporal"]


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
            "R_knee_flexion_ROM": 30, "L_knee_flexion_ROM": 30,
            "R_ankle_dorsiflexion_ROM": 10, "L_ankle_dorsiflexion_ROM": 10,
        }
        partial_metrics = {
            "R_hip_flexion_ROM": 40, "L_hip_flexion_ROM": 40,
        }
        full_domains = mqs_domain_scores(full_metrics)
        partial_domains = mqs_domain_scores(partial_metrics)
        assert partial_domains["kinematics"] > 0
        assert full_domains["kinematics"] != pytest.approx(
            partial_domains["kinematics"], abs=0.01
        ), "Partial metrics should produce different kinematics score than full"

    def test_completely_empty_metrics(self):
        from movement_analytics.kinematics.gait_metrics import movement_quality_score
        mqs = movement_quality_score({})
        assert mqs == pytest.approx(50.0)


class TestSignalCompleteness:
    """Verify signal completeness reporting."""

    def test_full_synthetic_completeness(self):
        params = GaitParameters()
        _, right, left, _ = generate_frames(params, fps=30, n_cycles=6)
        summary = compute_gait_summary(right, left, fps=30)
        assert summary["mqs_overall_completeness"] == pytest.approx(1.0)

    def test_empty_metrics_zero_completeness(self):
        from movement_analytics.kinematics.gait_metrics import mqs_signal_completeness
        c = mqs_signal_completeness({})
        assert c["kinematics"] == pytest.approx(0.0)
        assert c["smoothness"] == pytest.approx(0.0)
        assert c["symmetry"] == pytest.approx(0.0)

    def test_partial_metrics_partial_completeness(self):
        from movement_analytics.kinematics.gait_metrics import mqs_signal_completeness
        metrics = {
            "R_hip_flexion_ROM": 40,
            "L_hip_flexion_ROM": 40,
            "R_hip_flexion_SPARC": -1.5,
        }
        c = mqs_signal_completeness(metrics)
        assert 0 < c["kinematics"] < 1.0
        assert c["smoothness"] == pytest.approx(0.25)


class TestMQSConfidenceFactor:
    """Verify MQS confidence factor degrades with poor pose quality."""

    def test_synthetic_data_full_confidence(self):
        from movement_analytics.kinematics.gait_metrics import mqs_confidence_factor
        metrics = {"R_hip_flexion_ROM": 40}
        assert mqs_confidence_factor(metrics) == 1.0

    def test_perfect_video_confidence(self):
        from movement_analytics.kinematics.gait_metrics import mqs_confidence_factor
        metrics = {"pose_observed_fraction": 1.0, "pose_mean_confidence": 0.95}
        assert mqs_confidence_factor(metrics) == pytest.approx(0.95)

    def test_poor_detection_degrades(self):
        from movement_analytics.kinematics.gait_metrics import mqs_confidence_factor
        metrics = {"pose_observed_fraction": 0.5, "pose_mean_confidence": 0.6}
        cf = mqs_confidence_factor(metrics)
        assert cf == pytest.approx(0.3)
        assert cf < 1.0

    def test_zero_confidence(self):
        from movement_analytics.kinematics.gait_metrics import mqs_confidence_factor
        metrics = {"pose_observed_fraction": 0.0, "pose_mean_confidence": 0.0}
        assert mqs_confidence_factor(metrics) == 0.0


class TestSufficientEvidence:
    """Verify MQS sufficient evidence threshold."""

    def test_full_data_is_sufficient(self):
        from movement_analytics.kinematics.gait_metrics import mqs_sufficient_evidence
        params = GaitParameters()
        _, right, left, _ = generate_frames(params, fps=30, n_cycles=6)
        summary = compute_gait_summary(right, left, fps=30)
        assert mqs_sufficient_evidence(summary)
        assert summary["mqs_sufficient_evidence"] == 1.0

    def test_empty_data_is_insufficient(self):
        from movement_analytics.kinematics.gait_metrics import mqs_sufficient_evidence
        assert not mqs_sufficient_evidence({})

    def test_partial_data_threshold(self):
        from movement_analytics.kinematics.gait_metrics import mqs_sufficient_evidence
        metrics = {
            "R_hip_flexion_ROM": 40,
            "L_hip_flexion_ROM": 40,
            "R_hip_flexion_SPARC": -1.5,
        }
        assert not mqs_sufficient_evidence(metrics, min_completeness=0.5)


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

    def test_pelvic_obliquity_from_hip_heights(self):
        from movement_analytics.kinematics.joint_angles import compute_all_angles
        positions = {
            "pelvis": np.array([100, 200]),
            "shoulder": np.array([100, 100]),
            "right_hip": np.array([110, 200]),
            "right_knee": np.array([110, 300]),
            "right_ankle": np.array([110, 400]),
            "left_hip": np.array([90, 210]),
            "left_knee": np.array([90, 300]),
            "left_ankle": np.array([90, 400]),
        }
        angles = compute_all_angles(positions)
        assert "pelvic_obliquity" in angles
        assert angles["pelvic_obliquity"] > 0

    def test_video_key_mapping_correctness(self):
        key_mapping = {
            "right_hip_flexion": "hip_flexion",
            "right_knee_flexion": "knee_flexion",
            "right_ankle_dorsiflexion": "ankle_dorsiflexion",
            "left_hip_flexion": "hip_flexion",
            "left_knee_flexion": "knee_flexion",
            "left_ankle_dorsiflexion": "ankle_dorsiflexion",
        }
        for raw, mapped in key_mapping.items():
            assert mapped in ("hip_flexion", "knee_flexion", "ankle_dorsiflexion",
                              "elbow_flexion")

    def test_ankle_dorsiflexion_computed(self):
        from movement_analytics.kinematics.joint_angles import compute_all_angles
        positions = {
            "pelvis": np.array([100, 200]),
            "shoulder": np.array([100, 100]),
            "right_hip": np.array([110, 200]),
            "right_knee": np.array([110, 300]),
            "right_ankle": np.array([110, 400]),
            "right_toe": np.array([140, 400]),
            "left_hip": np.array([90, 200]),
            "left_knee": np.array([90, 300]),
            "left_ankle": np.array([90, 400]),
            "left_toe": np.array([60, 400]),
        }
        angles = compute_all_angles(positions)
        assert "right_ankle_dorsiflexion" in angles
        assert "left_ankle_dorsiflexion" in angles
        assert "right_ankle_angle" in angles


class TestNaNInterpolation:
    """Verify NaN interpolation in video processing angle arrays."""

    def test_nan_interpolation_fills_gaps(self):
        arr = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        valid = ~np.isnan(arr)
        indices = np.arange(len(arr))
        arr[~valid] = np.interp(indices[~valid], indices[valid], arr[valid])
        assert not np.any(np.isnan(arr))
        assert arr[1] == pytest.approx(2.0)
        assert arr[2] == pytest.approx(3.0)

    def test_all_nan_stays_nan(self):
        arr = np.full(5, np.nan)
        valid = ~np.isnan(arr)
        if np.any(valid):
            indices = np.arange(len(arr))
            arr[~valid] = np.interp(indices[~valid], indices[valid], arr[valid])
        assert np.all(np.isnan(arr))


class TestSensitivityAnalysis:
    """Verify MQS responds monotonically to continuous parameter degradation."""

    def test_knee_rom_sensitivity(self):
        mqs_scores = []
        for knee_rom in [60, 45, 30, 15]:
            params = GaitParameters(knee_rom=knee_rom)
            _, ar, al, _ = generate_frames(params, fps=30, n_cycles=4)
            s = compute_gait_summary(ar, al, fps=30)
            mqs_scores.append(s["movement_quality_score"])
        for i in range(len(mqs_scores) - 1):
            assert mqs_scores[i] >= mqs_scores[i + 1], (
                f"MQS should decrease as knee ROM decreases: "
                f"knee_rom step {i} ({mqs_scores[i]:.1f}) < step {i+1} ({mqs_scores[i+1]:.1f})"
            )

    def test_noise_level_sensitivity(self):
        mqs_scores = []
        for noise in [0, 1, 2, 4]:
            params = GaitParameters(noise_level=noise)
            _, ar, al, _ = generate_frames(params, fps=30, n_cycles=4)
            s = compute_gait_summary(ar, al, fps=30)
            mqs_scores.append(s["movement_quality_score"])
        for i in range(len(mqs_scores) - 1):
            assert mqs_scores[i] >= mqs_scores[i + 1], (
                f"MQS should decrease as noise increases: "
                f"noise step {i} ({mqs_scores[i]:.1f}) < step {i+1} ({mqs_scores[i+1]:.1f})"
            )

    def test_asymmetry_sensitivity(self):
        mqs_scores = []
        for asym in [0, 0.1, 0.2, 0.35]:
            params = GaitParameters(asymmetry=asym)
            _, ar, al, _ = generate_frames(params, fps=30, n_cycles=4)
            s = compute_gait_summary(ar, al, fps=30)
            mqs_scores.append(s["movement_quality_score"])
        for i in range(len(mqs_scores) - 1):
            assert mqs_scores[i] >= mqs_scores[i + 1], (
                f"MQS should decrease as asymmetry increases: "
                f"asym step {i} ({mqs_scores[i]:.1f}) < step {i+1} ({mqs_scores[i+1]:.1f})"
            )


    def test_pelvic_obliquity_sensitivity(self):
        mqs_scores = []
        for obliq in [5, 10, 15, 20]:
            params = GaitParameters(pelvic_obliquity=obliq)
            _, ar, al, _ = generate_frames(params, fps=30, n_cycles=4)
            s = compute_gait_summary(ar, al, fps=30)
            mqs_scores.append(s["movement_quality_score"])
        for i in range(len(mqs_scores) - 1):
            assert mqs_scores[i] >= mqs_scores[i + 1], (
                f"MQS should decrease as pelvic obliquity increases: "
                f"obliq step {i} ({mqs_scores[i]:.1f}) < step {i+1} ({mqs_scores[i+1]:.1f})"
            )

    def test_fps_stability(self):
        """MQS should be stable across frame rates for the same gait."""
        scores = []
        for fps in [15, 30, 60]:
            params = GaitParameters()
            _, ar, al, _ = generate_frames(params, fps=fps, n_cycles=6)
            s = compute_gait_summary(ar, al, fps=fps)
            scores.append(s["movement_quality_score"])
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                assert abs(scores[i] - scores[j]) < 5.0, (
                    f"MQS varies too much across FPS: "
                    f"{scores[i]:.1f} vs {scores[j]:.1f}"
                )


class TestSensitivityReport:
    """Test the sensitivity analysis report generation."""

    def test_sensitivity_report_creates_file(self):
        import os
        import tempfile

        from movement_analytics.cli import generate_sensitivity_report

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sensitivity.png")
            generate_sensitivity_report(path, fps=30, n_cycles=2)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000


class TestBenchmarkOutput:
    """Test the benchmark JSON generation."""

    def test_benchmark_creates_valid_json(self):
        import json

        from movement_analytics.cli import run_benchmark

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "benchmark.json")
            run_benchmark(output_path=path, fps=30, n_cycles=2)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["version"] == "1.4.0"
            assert data["n_domains"] == 6
            assert "normal" in data["profiles"]
            normal = data["profiles"]["normal"]
            assert "mqs" in normal
            assert "domains" in normal
            assert "completeness" in normal
            assert len(normal["domains"]) == 6

    def test_comparison_report_creates_image(self):
        from movement_analytics.cli import generate_comparison_report

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "compare.png")
            generate_comparison_report(path, fps=30, n_cycles=2)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000


class TestBenchmarkRegression:
    """Lock MQS scores to detect unintended regressions."""

    EXPECTED_MQS = {
        "normal": 98.3,
        "model_runway": 96.1,
        "fast": 90.1,
        "limp": 88.8,
        "trendelenburg": 87.3,
        "slow": 83.8,
        "stiff_knee": 79.9,
        "noisy": 67.7,
        "parkinsonian": 63.4,
    }

    MQS_TOLERANCE = 2.0

    def test_mqs_scores_match_baseline(self):
        for name, expected in self.EXPECTED_MQS.items():
            profile = GAIT_PROFILES[name]
            _, ar, al, _ = generate_frames(profile.params, fps=30, n_cycles=6)
            summary = compute_gait_summary(ar, al, fps=30)
            actual = summary["movement_quality_score"]
            assert abs(actual - expected) <= self.MQS_TOLERANCE, (
                f"{name}: MQS {actual:.1f} deviates from baseline "
                f"{expected:.1f} by {abs(actual - expected):.1f} "
                f"(tolerance {self.MQS_TOLERANCE})"
            )

    def test_ranking_preserved(self):
        scores = {}
        for name in self.EXPECTED_MQS:
            profile = GAIT_PROFILES[name]
            _, ar, al, _ = generate_frames(profile.params, fps=30, n_cycles=6)
            summary = compute_gait_summary(ar, al, fps=30)
            scores[name] = summary["movement_quality_score"]
        sorted_expected = sorted(
            self.EXPECTED_MQS.keys(),
            key=lambda k: self.EXPECTED_MQS[k],
            reverse=True,
        )
        sorted_actual = sorted(
            scores.keys(), key=lambda k: scores[k], reverse=True
        )
        assert sorted_actual == sorted_expected, (
            f"Ranking changed: expected {sorted_expected}, got {sorted_actual}"
        )


class TestVideoProcessingPipeline:
    """Test process_video data flow: angle mapping, NaN interpolation, metadata."""

    @staticmethod
    def _make_test_video(path, n_frames=60, fps=30):
        h, w = 480, 640
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for _ in range(n_frames):
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
        writer.release()

    @staticmethod
    def _fake_positions(frame_idx, n_frames):
        """Generate oscillating joint positions to simulate walking."""
        t = frame_idx / n_frames * 4 * np.pi
        hip_y = 200.0
        knee_y = 320.0
        ankle_y = 430.0
        shoulder_y = 120.0
        hip_offset = 30 * np.sin(t)
        return {
            "pelvis": np.array([320.0, hip_y]),
            "shoulder": np.array([320.0, shoulder_y]),
            "neck": np.array([320.0, shoulder_y - 10]),
            "head": np.array([320.0, 80.0]),
            "left_hip": np.array([300.0, hip_y]),
            "right_hip": np.array([340.0, hip_y]),
            "left_knee": np.array([300.0 - hip_offset, knee_y]),
            "right_knee": np.array([340.0 + hip_offset, knee_y]),
            "left_ankle": np.array([300.0 - hip_offset * 0.5, ankle_y]),
            "right_ankle": np.array([340.0 + hip_offset * 0.5, ankle_y]),
            "left_shoulder": np.array([280.0, shoulder_y]),
            "right_shoulder": np.array([360.0, shoulder_y]),
            "left_elbow": np.array([260.0, 180.0]),
            "right_elbow": np.array([380.0, 180.0]),
            "left_wrist": np.array([250.0, 230.0]),
            "right_wrist": np.array([390.0, 230.0]),
            "left_toe": np.array([290.0, 460.0]),
            "right_toe": np.array([350.0, 460.0]),
        }

    def test_process_video_returns_metadata(self):
        from movement_analytics.pose.estimator import process_video

        n_frames = 60
        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path = os.path.join(tmpdir, "test.mp4")
            self._make_test_video(vid_path, n_frames=n_frames)

            call_count = [0]

            def mock_process_frame(frame, min_visibility=0.5):
                idx = call_count[0]
                call_count[0] += 1
                if idx % 10 == 5:
                    return None, 0.0
                return self._fake_positions(idx, n_frames), 0.85

            with patch(
                "movement_analytics.pose.estimator.PoseEstimator"
            ) as MockEst:
                instance = MagicMock()
                instance.process_frame = mock_process_frame
                instance.__enter__ = lambda s: s
                instance.__exit__ = lambda s, *a: None
                MockEst.return_value = instance

                frames, ar, al, fps, meta = process_video(vid_path)

            assert len(frames) == n_frames
            assert "hip_flexion" in ar
            assert "hip_flexion" in al
            assert meta["observed_fraction"] < 1.0
            assert meta["mean_confidence"] > 0
            assert isinstance(meta["interpolation_fractions"], dict)
            assert meta["observed_fraction"] == pytest.approx(
                (n_frames - 6) / n_frames, abs=0.02
            )

    def test_process_video_nan_interpolation(self):
        from movement_analytics.pose.estimator import process_video

        n_frames = 30
        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path = os.path.join(tmpdir, "test.mp4")
            self._make_test_video(vid_path, n_frames=n_frames)

            call_count = [0]

            def mock_process_frame(frame, min_visibility=0.5):
                idx = call_count[0]
                call_count[0] += 1
                if idx in (10, 11, 12):
                    return None, 0.0
                return self._fake_positions(idx, n_frames), 0.9

            with patch(
                "movement_analytics.pose.estimator.PoseEstimator"
            ) as MockEst:
                instance = MagicMock()
                instance.process_frame = mock_process_frame
                instance.__enter__ = lambda s: s
                instance.__exit__ = lambda s, *a: None
                MockEst.return_value = instance

                _, ar, al, _, meta = process_video(vid_path)

            for key in ar:
                assert not np.any(np.isnan(ar[key])), f"NaN in right {key}"
            for key in al:
                assert not np.any(np.isnan(al[key])), f"NaN in left {key}"
            hip_interp = meta["interpolation_fractions"].get("hip_flexion", 0)
            assert hip_interp > 0, "Should have interpolated missing frames"

    def test_process_video_produces_valid_mqs(self):
        from movement_analytics.pose.estimator import process_video

        n_frames = 120
        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path = os.path.join(tmpdir, "test.mp4")
            self._make_test_video(vid_path, n_frames=n_frames, fps=30)

            call_count = [0]

            def mock_process_frame(frame, min_visibility=0.5):
                idx = call_count[0]
                call_count[0] += 1
                return self._fake_positions(idx, n_frames), 0.95

            with patch(
                "movement_analytics.pose.estimator.PoseEstimator"
            ) as MockEst:
                instance = MagicMock()
                instance.process_frame = mock_process_frame
                instance.__enter__ = lambda s: s
                instance.__exit__ = lambda s, *a: None
                MockEst.return_value = instance

                _, ar, al, fps, _ = process_video(vid_path)

            summary = compute_gait_summary(ar, al, fps=fps)
            mqs = summary["movement_quality_score"]
            assert 0 <= mqs <= 100, f"MQS out of range: {mqs}"
            assert not np.isnan(mqs)

    def test_process_video_empty_video(self):
        from movement_analytics.pose.estimator import process_video

        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path = os.path.join(tmpdir, "empty.mp4")
            self._make_test_video(vid_path, n_frames=5)

            def mock_process_frame(frame, min_visibility=0.5):
                return None, 0.0

            with patch(
                "movement_analytics.pose.estimator.PoseEstimator"
            ) as MockEst:
                instance = MagicMock()
                instance.process_frame = mock_process_frame
                instance.__enter__ = lambda s: s
                instance.__exit__ = lambda s, *a: None
                MockEst.return_value = instance

                frames, ar, al, _, meta = process_video(vid_path)

            assert ar == {}
            assert al == {}
            assert meta["observed_fraction"] == 0.0

    def test_angle_computation_round_trip(self):
        """Validate that compute_all_angles → key mapping is consistent."""
        from movement_analytics.kinematics.joint_angles import compute_all_angles

        positions = self._fake_positions(0, 100)
        angles = compute_all_angles(positions)

        assert "right_hip_flexion" in angles
        assert "left_hip_flexion" in angles
        assert "right_knee_flexion" in angles
        assert "right_ankle_dorsiflexion" in angles

        for key in angles:
            val = angles[key]
            assert np.isfinite(val), f"{key} is not finite: {val}"
            if "flexion" in key and "dorsiflexion" not in key:
                assert 0 <= val <= 180, f"{key} out of range: {val}"

    def test_pelvic_obliquity_from_video_positions(self):
        from movement_analytics.kinematics.joint_angles import compute_all_angles
        positions = self._fake_positions(0, 100)
        angles = compute_all_angles(positions)
        assert "pelvic_obliquity" in angles
        assert "pelvic_obliquity_signed" in angles
        assert angles["pelvic_obliquity"] >= 0

    def test_pelvic_obliquity_signed_direction(self):
        from movement_analytics.kinematics.joint_angles import compute_all_angles
        pos_tilted = self._fake_positions(0, 100).copy()
        pos_tilted["right_hip"] = np.array([340.0, 210.0])
        pos_tilted["left_hip"] = np.array([300.0, 200.0])
        angles = compute_all_angles(pos_tilted)
        assert angles["pelvic_obliquity_signed"] > 0
        pos_tilted["right_hip"] = np.array([340.0, 190.0])
        pos_tilted["left_hip"] = np.array([300.0, 200.0])
        angles2 = compute_all_angles(pos_tilted)
        assert angles2["pelvic_obliquity_signed"] < 0

    def test_trunk_lean_from_video_positions(self):
        from movement_analytics.kinematics.joint_angles import compute_all_angles
        positions = self._fake_positions(0, 100)
        angles = compute_all_angles(positions)
        assert "trunk_lean" in angles

    def test_process_video_includes_frontal_signals(self):
        from movement_analytics.pose.estimator import process_video

        n_frames = 60
        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path = os.path.join(tmpdir, "test.mp4")
            self._make_test_video(vid_path, n_frames=n_frames)

            call_count = [0]

            def mock_process_frame(frame, min_visibility=0.5):
                idx = call_count[0]
                call_count[0] += 1
                return self._fake_positions(idx, n_frames), 0.9

            with patch(
                "movement_analytics.pose.estimator.PoseEstimator"
            ) as MockEst:
                instance = MagicMock()
                instance.process_frame = mock_process_frame
                instance.__enter__ = lambda s: s
                instance.__exit__ = lambda s, *a: None
                MockEst.return_value = instance

                _, ar, al, _, _ = process_video(vid_path)

            assert "pelvis_obliquity" in ar
            assert "pelvis_obliquity" in al
            assert "trunk_lateral_lean" in ar

    def test_confidence_factor_with_video_metadata(self):
        from movement_analytics.kinematics.gait_metrics import mqs_confidence_factor
        metrics = {"pose_observed_fraction": 0.85, "pose_mean_confidence": 0.9}
        cf = mqs_confidence_factor(metrics)
        assert 0 < cf < 1.0
        assert cf == pytest.approx(0.85 * 0.9, abs=0.01)

    def test_confidence_factor_synthetic_is_1(self):
        from movement_analytics.kinematics.gait_metrics import mqs_confidence_factor
        metrics = {"cadence": 110, "stride_time_mean": 1.0}
        cf = mqs_confidence_factor(metrics)
        assert cf == 1.0
