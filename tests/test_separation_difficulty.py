import numpy as np

from ml.analyze_separation_difficulty import difficulty_scores


def test_identical_curves_have_maximum_difficulty_scores():
    curve = np.array([1.0, 2.0, 1.0])
    result = difficulty_scores(curve, curve, np.ones(3))
    assert result["temporal_overlap_percent"] == 100.0
    assert result["shared_activity_percent"] == 100.0
    assert result["amplitude_similarity_percent"] == 100.0
    assert result["entropy_ambiguity_percent"] == 100.0


def test_nonoverlapping_curves_have_zero_temporal_and_entropy_difficulty():
    fdg = np.array([1.0, 1.0, 0.0, 0.0])
    pbr = np.array([0.0, 0.0, 1.0, 1.0])
    result = difficulty_scores(fdg, pbr, np.ones(4))
    assert result["temporal_overlap_percent"] == 0.0
    assert result["shared_activity_percent"] == 0.0
    assert result["entropy_ambiguity_percent"] == 0.0
    assert result["amplitude_similarity_percent"] == 100.0


def test_temporal_overlap_is_scale_invariant_but_shared_activity_is_not():
    fdg = np.array([1.0, 2.0, 1.0])
    pbr = 0.5 * fdg
    result = difficulty_scores(fdg, pbr, np.ones(3))
    assert np.isclose(result["temporal_overlap_percent"], 100.0)
    assert result["shared_activity_percent"] < 100.0
    assert result["amplitude_similarity_percent"] < 100.0


def test_zero_curve_is_handled_safely():
    result = difficulty_scores(np.ones(3), np.zeros(3), np.ones(3))
    assert all(value == 0.0 for value in result.values())
