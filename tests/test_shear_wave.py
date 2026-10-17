import numpy as np
import pytest

from opencpt import shear_wave as sw

# Rows of research/cpt/vs_sdee/results/vs_compare_all.csv (12 SCPT sites of Assaf et al. 2023):
# z (m), qc, qt, fs (kPa), Ic, Vs A07 MLR, Vs MG15_qc MLR (m/s)
ROWS = np.array(
    [
        [4.02, 5710.25, 5711.981, 13.8, 1.653, 137.503, 137.516],
        [14.51, 8406.05, 8419.367, 84.45, 2.075, 196.02, 194.789],
        [8.07, 7161.0, 7174.44, 44.0, 1.849, 166.715, 166.759],
        [33.37, 3740.0, 3740.008, 44.55, 2.791, 217.172, 217.044],
    ]
)
z, qc, qt, fs, ic, vs_a07, vs_mg15 = ROWS.T


def test_a07_matches_research_comparison():
    assert sw.vs_assaf2023_a07(qt, ic, z) == pytest.approx(vs_a07, abs=0.01)


def test_mg15_qc_matches_research_comparison():
    assert sw.vs_assaf2023_mg15_qc(qc, fs, z) == pytest.approx(vs_mg15, abs=0.01)


def test_a07_power_form_by_hand():
    # Vs = e^3.25 qt^0.15 Ic^0.214 z^0.193
    expected = np.exp(3.25) * 10000**0.15 * 2.0**0.214 * 10**0.193
    assert sw.vs_assaf2023_a07(10000, 2.0, 10) == pytest.approx(expected)


def test_invalid_readings_give_nan():
    vs = sw.vs_assaf2023_a07([5000, -1, 5000, np.nan], [2.0, 2.0, 2.0, 2.0], [5, 5, 0, 5])
    assert np.isfinite(vs[0])
    assert np.isnan(vs[1:]).all()
    vs = sw.vs_assaf2023_mg15_qc([5000, 5000, 0], [-1, 10, 10], [5, 5, 5])
    assert np.isnan(vs[0]) and np.isfinite(vs[1]) and np.isnan(vs[2])


def test_mg15_qc_raises_zero_fs_to_minimum():
    assert sw.vs_assaf2023_mg15_qc(5000, 0.0, 5) == pytest.approx(sw.vs_assaf2023_mg15_qc(5000, sw.FS_MIN, 5))


def test_time_averaged_vs_by_hand():
    # 10 m at 100 m/s and 20 m at 200 m/s: 30 / (0.1 + 0.1) = 150 m/s
    assert sw.time_averaged_vs([10, 30], [100, 200]) == pytest.approx((150.0, 30.0))


def test_time_averaged_vs_stops_at_z_max():
    # The reading at 40 m covers 20-30 m only.
    assert sw.time_averaged_vs([20, 40], [100, 200]) == pytest.approx((30 / (0.2 + 0.05), 30.0))


def test_time_averaged_vs_of_a_shallow_sounding_covers_its_depth():
    assert sw.time_averaged_vs([5, 10, 20], [100, 100, 100]) == pytest.approx((100.0, 20.0))


def test_time_averaged_vs_skips_readings_without_vs():
    assert sw.time_averaged_vs([10, 20, 30], [100, np.nan, 200]) == pytest.approx((30 / (0.1 + 0.1), 30.0))
    assert sw.time_averaged_vs([10], [np.nan])[1] == 0.0
