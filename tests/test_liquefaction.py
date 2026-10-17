import math

import numpy as np
import pytest

from opencpt import liquefaction as liq
from opencpt import site


class TestSite:
    def test_corrected_cone_resistance_ignores_negative_u2(self):
        qt = site.corrected_cone_resistance([1000, 1000], [150, -20], area_ratio=0.8)
        assert qt == pytest.approx([1030, 1000])

    def test_vertical_stresses_by_hand(self):
        sv, u0, svp = site.vertical_stresses([1, 2, 3], 18.0, gwt=1.0)
        assert sv == pytest.approx([18, 36, 54])
        assert u0 == pytest.approx([0, 9.81, 19.62])
        assert svp == pytest.approx([18, 26.19, 34.38])

    def test_unit_weight_is_limited_to_14_22(self):
        gamma = site.unit_weight_robertson_cabal_2010([50, 50000], [0.1, 900])
        assert gamma == pytest.approx([14.0, 22.0])

    @pytest.mark.parametrize(
        "qt, fs, low, high",
        [
            (15000, 60, 1.0, 2.05),  # clean sand
            (600, 30, 2.95, 4.0),  # soft clay
        ],
    )
    def test_ic_falls_in_the_expected_soil_zone(self, qt, fs, low, high):
        sv, svp = 100.0, 60.0
        ic = site.soil_behaviour_type_index(np.array([qt]), np.array([fs]), np.array([sv]), np.array([svp]))
        assert low < ic[0] < high


class TestBoulangerIdriss2014:
    def test_fines_content_from_ic(self):
        assert liq.fines_content_bi2014([1.0, 2.6, 3.5]) == pytest.approx([0, 71, 100])
        assert liq.fines_content_bi2014([2.6], cfc=0.1) == pytest.approx([79])

    def test_crr_at_qc1ncs_100(self):
        q = 100.0
        expected = math.exp(q / 113 + (q / 1000) ** 2 - (q / 140) ** 3 + (q / 137) ** 4 - 2.80)
        assert liq.crr_m75_bi2014([q])[0] == pytest.approx(expected)
        assert liq.crr_m75_bi2014([q])[0] == pytest.approx(0.1373, abs=5e-4)

    def test_crr_is_capped_at_qc1ncs_211(self):
        assert liq.crr_m75_bi2014([300])[0] == liq.crr_m75_bi2014([211])[0]

    def test_msf_is_one_at_mw_7_5(self):
        assert liq.magnitude_scaling_factor_bi2014(7.5, [60, 100, 180]) == pytest.approx([1, 1, 1], abs=1e-3)

    def test_msf_max_is_capped_at_2_2_not_qc1ncs_at_180(self):
        # B&I (2014): MSFmax = 1.09 + (qc1Ncs/180)^3 <= 2.2, reached near qc1Ncs = 187.
        mw6 = liq.magnitude_scaling_factor_bi2014(6.0, [180, 200, 254])
        factor = 8.64 * math.exp(-6.0 / 4) - 1.325
        assert mw6 == pytest.approx([1 + 1.09 * factor, 1 + 1.2 * factor, 1 + 1.2 * factor])
        assert mw6[1] == pytest.approx(1.723, abs=1e-3)

    def test_msf_is_above_one_for_smaller_earthquakes(self):
        assert liq.magnitude_scaling_factor_bi2014(6.0, [100])[0] > 1

    def test_k_sigma_is_one_at_atmospheric_pressure(self):
        assert liq.overburden_correction_bi2014([site.PA], [100])[0] == pytest.approx(1.0)

    def test_k_sigma_is_capped_at_1_1(self):
        assert liq.overburden_correction_bi2014([1.0], [200])[0] == pytest.approx(1.1)

    def test_rd_refuses_depths_beyond_34_m(self):
        assert liq.stress_reduction_coefficient([34.0], 7.5)[0] == pytest.approx(0.619, abs=1e-3)
        with pytest.raises(ValueError, match="34 m"):
            liq.stress_reduction_coefficient([20.0, 34.5], 7.5)

    def test_triggering_refuses_max_depth_beyond_34_m(self):
        with pytest.raises(ValueError, match="34 m"):
            liq.triggering_bi2014([1.0, 2.0], [5000.0, 5000.0], [50.0, 50.0], gwt=0.5, mw=7.5, amax=0.3, max_depth=40)

    def test_rd_is_about_one_at_the_surface(self):
        assert liq.stress_reduction_coefficient([0.0], 7.5)[0] == pytest.approx(1.0, abs=0.01)

    def test_qc1ncs_needs_no_fines_adjustment_for_clean_sand(self):
        q1, cn = liq.qc1ncs_bi2014(np.array([10000.0]), np.array([site.PA]), np.array([0.0]))
        # At σ'v = pa, CN = 1, and FC = 0 gives Δqc1N ≈ 0.
        assert cn[0] == pytest.approx(1.0)
        assert q1[0] == pytest.approx(10000 / site.PA, abs=0.1)


class TestCriticalLayers:
    z = np.round(np.arange(0.05, 3.01, 0.05), 2)

    def test_keeps_runs_at_least_min_thickness(self):
        fsl = np.where((self.z >= 1.0) & (self.z <= 1.5), 0.5, 2.0)
        assert liq.critical_layers(self.z, fsl) == [(1.0, 1.5)]

    def test_drops_thin_runs(self):
        fsl = np.where((self.z >= 1.0) & (self.z <= 1.2), 0.5, 2.0)
        assert liq.critical_layers(self.z, fsl) == []

    def test_merges_across_one_interbedded_reading(self):
        fsl = np.where((self.z >= 1.0) & (self.z <= 2.0), 0.5, 2.0)
        fsl[np.isclose(self.z, 1.5)] = np.nan  # e.g. a clayey reading
        assert liq.critical_layers(self.z, fsl) == [(1.0, 2.0)]

    def test_treats_unassessed_readings_as_not_liquefiable(self):
        assert liq.critical_layers(self.z, np.full_like(self.z, np.nan)) == []


class TestInvalidReadings:
    z = np.round(np.arange(0.5, 6.01, 0.5), 2)

    def run(self, qt=None, fs=None, **kw):
        qt = np.full(self.z.size, 5000.0) if qt is None else qt
        fs = np.full(self.z.size, 30.0) if fs is None else fs
        return liq.triggering_bi2014(self.z, qt, fs, gwt=1.0, mw=7.5, amax=0.3, **kw)

    def test_a_missing_reading_does_not_spoil_the_readings_below(self):
        fs = np.full(self.z.size, 30.0)
        fs[3] = np.nan
        r, clean = self.run(fs=fs), self.run()
        assert np.isfinite(r.sigma_v).all()
        assert r.invalid.tolist() == [i == 3 for i in range(self.z.size)]
        assert np.isnan(r.fsl[3])
        # Readings below the gap match the complete profile (uniform soil, so interpolation is exact).
        assert r.fsl[4:] == pytest.approx(clean.fsl[4:])

    @pytest.mark.parametrize("qt_bad, fs_bad", [(-300.0, 30.0), (0.0, 30.0), (5000.0, -2.0)])
    def test_non_physical_readings_are_flagged_not_silently_blank(self, qt_bad, fs_bad):
        qt = np.full(self.z.size, 5000.0)
        fs = np.full(self.z.size, 30.0)
        qt[5], fs[5] = qt_bad, fs_bad
        r = self.run(qt=qt, fs=fs)
        assert r.invalid[5] and r.invalid.sum() == 1
        assert np.isnan(r.fsl[5])
        assert np.isfinite(r.fsl[6:]).all()

    def test_every_reading_invalid_is_an_error(self):
        with pytest.raises(ValueError, match="No reading"):
            self.run(qt=np.full(self.z.size, -1.0))

    def test_unit_weight_must_be_one_value(self):
        with pytest.raises(TypeError, match="one value"):
            self.run(unit_weight=np.full(self.z.size, 18.0))
        assert np.isfinite(self.run(unit_weight=18.0).fsl[2:]).all()

    def test_critical_qc1ncs_ignores_invalid_readings(self):
        qt = np.full(self.z.size, 3000.0)
        qt[6] = -1.0
        r = self.run(qt=qt)
        mean, _ = r.critical_qc1ncs
        assert np.isfinite(mean)
