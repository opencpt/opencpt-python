import math

import numpy as np
import pytest

from opencpt import liquefaction as liq
from opencpt import site


class TestSite:
    def test_corrected_cone_resistance_uses_u2_as_measured(self):
        # B&I (2014) Eq. 2.3: a negative u2 (dilative sand) reduces qt.
        qt = site.corrected_cone_resistance([1000, 1000], [150, -20], area_ratio=0.8)
        assert qt == pytest.approx([1030, 996])

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


class TestNormalizedParameters:
    def test_ic_qtn_and_n_by_hand_when_the_exponent_term_is_one(self):
        # σ′v = pa, so (pa/σ′v)^n = 1 and Qtn = qnet/pa for every n.
        # qnet = 100 pa, Fr = 1, σv = 200 kPa.
        pa = site.PA
        svp = pa
        qnet = 100.0 * pa
        sv = 200.0
        qt = sv + qnet
        fs = qnet / 100.0
        u0, u2 = 40.0, 150.0

        fr = fs / qnet * 100
        q = qnet / svp
        qtn = (qnet / pa) * (pa / svp) ** 1.0
        ic = math.sqrt((3.47 - math.log10(qtn)) ** 2 + (math.log10(fr) + 1.22) ** 2)
        # n = 0.381 Ic + 0.05 (σ′v/pa) - 0.15 ≤ 1. Here σ′v/pa = 1, so n = 0.381 Ic - 0.10.
        # The next pass repeats Qtn and Ic, so the iteration stops and this n is the one used.
        n = min(0.381 * ic + 0.05 * (svp / pa) - 0.15, 1.0)
        assert ic == pytest.approx(1.9103, abs=1e-4)
        assert n == pytest.approx(0.6278, abs=1e-4)
        bq = (u2 - u0) / qnet
        u = (u2 - u0) / svp
        qe = qt - u2

        got = site.normalized_parameters([qt], [fs], [sv], [svp], u0=[u0], u2=[u2])
        assert fr == pytest.approx(1.0)
        assert q == pytest.approx(100.0)
        assert qtn == pytest.approx(100.0)
        assert n < 1.0
        assert got.qnet == pytest.approx([qnet])
        assert got.qt1 == pytest.approx([q])
        assert got.qtn == pytest.approx([qtn])
        assert got.fr == pytest.approx([fr])
        assert got.ic == pytest.approx([ic])
        assert got.n == pytest.approx([n])
        assert got.bq == pytest.approx([bq])
        assert got.u == pytest.approx([u])
        assert got.u == pytest.approx(got.bq * got.qt1)
        assert got.qe == pytest.approx([qe])
        assert site.soil_behaviour_type_index([qt], [fs], [sv], [svp]) == pytest.approx(got.ic)

    def test_ic_qtn_and_n_by_hand_when_n_changes_qtn(self):
        # σ′v = 250 kPa ≠ pa, so Qtn moves once n drops below 1.
        pa = site.PA
        qt, fs, sv, svp = 20000.0, 100.0, 400.0, 250.0
        qnet = qt - sv
        # Fr = fs/(qt-σv)*100. qnet > 1 kPa and Fr > 0.01, so the Ic floors do not apply.
        fr = fs / qnet * 100
        n = 1.0
        # At the starting exponent, Qtn = (qnet/pa)*(pa/σ′v) = qnet/σ′v.
        qtn = (qnet / pa) * (pa / svp) ** n
        assert qtn == pytest.approx(qnet / svp)
        assert qtn == pytest.approx(78.4)
        for _ in range(30):
            qtn = max((qnet / pa) * (pa / svp) ** n, 1.0)
            ic = math.sqrt((3.47 - math.log10(qtn)) ** 2 + (math.log10(fr) + 1.22) ** 2)
            n_new = min(0.381 * ic + 0.05 * (svp / pa) - 0.15, 1.0)
            # Same stop as the library: NumPy allclose with atol 1e-4 (rtol 1e-5).
            if np.allclose(n, n_new, atol=1e-4):
                break
            n = n_new
        else:
            raise AssertionError("hand iteration did not converge")
        assert qtn == pytest.approx(110.292, abs=1e-3)
        assert ic == pytest.approx(1.7025, abs=1e-4)
        assert n == pytest.approx(0.6221, abs=1e-4)

        got = site.normalized_parameters([qt], [fs], [sv], [svp])
        assert got.qnet == pytest.approx([qnet])
        assert got.qt1 == pytest.approx([qnet / svp])
        assert got.fr == pytest.approx([fr])
        assert got.qtn == pytest.approx([qtn])
        assert got.n == pytest.approx([n])
        assert got.ic == pytest.approx([ic])
        # n < 1 and pa/σ′v < 1, so Qtn > Q.
        assert got.qtn[0] > got.qt1[0] + 1.0
        assert np.isnan(got.bq).all() and np.isnan(got.u).all() and np.isnan(got.qe).all()

    def test_stress_exponent_is_capped_at_one(self):
        # Soft clay. At n = 1, Qtn = qnet/σ′v = 500/60, and the uncapped n exceeds 1.
        pa = site.PA
        qnet, svp = 500.0, 60.0
        qtn = qnet / svp
        fr = 30.0 / qnet * 100
        ic = math.sqrt((3.47 - math.log10(qtn)) ** 2 + (math.log10(fr) + 1.22) ** 2)
        n_uncapped = 0.381 * ic + 0.05 * (svp / pa) - 0.15
        assert n_uncapped > 1.0
        got = site.normalized_parameters([100.0 + qnet], [30.0], [100.0], [svp])
        assert got.n == pytest.approx([1.0])
        assert got.qtn == pytest.approx([qtn])
        assert got.ic == pytest.approx([ic])


class TestPorePressureAndUnitWeightLimits:
    def test_user_u0_replaces_the_hydrostatic_profile(self):
        z = np.array([1.0, 2.0, 3.0])
        # Under-drained at 3 m: pore pressure above hydrostatic from a water table at the surface.
        supplied = np.array([0.0, 0.0, 50.0])
        sv, u0, svp = site.vertical_stresses(z, 18.0, u0=supplied)
        hydro_sv, hydro_u0, _ = site.vertical_stresses(z, 18.0, gwt=0.0)
        assert sv == pytest.approx(hydro_sv)
        assert u0 == pytest.approx(supplied)
        assert svp == pytest.approx([18.0, 36.0, 4.0])
        assert hydro_u0 == pytest.approx([9.81, 19.62, 29.43])
        with pytest.raises(TypeError, match="gwt"):
            site.vertical_stresses(z, 18.0)
        with pytest.raises(ValueError, match="u0"):
            site.vertical_stresses(z, 18.0, gwt=0.0, u0=[1.0, 2.0])

    def test_site_profile_returns_normalized_parameters_and_qE(self):
        # One reading at 5 m, γ = 18 kN/m³, water table at 2 m.
        # σv = 18*5 = 90 kPa
        # u0 = 9.81*(5-2) = 29.43 kPa
        # σ′v = 90 - 29.43 = 60.57 kPa
        z, qt, fs = [5.0], [8000.0], [40.0]
        u2 = [80.0]
        profile = site.site_profile(z, qt, fs, gwt=2.0, unit_weight=18.0, u2=u2)
        qnet = 8000.0 - 90.0
        assert profile.sigma_v == pytest.approx([90.0])
        assert profile.u0 == pytest.approx([29.43])
        assert profile.sigma_v_eff == pytest.approx([60.57])
        assert profile.qnet == pytest.approx([qnet])
        assert profile.qt1 == pytest.approx([qnet / 60.57])
        assert profile.fr == pytest.approx([40.0 / qnet * 100])
        assert profile.bq == pytest.approx([(80.0 - 29.43) / qnet])
        assert profile.u == pytest.approx([(80.0 - 29.43) / 60.57])
        assert profile.qe == pytest.approx([7920.0])
        direct = site.normalized_parameters(qt, fs, profile.sigma_v, profile.sigma_v_eff, u0=profile.u0, u2=u2)
        assert profile.ic == pytest.approx(direct.ic)
        assert profile.qtn == pytest.approx(direct.qtn)
        assert profile.n == pytest.approx(direct.n)
        # u2 does not feed Ic or the stresses.
        bare = site.site_profile(z, qt, fs, gwt=2.0, unit_weight=18.0)
        assert np.isnan(bare.bq).all() and np.isnan(bare.u).all() and np.isnan(bare.qe).all()
        assert bare.ic == pytest.approx(profile.ic)
        assert bare.sigma_v_eff == pytest.approx(profile.sigma_v_eff)

    def test_site_profile_u0_changes_effective_stress_and_ic(self):
        z = [1.0, 2.0, 3.0]
        qt = [3000.0, 5000.0, 8000.0]
        fs = [20.0, 40.0, 50.0]
        base = site.site_profile(z, qt, fs, gwt=1.0, unit_weight=18.0)
        assert base.u0 == pytest.approx([0.0, 9.81, 19.62])
        dry = site.site_profile(z, qt, fs, gwt=1.0, unit_weight=18.0, u0=[0.0, 0.0, 0.0])
        assert dry.sigma_v == pytest.approx(base.sigma_v)
        assert dry.sigma_v_eff == pytest.approx(dry.sigma_v)
        assert dry.ic[0] == pytest.approx(base.ic[0])
        assert not np.allclose(dry.ic, base.ic)
        # The same hydrostatic values, passed back in, reproduce the default profile.
        same = site.site_profile(z, qt, fs, unit_weight=18.0, u0=base.u0)
        assert same.ic == pytest.approx(base.ic)
        assert same.qtn == pytest.approx(base.qtn)
        assert same.sigma_v_eff == pytest.approx(base.sigma_v_eff)
        with pytest.raises(ValueError, match="u0"):
            site.site_profile(z, qt, fs, gwt=1.0, u0=[0.0])
        with pytest.raises(TypeError, match="gwt"):
            site.site_profile(z, qt, fs)

    def test_invalid_reading_leaves_normalized_parameters_undefined(self):
        z = [1.0, 2.0, 3.0]
        qt = [3000.0, -5.0, 4000.0]
        fs = [20.0, 20.0, 25.0]
        profile = site.site_profile(z, qt, fs, gwt=0.5, unit_weight=18.0, u2=[10.0, 10.0, 30.0])
        assert profile.invalid.tolist() == [False, True, False]
        assert np.isnan(profile.ic[1]) and np.isnan(profile.qtn[1]) and np.isnan(profile.fr[1])
        assert np.isnan(profile.bq[1]) and np.isnan(profile.qe[1])
        assert np.isfinite(profile.u0).all()
        assert np.isfinite(profile.qtn[0]) and np.isfinite(profile.bq[0]) and np.isfinite(profile.qe[0])

    def test_unit_weight_limits_default_to_14_22_and_can_open(self):
        # γ/γw = 0.27 log Rf + 0.36 log(qt/pa) + 1.236, Rf = fs/qt*100.
        # qt = 50 kPa, fs = 0.1 kPa sits below the default floor; qt = 50 MPa sits above it.
        def raw(qt, fs):
            rf = fs / qt * 100
            return site.GAMMA_W * (0.27 * math.log10(rf) + 0.36 * math.log10(qt / site.PA) + 1.236)

        low, high = raw(50.0, 0.1), raw(50000.0, 900.0)
        assert low < 14.0
        assert high > 22.0
        assert site.unit_weight_robertson_cabal_2010([50.0, 50000.0], [0.1, 900.0]) == pytest.approx([14.0, 22.0])
        opened = site.unit_weight_robertson_cabal_2010(
            [50.0, 50000.0], [0.1, 900.0], gamma_min=None, gamma_max=None
        )
        assert opened == pytest.approx([low, high])
        floor_only = site.unit_weight_robertson_cabal_2010([50.0, 50000.0], [0.1, 900.0], gamma_max=None)
        assert floor_only == pytest.approx([14.0, high])

    def test_site_profile_and_triggering_use_the_unit_weight_limits(self):
        z = np.array([1.0, 2.0, 3.0])
        qt = np.full(z.shape, 50.0)
        fs = np.full(z.shape, 0.1)
        clipped = site.site_profile(z, qt, fs, gwt=5.0)
        opened = site.site_profile(z, qt, fs, gwt=5.0, gamma_min=None, gamma_max=None)
        assert clipped.unit_weight == pytest.approx(np.full(z.shape, 14.0))
        assert opened.unit_weight == pytest.approx(
            site.unit_weight_robertson_cabal_2010(qt, fs, gamma_min=None, gamma_max=None)
        )
        assert np.all(opened.unit_weight < 14.0)
        assert opened.sigma_v[-1] < clipped.sigma_v[-1]
        # A unit weight supplied by the caller is kept, including below the default floor.
        given = site.site_profile(z, [5000.0, 5000.0, 5000.0], [30.0, 30.0, 30.0], gwt=1.0, unit_weight=12.0)
        assert given.unit_weight == pytest.approx([12.0, 12.0, 12.0])

        default = liq.triggering_bi2014(z, qt, fs, gwt=5.0, mw=7.5, amax=0.2)
        low = liq.triggering_bi2014(z, qt, fs, gwt=5.0, mw=7.5, amax=0.2, gamma_min=None)
        assert default.unit_weight == pytest.approx(clipped.unit_weight)
        assert low.unit_weight == pytest.approx(opened.unit_weight)

    def test_triggering_accepts_a_u0_profile(self):
        z = np.array([1.0, 2.0, 3.0, 4.0])
        qt = np.array([2000.0, 4000.0, 6000.0, 8000.0])
        fs = np.array([15.0, 25.0, 30.0, 40.0])
        kw = dict(gwt=1.5, mw=7.5, amax=0.25, unit_weight=18.0)
        base = liq.triggering_bi2014(z, qt, fs, **kw)
        hydrostatic = site.GAMMA_W * np.clip(z - 1.5, 0, None)
        same = liq.triggering_bi2014(z, qt, fs, u0=hydrostatic, **kw)
        assert same.sigma_v_eff == pytest.approx(base.sigma_v_eff)
        assert same.ic == pytest.approx(base.ic)
        assert same.qc1ncs == pytest.approx(base.qc1ncs)
        assert same.fsl == pytest.approx(base.fsl, nan_ok=True)
        assert same.critical_layers == base.critical_layers

        # Perched: dry through 2 m, then pore pressure well above hydrostatic.
        custom = liq.triggering_bi2014(z, qt, fs, u0=[0.0, 0.0, 80.0, 120.0], **kw)
        sigma_v = np.array([18.0, 36.0, 54.0, 72.0])
        assert custom.sigma_v == pytest.approx(sigma_v)
        assert custom.sigma_v_eff == pytest.approx(np.maximum(sigma_v - np.array([0.0, 0.0, 80.0, 120.0]), 1.0))
        assert np.isnan(custom.fsl[0]) and np.isnan(custom.fsl[1])
        assert np.isfinite(base.fsl[1])
        assert not np.allclose(custom.ic, base.ic)
        with pytest.raises(ValueError, match="u0"):
            liq.triggering_bi2014(z, qt, fs, u0=[0.0, 1.0], **kw)

    def test_triggering_u0_follows_the_max_depth_cut(self):
        z = np.arange(0.5, 30.01, 0.5)
        qt = np.full(z.size, 6000.0)
        fs = np.full(z.size, 40.0)
        u0 = 5.0 * z
        res = liq.triggering_bi2014(z, qt, fs, gwt=1.0, mw=7.5, amax=0.2, unit_weight=18.0, u0=u0, max_depth=20.0)
        kept = z <= 20.0
        sigma_v = np.cumsum(18.0 * np.diff(np.r_[0.0, z[kept]]))
        assert res.depth == pytest.approx(z[kept])
        assert res.sigma_v == pytest.approx(sigma_v)
        assert res.sigma_v_eff == pytest.approx(np.maximum(sigma_v - u0[kept], 1.0))


def test_site_profile_matches_the_triggering_result_above_max_depth():
    z = np.arange(0.5, 30.01, 0.5)
    qt = np.where(z < 8, 900.0, 6000.0)
    fs = np.where(z < 8, 35.0, 30.0)
    qt[10] = -1.0
    full = site.site_profile(z, qt, fs, gwt=2.0)
    res = liq.triggering_bi2014(z, qt, fs, gwt=2.0, mw=7.5, amax=0.3, max_depth=20.0)
    n = res.depth.size
    assert full.depth.size == z.size and n < z.size
    assert np.allclose(full.ic[:n], res.ic, equal_nan=True)
    assert np.allclose(full.sigma_v_eff[:n], res.sigma_v_eff)
    assert (full.invalid[:n] == res.invalid).all()
