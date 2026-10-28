"""Robertson & Wride (1998) / Robertson (2009) triggering.

Published checks, each intermediate recomputed from the cited equation:

- Youd et al. (2001) Table 3 magnitude scaling factors, and the written equation
  MSF = 10**2.24 / Mw**2.56.
- Andrus & Stokoe (1997) upper bound as fitted by Youd et al., MSF = (Mw/7.5)**(-3.3).
- Liao & Whitman (1986) / NCEER workshop rd at the published depth breaks.
- Robertson (2015) quotes Kc = 1.066 at Ic = 1.60 and 3.427 at Ic = 2.60. The
  polynomial printed by Youd et al. (2001) and Robertson & Wride gives 0.966 and
  3.327 there: the same polynomial with the constant term 0.1 higher. The constant
  kept here is -17.88. At Ic = 2 the polynomial is 1.30 exactly.
- Amiri, TRB webinar "CPT Technology" (Caltrans), NCEER CPT example at 15.26 ft:
  rd = 0.96, MSF = 1.19 at Mw 7, CRR7.5 = 0.14 at (qc1N)cs = 84.16. The slide's
  Kc = 2.03 at Ic = 2.30 does not match the polynomial (1.95). Their factor of
  safety 0.42 does not match (0.14/0.38)*1.19 = 0.44. Those two are recorded, not
  forced.
"""

import math

import numpy as np
import pytest

from opencpt import liquefaction as liq
from opencpt import site


def _kc_polynomial(ic: float) -> float:
    return -0.403 * ic**4 + 5.581 * ic**3 - 21.63 * ic**2 + 33.75 * ic - 17.88


def _atmospheric_profile(z: float, gwt: float, qtn: float, fr: float):
    """One reading whose effective stress is pa, so Qtn = qnet/pa for every n."""
    u0 = site.GAMMA_W * (z - gwt)
    sigma_v = site.PA + u0
    gamma = sigma_v / z
    qnet = qtn * site.PA
    return gamma, sigma_v, u0, qnet, sigma_v + qnet, fr / 100.0 * qnet


class TestKc:
    def test_flat_cap_and_polynomial_checksum(self):
        # Ic = 2 is an exact root of the published polynomial: Kc = 1.30.
        assert _kc_polynomial(2.0) == pytest.approx(1.30)
        assert liq.correction_factor_kc([1.0, 1.64, 2.0], [1.0, 1.0, 1.0]) == pytest.approx(
            [1.0, 1.0, 1.30]
        )
        # Just above the cap the polynomial is used, and it is continuous to about 0.004.
        assert _kc_polynomial(1.64) == pytest.approx(0.9961, abs=1e-4)
        assert liq.correction_factor_kc([1.65], [1.0])[0] == pytest.approx(_kc_polynomial(1.65))

    def test_published_ic_2_6_uses_the_youd_constant_not_the_robertson_2015_offset(self):
        # Robertson (2015) quotes Kc = 3.427 at Ic = 2.60 while comparing with Vs.
        # That is this polynomial with -17.78 instead of -17.88. Youd et al. (2001)
        # print -17.88, which is 3.327 at Ic = 2.60. Ic = 2.60 is still on the
        # sand curve (the cutoff is Ic > 2.6).
        assert _kc_polynomial(2.60) == pytest.approx(3.3267, abs=1e-4)
        assert liq.correction_factor_kc([2.60], [1.0])[0] == pytest.approx(3.3267, abs=1e-4)
        assert liq.correction_factor_kc([2.60], [1.0])[0] != pytest.approx(3.427, abs=0.01)

    def test_loose_sand_screen_holds_kc_at_one(self):
        # 1.64 < Ic < 2.36 is not what is passed in: the screen looks at Ic and Fr.
        ic = 2.0
        polynomial = _kc_polynomial(ic)
        assert polynomial > 1.2
        assert liq.correction_factor_kc([ic], [0.49])[0] == pytest.approx(1.0)
        assert liq.correction_factor_kc([ic], [0.50])[0] == pytest.approx(polynomial)
        assert liq.correction_factor_kc([ic], [0.49], loose_sand_screen=False)[0] == pytest.approx(
            polynomial
        )
        # At the Ic = 2.36 edge the screen stops and the polynomial is used.
        assert liq.correction_factor_kc([2.36], [0.1])[0] == pytest.approx(_kc_polynomial(2.36))
        assert liq.correction_factor_kc([1.64], [0.1])[0] == pytest.approx(1.0)


class TestCrrAndRdAndMsf:
    @pytest.mark.parametrize(
        "q, expected",
        [
            (0.0, 0.05),
            (20.0, 0.833 * 0.02 + 0.05),
            (50.0, 93.0 * (0.05) ** 3 + 0.08),
            (100.0, 0.173),
            (159.0, 93.0 * (0.159) ** 3 + 0.08),
        ],
    )
    def test_crr_branches_match_robertson_wride(self, q, expected):
        assert liq.crr75_robertson_wride([q])[0] == pytest.approx(expected, abs=1e-12)

    def test_crr_is_continuous_at_50_and_undefined_from_160(self):
        low = 0.833 * (50 / 1000) + 0.05
        high = 93 * (50 / 1000) ** 3 + 0.08
        assert low == pytest.approx(high, abs=1e-4)
        assert liq.crr75_robertson_wride([50.0])[0] == pytest.approx(high)
        assert math.isnan(liq.crr75_robertson_wride([160.0])[0])
        assert math.isnan(liq.crr75_robertson_wride([210.0])[0])

    def test_rd_follows_liao_whitman_then_the_nceer_extension(self):
        z = [0.0, 9.15, 10.0, 23.0, 30.0, 30.1, 40.0]
        rd = liq.stress_reduction_coefficient_nceer(z)
        assert rd[0] == pytest.approx(1.0)
        assert rd[1] == pytest.approx(1.0 - 0.00765 * 9.15)
        assert rd[2] == pytest.approx(1.174 - 0.0267 * 10.0)
        assert rd[3] == pytest.approx(1.174 - 0.0267 * 23.0)
        assert rd[4] == pytest.approx(0.744 - 0.008 * 30.0)
        assert rd[5] == pytest.approx(0.50)
        assert rd[6] == pytest.approx(0.50)
        # Not the Idriss rd used by Boulanger & Idriss.
        idriss = liq.stress_reduction_coefficient([10.0], 7.5)[0]
        assert rd[2] != pytest.approx(idriss, abs=0.01)
        with pytest.raises(ValueError, match="ground surface"):
            liq.stress_reduction_coefficient_nceer([-0.1])

    @pytest.mark.parametrize(
        "mw, table",
        [
            (5.5, 2.20),
            (6.0, 1.76),
            (6.5, 1.44),
            (7.0, 1.19),
            (7.5, 1.00),
            (8.0, 0.84),
            (8.5, 0.72),
        ],
    )
    def test_idriss_msf_matches_the_youd_equation_and_table(self, mw, table):
        # Table 3 is rounded to two decimals and is off by up to 0.02 (5.5 is 2.21,
        # 6.0 is 1.77, 8.0 is 0.85, 8.5 is 0.73). The equation is the source.
        got = liq.magnitude_scaling_factor_youd2001(mw)
        assert got == pytest.approx(10**2.24 / mw**2.56)
        assert got == pytest.approx(table, abs=0.02)

    def test_msf_is_one_at_mw_7_5_within_the_10_2_24_rounding(self):
        assert liq.magnitude_scaling_factor_youd2001(7.5) == pytest.approx(0.9996, abs=1e-4)
        assert liq.magnitude_scaling_factor_youd2001(7.5) == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize(
        "mw, table",
        [(5.5, 2.8), (6.0, 2.1), (6.5, 1.6), (7.0, 1.25), (7.5, 1.0), (8.0, 0.8), (8.5, 0.65)],
    )
    def test_andrus_stokoe_upper_bound_matches_the_youd_table(self, mw, table):
        got = liq.magnitude_scaling_factor_youd2001(mw, "andrus_stokoe")
        assert got == pytest.approx((mw / 7.5) ** -3.3)
        assert got == pytest.approx(table, abs=0.02)

    def test_msf_rejects_a_bad_bound_and_a_non_positive_magnitude(self):
        with pytest.raises(ValueError, match="bound"):
            liq.magnitude_scaling_factor_youd2001(7.5, "seed")
        with pytest.raises(ValueError, match="positive"):
            liq.magnitude_scaling_factor_youd2001(0.0)

    def test_k_sigma_is_one_at_atmospheric_pressure_and_falls_above_it(self):
        assert liq.overburden_correction_youd2001([site.PA], 0.7)[0] == pytest.approx(1.0)
        # f = 0.7, σ'v = 2 pa: Kσ = 2^(-0.3).
        assert liq.overburden_correction_youd2001([2 * site.PA], 0.7)[0] == pytest.approx(2.0**-0.3)
        # Below pa the Hynes & Olsen factor is greater than 1.
        assert liq.overburden_correction_youd2001([0.5 * site.PA], 0.8)[0] > 1.0


class TestCaltransNceerExample:
    def test_rd_msf_and_crr_match_the_slide_the_stated_kc_does_not(self):
        # Depth 15.26 ft = 4.65 m. rd = 1 - 0.00765*4.65 = 0.964, reported as 0.96.
        assert liq.stress_reduction_coefficient_nceer([4.65])[0] == pytest.approx(0.9644, abs=1e-4)
        assert liq.stress_reduction_coefficient_nceer([4.65])[0] == pytest.approx(0.96, abs=0.005)
        # MSF = 10**2.24 / 7**2.56 = 1.193, reported as 1.19.
        assert liq.magnitude_scaling_factor_youd2001(7.0) == pytest.approx(1.1927, abs=1e-4)
        assert liq.magnitude_scaling_factor_youd2001(7.0) == pytest.approx(1.19, abs=0.005)
        # (qc1N)cs = 84.16 is on the upper branch: CRR7.5 = 0.135, reported as 0.14.
        crr = liq.crr75_robertson_wride([84.16])[0]
        assert crr == pytest.approx(93 * (84.16 / 1000) ** 3 + 0.08)
        assert crr == pytest.approx(0.14, abs=0.005)
        # The slide says Ic = 2.30 gives Kc = 2.03, then 2.03*41.42 = 84.16.
        # The polynomial at Ic = 2.30 is 1.949. Kc = 2.032 is the polynomial at Ic ≈ 2.325.
        assert liq.correction_factor_kc([2.30], [1.5])[0] == pytest.approx(1.9487, abs=1e-4)
        assert 84.16 / 41.42 == pytest.approx(2.032, abs=1e-3)
        # FS = (0.14/0.38)*1.19 = 0.438. The slide prints 0.42.
        assert (0.14 / 0.38) * 1.19 == pytest.approx(0.4384, abs=1e-4)


class TestHandCalculation:
    def test_silty_sand_chain_at_atmospheric_effective_stress(self):
        # σ'v = pa, so Qtn does not depend on n. Fr = 1% puts Ic in the polynomial
        # band (above 1.64, Fr not below 0.5%). Qtn = 80 is on the upper CRR branch.
        z, gwt, qtn, fr = 8.0, 2.0, 80.0, 1.0
        gamma, sigma_v, _u0, qnet, qt, fs = _atmospheric_profile(z, gwt, qtn, fr)
        ic = math.sqrt((3.47 - math.log10(qtn)) ** 2 + (math.log10(fr) + 1.22) ** 2)
        assert 1.64 < ic < 2.36
        kc = _kc_polynomial(ic)
        qtn_cs = kc * qtn
        assert 50 <= qtn_cs < 160
        crr = 93 * (qtn_cs / 1000) ** 3 + 0.08
        rd = 1.0 - 0.00765 * z
        mw, amax = 7.0, 0.25
        msf = 10**2.24 / mw**2.56
        csr = 0.65 * amax * (sigma_v / site.PA) * rd
        fsl = crr / csr * msf

        res = liq.triggering_robertson2009(
            [z], [qt], [fs], gwt=gwt, mw=mw, amax=amax, unit_weight=gamma
        )
        assert res.sigma_v_eff == pytest.approx([site.PA], abs=1e-6)
        assert res.qtn == pytest.approx([qtn], rel=1e-6)
        assert res.fr == pytest.approx([fr])
        assert res.ic == pytest.approx([ic], abs=1e-6)
        assert res.n[0] < 1.0
        assert res.kc == pytest.approx([kc], abs=1e-6)
        assert res.qtn_cs == pytest.approx([qtn_cs], rel=1e-6)
        assert res.crr75 == pytest.approx([crr], rel=1e-6)
        assert res.msf == pytest.approx([msf])
        assert res.k_sigma == pytest.approx([1.0])
        assert res.rd == pytest.approx([rd])
        assert res.csr == pytest.approx([csr], rel=1e-6)
        assert res.fsl == pytest.approx([fsl], rel=1e-6)
        assert not res.not_susceptible[0]
        assert not res.not_liquefiable[0]
        assert not res.above_water_table[0]
        assert not res.loose_sand_zone[0]
        assert not res.invalid[0]

    def test_loose_clean_sand_uses_the_lower_crr_branch_and_kc_of_one(self):
        # Qtn = 20, Fr = 0.3%: Ic is between 1.64 and 2.36, so the screen sets Kc = 1
        # and Qtn,cs = 20 stays on the lower CRR branch.
        z, gwt, qtn, fr = 8.0, 2.0, 20.0, 0.3
        gamma, sigma_v, _u0, _qnet, qt, fs = _atmospheric_profile(z, gwt, qtn, fr)
        ic = math.sqrt((3.47 - math.log10(qtn)) ** 2 + (math.log10(fr) + 1.22) ** 2)
        assert 1.64 < ic < 2.36
        crr = 0.833 * (qtn / 1000) + 0.05
        rd = 1.0 - 0.00765 * z
        msf = 10**2.24 / 7.5**2.56
        csr = 0.65 * 0.30 * (sigma_v / site.PA) * rd

        res = liq.triggering_robertson2009(
            [z], [qt], [fs], gwt=gwt, mw=7.5, amax=0.30, unit_weight=gamma
        )
        screened = liq.triggering_robertson2009(
            [z],
            [qt],
            [fs],
            gwt=gwt,
            mw=7.5,
            amax=0.30,
            unit_weight=gamma,
            loose_sand_screen=False,
        )
        assert res.loose_sand_zone[0] and screened.loose_sand_zone[0]
        assert res.kc == pytest.approx([1.0])
        assert res.qtn_cs == pytest.approx([qtn])
        assert res.crr75 == pytest.approx([crr])
        assert res.fsl == pytest.approx([crr / csr * msf], rel=1e-6)
        assert screened.kc[0] == pytest.approx(_kc_polynomial(ic), abs=1e-4)
        assert screened.qtn_cs[0] > res.qtn_cs[0] + 1.0
        assert screened.fsl[0] > res.fsl[0]

    def test_clean_sand_below_the_kc_cap(self):
        # High Qtn and a low but not screened friction ratio: Ic <= 1.64, Kc = 1.
        z, gwt, qtn, fr = 8.0, 2.0, 120.0, 0.35
        gamma, _sv, _u0, _qnet, qt, fs = _atmospheric_profile(z, gwt, qtn, fr)
        ic = math.sqrt((3.47 - math.log10(qtn)) ** 2 + (math.log10(fr) + 1.22) ** 2)
        assert ic <= 1.64
        crr = 93 * (qtn / 1000) ** 3 + 0.08
        res = liq.triggering_robertson2009(
            [z], [qt], [fs], gwt=gwt, mw=7.5, amax=0.2, unit_weight=gamma
        )
        assert res.ic[0] == pytest.approx(ic, abs=1e-4)
        assert res.kc == pytest.approx([1.0])
        assert not res.loose_sand_zone[0]
        assert res.crr75 == pytest.approx([crr], rel=1e-6)
        assert np.isfinite(res.fsl[0])


class TestFlags:
    def _run(self, z, qt, fs, **kw):
        kw.setdefault("gwt", 1.0)
        kw.setdefault("mw", 7.5)
        kw.setdefault("amax", 0.3)
        kw.setdefault("unit_weight", 18.0)
        return liq.triggering_robertson2009(z, qt, fs, **kw)

    def test_above_the_water_table_has_no_factor_of_safety(self):
        res = self._run([0.5, 2.0], [8000.0, 8000.0], [40.0, 40.0], gwt=1.0)
        assert res.above_water_table.tolist() == [True, False]
        assert math.isnan(res.fsl[0])
        assert np.isfinite(res.kc[0]) and np.isfinite(res.csr[0])

    def test_ic_above_2_6_is_not_assessed_and_the_cutoff_is_strict(self):
        # Soft clay. Ic is well above 2.6, so there is no factor of safety.
        res = self._run([3.0], [700.0], [40.0])
        assert res.ic[0] > 2.6
        assert res.not_susceptible[0]
        assert math.isnan(res.fsl[0])
        # The same reading assessed against a cutoff equal to its own Ic is kept,
        # because the test is Ic > cutoff. A slightly lower cutoff drops it.
        ic = float(res.ic[0])
        at = self._run([3.0], [700.0], [40.0], ic_cutoff=ic)
        below = self._run([3.0], [700.0], [40.0], ic_cutoff=ic - 1e-6)
        assert not at.not_susceptible[0] and np.isfinite(at.fsl[0])
        assert below.not_susceptible[0] and math.isnan(below.fsl[0])

    def test_qtn_cs_at_or_above_160_is_not_liquefiable(self):
        res = self._run([3.0], [50000.0], [80.0], amax=0.5)
        assert res.qtn_cs[0] >= 160
        assert res.not_liquefiable[0]
        assert math.isinf(res.fsl[0])
        assert math.isnan(res.crr75[0])
        assert res.critical_layers == []
        # A limit just above the computed Qtn,cs puts the point back on the curve.
        opened = self._run([3.0], [50000.0], [80.0], amax=0.5, qtn_cs_limit=res.qtn_cs[0] + 1)
        assert not opened.not_liquefiable[0]
        assert np.isfinite(opened.fsl[0])

    def test_invalid_reading_is_flagged_and_does_not_spoil_the_one_below(self):
        z = np.array([1.5, 2.0, 2.5])
        qt = np.array([6000.0, -5.0, 6000.0])
        fs = np.array([30.0, 30.0, 30.0])
        res = self._run(z, qt, fs)
        clean = self._run(z, [6000.0, 6000.0, 6000.0], fs)
        assert res.invalid.tolist() == [False, True, False]
        assert math.isnan(res.fsl[1]) and math.isnan(res.kc[1])
        assert res.fsl[2] == pytest.approx(clean.fsl[2])

    def test_u0_profile_moves_the_dry_mask_and_the_stresses(self):
        z = np.array([1.0, 2.0, 3.0])
        qt = np.array([4000.0, 5000.0, 6000.0])
        fs = np.array([25.0, 30.0, 35.0])
        base = self._run(z, qt, fs, gwt=1.5)
        same = self._run(z, qt, fs, gwt=1.5, u0=site.GAMMA_W * np.clip(z - 1.5, 0, None))
        assert same.fsl == pytest.approx(base.fsl, nan_ok=True)
        assert same.qtn_cs == pytest.approx(base.qtn_cs, rel=1e-6)
        custom = self._run(z, qt, fs, gwt=1.5, u0=[0.0, 0.0, 40.0])
        assert custom.above_water_table.tolist() == [True, True, False]
        assert math.isnan(custom.fsl[0]) and math.isnan(custom.fsl[1])
        assert custom.sigma_v_eff == pytest.approx([18.0, 36.0, max(54.0 - 40.0, 1.0)])
        assert not np.allclose(custom.ic, base.ic)

    def test_k_sigma_scales_the_factor_of_safety_when_requested(self):
        # 15 m, γ = 18 kN/m³, water table at 1 m: σ'v ≈ 133 kPa, above pa, so Kσ < 1.
        z = [15.0]
        qt, fs = [9000.0], [50.0]
        base = self._run(z, qt, fs, gwt=1.0, k_sigma_f=None)
        corrected = self._run(z, qt, fs, gwt=1.0, k_sigma_f=0.7)
        expected = liq.overburden_correction_youd2001(base.sigma_v_eff, 0.7)
        assert base.k_sigma == pytest.approx([1.0])
        assert corrected.k_sigma == pytest.approx(expected)
        assert expected[0] < 1.0
        assert corrected.fsl == pytest.approx(base.fsl * expected, rel=1e-6)

    def test_andrus_stokoe_msf_raises_the_factor_of_safety_below_mw_7_5(self):
        z, qt, fs = [4.0], [7000.0], [40.0]
        idriss = self._run(z, qt, fs, mw=6.0, msf="idriss")
        andrus = self._run(z, qt, fs, mw=6.0, msf="andrus_stokoe")
        assert andrus.msf[0] > idriss.msf[0]
        assert andrus.fsl[0] == pytest.approx(idriss.fsl[0] * andrus.msf[0] / idriss.msf[0])
        # rd below 9.15 m does not depend on magnitude. The two runs share it.
        assert andrus.rd == pytest.approx(idriss.rd)

    def test_deep_reading_uses_the_extended_rd(self):
        res = self._run([25.0], [12000.0], [60.0], gwt=2.0)
        assert res.rd[0] == pytest.approx(0.744 - 0.008 * 25.0)
        deeper = self._run([32.0], [14000.0], [70.0], gwt=2.0)
        assert deeper.rd[0] == pytest.approx(0.50)

    def test_critical_layer_skips_clay_and_keeps_a_loose_sand_run(self):
        z = np.round(np.arange(0.5, 6.01, 0.5), 2)
        # Clay crust over a loose sand. The sand qt is low enough that FS drops below 1.
        qt = np.where(z < 2.0, 600.0, 2500.0)
        fs = np.where(z < 2.0, 25.0, 8.0)
        res = self._run(z, qt, fs, gwt=1.0, amax=0.45, mw=7.5)
        # The saturated clay (1.5 m) is above Ic 2.6. The 0.5 m reading is dry and
        # its Ic is just under 2.6, so the whole crust has no factor of safety
        # without every reading being flagged not susceptible.
        assert res.not_susceptible[(z >= 1.5) & (z < 2.0)].all()
        assert np.isnan(res.fsl[z < 2.0]).all()
        sand = z >= 2.0
        assert np.isfinite(res.fsl[sand]).all()
        assert (res.fsl[sand] < 1.0).all()
        assert res.critical_layers == [(2.0, 6.0)]
        mean, _std = res.critical_qtn_cs
        assert mean == pytest.approx(res.qtn_cs[sand].mean())
