"""CPT-based liquefaction triggering.

Boulanger, R.W., Idriss, I.M. (2014). CPT and SPT based liquefaction triggering
procedures. Report UCD/CGM-14/01, University of California, Davis.

Robertson, P.K., Wride, C.E. (1998). Evaluating cyclic liquefaction potential using
the cone penetration test. Canadian Geotechnical Journal 35(3): 442-459, as updated
by Robertson (2009) to normalised cone resistance Qtn and adopted for routine use by
Youd et al. (2001).

Inputs and outputs are NumPy arrays in SI units (m, kPa) unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .site import (
    PA,
    corrected_cone_resistance,
    site_profile,
)

C0_DETERMINISTIC = 2.80
"""CRR curve constant for the deterministic B&I (2014) relationship (PL ≈ 16%)."""

RD_MAX_DEPTH = 34.0
"""Depth (m) to which the rd equations apply (Idriss & Boulanger 2008, after Eq. 24)."""

RD_RECOMMENDED_DEPTH = 20.0
"""Below about this depth (m), Idriss & Boulanger (2008) recommend CSR from site response
analysis."""


def fines_content_bi2014(ic, cfc: float = 0.0):
    """Fines content (%) from Ic: FC = 80 (Ic + C_FC) - 137, limited to 0-100."""
    return np.clip(80 * (np.asarray(ic, dtype=float) + cfc) - 137, 0, 100)


def qc1ncs_bi2014(qt, sigma_v_eff, fc, max_iter: int = 50):
    """Clean-sand equivalent normalized cone resistance qc1Ncs, solved iteratively.

    qc1N = CN qt/pa with CN = (pa/σ'v)^m ≤ 1.7 and m = 1.338 - 0.249 qc1Ncs^0.264
    (qc1Ncs limited to 21-254), plus the fines adjustment Δqc1N.

    Boulanger & Idriss (2014) write qc1N = CN qc/pa, using the measured cone tip
    resistance qc. This function uses the corrected resistance qt instead
    (qc1N = CN qt/pa). In sand, u2 is near the equilibrium pore pressure, so qt and
    qc are close; qt is what `triggering_bi2014` has, and it is kept so existing
    results do not change. Pass qc as `qt` to evaluate the published form.

    Returns:
        (qc1ncs, cn)
    """
    qt = np.asarray(qt, dtype=float)
    sigma_v_eff = np.maximum(np.asarray(sigma_v_eff, dtype=float), 1.0)
    fc = np.asarray(fc, dtype=float)
    qc1ncs = np.full_like(qt, 100.0)
    cn = np.ones_like(qt)
    for _ in range(max_iter):
        m = np.clip(1.338 - 0.249 * np.clip(qc1ncs, 21, 254) ** 0.264, 0.264, 0.782)
        cn = np.minimum((PA / sigma_v_eff) ** m, 1.7)
        qc1n = cn * qt / PA  # qt, not the measured qc; see the docstring.
        dq = (11.9 + qc1n / 14.6) * np.exp(1.63 - 9.7 / (fc + 2) - (15.7 / (fc + 2)) ** 2)
        new = qc1n + dq
        if np.allclose(new, qc1ncs, atol=1e-3, equal_nan=True):
            qc1ncs = new
            break
        qc1ncs = new
    return qc1ncs, cn


def stress_reduction_coefficient(depth, mw: float):
    """Shear stress reduction coefficient rd (Idriss 1999), as used by B&I (2014).

    The equations apply to depths of at most 34 m; deeper they rise again and are not valid,
    so deeper depths raise ValueError (Idriss & Boulanger 2008).
    """
    z = np.asarray(depth, dtype=float)
    if np.any(z > RD_MAX_DEPTH):
        raise ValueError(
            f"rd is defined only to {RD_MAX_DEPTH:g} m depth; got {float(np.nanmax(z)):g} m."
        )
    alpha = -1.012 - 1.126 * np.sin(z / 11.73 + 5.133)
    beta = 0.106 + 0.118 * np.sin(z / 11.28 + 5.142)
    return np.exp(alpha + beta * mw)


def cyclic_stress_ratio(amax: float, sigma_v, sigma_v_eff, rd):
    """CSR = 0.65 (amax/g)(σv/σ'v) rd, with amax in g."""
    return 0.65 * amax * np.asarray(sigma_v) / np.maximum(np.asarray(sigma_v_eff), 1.0) * rd


def magnitude_scaling_factor_bi2014(mw: float, qc1ncs):
    """MSF = 1 + (MSFmax - 1)(8.64 e^(-M/4) - 1.325), MSFmax = 1.09 + (qc1Ncs/180)³ ≤ 2.2.

    The only limit is MSFmax ≤ 2.2 (reached near qc1Ncs = 187); qc1Ncs itself is not capped.
    """
    q = np.asarray(qc1ncs, dtype=float)
    msf_max = np.minimum(1.09 + (q / 180) ** 3, 2.2)
    return 1 + (msf_max - 1) * (8.64 * np.exp(-mw / 4) - 1.325)


def overburden_correction_bi2014(sigma_v_eff, qc1ncs):
    """Kσ = 1 - Cσ ln(σ'v/pa) ≤ 1.1, with Cσ = 1/(37.3 - 8.27 qc1Ncs^0.264) ≤ 0.3."""
    q = np.minimum(np.asarray(qc1ncs, dtype=float), 211)
    c_sigma = np.minimum(1 / (37.3 - 8.27 * q**0.264), 0.3)
    return np.minimum(1 - c_sigma * np.log(np.maximum(np.asarray(sigma_v_eff), 1.0) / PA), 1.1)


def crr_m75_bi2014(qc1ncs, c0: float = C0_DETERMINISTIC):
    """Cyclic resistance ratio for Mw 7.5 and σ'v = 1 atm.

    CRR = exp(q/113 + (q/1000)² - (q/140)³ + (q/137)⁴ - C0), with q = qc1Ncs ≤ 211.
    """
    q = np.minimum(np.asarray(qc1ncs, dtype=float), 211)
    return np.exp(q / 113 + (q / 1000) ** 2 - (q / 140) ** 3 + (q / 137) ** 4 - c0)


def correction_factor_kc(ic, fr=None, *, loose_sand_screen: bool = True):
    """Grain-characteristic correction Kc so that Qtn,cs = Kc Qtn.

    Kc = 1 for Ic <= 1.64. Above that,
    Kc = -0.403 Ic^4 + 5.581 Ic^3 - 21.63 Ic^2 + 33.75 Ic - 17.88
    (Robertson & Wride 1998; Youd et al. 2001).

    Where 1.64 < Ic < 2.36 and Fr < 0.5% the high Ic is treated as a very loose
    clean sand rather than fines, and Kc is set back to 1 (Robertson & Wride 1998;
    Robertson 2004, 2009). `fr` is that friction ratio in percent. Pass
    `loose_sand_screen=False` to keep the polynomial there. Fr is ignored when the
    screen is off.

    The polynomial at Ic = 1.64 is 0.996, so the flat cap of 1.0 is a small step.
    """
    ic = np.asarray(ic, dtype=float)
    kc = np.where(
        ic <= 1.64,
        1.0,
        -0.403 * ic**4 + 5.581 * ic**3 - 21.63 * ic**2 + 33.75 * ic - 17.88,
    )
    if loose_sand_screen and fr is not None:
        fr = np.asarray(fr, dtype=float)
        loose = (ic > 1.64) & (ic < 2.36) & (fr < 0.5)
        kc = np.where(loose, 1.0, kc)
    return np.where(np.isfinite(ic), kc, np.nan)


def clean_sand_qtn(qtn, kc):
    """Clean-sand equivalent normalised resistance Qtn,cs = Kc Qtn."""
    return np.asarray(kc, dtype=float) * np.asarray(qtn, dtype=float)


def crr75_robertson_wride(qtn_cs, upper: float = 160.0):
    """CRR for Mw 7.5 and σ'v = 1 atm, Robertson & Wride (1998).

    Qtn,cs < 50: 0.833 (Qtn,cs/1000) + 0.05.
    50 <= Qtn,cs < upper: 93 (Qtn,cs/1000)^3 + 0.08.
    Qtn,cs >= upper is outside the curve and returns NaN. The published curve
    ends at 160: denser than that is not liquefiable.
    """
    q = np.asarray(qtn_cs, dtype=float)
    crr = np.full(np.shape(q), np.nan, dtype=float)
    low = q < 50.0
    mid = (q >= 50.0) & (q < upper)
    crr = np.where(low, 0.833 * (q / 1000.0) + 0.05, crr)
    return np.where(mid, 93.0 * (q / 1000.0) ** 3 + 0.08, crr)


def stress_reduction_coefficient_nceer(depth):
    """Mean stress-reduction coefficient rd used with the Robertson & Wride CPT method.

    Liao & Whitman (1986), as recommended for routine use by the NCEER workshop
    (Youd & Idriss 1997; Youd et al. 2001), with the deeper extension from that
    workshop summary:

    rd = 1.0 - 0.00765 z for z <= 9.15 m
    rd = 1.174 - 0.0267 z for 9.15 m < z <= 23 m
    rd = 0.744 - 0.008 z for 23 m < z <= 30 m
    rd = 0.50 for z > 30 m

    z is depth in metres. The first two branches are Liao & Whitman. Robertson & Wride
    (1998) added the third; the value below 30 m was added in the NCEER workshop
    summary. Youd et al. (2001) endorse the Liao & Whitman branches for routine work
    and treat liquefaction calculations below about 23 m as outside the range the
    simplified procedure has been checked against. This is not the magnitude-dependent
    Idriss rd used by `stress_reduction_coefficient`.

    Depths above the ground surface raise ValueError.
    """
    z = np.asarray(depth, dtype=float)
    if np.any(z < 0):
        raise ValueError("rd is defined for depths at or below the ground surface.")
    return np.select(
        [z <= 9.15, z <= 23.0, z <= 30.0],
        [1.0 - 0.00765 * z, 1.174 - 0.0267 * z, 0.744 - 0.008 * z],
        default=0.50,
    )


def magnitude_scaling_factor_youd2001(mw: float, bound: str = "idriss"):
    """Magnitude scaling factor from Youd et al. (2001).

    The workshop recommended a range for Mw < 7.5 and a single factor above 7.5.
    Both ends are implemented. The ConeTec manual's MSF equations 9.19 and 9.20 are
    images in the source PDF and were not readable; these are the two equations in
    Youd et al. (2001) that a manual numbers as that pair. The default is the lower
    bound, which is also the factor Youd et al. require for Mw > 7.5 and the factor
    Robertson uses with this CPT method (Zhang, Robertson & Brachman 2004, written
    there as 174/Mw^2.56).

    `bound="idriss"` (default), revised Idriss factor, Youd et al. (2001):
        MSF = 10^2.24 / Mw^2.56
    At Mw = 7.5 this is 0.9996 rather than 1, because 7.5^2.56 is not exactly 10^2.24.
    (Mw/7.5)^(-2.56) is the same expression forced through 1 at Mw 7.5; the difference
    is about 0.04%. Robertson & Cabal guides sometimes round the numerator to 174
    (about 0.1% higher). A worked NCEER example (Amiri, TRB webinar) uses
    10^2.24/Mw^2.56 and gets 1.19 at Mw 7, which this form matches.

    `bound="andrus_stokoe"`, upper bound for Mw < 7.5, Andrus & Stokoe (1997) as
    fitted by Youd et al. (2001):
        MSF = (Mw/7.5)^(-3.3)
    Youd et al. do not recommend this upper bound for Mw > 7.5; the equation is still
    evaluated so the choice stays explicit.

    The later Idriss factor MSF = 6.9 exp(-Mw/4) - 0.058 is not used. Youd et al.
    (2001) set it aside because it belongs with a magnitude-dependent rd, not with
    Liao & Whitman rd.
    """
    if mw <= 0:
        raise ValueError("mw must be positive.")
    if bound == "idriss":
        return float(10.0**2.24 / mw**2.56)
    if bound == "andrus_stokoe":
        return float((mw / 7.5) ** -3.3)
    raise ValueError("bound must be 'idriss' or 'andrus_stokoe'.")


def overburden_correction_youd2001(sigma_v_eff, f: float):
    """Overburden factor Kσ = (σ'v/pa)^(f-1), Hynes & Olsen (1999).

    Recommended by Youd et al. (2001). f is about 0.7-0.8 at relative density 40-60%
    and 0.6-0.7 at 60-80%; the lower f is the less severe reduction at high stress.
    σ'v below pa gives Kσ above 1. Robertson (2009) does not apply this factor: the
    stress exponent inside Qtn already follows effective stress, and his case-history
    check used Kσ = 1. `triggering_robertson2009` follows that unless `k_sigma_f` is set.
    """
    sig = np.maximum(np.asarray(sigma_v_eff, dtype=float), 1.0)
    return (sig / PA) ** (f - 1.0)


def critical_layers(depth, fsl, min_thickness: float = 0.4, max_gap: float = 0.1):
    """Depth ranges where FSL ≤ 1.

    Runs of readings with FSL ≤ 1 are merged across gaps of at most `max_gap` (m), for
    example one interbedded clayey reading, and kept when they are at least
    `min_thickness` (m) thick, measured from the first to the last reading.

    Returns:
        List of (top, bottom) depths.
    """
    z = np.asarray(depth, dtype=float)
    liq = np.nan_to_num(np.asarray(fsl, dtype=float), nan=np.inf) <= 1.0
    runs: list[list[int]] = []
    start = None
    for i, flag in enumerate(np.r_[liq, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if runs and z[start] - z[runs[-1][1] - 1] <= max_gap + 1e-9:
                runs[-1][1] = i
            else:
                runs.append([start, i])
            start = None
    return [
        (float(z[s]), float(z[e - 1])) for s, e in runs if z[e - 1] - z[s] >= min_thickness - 1e-9
    ]


@dataclass
class TriggeringResult:
    """Boulanger & Idriss (2014) deterministic triggering for one earthquake scenario."""

    depth: np.ndarray
    qt: np.ndarray
    unit_weight: np.ndarray
    sigma_v: np.ndarray
    sigma_v_eff: np.ndarray
    ic: np.ndarray
    fc: np.ndarray
    qc1ncs: np.ndarray
    csr: np.ndarray
    crr: np.ndarray
    fsl: np.ndarray
    """Factor of safety; NaN where liquefaction is not assessed (above the water table, Ic ≥ cutoff,
    or an invalid reading)."""
    invalid: np.ndarray
    """True for readings that can't be interpreted: qt or fs missing, qt ≤ 0 or fs < 0."""
    critical_layers: list[tuple[float, float]]

    @property
    def critical_thickness(self) -> float:
        """Total thickness of the critical layers (m)."""
        return float(sum(bottom - top for top, bottom in self.critical_layers))

    @property
    def critical_qc1ncs(self) -> tuple[float, float]:
        """Mean and standard deviation of qc1Ncs over the readings in critical layers."""
        inside = np.zeros_like(self.depth, dtype=bool)
        for top, bottom in self.critical_layers:
            inside |= (self.depth >= top) & (self.depth <= bottom)
        q = self.qc1ncs[inside & np.isfinite(self.qc1ncs)]
        return (float(q.mean()), float(q.std())) if q.size else (float("nan"), float("nan"))


def triggering_bi2014(
    depth,
    qt,
    fs,
    gwt: float,
    mw: float,
    amax: float,
    *,
    unit_weight=None,
    u0=None,
    gamma_min: float | None = 14.0,
    gamma_max: float | None = 22.0,
    cfc: float = 0.0,
    ic_cutoff: float = 2.6,
    max_depth: float = 20.0,
    min_thickness: float = 0.4,
    c0: float = C0_DETERMINISTIC,
) -> TriggeringResult:
    """Deterministic liquefaction triggering by Boulanger & Idriss (2014).

    Normalized parameters (Q, Qtn, n, Fr, Bq, U, qnet, qE) are not repeated here;
    `opencpt.site.site_profile` returns them for the same inputs.

    Args:
        depth: Depth below ground surface (m), increasing.
        qt: Corrected cone resistance (kPa); see `opencpt.site.corrected_cone_resistance`.
        fs: Sleeve friction (kPa).
        gwt: Groundwater depth (m). Readings at or above it are not assessed, unless `u0`
            is supplied, in which case readings with u0 ≤ 0 are not assessed.
        mw: Earthquake moment magnitude.
        amax: Peak ground acceleration (g).
        unit_weight: One constant unit weight (kN/m³), or None for Robertson & Cabal (2010).
        u0: Pore pressure (kPa), one value per depth (aligned with `depth`, before the
            `max_depth` cut) or a scalar. None keeps the hydrostatic profile from `gwt`.
        gamma_min: Lower limit (kN/m³) on the Robertson & Cabal unit weight. Default 14.
            None leaves it open. Ignored when `unit_weight` is given.
        gamma_max: Upper limit (kN/m³). Default 22. None leaves it open.
        cfc: Fitting parameter C_FC of the fines content correlation.
        ic_cutoff: Readings with Ic at or above this are treated as not liquefiable.
        max_depth: Readings deeper than this are left out (m); at most 34 m, the limit of rd.
        min_thickness: Minimum critical layer thickness (m).
        c0: CRR curve constant; 2.80 for the deterministic relationship.
    """
    if max_depth > RD_MAX_DEPTH:
        raise ValueError(
            f"max_depth can be at most {RD_MAX_DEPTH:g} m, the depth to which rd is defined."
        )
    depth = np.asarray(depth, dtype=float)
    keep = depth <= max_depth
    if u0 is not None:
        u0 = np.asarray(u0, dtype=float)
        if u0.ndim != 0:
            if u0.shape != depth.shape:
                raise ValueError(
                    f"u0 must be one value per depth; got shape {u0.shape}, expected {depth.shape}."
                )
            u0 = u0[keep]
    site = site_profile(
        depth[keep],
        np.asarray(qt, dtype=float)[keep],
        np.asarray(fs, dtype=float)[keep],
        gwt,
        unit_weight,
        u0=u0,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )
    z, qt, gamma, invalid = site.depth, site.qt, site.unit_weight, site.invalid
    sigma_v, sigma_v_eff, ic = site.sigma_v, site.sigma_v_eff, site.ic
    fc = fines_content_bi2014(ic, cfc)
    q1, _ = qc1ncs_bi2014(qt, sigma_v_eff, fc)

    csr = cyclic_stress_ratio(amax, sigma_v, sigma_v_eff, stress_reduction_coefficient(z, mw))
    crr = (
        crr_m75_bi2014(q1, c0)
        * magnitude_scaling_factor_bi2014(mw, q1)
        * overburden_correction_bi2014(sigma_v_eff, q1)
    )
    fsl = crr / csr
    # Default keeps the z <= gwt mask. A supplied profile uses u0 <= 0, which is the
    # same mask when that profile is the hydrostatic one from gwt.
    unsaturated = (z <= gwt) if u0 is None else (site.u0 <= 0)
    fsl[(ic >= ic_cutoff) | unsaturated | invalid] = np.nan

    return TriggeringResult(
        depth=z,
        qt=qt,
        unit_weight=gamma,
        sigma_v=sigma_v,
        sigma_v_eff=sigma_v_eff,
        ic=ic,
        fc=fc,
        qc1ncs=q1,
        csr=csr,
        crr=crr,
        fsl=fsl,
        invalid=invalid,
        critical_layers=critical_layers(z, fsl, min_thickness),
    )


@dataclass
class RobertsonTriggeringResult:
    """Robertson (2009) triggering for one earthquake, on the Robertson & Wride curve.

    `fsl` is NaN where liquefaction is not assessed: above the water table, Ic above
    the cutoff (sample and test), or an invalid reading. It is infinite where
    Qtn,cs is at or above 160, which the curve classes as too dense to liquefy.
    `k_sigma` is 1 at every depth unless an exponent was passed to the triggering
    function. `loose_sand_zone` is the Fr < 0.5% band in which Kc is held at 1 when
    the screen is on.
    """

    depth: np.ndarray
    qt: np.ndarray
    fs: np.ndarray
    unit_weight: np.ndarray
    sigma_v: np.ndarray
    sigma_v_eff: np.ndarray
    u0: np.ndarray
    ic: np.ndarray
    qtn: np.ndarray
    fr: np.ndarray
    n: np.ndarray
    kc: np.ndarray
    qtn_cs: np.ndarray
    crr75: np.ndarray
    msf: np.ndarray
    k_sigma: np.ndarray
    rd: np.ndarray
    csr: np.ndarray
    fsl: np.ndarray
    not_susceptible: np.ndarray
    """True where Ic is above the cutoff. No factor of safety; test a sample."""
    not_liquefiable: np.ndarray
    """True where Qtn,cs >= 160. The clean-sand curve does not extend there."""
    above_water_table: np.ndarray
    """True where the reading is dry: z <= gwt, or u0 <= 0 when a profile was supplied."""
    loose_sand_zone: np.ndarray
    """True where 1.64 < Ic < 2.36 and Fr < 0.5%."""
    invalid: np.ndarray
    critical_layers: list[tuple[float, float]]

    @property
    def critical_thickness(self) -> float:
        """Total thickness of the critical layers (m)."""
        return float(sum(bottom - top for top, bottom in self.critical_layers))

    @property
    def critical_qtn_cs(self) -> tuple[float, float]:
        """Mean and standard deviation of Qtn,cs over the readings in critical layers."""
        inside = np.zeros_like(self.depth, dtype=bool)
        for top, bottom in self.critical_layers:
            inside |= (self.depth >= top) & (self.depth <= bottom)
        q = self.qtn_cs[inside & np.isfinite(self.qtn_cs)]
        return (float(q.mean()), float(q.std())) if q.size else (float("nan"), float("nan"))


def triggering_robertson2009(
    depth,
    qt,
    fs,
    gwt: float,
    mw: float,
    amax: float,
    *,
    unit_weight=None,
    u0=None,
    gamma_min: float | None = 14.0,
    gamma_max: float | None = 22.0,
    ic_cutoff: float = 2.6,
    qtn_cs_limit: float = 160.0,
    loose_sand_screen: bool = True,
    msf: str = "idriss",
    k_sigma_f: float | None = None,
    min_thickness: float = 0.4,
) -> RobertsonTriggeringResult:
    """Deterministic liquefaction triggering, Robertson & Wride (1998) / Robertson (2009).

    Qtn, Ic, n and Fr come from `opencpt.site.site_profile` (Robertson 2009). The
    resistance curve, Kc and the clean-sand equivalent are Robertson & Wride (1998).
    CSR uses Liao & Whitman / NCEER rd (`stress_reduction_coefficient_nceer`) and the
    Youd et al. (2001) magnitude scaling factor. Kσ is 1. Robertson (2009) built the
    stress exponent in Qtn so a separate overburden factor is not required, and the
    case-history check published with that update used Kσ = 1 with the NCEER rd and
    MSF. Pass `k_sigma_f` to apply the Hynes & Olsen factor from Youd et al. (2001)
    on top of that.

    This is not the 1998 stepped exponent (n = 1, then 0.5 or 0.7) and not qc1N.
    Robertson & Wride (1998) normalised qc without subtracting the total stress.
    Robertson (2009) uses Qtn = ((qt - σv)/pa) (pa/σ'v)^n.

    Args:
        depth: Depth below ground surface (m), increasing.
        qt: Corrected cone resistance (kPa).
        fs: Sleeve friction (kPa).
        gwt: Groundwater depth (m). Readings at or above it are not assessed, unless
            `u0` is supplied, in which case readings with u0 <= 0 are not assessed.
        mw: Earthquake moment magnitude.
        amax: Peak ground acceleration (g), the value without excess pore pressure.
        unit_weight: One constant unit weight (kN/m³), or None for Robertson & Cabal (2010).
        u0: Pore pressure (kPa), one value per depth or a scalar. None keeps hydrostatic u0.
        gamma_min: Lower limit (kN/m³) on the Robertson & Cabal unit weight. Default 14.
        gamma_max: Upper limit (kN/m³). Default 22.
        ic_cutoff: Ic above this value is not given a factor of safety (default 2.6).
            The comparison is strict. Those soils need a sample and a lab test.
        qtn_cs_limit: Upper end of the CRR curve. Qtn,cs at or above this is not
            liquefiable (default 160, the published end of the curve).
        loose_sand_screen: Set Kc = 1 for 1.64 < Ic < 2.36 and Fr < 0.5%.
        msf: "idriss" or "andrus_stokoe". See `magnitude_scaling_factor_youd2001`.
        k_sigma_f: Exponent f of the Youd et al. (2001) Kσ, or None to leave Kσ = 1.
        min_thickness: Minimum critical layer thickness (m).
    """
    depth = np.asarray(depth, dtype=float)
    supplied_u0 = None if u0 is None else np.asarray(u0, dtype=float)
    site = site_profile(
        depth,
        qt,
        fs,
        gwt,
        unit_weight,
        u0=supplied_u0,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )
    z = site.depth
    ic, qtn, fr = site.ic, site.qtn, site.fr
    loose = (ic > 1.64) & (ic < 2.36) & (fr < 0.5)
    kc = correction_factor_kc(ic, fr, loose_sand_screen=loose_sand_screen)
    qtn_cs = clean_sand_qtn(qtn, kc)
    crr75 = crr75_robertson_wride(qtn_cs, qtn_cs_limit)
    msf_value = magnitude_scaling_factor_youd2001(mw, msf)
    msf_profile = np.full(z.shape, msf_value)
    if k_sigma_f is None:
        k_sigma = np.ones(z.shape)
    else:
        k_sigma = overburden_correction_youd2001(site.sigma_v_eff, k_sigma_f)
    rd = stress_reduction_coefficient_nceer(z)
    csr = cyclic_stress_ratio(amax, site.sigma_v, site.sigma_v_eff, rd)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsl = crr75 * msf_profile * k_sigma / csr

    not_liquefiable = np.isfinite(qtn_cs) & (qtn_cs >= qtn_cs_limit)
    not_susceptible = np.isfinite(ic) & (ic > ic_cutoff)
    above_water_table = (z <= gwt) if supplied_u0 is None else (site.u0 <= 0)
    fsl = np.where(not_liquefiable, np.inf, fsl)
    withhold = not_susceptible | above_water_table | site.invalid
    fsl = np.where(withhold, np.nan, fsl)

    return RobertsonTriggeringResult(
        depth=z,
        qt=site.qt,
        fs=site.fs,
        unit_weight=site.unit_weight,
        sigma_v=site.sigma_v,
        sigma_v_eff=site.sigma_v_eff,
        u0=site.u0,
        ic=ic,
        qtn=qtn,
        fr=fr,
        n=site.n,
        kc=kc,
        qtn_cs=qtn_cs,
        crr75=crr75,
        msf=msf_profile,
        k_sigma=k_sigma,
        rd=rd,
        csr=csr,
        fsl=fsl,
        not_susceptible=not_susceptible,
        not_liquefiable=not_liquefiable,
        above_water_table=above_water_table,
        loose_sand_zone=loose & np.isfinite(ic),
        invalid=site.invalid,
        critical_layers=critical_layers(z, fsl, min_thickness),
    )


__all__ = [
    "C0_DETERMINISTIC",
    "RD_MAX_DEPTH",
    "RD_RECOMMENDED_DEPTH",
    "RobertsonTriggeringResult",
    "TriggeringResult",
    "clean_sand_qtn",
    "corrected_cone_resistance",
    "correction_factor_kc",
    "critical_layers",
    "crr75_robertson_wride",
    "crr_m75_bi2014",
    "cyclic_stress_ratio",
    "fines_content_bi2014",
    "magnitude_scaling_factor_bi2014",
    "magnitude_scaling_factor_youd2001",
    "overburden_correction_bi2014",
    "overburden_correction_youd2001",
    "qc1ncs_bi2014",
    "stress_reduction_coefficient",
    "stress_reduction_coefficient_nceer",
    "triggering_bi2014",
    "triggering_robertson2009",
]
