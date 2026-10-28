"""Site characterization from CPT data: cone correction, unit weight, stresses and Ic.

All functions take and return NumPy arrays in SI units: depth in m, stresses and
cone readings in kPa, unit weight in kN/m³.

Normalized parameters, with stresses at the same depth: Q = (qt - sigma_v) / sigma'_v
(also written Qt1), Qtn = ((qt - sigma_v) / pa) * (pa / sigma'_v)^n,
Fr = fs / (qt - sigma_v) * 100, Bq = (u2 - u0) / (qt - sigma_v),
U = (u2 - u0) / sigma'_v and qE = qt - u2. Ic, Qtn, n and Fr follow Robertson (2009).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PA = 101.325
"""Atmospheric pressure (kPa)."""

GAMMA_W = 9.81
"""Unit weight of water (kN/m³)."""


def _as_profile(name: str, values, like: np.ndarray) -> np.ndarray:
    """One value per reading, or a scalar broadcast across `like`."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.full(like.shape, float(values))
    if values.shape != like.shape:
        raise ValueError(
            f"{name} must be one value per reading; got shape {values.shape}, "
            f"expected {like.shape}."
        )
    return values.copy()


def _limit_unit_weight(gamma, gamma_min, gamma_max):
    if gamma_min is None and gamma_max is None:
        return gamma
    lo = -np.inf if gamma_min is None else gamma_min
    hi = np.inf if gamma_max is None else gamma_max
    return np.clip(gamma, lo, hi)


def corrected_cone_resistance(qc, u2, area_ratio: float = 0.8):
    """Corrected cone resistance qt = qc + u2 (1 - a) (kPa), Boulanger & Idriss (2014) Eq. 2.3.

    u2 is used as measured: a negative reading, as in dilative sand, reduces qt.

    Args:
        qc: Measured cone resistance (kPa).
        u2: Pore pressure behind the cone (kPa).
        area_ratio: Net area ratio *a* of the cone.
    """
    qc = np.asarray(qc, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    return qc + u2 * (1 - area_ratio)


def unit_weight_robertson_cabal_2010(
    qt, fs, *, gamma_min: float | None = 14.0, gamma_max: float | None = 22.0
):
    """Total unit weight from CPT, Robertson & Cabal (2010).

    γ/γw = 0.27 log Rf + 0.36 log(qt/pa) + 1.236, with Rf = fs/qt in percent.
    Rf is floored at 0.1% so the logarithm stays defined.

    The result is then limited to `gamma_min`–`gamma_max` (kN/m³). The defaults, 14 and 22,
    are this library's limits (the correlation itself has none); pass None on a side to leave
    it open. With the defaults the values match earlier versions of this function.

    Args:
        qt: Corrected cone resistance (kPa).
        fs: Sleeve friction (kPa).
        gamma_min: Lower limit (kN/m³), or None for no lower limit.
        gamma_max: Upper limit (kN/m³), or None for no upper limit.
    """
    qt = np.maximum(np.asarray(qt, dtype=float), 1.0)
    fs = np.asarray(fs, dtype=float)
    rf = np.clip(fs / qt * 100, 0.1, None)
    gamma = GAMMA_W * (0.27 * np.log10(rf) + 0.36 * np.log10(qt / PA) + 1.236)
    return _limit_unit_weight(gamma, gamma_min, gamma_max)


def vertical_stresses(depth, unit_weight, gwt: float | None = None, u0=None):
    """Total vertical stress, pore pressure and effective vertical stress (kPa).

    The unit weight of each reading applies to the interval above it, starting from the
    ground surface. Pore pressure is hydrostatic below the groundwater depth `gwt` (m),
    u0 = γw * max(z - gwt, 0), unless `u0` is given. A supplied profile is used as
    measured, including values above hydrostatic, below it, or negative (suction).

    Args:
        depth: Depth below ground surface (m).
        unit_weight: Total unit weight (kN/m³), one value or one per depth.
        gwt: Groundwater depth (m). Required when `u0` is omitted.
        u0: Pore pressure (kPa), one value per depth or a scalar. Replaces the
            hydrostatic profile. `gwt` is not used when this is given.

    Returns:
        (sigma_v, u0, sigma_v_eff)
    """
    depth = np.asarray(depth, dtype=float)
    unit_weight = np.broadcast_to(np.asarray(unit_weight, dtype=float), depth.shape)
    dz = np.diff(np.r_[0.0, depth])
    sigma_v = np.cumsum(unit_weight * dz)
    if u0 is None:
        if gwt is None:
            raise TypeError("gwt is required when u0 is not supplied.")
        u0 = GAMMA_W * np.clip(depth - gwt, 0, None)
    else:
        u0 = _as_profile("u0", u0, depth)
    return sigma_v, u0, sigma_v - u0


@dataclass
class NormalizedParameters:
    """Robertson (2009) normalized CPT parameters, one value per reading.

    Qtn, n and Fr are the values that enter Ic. Q (qt1) is the n = 1 form and is not
    floored. Bq, U and qE are NaN when `u2` was omitted; Bq and U are also NaN when
    `u0` was omitted.
    """

    ic: np.ndarray
    qnet: np.ndarray
    """Net cone resistance qnet = qt - σv (kPa)."""
    qt1: np.ndarray
    """Q = Qt1 = qnet / σ'v, normalized cone resistance with n = 1."""
    qtn: np.ndarray
    """Qtn = (qnet / pa) (pa / σ'v)^n, floored at 1."""
    n: np.ndarray
    """Stress exponent used to compute Qtn. Capped at 1."""
    fr: np.ndarray
    """Friction ratio Fr (%) used in Ic. See `normalized_parameters` for the floors."""
    bq: np.ndarray
    """Bq = (u2 - u0) / qnet. NaN when u2 or u0 is omitted."""
    u: np.ndarray
    """U = (u2 - u0) / σ'v. Equals Bq * Q. NaN when u2 or u0 is omitted."""
    qe: np.ndarray
    """Excess cone resistance qE = qt - u2 (kPa). NaN when u2 is omitted."""


def normalized_parameters(
    qt, fs, sigma_v, sigma_v_eff, max_iter: int = 30, *, u0=None, u2=None
) -> NormalizedParameters:
    """Ic and the normalized parameters Q, Qtn, n, Fr, Bq, U, qnet and qE.

    Ic = sqrt((3.47 - log Qtn)^2 + (log Fr + 1.22)^2), Robertson (2009), with
    Qtn = ((qt - σv) / pa) (pa / σ'v)^n, Fr = fs / (qt - σv) * 100 and
    n = 0.381 Ic + 0.05 (σ'v / pa) - 0.15, capped at 1. Iteration starts at n = 1
    and stops when n changes by less than 1e-4 (NumPy's default relative tolerance
    also applies), or after `max_iter` passes. The returned n is the exponent used
    for the returned Qtn, not the next estimate.

    Floors, so the logarithms stay defined: σ'v is at least 1 kPa, the Fr denominator
    qt - σv is at least 1 kPa, Fr is at least 0.01%, and Qtn is at least 1. Q, Bq, U,
    qnet and qE are not floored. Bq = (u2 - u0) / (qt - σv), U = (u2 - u0) / σ'v
    and qE = qt - u2. U equals Bq * Q where both are defined.

    Args:
        qt: Corrected cone resistance (kPa).
        fs: Sleeve friction (kPa).
        sigma_v: Total vertical stress (kPa).
        sigma_v_eff: Effective vertical stress (kPa).
        max_iter: Maximum n iterations.
        u0: Pore pressure (kPa). Required for Bq and U.
        u2: Pore pressure behind the cone (kPa). Required for Bq, U and qE.
    """
    qt = np.asarray(qt, dtype=float)
    fs = np.asarray(fs, dtype=float)
    sigma_v = np.asarray(sigma_v, dtype=float)
    sigma_v_eff = np.maximum(np.asarray(sigma_v_eff, dtype=float), 1.0)
    qnet = qt - sigma_v
    fr = np.clip(fs / np.maximum(qnet, 1.0) * 100, 0.01, None)
    n = np.ones_like(qt)
    ic = n
    for _ in range(max_iter):
        qtn = np.clip(qnet / PA * (PA / sigma_v_eff) ** n, 1.0, None)
        ic = np.sqrt((3.47 - np.log10(qtn)) ** 2 + (np.log10(fr) + 1.22) ** 2)
        n_new = np.minimum(0.381 * ic + 0.05 * (sigma_v_eff / PA) - 0.15, 1.0)
        # equal_nan: a missing reading must not keep the loop from ending early.
        if np.allclose(n, n_new, atol=1e-4, equal_nan=True):
            break
        n = n_new
    qt1 = qnet / sigma_v_eff
    if u2 is None:
        bq = np.full_like(qt, np.nan)
        u = np.full_like(qt, np.nan)
        qe = np.full_like(qt, np.nan)
    else:
        u2 = _as_profile("u2", u2, qt)
        qe = qt - u2
        if u0 is None:
            bq = np.full_like(qt, np.nan)
            u = np.full_like(qt, np.nan)
        else:
            u0 = _as_profile("u0", u0, qt)
            du = u2 - u0
            with np.errstate(divide="ignore", invalid="ignore"):
                bq = du / qnet
            u = du / sigma_v_eff
    return NormalizedParameters(ic, qnet, qt1, qtn, n, fr, bq, u, qe)


def soil_behaviour_type_index(qt, fs, sigma_v, sigma_v_eff, max_iter: int = 30):
    """Soil behaviour type index Ic, Robertson (2009), with the stress exponent n iterated.

    Ic = sqrt((3.47 - log Qtn)^2 + (log Fr + 1.22)^2), where
    Qtn = ((qt - σv)/pa)(pa/σ'v)^n, Fr = fs/(qt - σv) × 100 and
    n = 0.381 Ic + 0.05 (σ'v/pa) - 0.15 ≤ 1.

    Returns Ic only. Q, Qtn, n, Fr, Bq, U, qnet and qE are on `normalized_parameters`.
    """
    return normalized_parameters(qt, fs, sigma_v, sigma_v_eff, max_iter).ic


@dataclass
class SiteProfile:
    """Site characterization of a sounding; one value per reading.

    Ic, Q (qt1), Qtn, n and Fr are the Robertson (2009) parameters. Bq, U and qE
    are NaN unless `u2` was passed to `site_profile`.
    """

    depth: np.ndarray
    qt: np.ndarray
    """Corrected cone resistance (kPa); NaN for invalid readings."""
    fs: np.ndarray
    """Sleeve friction (kPa); NaN for invalid readings."""
    unit_weight: np.ndarray
    sigma_v: np.ndarray
    sigma_v_eff: np.ndarray
    """Effective vertical stress (kPa), at least 1 kPa."""
    ic: np.ndarray
    invalid: np.ndarray
    """True for readings that can't be interpreted: qt or fs missing, qt ≤ 0 or fs < 0."""
    u0: np.ndarray
    """Pore pressure (kPa): hydrostatic from `gwt`, or the supplied profile."""
    qnet: np.ndarray
    """Net cone resistance qt - σv (kPa). NaN for an invalid reading."""
    qt1: np.ndarray
    """Q = Qt1 = (qt - σv) / σ'v."""
    qtn: np.ndarray
    """Qtn = ((qt - σv) / pa) (pa / σ'v)^n, floored at 1."""
    n: np.ndarray
    """Stress exponent n ≤ 1 used for Qtn."""
    fr: np.ndarray
    """Friction ratio Fr (%) used in Ic."""
    bq: np.ndarray
    """Bq = (u2 - u0) / (qt - σv). NaN if u2 was not supplied."""
    u: np.ndarray
    """U = (u2 - u0) / σ'v. NaN if u2 was not supplied."""
    qe: np.ndarray
    """Excess cone resistance qE = qt - u2 (kPa). NaN if u2 was not supplied."""


def site_profile(
    depth,
    qt,
    fs,
    gwt: float | None = None,
    unit_weight=None,
    *,
    u0=None,
    u2=None,
    gamma_min: float | None = 14.0,
    gamma_max: float | None = 22.0,
) -> SiteProfile:
    """Unit weight, stresses, Ic and normalized CPT parameters for a sounding.

    Readings that can't be interpreted are flagged. Their unit weight is interpolated from the
    neighbouring readings, so the stresses of every reading below them stay defined.

    Pore pressure is hydrostatic from `gwt` unless `u0` is given, in which case that profile
    is the pore pressure. `u2` is only used for Bq, U and qE; Ic and the stresses do not
    depend on it. Without `u2`, those three are NaN.

    Args:
        depth: Depth below ground surface (m), increasing.
        qt: Corrected cone resistance (kPa).
        fs: Sleeve friction (kPa).
        gwt: Groundwater depth (m). Required unless `u0` is supplied.
        unit_weight: One constant unit weight (kN/m³), or None for Robertson & Cabal (2010).
            A supplied value is kept as given, including below 14 or above 22.
        u0: Pore pressure (kPa), one value per depth or a scalar. Replaces the hydrostatic
            profile from `gwt`.
        u2: Pore pressure behind the cone (kPa), one value per depth or a scalar.
            Needed for Bq, U and qE.
        gamma_min: Lower limit (kN/m³) on the Robertson & Cabal unit weight. None leaves
            the correlation unclipped below. Default 14. Ignored when `unit_weight` is given.
        gamma_max: Upper limit (kN/m³). Default 22. None leaves it unclipped above.
    """
    if gwt is None and u0 is None:
        raise TypeError("gwt is required unless a u0 profile is supplied.")
    if unit_weight is not None and np.ndim(unit_weight) != 0:
        raise TypeError(
            "unit_weight must be one value in kN/m³, or None for Robertson & Cabal (2010)."
        )
    z = np.asarray(depth, dtype=float)
    qt = np.asarray(qt, dtype=float)
    fs = np.asarray(fs, dtype=float)
    invalid = ~np.isfinite(qt) | ~np.isfinite(fs) | (qt <= 0) | (fs < 0)
    if invalid.size and invalid.all():
        raise ValueError("No reading has both qt > 0 and fs ≥ 0.")
    qt = np.where(invalid, np.nan, qt)
    fs = np.where(invalid, np.nan, fs)
    if u0 is not None:
        u0 = _as_profile("u0", u0, z)
    if u2 is not None:
        u2 = _as_profile("u2", u2, z)
    if unit_weight is None:
        gamma = unit_weight_robertson_cabal_2010(qt, fs, gamma_min=gamma_min, gamma_max=gamma_max)
        if invalid.any():
            gamma = np.interp(z, z[~invalid], gamma[~invalid])
    else:
        gamma = np.full_like(qt, unit_weight)
    sigma_v, u0, sigma_v_eff = vertical_stresses(z, gamma, gwt, u0)
    sigma_v_eff = np.maximum(sigma_v_eff, 1.0)
    norm = normalized_parameters(qt, fs, sigma_v, sigma_v_eff, u0=u0, u2=u2)
    return SiteProfile(
        z,
        qt,
        fs,
        gamma,
        sigma_v,
        sigma_v_eff,
        norm.ic,
        invalid,
        u0,
        norm.qnet,
        norm.qt1,
        norm.qtn,
        norm.n,
        norm.fr,
        norm.bq,
        norm.u,
        norm.qe,
    )
