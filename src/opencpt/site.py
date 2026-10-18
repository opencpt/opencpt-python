"""Site characterization from CPT data: cone correction, unit weight, stresses and Ic.

All functions take and return NumPy arrays in SI units: depth in m, stresses and
cone readings in kPa, unit weight in kN/m³.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PA = 101.325
"""Atmospheric pressure (kPa)."""

GAMMA_W = 9.81
"""Unit weight of water (kN/m³)."""


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


def unit_weight_robertson_cabal_2010(qt, fs):
    """Total unit weight from CPT, Robertson & Cabal (2010), limited to 14-22 kN/m³.

    γ/γw = 0.27 log Rf + 0.36 log(qt/pa) + 1.236, with Rf = fs/qt in percent.
    """
    qt = np.maximum(np.asarray(qt, dtype=float), 1.0)
    fs = np.asarray(fs, dtype=float)
    rf = np.clip(fs / qt * 100, 0.1, None)
    gamma = GAMMA_W * (0.27 * np.log10(rf) + 0.36 * np.log10(qt / PA) + 1.236)
    return np.clip(gamma, 14.0, 22.0)


def vertical_stresses(depth, unit_weight, gwt: float):
    """Total vertical stress, hydrostatic pore pressure and effective vertical stress (kPa).

    The unit weight of each reading applies to the interval above it, starting from the
    ground surface. Pore pressure is hydrostatic below the groundwater depth `gwt` (m).

    Returns:
        (sigma_v, u0, sigma_v_eff)
    """
    depth = np.asarray(depth, dtype=float)
    unit_weight = np.broadcast_to(np.asarray(unit_weight, dtype=float), depth.shape)
    dz = np.diff(np.r_[0.0, depth])
    sigma_v = np.cumsum(unit_weight * dz)
    u0 = GAMMA_W * np.clip(depth - gwt, 0, None)
    return sigma_v, u0, sigma_v - u0


def soil_behaviour_type_index(qt, fs, sigma_v, sigma_v_eff, max_iter: int = 30):
    """Soil behaviour type index Ic, Robertson (2009), with the stress exponent n iterated.

    Ic = sqrt((3.47 - log Qtn)² + (log Fr + 1.22)²), where
    Qtn = ((qt - σv)/pa)(pa/σ'v)^n, Fr = fs/(qt - σv) × 100 and
    n = 0.381 Ic + 0.05 (σ'v/pa) - 0.15 ≤ 1.
    """
    qt = np.asarray(qt, dtype=float)
    fs = np.asarray(fs, dtype=float)
    sigma_v = np.asarray(sigma_v, dtype=float)
    sigma_v_eff = np.maximum(np.asarray(sigma_v_eff, dtype=float), 1.0)
    fr = np.clip(fs / np.maximum(qt - sigma_v, 1.0) * 100, 0.01, None)
    n = np.ones_like(qt)
    ic = n
    for _ in range(max_iter):
        qtn = np.clip((qt - sigma_v) / PA * (PA / sigma_v_eff) ** n, 1.0, None)
        ic = np.sqrt((3.47 - np.log10(qtn)) ** 2 + (np.log10(fr) + 1.22) ** 2)
        n_new = np.minimum(0.381 * ic + 0.05 * (sigma_v_eff / PA) - 0.15, 1.0)
        # equal_nan: a missing reading must not keep the loop from ending early.
        if np.allclose(n, n_new, atol=1e-4, equal_nan=True):
            break
        n = n_new
    return ic


@dataclass
class SiteProfile:
    """Site characterization of a sounding; one value per reading."""

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


def site_profile(depth, qt, fs, gwt: float, unit_weight=None) -> SiteProfile:
    """Unit weight, stresses and Ic for a sounding.

    Readings that can't be interpreted are flagged. Their unit weight is interpolated from the
    neighbouring readings, so the stresses of every reading below them stay defined.

    Args:
        depth: Depth below ground surface (m), increasing.
        qt: Corrected cone resistance (kPa).
        fs: Sleeve friction (kPa).
        gwt: Groundwater depth (m).
        unit_weight: One constant unit weight (kN/m³), or None for Robertson & Cabal (2010).
    """
    if unit_weight is not None and np.ndim(unit_weight) != 0:
        raise TypeError("unit_weight must be one value in kN/m³, or None for Robertson & Cabal (2010).")
    z = np.asarray(depth, dtype=float)
    qt = np.asarray(qt, dtype=float)
    fs = np.asarray(fs, dtype=float)
    invalid = ~np.isfinite(qt) | ~np.isfinite(fs) | (qt <= 0) | (fs < 0)
    if invalid.size and invalid.all():
        raise ValueError("No reading has both qt > 0 and fs ≥ 0.")
    qt = np.where(invalid, np.nan, qt)
    fs = np.where(invalid, np.nan, fs)
    if unit_weight is None:
        gamma = unit_weight_robertson_cabal_2010(qt, fs)
        if invalid.any():
            gamma = np.interp(z, z[~invalid], gamma[~invalid])
    else:
        gamma = np.full_like(qt, unit_weight)
    sigma_v, _, sigma_v_eff = vertical_stresses(z, gamma, gwt)
    sigma_v_eff = np.maximum(sigma_v_eff, 1.0)
    ic = soil_behaviour_type_index(qt, fs, sigma_v, sigma_v_eff)
    return SiteProfile(z, qt, fs, gamma, sigma_v, sigma_v_eff, ic, invalid)
