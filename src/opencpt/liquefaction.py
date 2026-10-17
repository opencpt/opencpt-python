"""CPT-based liquefaction triggering, Boulanger & Idriss (2014).

Boulanger, R.W., Idriss, I.M. (2014). CPT and SPT based liquefaction triggering
procedures. Report UCD/CGM-14/01, University of California, Davis.

Inputs and outputs are NumPy arrays in SI units (m, kPa) unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .site import (
    PA,
    corrected_cone_resistance,
    soil_behaviour_type_index,
    unit_weight_robertson_cabal_2010,
    vertical_stresses,
)

C0_DETERMINISTIC = 2.80
"""CRR curve constant for the deterministic B&I (2014) relationship (PL ≈ 16%)."""


def fines_content_bi2014(ic, cfc: float = 0.0):
    """Fines content (%) from Ic: FC = 80 (Ic + C_FC) - 137, limited to 0-100."""
    return np.clip(80 * (np.asarray(ic, dtype=float) + cfc) - 137, 0, 100)


def qc1ncs_bi2014(qt, sigma_v_eff, fc, max_iter: int = 50):
    """Clean-sand equivalent normalized cone resistance qc1Ncs, solved iteratively.

    qc1N = CN qt/pa with CN = (pa/σ'v)^m ≤ 1.7 and m = 1.338 - 0.249 qc1Ncs^0.264
    (qc1Ncs limited to 21-254), plus the fines adjustment Δqc1N.

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
        qc1n = cn * qt / PA
        dq = (11.9 + qc1n / 14.6) * np.exp(1.63 - 9.7 / (fc + 2) - (15.7 / (fc + 2)) ** 2)
        new = qc1n + dq
        if np.allclose(new, qc1ncs, atol=1e-3):
            qc1ncs = new
            break
        qc1ncs = new
    return qc1ncs, cn


def stress_reduction_coefficient(depth, mw: float):
    """Shear stress reduction coefficient rd (Idriss 1999), as used by B&I (2014)."""
    z = np.asarray(depth, dtype=float)
    alpha = -1.012 - 1.126 * np.sin(z / 11.73 + 5.133)
    beta = 0.106 + 0.118 * np.sin(z / 11.28 + 5.142)
    return np.exp(alpha + beta * mw)


def cyclic_stress_ratio(amax: float, sigma_v, sigma_v_eff, rd):
    """CSR = 0.65 (amax/g)(σv/σ'v) rd, with amax in g."""
    return 0.65 * amax * np.asarray(sigma_v) / np.maximum(np.asarray(sigma_v_eff), 1.0) * rd


def magnitude_scaling_factor_bi2014(mw: float, qc1ncs):
    """MSF = 1 + (MSFmax - 1)(8.64 e^(-M/4) - 1.325), MSFmax = 1.09 + (qc1Ncs/180)³ ≤ 2.2."""
    qs = np.minimum(np.asarray(qc1ncs, dtype=float), 180)
    msf_max = np.minimum(1.09 + (qs / 180) ** 3, 2.2)
    return 1 + (msf_max - 1) * (8.64 * np.exp(-mw / 4) - 1.325)


def overburden_correction_bi2014(sigma_v_eff, qc1ncs):
    """Kσ = 1 - Cσ ln(σ'v/pa) ≤ 1.1, with Cσ = 1/(37.3 - 8.27 qc1Ncs^0.264) ≤ 0.3."""
    q = np.minimum(np.asarray(qc1ncs, dtype=float), 211)
    c_sigma = np.minimum(1 / (37.3 - 8.27 * q ** 0.264), 0.3)
    return np.minimum(1 - c_sigma * np.log(np.maximum(np.asarray(sigma_v_eff), 1.0) / PA), 1.1)


def crr_m75_bi2014(qc1ncs, c0: float = C0_DETERMINISTIC):
    """Cyclic resistance ratio for Mw 7.5 and σ'v = 1 atm.

    CRR = exp(q/113 + (q/1000)² - (q/140)³ + (q/137)⁴ - C0), with q = qc1Ncs ≤ 211.
    """
    q = np.minimum(np.asarray(qc1ncs, dtype=float), 211)
    return np.exp(q / 113 + (q / 1000) ** 2 - (q / 140) ** 3 + (q / 137) ** 4 - c0)


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
    return [(float(z[s]), float(z[e - 1])) for s, e in runs if z[e - 1] - z[s] >= min_thickness - 1e-9]


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
    """Factor of safety; NaN where liquefaction is not assessed (above the water table or Ic ≥ cutoff)."""
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
        q = self.qc1ncs[inside]
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
    cfc: float = 0.0,
    ic_cutoff: float = 2.6,
    max_depth: float = 20.0,
    min_thickness: float = 0.4,
    c0: float = C0_DETERMINISTIC,
) -> TriggeringResult:
    """Deterministic liquefaction triggering by Boulanger & Idriss (2014).

    Args:
        depth: Depth below ground surface (m), increasing.
        qt: Corrected cone resistance (kPa); see `opencpt.site.corrected_cone_resistance`.
        fs: Sleeve friction (kPa).
        gwt: Groundwater depth (m).
        mw: Earthquake moment magnitude.
        amax: Peak ground acceleration (g).
        unit_weight: Constant unit weight (kN/m³), or None for Robertson & Cabal (2010).
        cfc: Fitting parameter C_FC of the fines content correlation.
        ic_cutoff: Readings with Ic at or above this are treated as not liquefiable.
        max_depth: Readings deeper than this are left out (m).
        min_thickness: Minimum critical layer thickness (m).
        c0: CRR curve constant; 2.80 for the deterministic relationship.
    """
    depth = np.asarray(depth, dtype=float)
    keep = depth <= max_depth
    z = depth[keep]
    qt = np.asarray(qt, dtype=float)[keep]
    fs = np.asarray(fs, dtype=float)[keep]

    gamma = unit_weight_robertson_cabal_2010(qt, fs) if unit_weight is None else np.full_like(qt, unit_weight)
    sigma_v, _, sigma_v_eff = vertical_stresses(z, gamma, gwt)
    sigma_v_eff = np.maximum(sigma_v_eff, 1.0)
    ic = soil_behaviour_type_index(qt, fs, sigma_v, sigma_v_eff)
    fc = fines_content_bi2014(ic, cfc)
    q1, _ = qc1ncs_bi2014(qt, sigma_v_eff, fc)

    csr = cyclic_stress_ratio(amax, sigma_v, sigma_v_eff, stress_reduction_coefficient(z, mw))
    crr = crr_m75_bi2014(q1, c0) * magnitude_scaling_factor_bi2014(mw, q1) * overburden_correction_bi2014(sigma_v_eff, q1)
    fsl = crr / csr
    fsl[(ic >= ic_cutoff) | (z <= gwt)] = np.nan

    return TriggeringResult(
        depth=z, qt=qt, unit_weight=gamma, sigma_v=sigma_v, sigma_v_eff=sigma_v_eff, ic=ic, fc=fc,
        qc1ncs=q1, csr=csr, crr=crr, fsl=fsl, critical_layers=critical_layers(z, fsl, min_thickness),
    )


__all__ = [
    "C0_DETERMINISTIC",
    "TriggeringResult",
    "corrected_cone_resistance",
    "critical_layers",
    "crr_m75_bi2014",
    "cyclic_stress_ratio",
    "fines_content_bi2014",
    "magnitude_scaling_factor_bi2014",
    "overburden_correction_bi2014",
    "qc1ncs_bi2014",
    "stress_reduction_coefficient",
    "triggering_bi2014",
]
