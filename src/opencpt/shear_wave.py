"""Shear wave velocity from CPT, Assaf et al. (2023) multiple linear regressions.

Assaf, J., Molnar, S., El Naggar, M.H. (2023). CPT-Vs correlations for post-glacial
sediments in Metropolitan Vancouver. Soil Dynamics and Earthquake Engineering 165: 107693.
Coefficients from Assaf (2022), PhD thesis, Western University, Table B-1.

The regressions were fitted to post-glacial sediments in Metro Vancouver, with CPT readings
averaged over the 1 m seismic CPT intervals. Vs in m/s, cone readings in kPa, depth in m.
Readings that cannot enter the logarithms (non-positive or missing) give NaN.
"""

from __future__ import annotations

import numpy as np

FS_MIN = 0.1
"""Sleeve friction (kPa) below which fs is raised to this value, so ln(fs) stays finite."""


def _positive(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x) & (x > 0), x, np.nan)


def vs_assaf2023_a07(qt, ic, depth):
    """Vs (m/s), Assaf et al. (2023) A07 MLR, after the model form of Andrus et al. (2007).

    ln Vs = 3.25 + 0.15 ln qt + 0.214 ln Ic + 0.193 ln z, i.e. Vs = 25.8 qt^0.15 Ic^0.214 z^0.193.
    Recommended by Assaf et al. (2023) where a regression (not machine learning) model is used.

    Args:
        qt: Corrected cone resistance (kPa).
        ic: Soil behaviour type index, Robertson (2009).
        depth: Depth below ground surface (m).
    """
    return np.exp(3.25 + 0.15 * np.log(_positive(qt)) + 0.214 * np.log(_positive(ic)) + 0.193 * np.log(_positive(depth)))


def vs_assaf2023_mg15_qc(qc, fs, depth):
    """Vs (m/s), Assaf et al. (2023) MG15_qc MLR, after the model form of McGann et al. (2015).

    ln Vs = 3.81 + 0.0884 ln qc + 0.0132 ln fs + 0.226 ln z. Uses only the measured readings,
    the parameter combination Assaf et al. (2023) prefer. fs below `FS_MIN` is raised to it;
    negative fs gives NaN.

    Args:
        qc: Measured cone resistance (kPa).
        fs: Sleeve friction (kPa).
        depth: Depth below ground surface (m).
    """
    fs = np.asarray(fs, dtype=float)
    fs = np.where(fs >= 0, np.maximum(fs, FS_MIN), np.nan)
    return np.exp(3.81 + 0.0884 * np.log(_positive(qc)) + 0.0132 * np.log(fs) + 0.226 * np.log(_positive(depth)))
