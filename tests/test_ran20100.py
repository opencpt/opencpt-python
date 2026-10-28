"""Reproduction of Fig. 3 of Javanbakht et al. (2025), CPT RAN20100 in Richmond, BC.

Javanbakht, A., et al. (2025). Soil Dynamics and Earthquake Engineering 189: 109101.
Inputs as in the paper: B&I (2014) deterministic, GWT 3 m, amax 0.3 g, Mw 7.5 and 9.0,
unit weight from Robertson & Cabal (2010), C_FC = 0, Ic cutoff 2.6, to 20 m depth.

The CPT data come from the MVSMMP CPT database (Javanbakht et al. 2024,
doi:10.5683/SP3/XFJR0P) and are not distributed with this package. Point
OPENCPT_MVSMMP_DATA at the folder of CPT CSV files to run these tests; they are
skipped otherwise. The digitized paper curves (fig3_25.xlsx) are optional.
"""

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from opencpt import corrected_cone_resistance, triggering_bi2014

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("OPENCPT_MVSMMP_DATA", HERE.parents[1] / "research" / "cpt" / "data"))
DIGITIZED = Path(os.environ.get("OPENCPT_FIG3_DIGITIZED", DATA.parent / "fig3_25" / "fig3_25.xlsx"))
CSV = DATA / "RAN20100.csv"

pytestmark = pytest.mark.skipif(not CSV.exists(), reason="MVSMMP CPT data not available")


@pytest.fixture(scope="module")
def results():
    with open(CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    z = np.array([float(r["z(m)"]) for r in rows])
    qc = np.array([float(r["qc(kpa)"]) for r in rows])
    fs = np.array([float(r["fs(kpa)"]) for r in rows])
    u2 = np.array([float(r["u2(kpa)"]) for r in rows])
    qt = corrected_cone_resistance(qc, u2, area_ratio=0.8)
    return {mw: triggering_bi2014(z, qt, fs, gwt=3.0, mw=mw, amax=0.3) for mw in (7.5, 9.0)}


# Paper: Mw 7.5 Hcr = 13.8 m, qc1Ncs 101 ± 17; Mw 9.0 Hcr = 14.2 m, 101 ± 18.
# Ours, also given by research/cpt/fig3_25/replicate_fig3.py: 13.55 m and 13.70 m.
@pytest.mark.parametrize("mw, ours, paper", [(7.5, 13.55, 13.8), (9.0, 13.70, 14.2)])
def test_critical_layer_thickness(results, mw, ours, paper):
    hcr = results[mw].critical_thickness
    assert hcr == pytest.approx(ours, abs=0.01)
    assert abs(hcr - paper) <= 0.6


@pytest.mark.parametrize("mw, paper_mean", [(7.5, 101), (9.0, 101)])
def test_critical_layer_qc1ncs(results, mw, paper_mean):
    mean, std = results[mw].critical_qc1ncs
    assert abs(mean - paper_mean) <= 2
    assert 14 <= std <= 20


def test_critical_layers_mw_7_5(results):
    assert results[7.5].critical_layers == pytest.approx(
        [(3.75, 4.15), (4.90, 6.05), (7.15, 14.75), (14.95, 18.85), (19.25, 19.75)]
    )


@pytest.mark.parametrize("mw, sheet, max_rmse", [(7.5, "fos_75", 0.25), (9.0, "fos_90", 0.08)])
def test_fsl_matches_digitized_paper_curve(results, mw, sheet, max_rmse):
    pd = pytest.importorskip("pandas")
    if not DIGITIZED.exists():
        pytest.skip("digitized Fig. 3 curves not available")
    dig = pd.read_excel(DIGITIZED, sheet_name=None)
    d = (
        next(v for k, v in dig.items() if k.strip().lower() == sheet)
        .iloc[:, :2]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    a, b = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    paper_z, paper_fs = (a, b) if a.max() > b.max() else (b, a)

    r = results[mw]
    ok = np.isfinite(r.fsl)
    ours = np.interp(paper_z, r.depth[ok], r.fsl[ok])
    near = np.min(np.abs(paper_z[:, None] - r.depth[ok][None, :]), axis=1) < 0.05
    rmse = float(np.sqrt(np.mean((paper_fs[near] - ours[near]) ** 2)))
    assert near.sum() > 50
    assert rmse <= max_rmse
