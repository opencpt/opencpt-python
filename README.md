# opencpt

Open-source Python library for CPT (cone penetration test) analysis: site characterization,
liquefaction triggering by Boulanger & Idriss (2014) and shear wave velocity from CPT.
It is the engine of the [OpenCPT](https://github.com/opencpt) app.

**Alpha.** The API may change between 0.x releases.

```bash
pip install --pre opencpt
```

## Example

```python
import numpy as np
from opencpt import corrected_cone_resistance, triggering_bi2014
from opencpt.site import site_profile
from opencpt.shear_wave import vs_assaf2023_a07, time_averaged_vs

# A made-up 10 m sounding: loose sand under a 2 m clay crust (kPa, m).
depth = np.arange(0.5, 10.01, 0.5)
qc = np.where(depth < 2, 600.0, 4000.0 + 300 * depth)
fs = np.where(depth < 2, 25.0, 30.0)
u2 = np.where(depth < 2, 40.0, 9.81 * np.clip(depth - 1.5, 0, None))

qt = corrected_cone_resistance(qc, u2, area_ratio=0.8)
result = triggering_bi2014(depth, qt, fs, gwt=1.5, mw=7.5, amax=0.3)
result.critical_layers      # [(2.0, 10.0)]
result.critical_thickness   # 8.0 m
result.fsl                  # factor of safety per reading; NaN where not assessed

site = site_profile(depth, qt, fs, gwt=1.5)
vs = vs_assaf2023_a07(site.qt, site.ic, depth)
time_averaged_vs(depth, vs)  # (132.1, 10.0): Vs over the 10 m covered, m/s
```

Profile inputs accept array-like values; numerical result fields use NumPy arrays. Units are SI:
depth in m, cone readings and stresses in kPa, unit weight in kN/m³, Vs in m/s, amax in g.

## Methods

| Module | What | Source |
|---|---|---|
| `opencpt.site` | qt from u2, unit weight, stresses, Ic with the stress exponent iterated | Boulanger & Idriss (2014) Eq. 2.3; Robertson & Cabal (2010); Robertson (2009) |
| `opencpt.liquefaction` | Deterministic triggering: FC from Ic, qc1Ncs, rd, CSR, MSF, Kσ, CRR (C0 = 2.80), FSL and critical layers | Boulanger & Idriss (2014), Report UCD/CGM-14/01, Eqs. 2.2–2.30; rd from Idriss & Boulanger (2008) |
| `opencpt.shear_wave` | Vs from the A07 and MG15_qc regressions for Metro Vancouver post-glacial sediments; time-averaged Vs (Vs30) | Assaf, Molnar & El Naggar (2023), *Soil Dyn Earthq Eng* 165:107693; coefficients from Assaf (2022), Table B-1 |

Choices worth knowing:

- u2 is used as measured, so a negative reading reduces qt (Eq. 2.3).
- Readings with qt or fs missing, qt ≤ 0 or fs < 0 are flagged `invalid` and get no FSL.
  Their unit weight is interpolated so the stresses below stay defined.
- `triggering_bi2014` assesses readings down to `max_depth` (20 m by default) and leaves deeper
  ones out. `max_depth` can be at most 34 m, the depth to which rd is defined; a larger value
  raises `ValueError`, as does `stress_reduction_coefficient` for any depth deeper than 34 m. Below 20 m,
  Idriss & Boulanger (2008) recommend CSR from a site response analysis instead.
- A critical layer's thickness is measured from its first to its last reading with FSL ≤ 1.
  Runs are merged across gaps of up to 0.1 m and kept from 0.4 m thick.
- The Vs regressions were fitted to 1 m averages of CPT readings; they are applied here per reading.

## Validation

`tests/test_ran20100.py` reproduces Fig. 3 of Javanbakht et al. (2025), *Soil Dyn Earthq Eng*
189:109101, for CPT RAN20100 in Richmond, BC (GWT 3 m, amax 0.3 g):

| | Paper | opencpt |
|---|---|---|
| Critical thickness, Mw 7.5 | 13.8 m | 13.55 m |
| Critical thickness, Mw 9.0 | 14.2 m | 13.70 m |
| qc1Ncs over critical layers | 101 ± 17–18 | 100–101 ± 16–17 |

The thickness differs by up to 0.5 m, most likely because of how layer boundaries are measured.
The CPT data come from the MVSMMP CPT database (Javanbakht et al. 2024,
[doi:10.5683/SP3/XFJR0P](https://doi.org/10.5683/SP3/XFJR0P)) and are not distributed with this
package. Set `OPENCPT_MVSMMP_DATA` to the folder of CPT CSV files to run the test; it is skipped
otherwise.

## Development

```bash
uv run pytest
```

MIT license.
