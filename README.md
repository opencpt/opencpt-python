# opencpt

Open-source Python library for CPT (cone penetration test) analysis: site characterization,
liquefaction triggering by Boulanger & Idriss (2014) and by Robertson & Wride (1998) /
Robertson (2009), and shear wave velocity from CPT.
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
result.critical_layers  # [(2.0, 10.0)]
result.critical_thickness  # 8.0 m
result.fsl  # factor of safety per reading; NaN where not assessed

site = site_profile(depth, qt, fs, gwt=1.5)
vs = vs_assaf2023_a07(site.qt, site.ic, depth)
time_averaged_vs(depth, vs)  # (132.1, 10.0): Vs over the 10 m covered, m/s
```

Profile inputs accept array-like values; numerical result fields use NumPy arrays. Units are SI:
depth in m, cone readings and stresses in kPa, unit weight in kN/m³, Vs in m/s, amax in g.

## Methods

| Module | What | Source |
|---|---|---|
| `opencpt.site` | qt from u2, unit weight, stresses, Ic with the stress exponent iterated, and Q, Qtn, n, Fr, Bq, U, qnet, qE, u0 | Boulanger & Idriss (2014) Eq. 2.3; Robertson & Cabal (2010); Robertson (2009) |
| `opencpt.liquefaction` | Deterministic triggering: FC from Ic, qc1Ncs, rd, CSR, MSF, Kσ, CRR (C0 = 2.80), FSL and critical layers | Boulanger & Idriss (2014), Report UCD/CGM-14/01, Eqs. 2.2–2.30; rd from Idriss & Boulanger (2008) |
| `opencpt.liquefaction` | Robertson & Wride triggering on Robertson (2009) Qtn: Kc, Qtn,cs, CRR7.5, NCEER rd, Youd et al. MSF, FS | Robertson & Wride (1998); Robertson (2009); Youd et al. (2001); rd from Liao & Whitman (1986) / NCEER |
| `opencpt.shear_wave` | Vs from the A07 and MG15_qc regressions for Metro Vancouver post-glacial sediments; time-averaged Vs (Vs30) | Assaf, Molnar & El Naggar (2023), *Soil Dyn Earthq Eng* 165:107693; coefficients from Assaf (2022), Table B-1 |

Choices worth knowing:

- u2 is used as measured, so a negative reading reduces qt (Eq. 2.3).
- Pore pressure is hydrostatic from the groundwater depth unless a `u0` profile is passed
  (one value per reading). That profile is the equilibrium pore pressure used for effective
  stress, Bq and U.
- `site_profile` returns the normalized parameters beside `ic`: `qt1` (Q, also Qt1), `qtn`, `n`,
  `fr`, `qnet` and `u0`. Pass `u2` to fill `bq`, `u` (U) and `qe` (qE = qt - u2); otherwise those
  three are NaN. Bq = (u2 - u0)/(qt - sigma_v) and U = (u2 - u0)/sigma'_v. Qtn is floored at 1,
  the friction ratio used in Ic is floored at 0.01%, and n is capped at 1.
- Robertson & Cabal unit weight is limited to 14–22 kN/m³ by default. `gamma_min` and
  `gamma_max` change the limits (`None` leaves a side open). A unit weight you pass in is used as given.
- qc1N in Boulanger & Idriss triggering is CN·qt/pa. The 2014 procedure writes CN·qc/pa;
  qt is used here, and is close to qc in sand.
- Readings with qt or fs missing, qt ≤ 0 or fs < 0 are flagged `invalid` and get no FSL.
  Their unit weight is interpolated so the stresses below stay defined.
- `triggering_bi2014` assesses readings down to `max_depth` (20 m by default) and leaves deeper
  ones out. `max_depth` can be at most 34 m, the depth to which rd is defined; a larger value
  raises `ValueError`, as does `stress_reduction_coefficient` for any depth deeper than 34 m. Below 20 m,
  Idriss & Boulanger (2008) recommend CSR from a site response analysis instead.
- `triggering_robertson2009` is the Robertson & Wride (1998) clean-sand curve on Robertson
  (2009) Qtn and Ic, not the 1998 stepped stress exponent and not qc1N. Kc is 1 for
  Ic ≤ 1.64, the Robertson & Wride polynomial above that, and 1 again for
  1.64 < Ic < 2.36 and Fr < 0.5%. Ic > 2.6 is not given a factor of safety. Qtn,cs ≥ 160
  is classed as not liquefiable. CSR uses the NCEER rd, not the Idriss rd used by
  Boulanger & Idriss. The magnitude scaling factor defaults to 10^2.24/Mw^2.56
  (Youd et al. 2001). Kσ is 1 unless `k_sigma_f` is set; the Qtn exponent is the
  overburden correction Robertson (2009) uses.
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
