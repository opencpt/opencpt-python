# Changelog

## 0.1.0a3 (2026-10-28)

- Liquefaction triggering by Robertson & Wride (1998), on Robertson (2009) Qtn and Ic:
  `triggering_robertson2009`. Returns Kc, Qtn,cs, CRR7.5, MSF, Kσ, rd, CSR and the
  factor of safety at each depth.
- Kc is 1 at Ic ≤ 1.64, the Robertson & Wride polynomial above that, and 1 again where
  1.64 < Ic < 2.36 and Fr < 0.5%. Ic > 2.6 has no factor of safety. Qtn,cs ≥ 160 is
  not liquefiable.
- CSR uses the NCEER rd (Liao & Whitman, extended below 23 m as in the workshop summary)
  and the Youd et al. (2001) magnitude scaling factor. Kσ stays 1 unless an exponent is passed.

## 0.1.0a2 (2026-10-28)

- `site_profile` returns Q (Qt1), Qtn, n, Fr, Bq, U, qnet, qE and u0 with Ic.
  Bq, U and qE are NaN unless `u2` is supplied. `normalized_parameters` returns the same
  set from stresses already in hand. `soil_behaviour_type_index` still returns only Ic.
- Optional pore-pressure profile `u0` on `vertical_stresses`, `site_profile` and
  `triggering_bi2014`. Omitted, u0 stays hydrostatic from the groundwater depth.
- Robertson & Cabal unit weight accepts `gamma_min` and `gamma_max` (default 14 and 22 kN/m³).
  Pass None to leave a side open. A unit weight supplied by the caller is not clipped.
- `qc1ncs_bi2014` still normalizes qt rather than the measured qc. The docstring records that choice.

## 0.1.0a1 (2026-10-17)

First alpha.

- Site characterization: qt from u2, unit weight (Robertson & Cabal 2010), vertical stresses,
  Ic (Robertson 2009), and `site_profile` for a whole sounding with invalid readings flagged.
- Deterministic liquefaction triggering by Boulanger & Idriss (2014): `triggering_bi2014`,
  with FSL per reading and critical layers, to `max_depth` (20 m by default, at most 34 m, the limit of rd).
- Vs from CPT: Assaf et al. (2023) A07 and MG15_qc regressions, and time-averaged Vs (Vs30).
- u2 is used as measured in qt (B&I 2014 Eq. 2.3); a negative reading reduces qt.
- Reproduces Fig. 3 of Javanbakht et al. (2025) for CPT RAN20100.

## 0.0.1

Name placeholder.
