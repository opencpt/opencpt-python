# Changelog

## 0.1.0a1 (unreleased)

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
