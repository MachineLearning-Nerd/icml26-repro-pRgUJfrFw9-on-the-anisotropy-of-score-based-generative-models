---
title: "Repro - Score Anisotropy Directions"
emoji: 🎯
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
tags:
 - trackio
 - trackio-logbook
 - open-experiment
 - icml2026-repro
 - paper-pRgUJfrFw9
---

# Repro - Score Anisotropy Directions

Current evaluator entrypoint: [claim-by-claim verification](pages/index.md).

The live judged score remains **5/12** at revision
`33d98421a79a7686497639f2d4d6163fc5acfa3f`. This candidate forecasts a
possible `6/12–8/12`; that range is not a judge result.

Current verdicts: Claims 2, 3, and 6 are **VERIFIED**; Claims 1, 4, and 5 are
**BLOCKED**. The exact judged artifact is preserved unchanged under
`historical/judged-33d98421/`, labeled **Historical rejected baseline**.
