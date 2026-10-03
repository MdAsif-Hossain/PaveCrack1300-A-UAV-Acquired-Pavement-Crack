# Research — label-efficiency study

**Question:** how much labelled data do foundation models and self-supervised pretraining
actually save for UAV pavement-crack segmentation — and do the recent *annotation-free*
claims hold up under a powered protocol?

**Output:** a conference paper (venue chosen by the supervisor).

> **Dataset attribution.** PaveCrack1300 is third-party data used under CC BY 4.0:
> Liang D., Kong X., Liu J., He X., Zhang Y., Xue X., Xu L., *PaveCrack1300: A
> UAV-Acquired Pavement Crack Segmentation Dataset*, Mendeley Data (2026),
> doi:10.17632/8b27pdcxv7. It is **not** authored by this group — cite it everywhere.

---

## Documents

| File | What it is |
|---|---|
| [`CONFERENCE_PLAN.md`](CONFERENCE_PLAN.md) | The plan: scope, arms, matrix, figures, phases |
| [`LITERATURE.md`](LITERATURE.md) | 15 verified peer-reviewed papers (DOIs checked via Crossref) + preprints kept separate |
| [`STUDY_NOTES.md`](STUDY_NOTES.md) | Deep read of the closest competitor; the full methodology critique |
| [`RUN_GUIDE.md`](RUN_GUIDE.md) | Execution: datasets, settings, order, the saving contract |
| [`FINAL_PLAN.md`](FINAL_PLAN.md) | Longer-form plan incl. the journal follow-up |

## Code

```
code/
  crackssl/            installable package (upload to Kaggle as `crackssl-lib`)
    metrics.py         IoU · Dice · AP · clDice · Boundary IoU · Betti-0 · threshold sweep
    data.py            split guard + fingerprint · nested stratified label subsets
    models.py          5 arms, one shared ASPP decoder, token->grid adapter
    train.py           fixed-iteration training · threshold tuning · saving contract
    viz.py             5 paper figures, CVD-validated palette
  notebooks/
    R0_prepare.ipynb        CPU   split fingerprint, density, leakage audit
    R1_label_curve.ipynb    GPU   the core experiment (175 runs)
    R2_sam_zeroshot.ipynb   GPU   SAM 2, three prompt regimes
    R3_analysis.ipynb       CPU   statistics + every figure and table
  build_R*.py          notebook generators (edit these, not the .ipynb)
  smoke_test_R0.py     runs R0 against a synthetic input tree
```

**Notebooks are generated, not hand-edited.** Change `build_RX.py`, re-run it, re-upload.

## Experiment matrix

5 arms × {1, 5, 10, 25, 100}% labels × {10, 10, 6, 6, 3} seeds = **175 runs ≈ 76 GPU-h**

Seed counts follow statistical power, not convenience: 3 seeds cannot resolve a 0.02 mIoU
difference, and a paired nonparametric test *cannot* reach p<0.05 at n=3. The scarce-label
end — where the variance is largest and the claim lives — gets the most seeds.

Storage: ~3.4 GB probabilities (all runs) + ~8.2 GB checkpoints (seed 0 only) ≈ **11.8 GB**,
inside Kaggle's ~20 GB cap.

## What makes this publishable rather than just another benchmark

1. **Variance is reported.** No paper in the surveyed literature reports multi-seed
   mean ± CI for crack segmentation. We do, and we say out loud which differences are
   *not* resolvable.
2. **The confounds are removed, not hidden.** Fixed iterations (not epochs), validation
   carved from the label budget, test touched once, per-arm threshold tuning.
3. **Topology-aware metrics.** IoU is dominated by predicted thickness — a 3px dilation of
   a 1px line scores clDice 0.982 but IoU 0.333. The rank-reversal test is pre-registered.
4. **Leakage is measured, not assumed.** dHash is a global descriptor and misses
   overlapping UAV frames; the nearest-neighbour probe figure is the certificate.
5. **It tests a published claim.** *Automation in Construction* (2026) reports
   annotation-free crack segmentation beating 13 supervised methods. Nobody has checked it.

## Status

- [x] Library written and tested (metrics verified against sklearn; Holm against hand-computed values)
- [x] All four notebooks generated and syntax-validated
- [x] R0 smoke-tested end to end against a synthetic dataset
- [ ] Upload `crackssl/` to Kaggle as `crackssl-lib`
- [ ] Run R0, paste its fingerprint into R1 and R2
- [ ] Pilot one R1 cell to measure true per-run cost
- [ ] Run the matrix in slices
- [ ] R3 → figures → paper

## Open items

1. **Fix the Part-B report citation** — it credits `M. Mahi` for the dataset.
2. **Do raw UAV frames with EXIF/GPS exist?** Decides whether flight-level splitting is
   possible; otherwise geometric verification is the fallback.
