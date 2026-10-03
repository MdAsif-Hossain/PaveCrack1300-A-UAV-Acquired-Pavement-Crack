# Conference Paper Plan — v1
*Decision: conference track. Target IEEE ICIP (fallback ICCIT). 6–8 pages, one sharp contribution.*
*Resourcing: 2 contributors (Asif + Claude), unlimited Kaggle accounts. Asif = first author.*

---

## 1. The paper

**Working title**
> *Do Foundation Models Remove the Need for Crack Annotations?
> A Statistically Powered Label-Efficiency Study for Thin-Structure Segmentation*

**The hook (why now, why this venue)**
Two 2026 publications claim labels are no longer needed for crack segmentation:
- **Crack-Segmenter** (*Automation in Construction* 2026, doi:10.1016/j.autcon.2025.106591) — "fully
  self-supervised… outperforms 13 supervised methods across ten datasets" (0.9332 mIoU on Crack500,
  ~15 points above published supervised SOTA).
- **SAM3-SERD** (preprint 2026) — training-free crack measurement.

Nobody has independently tested these under a powered protocol. **We do.**

**One-sentence claim**
> We measure, with confidence intervals, how much labelled data foundation models and self-supervised
> pretraining actually save for UAV pavement-crack segmentation — and at which label budget each
> approach stops paying off.

**Why it fits ICIP:** thin-structure segmentation, evaluation methodology, and image-processing metrics
are core ICIP territory. A rigorous evaluation with a negative or cautionary result is welcome there in
a way it is not at an applications venue.

---

## 2. Scope discipline (conference = ONE contribution)

### IN
- The **label-efficiency curve** — the paper *is* this figure.
- **Topology-aware evaluation** (clDice + Boundary IoU) alongside IoU.
- **Threshold/calibration control** — tuned per method, AP reported.
- Honest CIs and an explicit statement of which differences are unresolvable.

### OUT (explicitly deferred to the journal version)
- Cross-dataset generalization
- Augmentation ablations
- The full 4-family SSL taxonomy (MoCo v3 / DINO / MAE / iBOT)
- Any manual annotation work

---

## 3. Arms — 6 total, chosen so each answers a question a reviewer will ask

| Arm | Question it answers |
|---|---|
| **SAM 2 zero-shot** (+ simple prompting) | Does a foundation model work with *zero* crack labels? |
| **DINOv2 frozen + linear/ASPP head** | Does a frozen foundation encoder suffice? |
| **DINOv2 fine-tuned** | What does adaptation buy? |
| **MAE (ViT-B/16, IN-1k)** | Does a non-foundation SSL encoder keep up? |
| **ImageNet-supervised init** | Does any of this beat plain supervised transfer? |
| **Random init** | What is the floor? |

All ViT-B/16 where applicable. One decoder (ASPP) held fixed. One resolution.

---

## 4. Experiment matrix (tight enough for 6 pages)

| Block | Design | Runs | GPU-h |
|---|---|---|---|
| ~~**E0** Threshold re-analysis of existing predictions~~ | **IMPOSSIBLE — see note** | — | — |
| **E1** Label curve | 5 trainable arms x {1, 5, 10, 25, 100}% x 10 seeds | 250 | ~75 |
| **E2** SAM 2 zero-shot baseline | inference only, whole test set | 1 | ~1 |
| **E3** Topology metrics + mixed-model stats | CPU | — | 0 |
| | | **~251 runs** | **~76 GPU-h** |

At 120 GPU-h/week across accounts: **~1 week of compute.** Everything else is build + analysis + writing.

> **E0 cancelled (verified 2026-10-02).** The Part-B notebooks call `torch.save` **zero
> times**, and the saved `.npz` files contain `argmax` output — binary masks for 5 images,
> not probabilities. The trained models no longer exist, so the calibration-vs-representation
> question cannot be settled without retraining. It is therefore folded into R1: every new run
> dumps full-test-split probabilities, which makes threshold sweeps, AP and calibration
> analysis free forever after. See `RUN_GUIDE.md` §1 (the saving contract).

### Protocol (non-negotiable — this is the paper's credibility)
- Fix **iterations**, not epochs.
- Validation carved from the labelled budget (80/20); **test touched once**.
- Label subsets: resampled per seed, nested within seed, stratified by crack-density quintile.
- Threshold tuned on validation per method; AP reported alongside.
- Stats: `IoU ~ method + (1|image) + (1|seed)`; Holm-corrected; seed SD + BCa bootstrap CI over images.
- Metrics: IoU (dataset + image-level, stated) · clDice (pinned skeletonizer) · Boundary IoU (θ=2px).
- Leakage: NN-distance probe figure (test IoU vs cosine distance to nearest train image) as the certificate.

---

## 5. Figures (a 6-page paper supports ~4)

1. **The label-efficiency curve** — the money figure. x = label count (log), y = crack IoU, 6 arms, CI bands.
2. **Qualitative grid** — same images across arms at 5% and 100% labels.
3. **IoU vs clDice rank comparison** — does IoU misrank thin-structure methods?
4. **NN-distance leakage probe** — the credibility figure.

---

## 6. Phases (work backwards from the chosen deadline)

| Phase | Work | Owner |
|---|---|---|
| **P0** | E0 threshold re-analysis — decides framing (calibration vs representation) | Claude builds, Asif runs |
| **P1** | Fix dataset citation; freeze `splits.json` + hashes; shared data module; leakage probe | Both |
| **P2** | Build the runner: resumable, 1 CSV row per run, config + seed + SHA recorded | Claude |
| **P3** | Pilot 1 cell end-to-end → measure true per-run cost → finalise matrix | Both |
| **P4** | Batch E1/E2 across accounts | Asif |
| **P5** | Analysis, figures, stats | Claude |
| **P6** | Write, internal review, arXiv, submit | Both |

**Hard requirement: pick the target conference and its deadline first.** Everything above is scheduled
backwards from it. ICIP deadlines are typically ~January for a September conference; ICCIT ~July for
December. Verify current dates before committing.

---

## 7. Carried-forward blockers

1. **Dataset citation in the submitted Part B report is wrong** (credits `M. Mahi`; real authors are
   Liang Deyu et al.). Fix and disclose.
2. **Do raw UAV frames with EXIF/GPS exist?** Decides whether flight-level splitting is possible.
3. **Faculty co-author** — not required for a conference, but improves the journal follow-up.

---

## 8. Why this is the right conference paper

- **One figure carries it.** Conference papers succeed on a single clear message.
- **The hook is current and concrete** — testing two named 2026 claims, not a vague gap.
- **Compute-heavy, labour-light** — exactly our resource profile.
- **A negative result still publishes.** "Foundation models do not solve thin structures below X labels"
  is a useful, citable finding.
- **It extends cleanly into the journal version** — add the SSL taxonomy and cross-dataset work later,
  with >=30% new content.
