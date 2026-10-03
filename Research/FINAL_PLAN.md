# Final Research Plan v3 — compute-rich, labour-poor
*After literature study (15 peer-reviewed papers), three-reviewer council critique, and the
resourcing constraint: **2 working people (user + Claude), 5 authors, unlimited Kaggle accounts**.*
*Verified 2026-10-02. Supersedes v2 and PLAN_DRAFT.md.*

---

## PART 0 — Unresolved blockers (carried forward from v2)

### 0.1 Dataset attribution — STILL OPEN, must be settled
DataCite **10.17632/8b27pdcxv7.1** authors: *Liang Deyu, Kong Xiangyu, Liu Jinlong, He Xi, Zhang Yuzhuo,
Xue Xingwei, Xu Lei* (Shenyang, China; CC BY 4.0). No group member is listed. The Kaggle entry is a
re-upload.
- The submitted Part B report cites `M. Mahi` as the dataset author. **This needs correcting.**
- Any dataset-description paper (Data in Brief etc.) is off the table.
- CC BY 4.0 permits the research with attribution — only the authorship claim was wrong.

### 0.2 Authorship composition — flag, decide deliberately
5 authors, 2 contributors. Most venues (and IEEE/Elsevier policy) require all listed authors to have made
a substantive contribution and to approve the submission. Either give the other three real, documented
roles (annotation QA, replication runs, figure production, writing sections) or reduce the author list.
This is cheap to fix now and very expensive to fix after submission.

### 0.3 Two facts still needed before the protocol can be finalised
1. **Do raw UAV frames with EXIF/GPS exist, or only the 1,300 released crops?** Decides whether
   flight-level splitting (the only real leakage fix) is possible, or whether we fall back to geometric
   verification.
2. **Is a faculty co-author with Q1 experience available?** Decides whether the Q1 path is live.

---

## PART 1 — What the resourcing change means

| Resource | Status | Design consequence |
|---|---|---|
| GPU compute | **Effectively unlimited** (multiple accounts x 30 h/wk) | Buy statistical power. 10 seeds, not 3. More fractions. Full HP grids. |
| Human labour | **1 person + Claude** | Zero manual annotation. Everything scripted, resumable, batch-submitted. |
| Calendar | Months | Sequence so something is submittable early. |

**Therefore, decisively:**
- **DROP** everything needing human annotation: the 50-image re-annotation study, inter-annotator
  agreement, manual mask-convention harmonisation. These were the hidden labour sinks.
- **EXPAND** everything compute-bought: seeds (3 -> 10), label fractions, baseline arms, HP grids,
  threshold sweeps. The statistician's advice was "spend budget on seeds"; we can now afford to.
- **AUTOMATE**: one shared data module, one runner script, one results CSV, resume-from-checkpoint by
  default, every run self-describing (seed + config + git SHA).

*Kaggle condition: this is legitimate provided each account belongs to its actual owner. One person
operating several accounts breaches Kaggle's ToS and would taint the paper.*

---

## PART 2 — Step 1: the zero-GPU experiment that may change the paper

**Hypothesis to kill first:** our headline finding — precision collapses 0.821 -> 0.576 while recall
holds — **may be a decision-threshold artifact, not a representation property.**

We already have `{method}_qual_preds.npz` saved for all four methods. So, on CPU, today:

1. Sweep the threshold 0.05 -> 0.95 on every saved prediction.
2. Recompute precision / recall / IoU / clDice at each.
3. Report **AP** (threshold-free) and the IoU at each method's *own* optimal threshold.

**Decision rule:**
- If the precision gap **closes** at tuned thresholds -> the Part B "insight" was calibration, not
  representation. That is itself a publishable methodological finding, and it reframes the paper.
- If the gap **persists** -> the finding is real and survives the strongest objection raised against it.

Either outcome is informative and costs nothing. **This is the first thing we do.**

---

## PART 3 — The two structural fixes (unchanged from v2, now affordable)

### 3.1 A real unlabelled corpus
Pretraining on the same 909 images we fine-tune on is not SSL — it is pretext-task regularisation, and it
makes a label-efficiency x-axis meaningless. Pretrain on **>=50k unlabelled pavement crops** (CSF-50K /
UDTIRI / RDD / raw UAV frames), fine-tune and evaluate on PaveCrack1300.

### 3.2 One backbone, one corpus — all four verified available
| Family | Method | Checkpoint | Status |
|---|---|---|---|
| Contrastive | MoCo v3 | `nyu-visionx/moco-v3-vit-b` | mirror — checksum vs `facebookresearch/moco-v3` |
| Self-distillation | DINO v1 | `facebook/dino-vitb16` | official, 468k dl/mo |
| Reconstructive | MAE | `facebook/vit-mae-base` | official, 45k dl/mo |
| Hybrid | iBOT | `OK-AI/ibot-vitb16-pretrain-in1k` | mirror — checksum vs `bytedance/ibot` |

All ViT-B/16, all ImageNet-1k, all patch-16. DINOv2 (LVD-142M, distilled from ViT-g) is reported
**separately** as a labelled *foundation-model upper bound*.

---

## PART 4 — Experiment matrix v3 (compute-rich)

| Block | Design | Runs | GPU-h |
|---|---|---|---|
| **B0** Threshold/AP re-analysis of existing preds | CPU | — | **0** |
| **B1** Label curve | 4 SSL + 3 baselines (ImageNet-sup, random-init, semi-supervised) x {1,5,10,25,50,100}% x 10 seeds | 420 | ~120 |
| **B2** SSL pretraining on >=50k corpus | 4 methods, defensible schedule | 4 | ~40 |
| **B3** HP grid (LR x layer-decay, per method x regime) | selected on val | ~48 | ~15 |
| **B4** Frozen-vs-finetuned, properly tuned | 4 methods x 2 regimes x 5 seeds | 40 | ~12 |
| **B5** Topology metrics + stats | CPU | — | 0 |
| **B6** DINOv2 upper-bound arm | 6 fractions x 10 seeds | 60 | ~18 |
| | | **~570 runs** | **~205 GPU-h** |

At 120 GPU-h/week across accounts: **~2 weeks of compute**, spread over a longer calendar for
debugging. The binding constraint is now *my* notebook-building and *your* run-shepherding, not GPUs.

### Protocol (non-negotiable, from the council)
- Fix **iterations**, not epochs.
- Validation carved from the labelled budget (80/20), never the 195-image split on top.
- Label subsets: resampled per seed, **nested** within a seed, **stratified** by crack-density quintile.
- Threshold tuned per method on validation; report AP alongside.
- Stats: `IoU ~ method + (1|image) + (1|seed)` mixed model, Holm-corrected; seed SD + BCa bootstrap CI
  over test images. State which differences are unresolvable.
- Metrics: IoU (dataset + image-level, stated) · clDice (pinned skeletonizer, declared empty-mask rule) ·
  Boundary IoU (theta=2px) · Betti-0 error.
- Leakage: flight-level split if EXIF exists; otherwise SIFT/LoFTR + RANSAC geometric verification of
  cross-split pairs, plus the **NN-distance probe figure** (test IoU vs cosine distance to nearest train
  image) as the leakage certificate.

---

## PART 5 — Contributions

| | Contribution | Status |
|---|---|---|
| C1 | Controlled 4-family SSL comparison, one backbone / one corpus / real unlabelled pool | Keep |
| C2 | Label-efficiency curves with 10 seeds, honest CIs, and ImageNet-sup / random / semi-supervised baselines at every fraction | **Core** |
| C3 | Topology-aware evaluation; pre-registered test of whether IoU misranks thin-structure methods | Keep |
| C6 | An independent, powered test of the annotation-free claim published in *Automation in Construction* 2026 | **Sharpest hook** |
| C5 | Precision/recall decomposition — **pending B0**; may become a calibration finding instead | Conditional |
| C4 | Cross-dataset generalization | **Dropped** -> second paper (labour-heavy, weak expected finding) |

---

## PART 6 — Sequence

| Phase | When | What | Who |
|---|---|---|---|
| **P0** | Now | B0 threshold re-analysis (0 GPU) | Claude builds, user runs |
| **P1** | Weeks 0–2 | Fix citation; freeze `splits.json` + hashes; shared data module; leakage audit | Both |
| **P2** | Weeks 1–4 | Assemble >=50k unlabelled corpus; checksum the 4 checkpoints; pilot 1 cell end-to-end to measure true per-run cost | Claude builds |
| **P3** | Weeks 3–10 | B1–B4 batch runs across accounts | User shepherds |
| **P4** | Weeks 6–12 | ICCIT submission (supervised benchmark, re-run correctly) | Both |
| **P5** | Weeks 10–20 | Analysis, figures, paper -> arXiv -> *Measurement Science & Technology* | Both |

---

## PART 7 — Honest expected value (unchanged)

| Outcome | Probability |
|---|---|
| Automation in Construction / IEEE T-ITS | **<10%** |
| ICCIT + arXiv + MST under review within 8 months | **>70%** |

Compute abundance raises the *quality ceiling* but does not change venue selectivity. What moves the
odds most is a Q1-experienced faculty co-author, which remains unresolved (0.3).
