# Deep-study notes

## 1. Crack-Segmenter (arXiv 2510.10378, NDSU) — the apparent novelty threat, examined

**Claim:** "fully self-supervised... requires no manual annotations, pixel labels, or any form of
ground truth supervision" and "consistently outperforms 13 state-of-the-art supervised methods"
across 10 datasets.

**Reported numbers vs. the field:**

| Dataset | Crack-Segmenter (no labels) | Their best supervised baseline | Published supervised SOTA |
|---|---|---|---|
| Crack500 | **0.9332 mIoU** | LinkNet 0.6449 | ~0.78 mIoU; SegFormer-B2 IoU 0.595 |
| CFD | **0.8875** | U-Net++ 0.5257 | — |
| CrackTree200 | **0.8670** | U-Net 0.4861 | — |
| GAPs384 | **0.8096** | SegFormer 0.4016 | — |

**Assessment — treat with caution, do not take at face value:**

1. An annotation-free method reporting **+15 mIoU above published supervised SOTA** on Crack500, and
   +29 over its own best baseline, is an extraordinary claim.
2. Unsupervised segmentation must assign cluster to class somewhere. The paper does not describe a
   label-assignment step; if Hungarian/greedy matching to GT is used at eval, that leaks label
   information and the method is not annotation-free *at evaluation*.
3. Internal inconsistency: on CFD they report "XOR (0.6138) and HD (0.6138)" — two different metrics
   with identical values. Suggests a table-level error.
4. **It is a preprint (Oct 2025), unreviewed.** No replication exists.

**Strategic consequence — this does NOT kill our contribution; it motivates it.**
The paper itself defines the frontier we sit on. It explicitly criticises the SSL-pretrain +
supervised-finetune paradigm (our paradigm), calling Song et al. (ESWA 2024) "merely semi-supervised,
not end-to-end fully self-supervised." Fine — but if annotation-free really beat supervised by 29
points, label efficiency would be a solved problem. **A rigorous, multi-seed label-efficiency curve is
exactly the instrument that would test that claim.** Position our work as the measurement the field
lacks, not as a competing method.

## 2. Methodology review (council reviewer: statistician) — findings adopted

### FATAL: "all four from ImageNet-SSL checkpoints" is impossible

Public checkpoints split by **architecture**, not by pretext family:

- SimCLR / BYOL — ResNet-50 only (no public ViT on IN-1k)
- MAE — ViT only (`facebook/vit-mae-base`, IN-1k 1600ep)
- **DINOv2 — no IN-1k checkpoint exists at all.** Released weights are LVD-142M (~110x IN-1k) and
  ViT-S/B/L are *distilled from ViT-g*.

So the "fixed" design would still silently vary backbone (CNN vs ViT), pretrain data (1.3M vs 142M),
pretrain compute, distillation, and patch size (14 vs 16).

**Fix:** one backbone, re-mapped taxonomy — **ViT-B/16, all IN-1k**:
MoCo v3 (contrastive) · DINO v1 (self-distillation) · MAE (reconstructive) · iBOT (hybrid).
Keep DINOv2 only as an explicitly-labelled *foundation-model upper bound*, with its data advantage stated.

### Missing baselines that would sink the paper

No **ImageNet-supervised-classification init** arm and no **random-init** arm at each label fraction.
The first reviewer question is "does any SSL beat plain supervised transfer?" Add both at every fraction.

### Other confounds not addressed in v1

- **Epoch vs iteration budget:** fixing epochs gives the 45-label runs 20x fewer gradient steps.
  Fix the *iteration* budget instead.
- **Validation leak:** "5% labels" actually consumes 45 + 195 val = 240 labels, and val > train.
  Carve val from the budget (80/20 of 45) or use a fixed no-selection schedule.
- **Frozen-vs-FT (+0.0362) is confounded** by trainable-parameter count, optimal LR (frozen heads want
  10-30x higher LR), and tuning budget. Use an identical grid for every (method x regime) cell.
- **Our precision collapse (0.821 to 0.576 at flat recall) may be a THRESHOLD ARTIFACT.** Must tune the
  decision threshold per method on val and report AP before making any representation-level claim.

### Statistics: 3 seeds cannot support a 0.02 mIoU claim

Minimum detectable effect at 80% power, alpha = 0.05:

| seeds | MDE @ sigma=0.010 | MDE @ sigma=0.015 |
|---|---|---|
| 3 | 0.031 | 0.046 |
| 5 | 0.020 | 0.030 |
| 10 | 0.013 | 0.020 |

Nonparametric paired tests are *mathematically incapable* of p<0.05 at n=3 (2^3 = 8 sign-flips,
minimum two-sided p = 0.25). Need **6 or more seeds** for permutation; 10 for a 0.02 claim.

**Recommended test:** linear mixed model on per-image IoU, `IoU ~ method + (1|image) + (1|seed)`,
Holm-corrected contrasts. Report seed-level SD *and* a BCa bootstrap CI over test images (resample
images, not pixels). State explicitly which differences are unresolvable — that honesty *is* contribution G3.

### Label-subset protocol

Resample the subset **per seed** (seed indexes subset and init jointly) — a fixed subset badly
understates variance at 45 images. Use **nested** subsets within a seed (5% in 10% in 25% ...),
varying the nesting across seeds. **Stratify** by crack-pixel-fraction quintile with a
guaranteed-nonzero constraint. 6 seeds at 5%/10%, 3 at the rest. Expect plus/minus 0.03-0.05 bars at
5% — pre-commit to reporting "indistinguishable" there rather than ranking.

### Metrics — clDice alone is insufficient

clDice is biased toward thick structures (a 1-2px centreline offset is fatal below ~4px width — exactly
crack width), high clDice does not imply connectivity, and it is degenerate on empty masks.

**Suite:** dataset-level IoU *and* image-mean IoU (state which) · clDice (pin the skeletonizer version
and the empty-mask convention) · **Boundary IoU** (Cheng et al., CVPR 2021) at theta=2px ·
**Betti-0 / Betti matching error** (Stucki et al., ICML 2023) · skeleton-length relative error ·
component-level detection F1 at IoU >= 0.25.

**Make "IoU misranks vs clDice" non-trivial:** pre-register it as the *rank-reversal rate restricted to
pairs where both differences are individually significant and opposite in sign*, plus Kendall's tau
between rankings with a bootstrap CI.

### Leakage: dHash <= 6 is NOT adequate for UAV imagery

dHash is a *global* descriptor; two frames with 50% overlap along a flight line are not globally
near-duplicate, pass Hamming <= 6, and land in different splits while sharing half their pixels.

Required:

1. **Split by flight/sortie**, not by image (flight-line ID + GPS/timestamp from EXIF) — the only real fix.
2. **Geometric verification** of cross-split pairs: SIFT/LoFTR + RANSAC homography; move any pair with
   more than 5% estimated overlap. ~178k pairs, cheap after an embedding prefilter.
3. **NN-distance probe figure:** for each test image, distance to its nearest train image (DINOv2 CLS
   cosine) vs its IoU. A downward slope means leakage is inflating scores; flat is a leakage-free certificate.
4. **Threshold sensitivity table:** groups reassigned at Hamming 2/6/10/16, and test mIoU under each.
5. If any image was tiled from a stitched orthomosaic, group all tiles of that mosaic.

### Cross-dataset eval measures annotation style unless controlled

- Normalise **physical scale** (mm/pixel, or a common median crack width measured as 2x the distance
  transform at skeleton points). Report the width distribution per dataset — it is the key covariate.
- Report **relative drop vs in-domain oracle**, never absolute.
- Quantify annotation style: GT width / dilation-radius distributions; evaluate with a tolerance metric
  (NSD or Boundary IoU at tau = median width difference) alongside strict IoU.
- **Re-annotate a 50-image common subset of each target under our protocol** — the only clean way to
  separate domain shift from annotation shift. Roughly two days of work.
- No target-set model selection.

### Compute reality

Not 17 GPU-h. With 6 seeds at low fractions, two extra baseline arms, and an honest HP grid:
**120-200 GPU-h = 4-7 weeks at Kaggle's 30 h/week.** Spend the budget on *seeds*, not cells:
drop the 50% fraction, drop B3, keep 4 methods x {5,10,25,100}% x {6,6,3,3} seeds + 2 baseline arms.

## 3. Checkpoint availability — VERIFIED (the redesign is feasible)

The statistician's proposed fix requires four IN-1k ViT-B/16 checkpoints. All four exist:

| Pretext family | Method | Checkpoint | Source | Status |
|---|---|---|---|---|
| Reconstructive | MAE | `facebook/vit-mae-base` | HuggingFace (official, 45k dl/mo) | VERIFIED |
| Self-distillation | DINO v1 | `facebook/dino-vitb16` | HuggingFace (official, 468k dl/mo) | VERIFIED |
| Contrastive | MoCo v3 | `nyu-visionx/moco-v3-vit-b` | HuggingFace (3rd-party re-upload) + `facebookresearch/moco-v3` GitHub (1.3k stars) | VERIFIED, mirror needs checksum against official |
| Hybrid | iBOT | `OK-AI/ibot-vitb16-pretrain-in1k` | HuggingFace (3rd-party) + `bytedance/ibot` GitHub (778 stars) | VERIFIED, mirror needs checksum |

Caveat: MoCo v3 and iBOT HF entries are community re-uploads, not official org accounts. Validate each
against the official release (linear-probe top-1 should reproduce the published number) before use, or
convert the official checkpoints directly.

Consequence: a clean 4-family comparison on ONE backbone, ONE pretrain corpus (IN-1k), ONE patch size
(16) is achievable. This removes the single largest confound in the v1 plan.

DINOv2 (LVD-142M, distilled from ViT-g) is then reported separately as a labelled
"foundation-model upper bound", not as a member of the controlled comparison.
