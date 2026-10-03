# Draft Research Plan v1 — for council review

## What we already have (assets)
- **PaveCrack1300**: 1,300 UAV image–mask pairs, 512², ~11% crack pixels. Published by a team member on Mendeley Data (DOI 10.17632/8b27pdcxv7). Leakage-safe dHash-grouped 70/15/15 split (909/195/196).
- **Part A**: supervised benchmark of DeepLabV3-R50 (0.8240), SegFormer-B0 (0.8048), YOLOv26s-Sem (0.8319 mIoU).
- **Part B**: SimCLR / BYOL / MAE / DINOv2 pretrained 50 ep on 909 unlabelled, DeepLabV3 ASPP head, fine-tuned on 195 labels. DINOv2 0.7144 (85.9% of supervised); frozen-vs-FT +0.0362; precision 0.821→0.576 while recall 0.833→0.846.
- All code built and runs on Kaggle T4.

## Gaps found in the verified literature

**G1 — No controlled multi-family SSL comparison on crack data.** Every SSL/weakly-supervised crack paper (T-ITS 2025 spatial-contexts; ESWA 2024 contrastive U-Net; KBS 2026 fractal diffusion) proposes *one* method and compares against supervised baselines. None holds dataset+decoder+budget fixed and varies only the pretext family across contrastive / negative-free / reconstructive / self-distillation.

**G2 — Label efficiency is claimed at a single operating point, never as a curve.** RMPD 2023 reports "60% of annotations"; Crack-Segmenter (preprint) claims annotation-free. Nobody sweeps the label fraction, so nobody knows *where* SSL stops paying off.

**G3 — No variance reporting.** Across all 15 verified papers, none reports multi-seed mean±std for segmentation metrics. Single-run numbers are the norm — differences of <1 mIoU are routinely claimed as improvements.

**G4 — IoU-only evaluation of thin structures.** Crack SSL papers report IoU/Dice only. Topology metrics (clDice) appear in thin-structure work (FGOS-Net reports 97.1% clDice on DeepCrack) but have not been applied to SSL crack evaluation. IoU is dominated by boundary thickness, which is exactly where our Part-B models fail (precision collapse at flat recall).

**G5 — UAV + SSL is empty.** UAV crack papers (T-ITS 2024 UAV-Crack500; MST 2025 DSA-Net; Sensors 2024) are all fully supervised. SSL crack papers all use ground-level/handheld imagery. No paper studies SSL label efficiency on UAV pavement imagery.

**G6 — Cross-dataset generalization untested for SSL crack models.** CrackUDA and DSDGNet do domain adaptation/generalization but for supervised/weakly-supervised models.

## Proposed contributions
- **C1** Controlled 4-family SSL benchmark, all confounds removed (identical init regime, resolution, decoder, schedule, budget).
- **C2** Label-efficiency **curves**: {5, 10, 25, 50, 100}% of the 909-image labelled pool × 4 methods × 3 seeds, mean±std.
- **C3** Topology-aware evaluation: report clDice + boundary-F1 alongside IoU; test whether IoU **misranks** methods relative to clDice on thin structures.
- **C4** Cross-dataset transfer: PaveCrack1300 → Crack500 / DeepCrack / CrackSeg9k subsets (and reverse).
- **C5** Precision/recall decomposition as a reusable diagnostic for label-scarce segmentation.

## Experiment matrix (v1)
| Block | Runs | Est. GPU-h |
|---|---|---|
| B1 label curve: 5 fracs × 4 methods × 3 seeds | 60 FT runs | ~6 |
| B2 confound removal: all 4 from ImageNet-SSL ckpt, 224² supervised baseline | 4 pretrain + 4 FT + 3 baseline | ~8 |
| B3 augmentation ablation (SimCLR ± photometric) | 2 pretrain + 2 FT | ~2 |
| B4 cross-dataset eval (inference only) | 4 methods × 3 datasets | ~1 |
| B5 clDice/boundary-F1 recompute | CPU | 0 |
| **Total** | | **~17 GPU-h** |

## Target venues (peer-reviewed only)
1. **Automation in Construction** (Q1, IF ~10) — publishes exactly this; CrackSegFlow 2026 is there.
2. **IEEE T-ITS** — two of our closest competitors are there.
3. **Measurement Science & Technology** / **Int. J. Pavement Engineering** — realistic fallback.
4. **Data in Brief** — separate short dataset paper for PaveCrack1300.

## Known weaknesses of this plan (self-identified)
- 1,300 images is small vs CSF-50K (50k) and CrackSeg9k (9.2k).
- No novel architecture — contribution is protocol/empirical, which some reviewers discount.
- Test-set checkpoint selection in existing runs must be redone.
- Team is undergraduate with Kaggle-only compute (~30 GPU-h/week).
