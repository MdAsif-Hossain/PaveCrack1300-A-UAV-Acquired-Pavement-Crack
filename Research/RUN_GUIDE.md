# Research Run Guide — label-efficiency study

Everything needed to execute the experiments on Kaggle: what to attach, which settings,
what order, and — critically — **what each run must save**.

> **Read §1 first.** The single reason the Part-B results cannot be reused is that
> nothing was saved except summary numbers. Do not repeat that.

---

## 1. The saving contract (non-negotiable)

The Part-B notebooks called `torch.save` **zero times**. The `.npz` files stored
`argmax` output — binary masks for 5 images. Consequence: the trained models are gone,
probabilities are gone, and no re-analysis of any kind is possible without retraining.

**Every run in this study must write all six of these**, or the run does not count:

| Artifact | Why |
|---|---|
| `model_final.pt` (+ `model_best.pt`) | Without it the run cannot be re-analysed, ever |
| `probs_test.npy` — float16 `[N,H,W]` probabilities on the **full** test split | Enables threshold sweeps, AP, calibration analysis with **zero** extra GPU |
| `per_image_metrics.csv` — one row per test image | Required for the mixed-effects model and bootstrap CIs |
| `run_config.json` — seed, fraction, arm, LR, iterations, split fingerprint, git SHA | Reproducibility; a reviewer will ask |
| `history.csv` — per-iteration loss + val metric | Shows convergence; proves the iteration budget was honoured |
| one row appended to `results_master.csv` | The single table every figure is built from |

`probs_test.npy` at float16, 196 × 224 × 224 ≈ **19 MB** per run. For ~250 runs that is
~5 GB — fine as a handful of Kaggle Datasets, and it buys unlimited free re-analysis.

---

## 2. Datasets to attach

| Alias | Source | Used by |
|---|---|---|
| **RAW** | `mantashamahi/pavecrack1300-a-uav-acquired-pavement-crack` | every notebook |
| **SPLIT** | committed output of `R0_prepare` (contains `split.json` + fingerprint) | R1–R4 |
| **CODE** | the `crackssl` package uploaded as a Kaggle Dataset | every notebook |
| **UNLABELLED** | ≥50k unlabelled pavement crops (CSF-50K / UDTIRI / RDD) | only if we add an in-domain SSL arm |

> **Attribution.** PaveCrack1300 is **not ours**. Cite
> Liang D., Kong X., Liu J., He X., Zhang Y., Xue X., Xu L., *PaveCrack1300*, Mendeley
> Data, 2026, doi:10.17632/8b27pdcxv7, CC BY 4.0 — in the paper, the notebooks, and the repo.

### Uploading the code package
1. Zip `Research/code/crackssl/` → upload as a Kaggle Dataset named `crackssl-lib`
2. In every notebook:
   ```python
   import sys; sys.path.insert(0, "/kaggle/input/crackssl-lib")
   from crackssl import metrics, viz, data
   ```
3. Re-upload a new **version** whenever the library changes; pin the version in `run_config.json`.

---

## 3. Notebook order

| # | Notebook | Attach | GPU | Internet | ~Time |
|---|---|---|---|---|---|
| **R0** | `R0_prepare` | RAW, CODE | ❌ CPU | On | ~5 min |
| **R1** | `R1_label_curve` | RAW, SPLIT, CODE | ✅ T4 | On | ~20 min / cell |
| **R2** | `R2_sam_zeroshot` | RAW, SPLIT, CODE | ✅ T4 | ✅ **required** | ~15 min |
| **R3** | `R3_analysis` | all R1/R2 outputs | ❌ CPU | On | ~10 min |

**R0 must run first and be committed** — everything else consumes its `split.json`.

### R0 produces
- `split.json` — the frozen Part-A split, **plus a SHA-256 fingerprint** printed to stdout
- `density.json` — crack-pixel fraction per image (the stratification variable; cached because every seed needs it)
- `leakage_probe.csv` — nearest-neighbour distance from each test image to the training set

### R1 — one notebook, many cells
Each cell = one `(arm, fraction, seed)`. **Cap every cell under 4 hours** so three fit in a
12-hour session and a preemption costs one cell, not the run.

---

## 4. Kaggle settings

| Setting | Value | Why |
|---|---|---|
| Accelerator | **GPU T4 ×2** (R1, R2) · **None** (R0, R3) | Don't burn GPU quota on CPU work |
| Internet | **On** | HuggingFace checkpoint downloads |
| Persistence | **Variables and Files** | Survives a session restart |
| Environment | Pin to the **latest** image, record its version in `run_config.json` | Library drift silently changes metrics |

**Checkpoints to pre-download once** (then upload as a Kaggle Dataset to avoid repeated pulls):

| Arm | Checkpoint |
|---|---|
| DINOv2 | `facebook/dinov2-base` |
| MAE | `facebook/vit-mae-base` |
| SAM 2 | `facebook/sam2-hiera-base-plus` |
| ImageNet-supervised | `google/vit-base-patch16-224` (or torchvision ResNet-50) |

---

## 5. Protocol settings (these ARE the paper's credibility)

| Setting | Value | Reason |
|---|---|---|
| Budget | **Fixed ITERATIONS, not epochs** | Otherwise 45-label runs get 20× fewer updates and the curve measures schedule length |
| Validation | Carved **from** the label budget (80/20) | "5% labels" must not secretly consume 195 extra val labels |
| Test split | **Touched once**, at the very end | Selection on test is what invalidated Part B |
| Label subsets | `data.nested_subsets(...)` — per-seed, nested, density-stratified | Fixed subsets hide the dominant variance source |
| Threshold | Tuned **per arm on validation**; AP also reported | A shared 0.5 threshold confounds calibration with representation |
| Seeds | **10** at 1%/5%, **6** at 10%/25%, **3** at 100% | 3 seeds cannot resolve 0.02 mIoU; paired tests can't reach p<0.05 at n=3 |
| Metrics | IoU (dataset **and** image-level, stated) · clDice · Boundary IoU (θ=2) · Betti-0 · AP | IoU alone is dominated by predicted thickness |

---

## 6. Sanity checks before letting a batch run

```
split fingerprint : b3a2dbd0...        <- must match R0's, every time
train/val/test    : 909 / 195 / 196    <- unchanged from Part A
arm               : DINOv2 fine-tuned
fraction          : 0.05  -> 45 labelled = 36 train + 9 val
iterations        : 2000 (fixed across all fractions)
crack density     : pool 0.1086 | subset 0.1104   <- stratification working
```

If the fingerprint differs, **stop** — the split changed and results are not comparable.

---

## 7. Troubleshooting

**`CUDA out of memory`** — lower batch size; ViT-B/16 at 224² needs ~8 GB at batch 16.
Reduce to 8 and double gradient accumulation to keep the effective batch fixed.

**Session died mid-batch** — resume from `model_best.pt`; each cell is independent, so
re-run only the failed `(arm, fraction, seed)`.

**Metrics differ from a previous run of the same config** — check the environment version
and `metrics.PROVENANCE` (the skimage version changes `skeletonize`, which changes clDice).

**SAM 2 produces empty masks** — expected without prompting. Report the prompting strategy
explicitly; "SAM zero-shot fails on thin structures" is a legitimate finding, but only if
the prompt protocol is stated.

---

## 8. What to do when a batch finishes

1. Commit the notebook (**Save & Run All**) so the output becomes attachable
2. Append rows to `results_master.csv`
3. Run **R3** to regenerate every figure from the master table
4. Never hand-edit a results file — if a number is wrong, fix the code and re-run
