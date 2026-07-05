# Reusable Guide — 4-Notebook Semantic-Segmentation Benchmark

**Purpose.** This is a battle-tested playbook for the assignment type *"benchmark three semantic-segmentation
models (DeepLabV3 / SegFormer-B0 / YOLOv26-sem) on an assigned dataset, Tasks A–H, 4 Kaggle notebooks."* It
was distilled from a completed medical (LiTS) run and is written to be **domain-agnostic** — reuse it for any
dataset (street scenes, satellite, documents, industrial, etc.).

> **How to use it (human):** drop this file + the assignment PDF + the dataset into the new project, then tell
> Claude Code: *"Follow GUIDE.md. Read the PDF for exact requirements, run the workflow, and convene a Claude
> council for any open decision."* Rename this file to **`CLAUDE.md`** in the project root so Claude Code
> auto-loads it as context.
>
> **How to use it (Claude Code):** treat §1–2 as invariants, §3 as the execution order, §4 as the decisions to
> resolve *from the new data* (convene a council when flagged), §5 as ready code, §6 as bugs to pre-empt.

---

## 1. The assignment shape (usually fixed — but always re-read the PDF)

- **4 Kaggle notebooks:** `eda-and-data-prep` (NB0), `seg-deeplabv3` (NB1), `seg-segformer-b0` (NB2),
  `seg-yolov26-semantic` (NB3). **Use these exact titles on Kaggle.**
- **3 models:** DeepLabV3-ResNet50 (torchvision, COCO) · SegFormer-B0 (`nvidia/segformer-b0-finetuned-ade-512-512`,
  HuggingFace) · `yolo26s-sem.pt` (Ultralytics, semantic `-sem`, **not** `-seg`).
- **Tasks:** A EDA · B leakage-safe split (saved once, reused) · C augmentation (train-only, synchronized) ·
  D sanity viz · E train ≥50 epochs (config cell) · F test metrics (mIoU, per-class IoU, pixel acc, mean pixel
  acc, Dice, confusion matrix) · G error analysis (worst images, confused pairs, same/different failures) ·
  H final comparison (table, bar/radar, side-by-side grid ≥5, verdict).
- **Deliverable extras we add:** figure-rich `README.md`, `CODING_QUESTIONS.md` answering §6.1 against the code.
- **First action, always:** read the assignment PDF fully and build a task→cell compliance map. Mandates vs
  choices — you only get marks for *justifying the choices*, so justify every one in markdown.

## 2. Golden rules (non-negotiable — these are where marks and bugs live)

1. **Leakage-safe split by GROUP, not by sample.** Identify the grouping signal (patient/volume, video, scene,
   source image, tile, document) and split so no group crosses train/val/test. Fixed `SEED=42`. **Assert** zero
   overlap and print it. Save once to `split.json`; **reuse identically** in NB1–NB3.
2. **One shared, identical pipeline** across the two PyTorch models (dataset, augmentation, loss, metrics) — only
   the model differs. Any per-model difference (e.g. weight decay) is a **fairness bug**.
3. **Augmentation train-only, synchronized image+mask** (one `A.Compose(image=…, mask=…)` call).
4. **Report all Task-F metrics** even if you add more. Adding metrics is fine; dropping required ones is not.
5. **Everything reproducible & version-pinned.** Save best checkpoint, `results.json`, curves, figures per run.

## 3. Workflow (execution order)

**Phase 0 — Understand the data (never assume structure).**
- Write a one-cell **EDA probe** (see §5.1) the user runs on Kaggle (CPU). It auto-discovers the folder tree,
  pairs images↔masks, reports the **filename→group convention**, mask encoding (values, #classes), resolution,
  and the pixel-level class distribution. **Verify the output before writing anything else** — the mask format
  and the group key drive the entire design.

**Phase 1 — NB0 (`eda-and-data-prep`), Tasks A–D.**
- Build a per-sample manifest (cache to CSV). Pixel-class distribution, size/aspect, representative pairs per
  class, corrupt/mismatch/**near-duplicate** checks (perceptual hash on adjacent group members).
- Group split 70/15/15, seed 42, asserts, save `split.json` + `class_weights.json` + `manifest.csv`.
- Augmentation pipeline + `Dataset` class + post-aug **sanity grid** (image | mask | overlay). This is a graded
  quality gate — it must appear before any training.

**Phase 2 — Models.**
- NB1 (DeepLabV3) and NB2 (SegFormer) share §5 code; run in **parallel** on GPU.
- NB3 (YOLO) converts the split to YOLO-semantic format, trains, then does **Task H** loading NB1+NB2
  checkpoints for the side-by-side comparison.

**Phase 3 — Verify & iterate.**
- Extract each run's outputs/figures and **read them** (curves for over/underfit, confusion for the hard class,
  worst-grids). If a result underperforms or overfits, **convene a council** (§8) and run one corrected
  iteration. Keep the old run — it becomes your **ablation** (baseline → change → result).

**Phase 4 — Finalize.**
- Figure-rich `README.md`, `CODING_QUESTIONS.md`, exact Kaggle titles, Public, submit.

## 4. Decision framework — resolve these FROM the new data (🧠 = convene a council)

| Decision | Default | When to reconsider / 🧠 |
|---|---|---|
| **Mask parsing** | class-index PNG `{0..C}` | Separate binary masks → fuse; RGB-coded → map colors→ids; COCO polygons → rasterize. Confirm from the probe. |
| **Group key (leakage)** | filename prefix (patient/video/scene) | If none exists, find any grouping signal; if truly none, document the risk. **This is the #1 thing to get right.** |
| **Class imbalance** | Dice + light weighted-CE `[~0.3,1,~6]` + minority oversampling + **F-beta checkpoint** | Severe minority (<1%): 🧠 loss/sampling/selection. Don't over-weight CE (gradient spikes). |
| **Augmentation** | flips + affine + elastic/grid + brightness/contrast + noise | Flips are **strong regularizers — keep them for small datasets** unless the domain is orientation-critical *and* you have evidence they hurt (we learned removing them caused overfitting). 🧠 if unsure. |
| **Input representation** | 3-ch (replicate gray, or native RGB) | **Volumetric / sequential** data (slices, video) → **2.5D** (neighbors `[i-1,i,i+1]` as channels). Leakage-safe if neighbors are same-group. |
| **Checkpoint metric** | best **val minority-class F2** (recall-weighted) | Selecting by mIoU picks the least-sensitive-on-minority epoch. Report mIoU too. |
| **Model sizes** | DeepLabV3-R50, SegFormer-B0 (fixed), `yolo26s-sem` | Tight GPU → `yolo26n`; more compute → bigger. |
| **imgsz / resolution** | dataset-native | Don't upscale away tiny objects; don't downscale below native. |

## 5. Reusable code patterns (drop-in, domain-agnostic)

### 5.1 EDA probe (Phase 0) — auto-discover structure
Walk `/kaggle/input` with `rglob`; group image files by parent folder; classify image vs mask folders by name
tokens (⚠️ **don't** match `"seg"` — it hits `"segmentation"` in dataset names; match exact folder basenames);
pair by the integer signature `tuple(re.findall(r"\d+", stem))`; parse group id via regex; sample masks →
`np.unique` for encoding + `bincount` for class distribution; show 10 image/mask overlays. Print a copy-paste
**SUMMARY** block.

### 5.2 Leakage-safe group split
```python
rng = np.random.RandomState(42); groups = sorted(df.group.unique()); rng.shuffle(groups)
n=len(groups); tr,va = int(.7*n), int(.15*n)
train_g, val_g, test_g = set(groups[:tr]), set(groups[tr:tr+va]), set(groups[tr+va:])
assert not (train_g&val_g) and not (train_g&test_g) and not (val_g&test_g)   # print "LEAKAGE CHECK PASSED"
# save split.json with native ints:  [int(g) for g in sorted(train_g)]   (numpy int64 is NOT json-serializable!)
```

### 5.3 Metrics — one confusion matrix is the source of truth
```python
class ConfMat:
    def __init__(s,n): s.n=n; s.mat=torch.zeros(n,n,dtype=torch.int64)
    def update(s,p,t):
        k=(t>=0)&(t<s.n); s.mat+=torch.bincount(s.n*t[k]+p[k],minlength=s.n**2).reshape(s.n,s.n)
    def compute(s):
        m=s.mat.double(); tp=m.diag(); fp=m.sum(0)-tp; fn=m.sum(1)-tp
        iou=tp/(tp+fp+fn).clamp(min=1e-9); dice=2*tp/(2*tp+fp+fn).clamp(min=1e-9)
        return dict(iou=iou.tolist(), miou=iou.mean().item(), dice=dice.tolist(),
                    mdice=dice.mean().item(), pixel_acc=(tp.sum()/m.sum()).item(),
                    mean_pixel_acc=(tp/m.sum(1).clamp(min=1e-9)).mean().item())
# minority F2 (checkpoint metric): rec=tp/(tp+fn); prec=tp/(tp+fp); f2=5*prec*rec/(4*prec+rec)
```

### 5.4 Loss (Dice + light weighted CE) & training
`ce = CrossEntropyLoss(weight=tensor([0.3,1,6]))`; `dice_loss` from softmax vs one-hot; `criterion = ce+dice`
(DeepLabV3 also `+0.4*criterion(aux)`). AdamW, `SequentialLR(LinearLR warmup 3 → CosineAnnealing)`, AMP, **unify
`weight_decay` across models**, `WeightedRandomSampler` for minority oversampling, save best-by-minority-F2, hflip
**TTA** at test. SegFormer: `model(pixel_values=x).logits` → `F.interpolate` to full res (compute your OWN loss,
don't pass `labels=`).

### 5.5 Shared-split mechanism on Kaggle
NB0 saves `split.json`+`manifest.csv` to `/kaggle/working` → **Save Version (commit)** → its output becomes a
Dataset → **+ Add Input** to NB1–NB3, which find it via `glob.glob("/kaggle/input/**/split.json", recursive=True)`
(name-agnostic). Split is deterministic, so re-committing NB0 never changes it.

### 5.6 YOLO semantic conversion (NB3)
Export split → `images/{train,val,test}` + parallel `masks/{...}` (single-channel PNG, **pixel=class id, 255=ignore**),
plus `data.yaml` (`path, train, val, test, masks_dir, names`). Train `YOLO("yolo26s-sem.pt").train(data=…, imgsz=native,
epochs=50, fliplr=.5, flipud=.5, hsv_*=0)`. Predict → `result.semantic_mask.data` → `ConfMat` vs GT. If you used 2.5D,
**predict on the exported 2.5D PNGs** (same read path as training). Add a **skip-train branch**: if a prior `best.pt` is
attached, load it and export only `test` (turns a 6 h re-commit into ~15 min).

### 5.7 Notebook authoring & figures
Author each notebook via a **Python builder** (`code(r'''…'''); md(r'''…''')` → `json.dump` to `.ipynb`), then
**syntax-check every code cell** with `compile()` before handing over. After runs, extract figures from the
`.ipynb` outputs (decode `image/png`, name by the cell's `savefig(WORK/"…")` target) into `figures/` for the README.

## 6. Pitfalls & lessons (pre-empt these — we hit every one)

- **`numpy int64` is not JSON-serializable** → cast split ids to `int()` (and add a `default=` on `json.dump`).
- **Removing flips caused overfitting** (train loss 0.16→0.07, val loss diverged) → *regularization beats
  augmentation-realism on small datasets.* Keep flips by default.
- **Weight-decay must be identical across models** — a 1e-4 vs 1e-2 mismatch is unfair and made one model overfit.
- **Checkpoint by val-mIoU under-selects the minority class** (mIoU ≈ dominated by easy classes; minority IoU peaks at
  balanced P/R). Select by **minority F2**; report mIoU alongside.
- **Focal-Tversky is easy to break:** the paper uses exponent `1/γ` (≈0.75, <1); `**gamma` (>1) inverts it. Averaging the
  region term over background dilutes the asymmetry. If you use it, apply it foreground-only.
- **`results.json` must MERGE** all models in NB3 (`for rp in glob(...): res.update(json.load(rp))`), not load just one —
  else the final table is missing a model.
- **YOLO train/predict channel consistency:** train and predict via the same file-read path (Ultralytics uses cv2/BGR).
- **Folder-token trap:** classifying "mask" folders by substring `"seg"` matches `"…segmentation…"` in the dataset name.
- **Green tint in sanity grid** = harmless denorm display artifact; show `img[...,center_channel]` in grayscale.
- **Kaggle ops:** EDA on **CPU** (save quota); training on **GPU T4×2** with **Save & Run All (Commit)** (survives
  disconnect, saves outputs); ~30 GPU-h/week; do a 3-min interactive epoch-1 check before committing a long run;
  **rename Kaggle notebooks to the exact required titles** before submission.

## 7. The Claude council protocol (for hard decisions)

Convene when a decision is genuinely open or a result underperforms/overfits. Spawn **3 specialists in parallel**
with the *same full context* (dataset facts, constraints, the exact numbers of any prior runs) and distinct mandates —
e.g. **(1) augmentation/regularization, (2) loss/imbalance, (3) training-strategy/architecture.** Ask each for: top
root-cause, a ranked list of concrete changes with exact settings, and the single recipe to run next. Then **synthesize**
into one bundled "Run-A" recipe, implement, syntax-check, and run once. Frame prior vs new as a clean **ablation**.

## 8. Definition of done

- [ ] 4 notebooks, exact Kaggle titles, run end-to-end with visible outputs, Public.
- [ ] Leakage-safe group split saved once, reused, asserted.
- [ ] All Task-F metrics + confusion matrix per model; Task G worst-cases + confused pairs + same/different failures.
- [ ] Task H: table + bar/radar + per-class + side-by-side grid + confusion + verdict.
- [ ] Figure-rich `README.md` + `CODING_QUESTIONS.md` (every §6.1 question answered vs the code).
- [ ] Intro of each NB: course, department, **group members + IDs**.
- [ ] Compliance re-audited against the PDF (mandates intact, choices justified).

*Adapt the domain-specific choices (§4) to the new data; keep the invariants (§1–2) and engineering (§5) as-is.*
