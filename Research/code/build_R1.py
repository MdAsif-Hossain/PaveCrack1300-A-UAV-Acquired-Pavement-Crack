"""Generate R1_label_curve.ipynb — the label-efficiency runner (the core experiment)."""

import json
from pathlib import Path

OUT = Path(__file__).parent / "notebooks" / "R1_label_curve.ipynb"


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": s.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md("""
# R1 — Label-efficiency curve

**The core experiment.** One cell = one `(arm, fraction, seed)` run. Each run trains for a
**fixed iteration budget**, selects on **validation**, then touches the test split exactly
once and writes the six required artifacts.

### Protocol, and why each rule exists

| Rule | Reason |
|---|---|
| Fixed **iterations**, not epochs | With an epoch budget a 45-label run gets ~20× fewer updates, so the curve would measure schedule length, not labels |
| Validation **carved from** the label budget | Otherwise "5% labels" secretly also consumes 195 validation labels, and the x-axis is a lie |
| Test split touched **once**, after training | Selecting on test is exactly what invalidated the previous study |
| Threshold tuned **per arm on validation** | A shared 0.5 cut confounds calibration with representation |
| Label subsets **nested, stratified, resampled per seed** | A fixed subset hides the dominant variance source at 45 images |
| **Probabilities saved** for the full test split | Makes every later re-analysis free; the previous study saved `argmax` and became unanalysable |

### Arms

| Arm | Question it answers |
|---|---|
| `dinov2_ft` | What does a foundation encoder buy when adapted? |
| `dinov2_frozen` | Is a frozen foundation encoder enough? |
| `mae_ft` | Does a non-foundation (ImageNet-1k) SSL encoder keep up? |
| `imagenet_sup` | Does any of this beat plain supervised transfer? |
| `random_init` | What is the floor? |

> SAM 2 zero-shot is a separate notebook (R2) because it requires no training.

### Dataset attribution
Liang D., Kong X., Liu J., He X., Zhang Y., Xue X., Xu L., *PaveCrack1300*,
Mendeley Data (2026), doi:10.17632/8b27pdcxv7, CC BY 4.0.
"""))

cells.append(md("## 1 · Setup and the split guard"))
cells.append(code("""
import sys, json, os, gc, time
from pathlib import Path

sys.path.insert(0, "/kaggle/input/crackssl-lib")

import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from crackssl import data as D, models as Mo, train as T, metrics as M, viz

viz.apply_style()

WORK = Path("/kaggle/working")
RUNS = WORK / "runs"; RUNS.mkdir(exist_ok=True, parents=True)
MASTER = WORK / "results_master.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# ---- paste the fingerprint R0 printed -------------------------------------
EXPECTED_FINGERPRINT = ""      # <-- REQUIRED. R1 refuses to run without it.

print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
print("torch :", torch.__version__)
"""))

cells.append(code("""
INPUT = Path("/kaggle/input")

def find_one(pattern, what):
    hits = sorted(INPUT.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"missing {what} ({pattern}); attached: {[p.name for p in INPUT.iterdir()]}")
    return hits[0]

split = D.load_split(find_one("**/split.json", "R0 split.json"))
density = json.loads(find_one("**/density.json", "R0 density.json").read_text())

fp = split.fingerprint()
if not EXPECTED_FINGERPRINT:
    raise SystemExit(f"Set EXPECTED_FINGERPRINT = \\"{fp}\\" (from R0) and re-run this cell.")
if fp != EXPECTED_FINGERPRINT:
    raise SystemExit(f"SPLIT MISMATCH: got {fp}, expected {EXPECTED_FINGERPRINT}. "
                     "Results would not be comparable. Stop and investigate.")
print("split fingerprint OK:", fp, "|", split.counts)

img_dir  = find_one("**/[Ii]mages", "image directory")
mask_dir = next((p for p in [*INPUT.glob("**/[Mm]asks"), *INPUT.glob("**/[Ll]abels")]), None)
IMAGES = {p.stem: p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
MASKS  = {p.stem: p for p in sorted(mask_dir.rglob("*")) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
print(f"{len(IMAGES)} images / {len(MASKS)} masks resolved")
"""))

cells.append(md("""
## 2 · Dataset and loaders

Augmentation is deliberately mild and **identical across arms** — flips and small
photometric jitter only. Anything stronger would interact with the label budget (heavy
augmentation helps most when labels are scarcest), which would confound the curve.
"""))
cells.append(code("""
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

class CrackSeg(Dataset):
    def __init__(self, ids, train: bool, size: int = IMG_SIZE, seed: int = 0):
        self.ids, self.train, self.size = list(ids), train, size
        self.rng = np.random.default_rng(seed)

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        img = np.asarray(Image.open(IMAGES[sid]).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR), np.float32) / 255.0
        msk = np.asarray(Image.open(MASKS[sid]).resize(
            (self.size, self.size), Image.NEAREST))
        msk = (msk > 0).astype(np.int64)

        if self.train:
            if self.rng.random() < 0.5: img, msk = img[:, ::-1], msk[:, ::-1]
            if self.rng.random() < 0.5: img, msk = img[::-1], msk[::-1]   # near-nadir: no privileged up
            if self.rng.random() < 0.3:
                img = np.clip(img * self.rng.uniform(0.85, 1.15)
                              + self.rng.uniform(-0.06, 0.06), 0, 1)

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(msk)))

def loaders_for(budget, batch_size, seed):
    tr = DataLoader(CrackSeg(budget.train_ids, True, seed=seed), batch_size=batch_size,
                    shuffle=True, drop_last=len(budget.train_ids) > batch_size,
                    num_workers=2, pin_memory=True, persistent_workers=True)
    va = DataLoader(CrackSeg(budget.val_ids, False), batch_size=batch_size,
                    shuffle=False, num_workers=2)
    return tr, va

TEST_LOADER = DataLoader(CrackSeg(split.test, False), batch_size=8, shuffle=False, num_workers=2)
TEST_MASKS = np.stack([
    (np.asarray(Image.open(MASKS[s]).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 0)
    for s in split.test])
print("test masks:", TEST_MASKS.shape, "| crack fraction:", round(float(TEST_MASKS.mean()), 4))
"""))

cells.append(md("""
## 3 · The run function

One call = one complete `(arm, fraction, seed)` cell of the experiment matrix.
It is **idempotent**: if the run directory already has a `summary.json`, it returns
immediately. That makes a preempted Kaggle session cost one run, not the batch.
"""))
cells.append(code("""
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.00]
ITERATIONS = 2000          # FIXED across every fraction -- see the protocol table
BATCH = 8

def run_one(arm: str, fraction: float, seed: int, iterations: int = ITERATIONS,
            batch: int = BATCH, force: bool = False):
    run_id = f"{arm}__f{fraction:g}__s{seed}"
    out = RUNS / run_id
    if (out / "summary.json").exists() and not force:
        print(f"[skip] {run_id} already complete"); return json.loads((out / "summary.json").read_text())

    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)

    budget = D.nested_subsets(split.train, density, [fraction], seed=seed, val_frac=0.2)[fraction]
    print(f"\\n=== {run_id} ===")
    print(f"  labels: {budget.n_total} ({budget.n_train} train + {budget.n_val} val)"
          f"  | subset density {np.mean([density[i] for i in budget.train_ids + budget.val_ids]):.4f}")

    model, spec = Mo.build_arm(arm, img_size=IMG_SIZE)
    cfg = T.TrainConfig(
        arm=arm, fraction=fraction, seed=seed, iterations=iterations, batch_size=batch,
        split_fingerprint=fp, lib_version=__import__("crackssl").__version__,
        extra={"n_labels": budget.n_total, "n_train": budget.n_train,
               "n_val": budget.n_val, "family": spec.family, "hf_id": spec.hf_id},
    )
    tr, va = loaders_for(budget, batch, seed)
    best, hist = T.train_arm(model, tr, va, cfg, device=DEVICE)

    model.load_state_dict(best["state"])
    val_probs = T.predict_probs(model, va, device=DEVICE)
    val_masks = np.stack([
        (np.asarray(Image.open(MASKS[s]).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 0)
        for s in budget.val_ids])
    thr, sweep = T.tune_threshold(val_probs, val_masks)

    test_probs = T.predict_probs(model, TEST_LOADER, device=DEVICE)   # test: touched once
    summary, per_image = T.evaluate(test_probs, TEST_MASKS, thr, image_ids=split.test)
    summary["minutes"] = round((time.time() - t0) / 60, 2)

    # Checkpoints are ~330 MB each; the full matrix would be ~58 GB against Kaggle's
    # ~20 GB cap. Keep seed 0 (enough to re-run inference for any arm x fraction) and
    # rely on probs_test.npy for everything else.
    T.save_run(RUNS, cfg, best, hist, summary, per_image, test_probs,
               val_sweep=sweep, master_csv=MASTER, save_model=(seed == 0))

    print(f"  tuned threshold {thr:.3f} | test IoU {summary['iou_dataset']:.4f} "
          f"(at 0.5: {summary['iou_dataset_at_0.5']:.4f}) | clDice {summary['cl_dice_image_mean']:.4f} "
          f"| {summary['minutes']:.1f} min")

    del model, best; gc.collect(); torch.cuda.empty_cache()
    return summary
"""))

cells.append(md("""
## 4 · Pilot — run this FIRST

One cell, to measure the true per-run cost before committing a batch. Multiply the
reported minutes by the matrix size to decide how many seeds you can afford.
"""))
cells.append(code("""
_ = run_one("dinov2_ft", 0.05, seed=0, iterations=200)   # short pilot
print("\\nScale this up: full runs use ITERATIONS =", ITERATIONS)
"""))

cells.append(md("""
## 5 · The batch

Seed counts follow statistical power, not convenience: 3 seeds cannot resolve a 0.02 mIoU
difference, and a paired nonparametric test *cannot* reach p<0.05 at n=3 (2³ = 8 sign
flips → minimum two-sided p = 0.25). So the scarce-label end — where the variance is
largest and the paper's claim lives — gets the most seeds.

**Run in slices.** Keep each Kaggle session under ~4 hours so a preemption is cheap.
Completed runs are skipped automatically, so re-running this cell resumes.
"""))
cells.append(code("""
SEEDS_BY_FRACTION = {0.01: 10, 0.05: 10, 0.10: 6, 0.25: 6, 1.00: 3}
ARMS = ["dinov2_ft", "dinov2_frozen", "mae_ft", "imagenet_sup", "random_init"]

plan = [(a, f, s) for f, n in SEEDS_BY_FRACTION.items() for s in range(n) for a in ARMS]
done = {p.name for p in RUNS.glob("*") if (p / "summary.json").exists()}
todo = [(a, f, s) for a, f, s in plan if f"{a}__f{f:g}__s{s}" not in done]

print(f"matrix: {len(plan)} runs | done: {len(done)} | remaining: {len(todo)}")
print("\\nper fraction:")
for f, n in SEEDS_BY_FRACTION.items():
    print(f"  {f:>5.0%}  {n:>2} seeds x {len(ARMS)} arms = {n*len(ARMS):>3} runs")
"""))

cells.append(code("""
# ---- edit this slice, run, commit, repeat --------------------------------
SLICE = slice(0, 12)          # e.g. slice(0,12) then slice(12,24) ...

for arm, frac, seed in todo[SLICE]:
    try:
        run_one(arm, frac, seed)
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM on {arm} f{frac} s{seed} -- retrying at batch 4")
        torch.cuda.empty_cache()
        run_one(arm, frac, seed, batch=4)
    except Exception as e:
        print(f"  FAILED {arm} f{frac} s{seed}: {type(e).__name__}: {e}")
"""))

cells.append(md("## 6 · Progress and a first look"))
cells.append(code("""
if MASTER.exists():
    m = pd.read_csv(MASTER).drop_duplicates("run_id", keep="last")
    print(f"{len(m)} runs recorded\\n")
    piv = m.pivot_table(index="arm", columns="fraction", values="iou_dataset",
                        aggfunc=["mean", "count"])
    display(piv.round(4))

    agg = (m.groupby(["arm", "fraction", "n_labels"], as_index=False)
             .agg(iou_crack=("iou_dataset", "mean"), iou_crack_sd=("iou_dataset", "std"),
                  n=("iou_dataset", "size")))
    agg["iou_crack_sd"] = agg["iou_crack_sd"].fillna(0)

    NICE = {"dinov2_ft": "DINOv2 fine-tuned", "dinov2_frozen": "DINOv2 frozen",
            "mae_ft": "MAE", "imagenet_sup": "ImageNet-sup.", "random_init": "Random init"}
    agg["arm"] = agg["arm"].map(lambda a: NICE.get(a, a))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    viz.label_efficiency_curve(agg, ax=ax)
    viz.savefig(fig, str(WORK / "r1_progress_curve.png")); plt.show()
    print("\\nBands are +/-1 SD across seeds -- NOT a confidence interval.")
    print("R3 computes proper BCa bootstrap CIs and the mixed-effects test.")
else:
    print("no runs yet")
"""))

cells.append(md("## 7 · Before committing"))
cells.append(code("""
n_runs = len(list(RUNS.glob("*/summary.json")))
size_mb = sum(p.stat().st_size for p in RUNS.rglob("*")) / 1e6
print(f"{n_runs} completed runs | {size_mb:.0f} MB in /kaggle/working")
print()
ckpts = list(RUNS.glob("*/model_best.pt"))
print(f"checkpoints kept: {len(ckpts)} (seed 0 only, by design)")
print()
print("Kaggle's output cap is ~20 GB. Budget for the full 175-run matrix:")
print(f"  probabilities  ~3.4 GB   <- required by R3, never delete")
print(f"  checkpoints    ~{len(ckpts) * 0.33:.1f} GB   <- seed 0 only")
print("If you still approach the cap, drop checkpoints last:")
print("  for p in RUNS.glob('*/model_best.pt'): p.unlink()")
print()
print("=" * 66)
print("Now: Save Version -> Save & Run All (Commit), then attach this output to R3.")
print("=" * 66)
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
