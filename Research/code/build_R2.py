"""Generate R2_sam_zeroshot.ipynb — the training-free foundation-model baseline."""

import json
from pathlib import Path

OUT = Path(__file__).parent / "notebooks" / "R2_sam_zeroshot.ipynb"


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": s.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md("""
# R2 — SAM 2 zero-shot: the training-free baseline

Tests the claim that a promptable foundation model segments cracks **without any
labels**. No training happens here; it is inference only.

### The protocol question that decides whether this is meaningful

SAM is *promptable*: its output depends entirely on what you ask for. "SAM scores X on
cracks" is not a well-defined statement until the prompt protocol is fixed, and a
hostile reviewer will say so. We therefore evaluate **three** clearly separated regimes:

| Regime | Prompt | What it isolates |
|---|---|---|
| **A · automatic** | dense point grid, no human input | the honest zero-label condition |
| **B · grid-prompted** | regular point grid, best-matching mask kept | removes mask *selection* as a failure cause |
| **C · oracle-prompted** | points sampled from ground-truth crack pixels | **upper bound** — separates "cannot *find* cracks" from "cannot *segment* them" |

Regime C is not a usable method — it consumes labels. It exists to make the diagnosis
precise: if C is high and A is low, SAM can segment cracks but cannot locate them, and
the fix is a detector rather than a better segmenter. If C is *also* low, SAM simply does
not represent thin structures, which is a much stronger finding.

### Dataset attribution
Liang D. et al., *PaveCrack1300*, Mendeley Data (2026), doi:10.17632/8b27pdcxv7, CC BY 4.0.
"""))

cells.append(md("## 1 · Setup"))
cells.append(code("""
import sys, json, time
from pathlib import Path

sys.path.insert(0, "/kaggle/input/crackssl-lib")

import numpy as np, pandas as pd, torch
from PIL import Image

from crackssl import data as D, metrics as M, viz
viz.apply_style()

WORK = Path("/kaggle/working"); WORK.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
EXPECTED_FINGERPRINT = ""      # <-- paste from R0

INPUT = Path("/kaggle/input")
def find_one(pat, what):
    hits = sorted(INPUT.glob(pat))
    if not hits: raise FileNotFoundError(f"missing {what} ({pat})")
    return hits[0]

split = D.load_split(find_one("**/split.json", "R0 split.json"))
fp = split.fingerprint()
if not EXPECTED_FINGERPRINT: raise SystemExit(f'Set EXPECTED_FINGERPRINT = "{fp}"')
assert fp == EXPECTED_FINGERPRINT, f"SPLIT MISMATCH {fp} != {EXPECTED_FINGERPRINT}"
print("split OK:", fp, "| test images:", len(split.test))

img_dir  = find_one("**/[Ii]mages", "images")
mask_dir = next((p for p in [*INPUT.glob("**/[Mm]asks"), *INPUT.glob("**/[Ll]abels")]), None)
IMAGES = {p.stem: p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in {".jpg",".jpeg",".png"}}
MASKS  = {p.stem: p for p in sorted(mask_dir.rglob("*")) if p.suffix.lower() in {".png",".jpg",".jpeg"}}
print("device:", DEVICE)
"""))

cells.append(md("""
## 2 · Load SAM 2

Requires Internet. If the checkpoint is unavailable the notebook records the failure
explicitly rather than silently skipping the arm — an absent baseline must be visible.
"""))
cells.append(code("""
SAM_ID = "facebook/sam2.1-hiera-base-plus"
sam_ok, sam_err = False, None
try:
    from transformers import Sam2Model, Sam2Processor
    sam_proc = Sam2Processor.from_pretrained(SAM_ID)
    sam = Sam2Model.from_pretrained(SAM_ID).to(DEVICE).eval()
    sam_ok = True
    print("loaded", SAM_ID)
except Exception as e:
    sam_err = f"{type(e).__name__}: {e}"
    print("SAM 2 unavailable:", sam_err)
    print("Fallback: try SAM 1 ('facebook/sam-vit-base') in the next cell.")
"""))

cells.append(code("""
if not sam_ok:
    try:
        from transformers import SamModel, SamProcessor
        SAM_ID = "facebook/sam-vit-base"
        sam_proc = SamProcessor.from_pretrained(SAM_ID)
        sam = SamModel.from_pretrained(SAM_ID).to(DEVICE).eval()
        sam_ok = True
        print("fell back to", SAM_ID, "-- record this in the paper")
    except Exception as e:
        raise SystemExit(f"no SAM checkpoint available: {e}")
"""))

cells.append(md("""
## 3 · The three prompt regimes

Each returns a per-image probability map, so the output is directly comparable with the
trained arms from R1 and can go through the same threshold sweep and metric suite.
"""))
cells.append(code("""
def load_pair(sid):
    img = Image.open(IMAGES[sid]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    msk = np.asarray(Image.open(MASKS[sid]).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 0
    return img, msk

@torch.no_grad()
def sam_prob(img, points):
    \"\"\"Union of SAM's mask probabilities over a set of positive point prompts.\"\"\"
    if len(points) == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    pts = [[[[int(x), int(y)]] for x, y in points]]        # [batch][objects][points][xy]
    labs = [[[1]] * len(points)]
    inputs = sam_proc(images=img, input_points=pts, input_labels=labs,
                      return_tensors="pt").to(DEVICE)
    out = sam(**inputs, multimask_output=True)
    masks = sam_proc.post_process_masks(
        out.pred_masks.float().cpu(),
        inputs["original_sizes"].cpu(),
        **({"reshaped_input_sizes": inputs["reshaped_input_sizes"].cpu()}
           if "reshaped_input_sizes" in inputs else {}),
    )[0].numpy()                                            # [objects, 3, H, W]
    scores = out.iou_scores.float().cpu().numpy()[0]        # [objects, 3]
    prob = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    for o in range(masks.shape[0]):
        k = int(scores[o].argmax())
        m = masks[o, k].astype(np.float32)
        if m.shape != prob.shape:
            m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                           .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST), np.float32) / 255.0
        prob = np.maximum(prob, m * float(scores[o, k]))
    return prob

def grid_points(step=32, size=IMG_SIZE):
    half = step // 2
    return [(x, y) for y in range(half, size, step) for x in range(half, size, step)]

def oracle_points(mask, n=8, seed=0):
    \"\"\"Points sampled from GT crack pixels. Uses labels -> an UPPER BOUND, not a method.\"\"\"
    ys, xs = np.nonzero(mask)
    if len(ys) == 0: return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    return [(int(xs[i]), int(ys[i])) for i in idx]
"""))

cells.append(md("## 4 · Run all three regimes over the test split"))
cells.append(code("""
REGIMES = {
    "A_automatic":   dict(desc="dense 16px point grid, no human input", labels_used=0),
    "B_grid":        dict(desc="coarse 32px point grid",                 labels_used=0),
    "C_oracle":      dict(desc="8 points sampled from GT crack pixels",  labels_used="GT (upper bound)"),
}

probs = {k: [] for k in REGIMES}
gts = []
t0 = time.time()

for i, sid in enumerate(split.test):
    img, msk = load_pair(sid)
    gts.append(msk)
    probs["A_automatic"].append(sam_prob(img, grid_points(16)))
    probs["B_grid"].append(sam_prob(img, grid_points(32)))
    probs["C_oracle"].append(sam_prob(img, oracle_points(msk, n=8, seed=0)))
    if (i + 1) % 20 == 0:
        print(f"\\r  {i+1}/{len(split.test)} images  ({(time.time()-t0)/60:.1f} min)", end="")
print(f"\\ndone in {(time.time()-t0)/60:.1f} min")

GT = np.stack(gts)
for k in probs:
    probs[k] = np.stack(probs[k]).astype(np.float16)
    np.save(WORK / f"sam_probs_{k}.npy", probs[k])
print("saved probability maps:", {k: v.shape for k, v in probs.items()})
"""))

cells.append(md("""
## 5 · Evaluate

Each regime gets its **own** tuned threshold, chosen on the first 40 test images and
applied to the remaining 156. Tuning on the full test set would be selection on test —
the error that invalidated the earlier study. The split is reported so the asymmetry is
visible.
"""))
cells.append(code("""
from crackssl import train as T

TUNE_N = 40
tune_idx, eval_idx = np.arange(TUNE_N), np.arange(TUNE_N, len(GT))
rows, per_image_all = [], []

for k, meta in REGIMES.items():
    p = probs[k].astype(np.float32)
    thr, sweep = T.tune_threshold(p[tune_idx], GT[tune_idx])
    summ, per = T.evaluate(p[eval_idx], GT[eval_idx], thr,
                           image_ids=[split.test[i] for i in eval_idx])
    summ.update(regime=k, description=meta["desc"], labels_used=str(meta["labels_used"]),
                threshold=thr, sam_checkpoint=SAM_ID,
                n_tune=TUNE_N, n_eval=len(eval_idx))
    rows.append(summ)
    per["regime"] = k
    per_image_all.append(per)
    print(f"{k:14s} thr {thr:.3f}  IoU {summ['iou_dataset']:.4f}  "
          f"clDice {summ['cl_dice_image_mean']:.4f}  AP {summ['ap_image_mean']:.4f}")

res = pd.DataFrame(rows)
res.to_csv(WORK / "sam_zeroshot_results.csv", index=False)
pd.concat(per_image_all).to_csv(WORK / "sam_per_image.csv", index=False)
display(res[["regime", "labels_used", "threshold", "iou_dataset",
             "precision_dataset", "recall_dataset", "cl_dice_image_mean", "ap_image_mean"]].round(4))
"""))

cells.append(md("## 6 · The diagnosis"))
cells.append(code("""
a = res.set_index("regime").loc["A_automatic", "iou_dataset"]
c = res.set_index("regime").loc["C_oracle", "iou_dataset"]

print(f"automatic (no labels)      : IoU {a:.4f}")
print(f"oracle-prompted (GT points): IoU {c:.4f}")
print(f"gap                        : {c - a:+.4f}\\n")

if c - a > 0.15 and c > 0.35:
    verdict = ("SAM can SEGMENT cracks once told where they are, but cannot FIND them "
               "unprompted. The bottleneck is localisation, not representation -- a "
               "detector front-end would help; a better mask decoder would not.")
elif c < 0.35:
    verdict = ("SAM scores poorly EVEN WITH ground-truth prompts. Thin structures are "
               "not well represented by its mask decoder. This is the stronger finding: "
               "prompting cannot rescue it.")
else:
    verdict = ("Automatic and oracle prompting are close -- prompt strategy is not the "
               "limiting factor here. Report both and avoid over-claiming either way.")
print("DIAGNOSIS:", verdict)

json.dump({"automatic_iou": float(a), "oracle_iou": float(c), "gap": float(c - a),
           "verdict": verdict, "checkpoint": SAM_ID},
          open(WORK / "sam_diagnosis.json", "w"), indent=1)
"""))

cells.append(code("""
import matplotlib.pyplot as plt

k = min(4, len(split.test))
fig, axes = plt.subplots(k, 5, figsize=(11, 2.1 * k), squeeze=False)
for r in range(k):
    sid = split.test[r]
    img, msk = load_pair(sid)
    axes[r][0].imshow(img)
    axes[r][1].imshow(msk, cmap="gray")
    for j, reg in enumerate(REGIMES):
        axes[r][2 + j].imshow(probs[reg][r].astype(np.float32), cmap="magma", vmin=0, vmax=1)
    for ax in axes[r]:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
for j, t in enumerate(["image", "ground truth", *REGIMES]):
    axes[0][j].set_title(t, fontsize=8, color=viz.PALETTE["ink"])
fig.suptitle("SAM zero-shot probability maps", y=1.01, fontsize=10)
fig.tight_layout(); viz.savefig(fig, str(WORK / "r2_sam_qualitative.png")); plt.show()
"""))

cells.append(code("""
print("=" * 66)
print("R2 COMPLETE -- Save Version -> Save & Run All (Commit)")
print("=" * 66)
print("Outputs:", sorted(p.name for p in WORK.glob('*') if p.is_file()))
print()
print("Attach this output to R3 alongside R1's runs.")
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
