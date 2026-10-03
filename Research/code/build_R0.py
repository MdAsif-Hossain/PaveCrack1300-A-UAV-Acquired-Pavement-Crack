"""Generate R0_prepare.ipynb — the foundation notebook every other run depends on."""

import json
from pathlib import Path

OUT = Path(__file__).parent / "notebooks" / "R0_prepare.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip("\n").splitlines(keepends=True),
    }


cells = []

# ---------------------------------------------------------------- 0. header
cells.append(md("""
# R0 — Prepare: frozen split, crack density, leakage audit

**This notebook runs first and is committed. Everything else consumes its output.**

It produces four artifacts:

| Output | Used by | Why it exists |
|---|---|---|
| `split.json` + **SHA-256 fingerprint** | R1, R2 | Every later run asserts this fingerprint. If it differs, the split changed and results are not comparable. |
| `density.json` | R1 | Crack-pixel fraction per image — the stratification variable for label subsets. Cached because every seed needs it. |
| `leakage_audit.csv` | the paper | dHash threshold-sensitivity table: how many images move between splits at Hamming 2/6/10/16. |
| `leakage_probe.csv` | Figure 4 | Per-test-image: distance to its nearest *training* image, and that image's identity. A flat IoU-vs-distance slope is the leakage-free certificate. |

---

### Dataset attribution

PaveCrack1300 is **third-party data**, used here under CC BY 4.0:

> Liang D., Kong X., Liu J., He X., Zhang Y., Xue X., Xu L.
> *PaveCrack1300: A UAV-Acquired Pavement Crack Segmentation Dataset.*
> Mendeley Data (2026). doi:10.17632/8b27pdcxv7 — CC BY 4.0

Cite this in the paper, the repository, and every notebook.

### Why the leakage audit is here at all

The Part-A split grouped near-duplicates by difference-hash (dHash, Hamming ≤ 6) before
splitting. dHash is a **global** descriptor: two UAV frames that overlap by 50% along a
flight line are not globally near-duplicate, so they pass the filter and land in different
splits while sharing half their pixels. This notebook quantifies how much that matters
instead of assuming it away.
"""))

# ---------------------------------------------------------------- 1. setup
cells.append(md("## 1 · Setup"))
cells.append(code("""
import sys, json, hashlib, warnings
from pathlib import Path

sys.path.insert(0, "/kaggle/input/crackssl-lib")   # the uploaded crackssl package

import numpy as np
import pandas as pd
from PIL import Image

from crackssl import data as D
from crackssl import viz

viz.apply_style()
warnings.filterwarnings("ignore", category=UserWarning)

WORK = Path("/kaggle/working")
SEED = 42
rng = np.random.default_rng(SEED)

print("crackssl loaded from:", Path(D.__file__).parent)
"""))

# ---------------------------------------------------------------- 2. locate
cells.append(md("""
## 2 · Locate the dataset and the Part-A split

Discovery is glob-based so the notebook does not break when Kaggle renames an input
folder. It **fails loudly** rather than silently falling back to a different split —
a wrong split is the one error that invalidates everything downstream.
"""))
cells.append(code("""
INPUT = Path("/kaggle/input")

def find_one(pattern, what, required=True):
    hits = sorted(INPUT.glob(pattern))
    if not hits:
        if required:
            raise FileNotFoundError(
                f"could not find {what} (pattern {pattern!r}).\\n"
                f"attached inputs: {[p.name for p in INPUT.iterdir()]}"
            )
        return None
    if len(hits) > 1:
        print(f"  ! {len(hits)} candidates for {what}, using the first: {hits[0]}")
    return hits[0]

split_path = find_one("**/split.json", "the Part-A split.json")
print("split.json   :", split_path)

# image / mask directories — accept the common layout variants
img_dir  = find_one("**/[Ii]mages", "the image directory")
mask_dir = find_one("**/[Mm]asks", "the mask directory", required=False) \\
        or find_one("**/[Ll]abels", "the mask directory")
print("images       :", img_dir)
print("masks        :", mask_dir)

IMAGES = {p.stem: p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
MASKS  = {p.stem: p for p in sorted(mask_dir.rglob("*")) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
print(f"found {len(IMAGES)} images, {len(MASKS)} masks")
"""))

# ---------------------------------------------------------------- 3. split
cells.append(md("""
## 3 · Load, verify and fingerprint the split

`crackssl.data.Split` raises on any train/val/test overlap, so this cell is also the
leakage guard. The printed fingerprint is the value every later notebook asserts.
"""))
cells.append(code("""
split = D.load_split(split_path)
FINGERPRINT = split.fingerprint()

print("counts      :", split.counts)
print("fingerprint :", FINGERPRINT, "  <-- assert this in R1/R2")

# every id must actually resolve to a file
all_ids = split.train + split.val + split.test
missing_img = [i for i in all_ids if i not in IMAGES]
missing_msk = [i for i in all_ids if i not in MASKS]
assert not missing_img, f"{len(missing_img)} ids have no image, e.g. {missing_img[:5]}"
assert not missing_msk, f"{len(missing_msk)} ids have no mask, e.g. {missing_msk[:5]}"
print(f"all {len(all_ids)} ids resolve to image+mask files")

(WORK / "split.json").write_text(json.dumps({
    "train": split.train, "val": split.val, "test": split.test,
    "fingerprint": FINGERPRINT, "source": str(split_path),
    "counts": split.counts,
}, indent=1))
print("wrote split.json")
"""))

# ---------------------------------------------------------------- 4. density
cells.append(md("""
## 4 · Crack density — the stratification variable

Crack-pixel fraction is heavy-tailed: an *unstratified* draw of 45 images routinely lands
far from the pool's ~11%. `data.nested_subsets` stratifies on this, so it has to be
computed once and cached.
"""))
cells.append(code("""
density = D.crack_density({i: MASKS[i] for i in all_ids})
json.dump(density, open(WORK / "density.json", "w"))

dens = pd.Series(density)
pool = dens.loc[split.train]
print(f"pool (train) crack fraction: mean {pool.mean():.4f}  median {pool.median():.4f}")
print(f"  min {pool.min():.4f}   max {pool.max():.4f}")
print(f"  images with zero crack pixels: {(pool == 0).sum()}")
print()
print("quintile edges used for stratification:")
for q, v in zip([0, 20, 40, 60, 80, 100], np.percentile(pool, [0, 20, 40, 60, 80, 100])):
    print(f"    p{q:<3} {v:.4f}")
"""))

cells.append(code("""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
axes[0].hist(pool * 100, bins=40, color=viz.PALETTE["blue"], edgecolor=viz.PALETTE["surface"])
axes[0].set_xlabel("crack pixels per image (%)"); axes[0].set_ylabel("images")
axes[0].set_title("Crack density is heavy-tailed", loc="left", color=viz.PALETTE["ink"])

for name, ids, c in [("train", split.train, viz.PALETTE["blue"]),
                     ("val",   split.val,   viz.PALETTE["orange"]),
                     ("test",  split.test,  viz.PALETTE["aqua"])]:
    s = np.sort(dens.loc[ids].values) * 100
    axes[1].plot(s, np.linspace(0, 1, len(s)), label=f"{name} (n={len(ids)})", color=c)
axes[1].set_xlabel("crack pixels per image (%)"); axes[1].set_ylabel("cumulative fraction")
axes[1].set_title("Splits are density-comparable", loc="left", color=viz.PALETTE["ink"])
axes[1].legend()
fig.tight_layout(); viz.savefig(fig, str(WORK / "r0_density.png")); plt.show()
"""))

# ---------------------------------------------------------------- 5. dhash audit
cells.append(md("""
## 5 · Leakage audit I — dHash threshold sensitivity

Part A grouped near-duplicates at Hamming ≤ 6. That cutoff was never justified empirically.
Here we recompute the grouping at 2 / 6 / 10 / 16 and report **how many cross-split pairs
each threshold would have caught**. If the count explodes above 6, the split is sensitive to
an arbitrary choice and the paper must say so.
"""))
cells.append(code("""
def dhash(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()

print("computing dHash for all images ...")
H = {i: dhash(IMAGES[i]) for i in all_ids}
ids_arr = np.array(all_ids)
bits = np.stack([H[i] for i in all_ids])            # [N, 64] bool

split_of = {i: "train" for i in split.train}
split_of.update({i: "val" for i in split.val})
split_of.update({i: "test" for i in split.test})
grp = np.array([split_of[i] for i in all_ids])

# pairwise Hamming via matrix product on +/-1 encoding
# +/-1 encoding turns Hamming distance into a matrix product:
#   B @ B.T = (#matches - #mismatches) = BITS - 2*hamming
# int16 (not int8) so the sentinel on the diagonal can never overflow.
B = bits.astype(np.int16) * 2 - 1
ham = ((bits.shape[1] - B @ B.T) // 2).astype(np.int16)
np.fill_diagonal(ham, 999)

rows = []
for thr in [2, 6, 10, 16]:
    close = np.argwhere(ham <= thr)
    close = close[close[:, 0] < close[:, 1]]
    cross = sum(1 for a, b in close if grp[a] != grp[b])
    rows.append({"hamming_threshold": thr,
                 "near_duplicate_pairs": len(close),
                 "cross_split_pairs": cross,
                 "images_involved": len(set(close.ravel().tolist()))})
audit = pd.DataFrame(rows)
audit.to_csv(WORK / "leakage_audit.csv", index=False)
display(audit)
print()
print("Reading: 'cross_split_pairs' at threshold 6 should be 0 by construction —")
print("Part A grouped at <=6. Non-zero counts at 10/16 are pairs that a stricter")
print("threshold would have separated; report them as a stated limitation.")
"""))

# ---------------------------------------------------------------- 6. nn probe
cells.append(md("""
## 6 · Leakage audit II — nearest-neighbour probe (the certificate figure)

The decisive check the council asked for. For every **test** image, find its nearest
**training** image in an embedding space, and record the distance. Later, in R3, we plot
that distance against the model's per-image IoU:

* **downward slope** → test images that resemble training images score higher → leakage is inflating the result
* **flat slope** → no near-duplicate advantage → this is the leakage-free certificate

Embeddings use DINOv2-small CLS tokens. It runs on CPU in a few minutes; if a GPU is
attached it is used automatically.
"""))
cells.append(code("""
import torch
from transformers import AutoImageProcessor, AutoModel

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
enc = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()

@torch.no_grad()
def embed(paths, bs=32):
    out = []
    for k in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[k:k + bs]]
        px = proc(images=ims, return_tensors="pt").to(dev)
        cls = enc(**px).last_hidden_state[:, 0, :]          # CLS token
        out.append(torch.nn.functional.normalize(cls, dim=1).cpu())
        print(f"\\r  embedded {min(k + bs, len(paths))}/{len(paths)}", end="")
    print()
    return torch.cat(out).numpy()

E_train = embed([IMAGES[i] for i in split.train])
E_test  = embed([IMAGES[i] for i in split.test])

sim = E_test @ E_train.T            # cosine similarity (rows are unit-norm)
nn_idx = sim.argmax(1)
nn_cos = sim.max(1)

probe = pd.DataFrame({
    "test_id": split.test,
    "nearest_train_id": [split.train[j] for j in nn_idx],
    "cosine_similarity": nn_cos,
    "cosine_distance": 1.0 - nn_cos,
    "test_crack_fraction": [density[i] for i in split.test],
})
probe.to_csv(WORK / "leakage_probe.csv", index=False)

print(f"\\nnearest-neighbour cosine similarity (test -> train):")
print(f"  mean {nn_cos.mean():.4f}   median {np.median(nn_cos):.4f}   max {nn_cos.max():.4f}")
print(f"  pairs above 0.95 similarity: {(nn_cos > 0.95).sum()} of {len(nn_cos)}")
display(probe.nlargest(5, "cosine_similarity"))
"""))

cells.append(code("""
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.hist(probe["cosine_distance"], bins=40, color=viz.PALETTE["blue"],
        edgecolor=viz.PALETTE["surface"])
ax.set_xlabel("cosine distance from test image to its nearest training image")
ax.set_ylabel("test images")
ax.set_title("How close is the test set to the training set?", loc="left",
             color=viz.PALETTE["ink"])
fig.tight_layout(); viz.savefig(fig, str(WORK / "r0_nn_distance.png")); plt.show()

print("A mass concentrated near zero would mean many test images have a near-twin in")
print("training. The IoU-vs-distance slope in R3 is what decides whether that mattered.")
"""))

# ---------------------------------------------------------------- 7. budgets
cells.append(md("""
## 7 · Label-budget preview

Sanity-check the subsets R1 will train on. Three properties must hold, and each is
printed below so a reviewer can see it was checked rather than assumed:

1. **nested** — every budget is a subset of the next
2. **stratified** — subset crack density tracks the pool
3. **validation carved from the budget**, never added on top (so the x-axis is the true annotation cost)
"""))
cells.append(code("""
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.00]

budgets = D.nested_subsets(
    pool_ids=split.train, density=density, fractions=FRACTIONS,
    seed=SEED, val_frac=0.2, n_strata=5,
)

pool_mean = float(np.mean([density[i] for i in split.train]))
rows = []
for f in FRACTIONS:
    b = budgets[f]
    sub = b.train_ids + b.val_ids
    rows.append({
        "fraction": f, "labels_total": b.n_total, "train": b.n_train, "val": b.n_val,
        "subset_density": np.mean([density[i] for i in sub]),
        "pool_density": pool_mean,
        "zero_crack_imgs": sum(1 for i in sub if density[i] == 0),
    })
display(pd.DataFrame(rows).round(4))

nested_ok = all(
    set(budgets[a].train_ids) | set(budgets[a].val_ids)
    <= set(budgets[b].train_ids) | set(budgets[b].val_ids)
    for a, b in zip(FRACTIONS, FRACTIONS[1:])
)
print("nested across fractions      :", nested_ok)
print("train/val disjoint everywhere:",
      all(not (set(budgets[f].train_ids) & set(budgets[f].val_ids)) for f in FRACTIONS))

# different seeds must give genuinely different subsets, not prefixes of one order
other = D.nested_subsets(split.train, density, [0.05], seed=SEED + 1)[0.05]
ov = len(set(budgets[0.05].train_ids) & set(other.train_ids))
print(f"overlap between seed {SEED} and {SEED+1} at 5%: {ov}/{budgets[0.05].n_train} images")
"""))

# ---------------------------------------------------------------- 8. summary
cells.append(md("## 8 · Summary — copy this block into R1"))
cells.append(code("""
from crackssl import metrics as M

summary = {
    "fingerprint": FINGERPRINT,
    "counts": split.counts,
    "pool_crack_density": round(pool_mean, 6),
    "fractions": FRACTIONS,
    "seed_used_for_preview": SEED,
    "n_images": len(IMAGES),
    "metric_provenance": M.PROVENANCE,
    "dataset_citation": (
        "Liang D., Kong X., Liu J., He X., Zhang Y., Xue X., Xu L., "
        "PaveCrack1300: A UAV-Acquired Pavement Crack Segmentation Dataset, "
        "Mendeley Data (2026), doi:10.17632/8b27pdcxv7, CC BY 4.0"
    ),
}
json.dump(summary, open(WORK / "r0_summary.json", "w"), indent=1)

print(json.dumps(summary, indent=1))
print()
print("=" * 68)
print("R0 COMPLETE — now: Save Version -> Save & Run All (Commit)")
print("=" * 68)
print("Outputs:", sorted(p.name for p in WORK.glob("*") if p.is_file()))
print()
print(f"Paste this into R1:   EXPECTED_FINGERPRINT = \\"{FINGERPRINT}\\"")
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
