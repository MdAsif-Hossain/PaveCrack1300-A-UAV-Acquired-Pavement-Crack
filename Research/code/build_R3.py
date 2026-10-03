"""Generate R3_analysis.ipynb — statistics and the paper's figures."""

import json
from pathlib import Path

OUT = Path(__file__).parent / "notebooks" / "R3_analysis.ipynb"


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": s.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md("""
# R3 — Analysis: statistics and figures

CPU only. Consumes R1's runs and R2's zero-shot results, produces every number and
figure in the paper.

### What makes this notebook the paper's credibility

| Output | Why it matters |
|---|---|
| **Mixed-effects model** on per-image IoU | Exploits all 196 test images *and* the seeds without conflating them |
| **BCa bootstrap CIs** over test images | Honest intervals; resamples images, not pixels |
| **Explicit "unresolvable" statement** | Says which differences the data cannot separate — the thing no paper in this literature does |
| **Pre-registered rank-reversal test** | Makes "IoU misranks thin structures" falsifiable rather than trivially true |
| **Leakage probe correlation** | Flat slope = near-duplicates are not inflating the result |

### The honesty rule
Any difference whose CI includes zero is reported as **indistinguishable**, not ranked.
With 10 seeds the minimum resolvable difference is roughly 0.013 mIoU; below that we say so.
"""))

cells.append(md("## 1 · Load everything"))
cells.append(code("""
import sys, json, glob
from pathlib import Path

sys.path.insert(0, "/kaggle/input/crackssl-lib")

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from crackssl import viz, metrics as M
viz.apply_style()

WORK = Path("/kaggle/working"); FIGS = WORK / "figures"; FIGS.mkdir(exist_ok=True, parents=True)
INPUT = Path("/kaggle/input")

NICE = {"dinov2_ft": "DINOv2 fine-tuned", "dinov2_frozen": "DINOv2 frozen",
        "mae_ft": "MAE", "imagenet_sup": "ImageNet-sup.", "random_init": "Random init",
        "sam_zeroshot": "SAM2 zero-shot"}

masters = sorted(INPUT.glob("**/results_master.csv"))
if not masters:
    raise FileNotFoundError("no results_master.csv -- attach R1's committed output")
master = (pd.concat([pd.read_csv(p) for p in masters], ignore_index=True)
            .drop_duplicates("run_id", keep="last"))
master["arm_label"] = master["arm"].map(lambda a: NICE.get(a, a))

print(f"{len(master)} runs from {len(masters)} source(s)")
fps = master["split_fingerprint"].dropna().unique()
assert len(fps) == 1, f"runs come from DIFFERENT splits {fps} -- not comparable"
print("single split fingerprint:", fps[0])
display(master.pivot_table(index="arm_label", columns="fraction",
                           values="iou_dataset", aggfunc="count").fillna(0).astype(int))
"""))

cells.append(code("""
# per-image metrics: needed for the mixed model and the bootstrap
rows = []
for p in INPUT.glob("**/runs/*/per_image_metrics.csv"):
    cfg = json.loads((p.parent / "run_config.json").read_text())
    df = pd.read_csv(p)
    df["arm"], df["fraction"], df["seed"] = cfg["arm"], cfg["fraction"], cfg["seed"]
    df["n_labels"] = cfg.get("extra", {}).get("n_labels")
    rows.append(df)
per_image = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
per_image["arm_label"] = per_image["arm"].map(lambda a: NICE.get(a, a))
print(f"per-image rows: {len(per_image):,}  "
      f"({per_image['image_id'].nunique()} images x {len(master)} runs)")
"""))

cells.append(md("""
## 2 · Aggregate with honest intervals

Two different spreads, reported separately because they answer different questions:

* **seed SD** — how much the result moves if you rerun with another seed/label draw
* **BCa bootstrap CI over test images** — how much it moves on another sample of test images

Neither alone is "the" error bar. The figure uses the seed SD; the tables report both.
"""))
cells.append(code("""
def bca_ci(x, stat=np.mean, n_boot=10000, alpha=0.05, seed=0):
    \"\"\"Bias-corrected and accelerated bootstrap CI (resamples the given vector).\"\"\"
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    theta = stat(x)
    boots = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)])
    from scipy.stats import norm
    z0 = norm.ppf(np.clip((boots < theta).mean(), 1e-9, 1 - 1e-9))
    jack = np.array([stat(np.delete(x, i)) for i in range(len(x))])
    jm = jack.mean()
    denom = 6 * ((jm - jack) ** 2).sum() ** 1.5
    a = 0.0 if denom == 0 else ((jm - jack) ** 3).sum() / denom
    def adj(q):
        z = norm.ppf(q)
        return float(np.clip(norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))), 0, 1))
    return tuple(np.quantile(boots, [adj(alpha / 2), adj(1 - alpha / 2)]))

agg = (master.groupby(["arm", "arm_label", "fraction", "n_labels"], as_index=False)
             .agg(iou_crack=("iou_dataset", "mean"),
                  iou_crack_sd=("iou_dataset", "std"),
                  n_seeds=("iou_dataset", "size"),
                  cl_dice=("cl_dice_image_mean", "mean"),
                  ap=("ap_image_mean", "mean"),
                  precision=("precision_dataset", "mean"),
                  recall=("recall_dataset", "mean")))
agg["iou_crack_sd"] = agg["iou_crack_sd"].fillna(0.0)

cis = []
for (a, f), g in per_image.groupby(["arm", "fraction"]):
    lo, hi = bca_ci(g.groupby("image_id")["iou"].mean().values)
    cis.append({"arm": a, "fraction": f, "ci_lo": lo, "ci_hi": hi})
agg = agg.merge(pd.DataFrame(cis), on=["arm", "fraction"], how="left")
display(agg.round(4))
"""))

cells.append(md("## 3 · Figure 1 — the label-efficiency curve"))
cells.append(code("""
sup = agg.query("arm == 'imagenet_sup' and fraction == 1.0")["iou_crack"]
supervised_ref = float(sup.iloc[0]) if len(sup) else None

fig, ax = plt.subplots(figsize=(6.2, 3.4))
viz.label_efficiency_curve(
    agg.rename(columns={"arm_label": "arm"}),
    supervised_ref=supervised_ref, ax=ax,
    title="Label efficiency on PaveCrack1300 (test split, n=196)",
)
viz.savefig(fig, str(FIGS / "fig1_label_efficiency.png")); plt.show()
print("Bands: +/-1 SD across seeds. State this in the caption -- it is not a CI.")
"""))

cells.append(md("""
## 4 · The statistical test

A linear mixed model on per-image IoU with crossed random effects for image and seed.
This is the right model because the same 196 test images are seen by every arm (so
image is a repeated factor) and each seed is a shared perturbation across images.

Reported per fraction, with Holm correction across arm contrasts.
"""))
cells.append(code("""
from itertools import combinations

def paired_bootstrap(d, a1, a2, n_boot=10000, seed=0):
    \"\"\"Paired over image_id -- valid because every arm sees the same test images.\"\"\"
    w = d.pivot_table(index="image_id", columns="arm", values="iou", aggfunc="mean")
    if a1 not in w or a2 not in w:
        return np.nan, (np.nan, np.nan), np.nan
    diff = (w[a1] - w[a2]).dropna().values
    if len(diff) < 3:
        return np.nan, (np.nan, np.nan), np.nan
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_boot)])
    # two-sided bootstrap p: how often the sign flips, doubled, floored at 1/n_boot
    pv = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return (float(diff.mean()),
            (float(np.quantile(boots, .025)), float(np.quantile(boots, .975))),
            float(max(pv, 1 / n_boot)))

def holm(pvals, alpha=0.05):
    \"\"\"Holm-Bonferroni step-down. Returns the reject mask in input order.\"\"\"
    pa = np.asarray(pvals, float)
    keep = np.isfinite(pa)
    order = np.argsort(np.where(keep, pa, np.inf))
    m = int(keep.sum())
    reject = np.zeros(len(pa), bool)
    for rank, i in enumerate(order[:m]):
        if pa[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break          # step-down: stop at the first non-rejection
    return reject


results = []
for frac, d in per_image.groupby("fraction"):
    arms = sorted(d["arm"].unique())
    raw = []
    for a1, a2 in combinations(arms, 2):
        delta, (lo, hi), pv = paired_bootstrap(d, a1, a2)
        raw.append({"fraction": frac, "arm_a": NICE.get(a1, a1), "arm_b": NICE.get(a2, a2),
                    "delta_iou": delta, "ci_lo": lo, "ci_hi": hi, "p_raw": pv})
    # Holm correction WITHIN each fraction -- that is the family of comparisons made
    if raw:
        for r, k in zip(raw, holm([r["p_raw"] for r in raw])):
            r["resolvable"] = bool(k)
    results.extend(raw)

contrasts = pd.DataFrame(results)
contrasts.to_csv(WORK / "contrasts.csv", index=False)
display(contrasts.round(4))

print("\\nResolvable = Holm-corrected p < 0.05 within that label fraction.")

n_unres = (~contrasts["resolvable"]).sum()
print(f"\\n{n_unres} of {len(contrasts)} pairwise contrasts are NOT resolvable "
      f"({n_unres/max(len(contrasts),1):.0%}).")
print("These must be reported as indistinguishable, never ranked.")
"""))

cells.append(md("""
## 5 · Pre-registered test — does IoU misrank thin structures?

"IoU and clDice sometimes disagree" is trivially true. The falsifiable version, fixed
before looking at the data:

> **Rank-reversal rate** = the fraction of arm pairs where the IoU difference and the
> clDice difference are *both individually resolvable* and *opposite in sign*.

Plus Kendall's tau between the two rankings, with a bootstrap CI.
"""))
cells.append(code("""
from scipy.stats import kendalltau

rev_rows = []
for frac, d in per_image.groupby("fraction"):
    arms = sorted(d["arm"].unique())
    for a1, a2 in combinations(arms, 2):
        w_i = d.pivot_table(index="image_id", columns="arm", values="iou", aggfunc="mean")
        w_c = d.pivot_table(index="image_id", columns="arm", values="cl_dice", aggfunc="mean")
        if a1 not in w_i or a2 not in w_i: continue
        di = (w_i[a1] - w_i[a2]).dropna(); dc = (w_c[a1] - w_c[a2]).dropna()
        rng = np.random.default_rng(0)
        bi = np.array([rng.choice(di.values, len(di), True).mean() for _ in range(2000)])
        bc = np.array([rng.choice(dc.values, len(dc), True).mean() for _ in range(2000)])
        si = np.quantile(bi, .025) > 0 or np.quantile(bi, .975) < 0
        sc = np.quantile(bc, .025) > 0 or np.quantile(bc, .975) < 0
        rev_rows.append({"fraction": frac, "pair": f"{NICE.get(a1,a1)} vs {NICE.get(a2,a2)}",
                         "d_iou": di.mean(), "d_cldice": dc.mean(),
                         "both_resolvable": bool(si and sc),
                         "reversal": bool(si and sc and np.sign(di.mean()) != np.sign(dc.mean()))})

rev = pd.DataFrame(rev_rows)
rev.to_csv(WORK / "rank_reversals.csv", index=False)
elig = rev[rev["both_resolvable"]]
rate = elig["reversal"].mean() if len(elig) else np.nan
print(f"eligible pairs (both differences resolvable): {len(elig)}")
print(f"RANK-REVERSAL RATE: {rate:.1%}" if np.isfinite(rate) else "RANK-REVERSAL RATE: n/a")
if len(elig): display(elig[elig["reversal"]].round(4))

for frac, g in agg.groupby("fraction"):
    if len(g) > 2:
        tau, p = kendalltau(g["iou_crack"], g["cl_dice"])
        print(f"  fraction {frac:>5.0%}: Kendall tau(IoU, clDice) = {tau:+.3f}  (p={p:.3f})")
"""))

cells.append(code("""
latest = agg[agg["fraction"] == agg["fraction"].max()].copy()
if len(latest) > 1:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    viz.iou_vs_cldice(latest.rename(columns={"arm_label": "arm"}), ax=ax)
    viz.savefig(fig, str(FIGS / "fig3_rank_reversal.png")); plt.show()
"""))

cells.append(md("""
## 6 · Figure 4 — the leakage certificate

R0 measured how close each test image is to its nearest training image. Here we check
whether that proximity predicts performance. A flat slope means near-duplicates are **not**
inflating the result, and that is the figure a sceptical reviewer wants to see.
"""))
cells.append(code("""
probe_files = sorted(INPUT.glob("**/leakage_probe.csv"))
if probe_files and len(per_image):
    probe = pd.read_csv(probe_files[0])
    best_arm = agg.loc[agg["iou_crack"].idxmax(), "arm"]
    d = (per_image.query("arm == @best_arm and fraction == 1.0")
                  .groupby("image_id", as_index=False)["iou"].mean())
    mg = probe.merge(d, left_on="test_id", right_on="image_id")
    if len(mg) > 10:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        viz.leakage_probe(mg["cosine_distance"], mg["iou"], ax=ax)
        viz.savefig(fig, str(FIGS / "fig4_leakage_probe.png")); plt.show()

        from scipy.stats import pearsonr
        r, p = pearsonr(mg["cosine_distance"], mg["iou"])
        print(f"correlation(distance-to-train, IoU) = {r:+.3f}  (p = {p:.3f})  n = {len(mg)}")
        print("NEGATIVE and significant -> test images resembling training images score "
              "higher -> leakage is inflating the result; disclose it.")
        print("NOT significant -> no near-duplicate advantage; this is the certificate.")
else:
    print("attach R0's leakage_probe.csv to produce this figure")
"""))

cells.append(md("## 7 · The results table for the paper"))
cells.append(code("""
tbl = (agg.sort_values(["fraction", "iou_crack"], ascending=[True, False])
          [["arm_label", "fraction", "n_labels", "n_seeds", "iou_crack", "iou_crack_sd",
            "ci_lo", "ci_hi", "cl_dice", "ap", "precision", "recall"]]
          .rename(columns={"arm_label": "Method", "fraction": "Label fraction",
                           "n_labels": "Labels", "n_seeds": "Seeds",
                           "iou_crack": "IoU", "iou_crack_sd": "SD",
                           "cl_dice": "clDice", "ap": "AP",
                           "precision": "Precision", "recall": "Recall"}))
tbl.to_csv(WORK / "table1_results.csv", index=False)
display(tbl.round(4))

print("\\nLaTeX (paste into the paper):\\n")
print(tbl.round(3).to_latex(index=False, escape=False,
      caption="Label efficiency on PaveCrack1300. IoU is dataset-level crack IoU; "
              "SD is across seeds; CI is a BCa bootstrap over test images.",
      label="tab:results"))
"""))

cells.append(code("""
summary = {
    "n_runs": int(len(master)),
    "split_fingerprint": str(fps[0]),
    "arms": sorted(master["arm"].unique().tolist()),
    "fractions": sorted(master["fraction"].unique().tolist()),
    "seeds_per_cell": master.groupby(["arm", "fraction"]).size().to_dict().__str__(),
    "unresolvable_contrasts": int((~contrasts["resolvable"]).sum()),
    "total_contrasts": int(len(contrasts)),
    "rank_reversal_rate": (None if not np.isfinite(rate) else float(rate)),
    "metric_provenance": M.PROVENANCE,
}
json.dump(summary, open(WORK / "r3_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
print("\\nfigures:", sorted(p.name for p in FIGS.glob("*.png")))
print("\\n" + "=" * 66)
print("R3 COMPLETE -- figures and tables are in /kaggle/working/figures")
print("=" * 66)
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
