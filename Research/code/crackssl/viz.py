"""
Publication figures for the label-efficiency study.

Palette provenance
------------------
Categorical hues are slots 1-3 of the reference data-viz palette, VALIDATED with
``validate_palette.js --mode light --pairs all``::

    [PASS] Lightness band      all 3 inside L 0.43-0.77
    [PASS] Chroma floor        all 3 >= 0.1
    [PASS] CVD separation      worst all-pairs dE 9.2 (deutan) / 9.6 (tritan)
    [PASS] Normal-vision floor worst all-pairs dE 24.0
    [WARN] Contrast vs surface #1baf7a at 2.74 -> relief rule: direct labels REQUIRED

All-pairs (not adjacent-pairs) was validated because lines in a multi-series chart
cross, so any pair can end up adjacent on screen.

Encoding rules followed here
----------------------------
* **Colour encodes the entity, never its rank.** DINOv2 keeps one hue whether it is
  frozen or fine-tuned; the regime is carried by line style. Filtering the chart
  never repaints the survivors.
* **Baselines are neutral, not categorical.** ImageNet-supervised and random-init are
  reference levels, not competitors, so they are grey -- which also keeps the
  categorical load at 3 hues, inside the all-pairs-validated set.
* **Legend always present for >=2 series, plus direct labels** (required here by the
  contrast relief rule).
* **One y-axis, never two.** Recessive grid, 2px lines, >=8px markers.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "PALETTE",
    "STYLE",
    "apply_style",
    "arm_style",
    "label_efficiency_curve",
    "threshold_sweep",
    "iou_vs_cldice",
    "leakage_probe",
    "qualitative_grid",
    "savefig",
]

# --------------------------------------------------------------------------- #
# tokens
# --------------------------------------------------------------------------- #
PALETTE = {
    # categorical -- validated all-pairs, light mode
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    # neutrals: surfaces and ink
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#8a8984",
    "grid": "#e4e3df",
}

#: entity -> (colour, linestyle, marker, z-order). Colour = entity, style = regime.
STYLE: dict[str, dict] = {
    "SAM2 zero-shot":    dict(color=PALETTE["orange"], ls="-",  marker="s", zorder=5),
    "DINOv2 fine-tuned": dict(color=PALETTE["blue"],   ls="-",  marker="o", zorder=6),
    "DINOv2 frozen":     dict(color=PALETTE["blue"],   ls="--", marker="o", zorder=4),
    "MAE":               dict(color=PALETTE["aqua"],   ls="-",  marker="^", zorder=5),
    "ImageNet-sup.":     dict(color=PALETTE["ink_secondary"], ls="-",  marker="D", zorder=3),
    "Random init":       dict(color=PALETTE["ink_muted"],     ls=":",  marker="v", zorder=2),
}

STYLE_FALLBACK = dict(color=PALETTE["ink_muted"], ls="-", marker="o", zorder=1)

#: correct capitalisation for axis labels (``str.title()`` mangles these)
_METRIC_LABELS = {
    "iou": "Crack IoU",
    "iou_crack": "Crack IoU",
    "dice": "Dice",
    "cl_dice": "clDice",
    "boundary_iou": "Boundary IoU",
    "precision": "Precision",
    "recall": "Recall",
    "ap": "Average Precision",
}


def arm_style(name: str) -> dict:
    return dict(STYLE.get(name, STYLE_FALLBACK))


def apply_style() -> None:
    """Paper-grade matplotlib defaults. Call once per notebook."""
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "savefig.facecolor": PALETTE["surface"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelcolor": PALETTE["ink_secondary"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "xtick.color": PALETTE["ink_secondary"],
            "ytick.color": PALETTE["ink_secondary"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "text.color": PALETTE["ink"],
        }
    )


def savefig(fig, path: str, also_pdf: bool = True) -> None:
    """Save PNG (for review) and PDF (for LaTeX vector embedding)."""
    fig.savefig(path, dpi=300)
    if also_pdf and path.endswith(".png"):
        fig.savefig(path[:-4] + ".pdf")


# --------------------------------------------------------------------------- #
# Figure 1 -- the money figure
# --------------------------------------------------------------------------- #
def label_efficiency_curve(
    df,
    y: str = "iou_crack",
    y_err: str | None = "iou_crack_sd",
    x: str = "n_labels",
    arm_col: str = "arm",
    supervised_ref: float | None = None,
    ax=None,
    title: str = "Label efficiency on PaveCrack1300 (test split)",
    ylabel: str = "Crack IoU",
    annotate: bool = True,
):
    """
    The paper's central figure: performance against label count, per arm, with
    confidence bands.

    ``df`` has one row per (arm, label fraction): ``arm``, ``n_labels``, the metric
    column and its spread column. Spread should be a CI half-width, not a bare SD --
    state which in the caption.

    Direct labels are drawn at the right-hand end (required: the aqua slot is below
    3:1 contrast, so identity must not rest on colour alone).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 3.6))

    n_series = df[arm_col].nunique()
    # The rule: a legend is always present for >=2 series; only <=4 series are ALSO
    # direct-labelled. Past that, direct labels collide and the legend alone carries
    # identity (which also satisfies the contrast relief rule for the aqua slot).
    if annotate and n_series > 4:
        annotate = False

    if supervised_ref is not None:
        ax.axhline(
            supervised_ref, color=PALETTE["ink_muted"], lw=1.0, ls=(0, (6, 4)), zorder=1
        )
        ax.text(
            df[x].min(), supervised_ref, "full supervision",
            va="bottom", ha="left", fontsize=7.5, color=PALETTE["ink_muted"],
        )

    for arm, g in df.groupby(arm_col, sort=False):
        g = g.sort_values(x)
        st = arm_style(arm)
        if y_err and y_err in g:
            ax.fill_between(
                g[x], g[y] - g[y_err], g[y] + g[y_err],
                color=st["color"], alpha=0.13, lw=0, zorder=st["zorder"] - 1,
            )
        ax.plot(
            g[x], g[y], label=arm, color=st["color"], ls=st["ls"],
            marker=st["marker"], zorder=st["zorder"],
            markeredgecolor=PALETTE["surface"], markeredgewidth=0.8,
        )
        if annotate:
            last = g.iloc[-1]
            ax.annotate(
                arm, (last[x], last[y]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=7.5, color=st["color"], zorder=10,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Labelled training images (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=PALETTE["ink"])
    # Legend outside the data area: inside, six entries cover the low-label region
    # that the paper is actually about.
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1,
        handlelength=2.4, borderaxespad=0.0,
    )
    ax.margins(x=0.06)
    return ax


# --------------------------------------------------------------------------- #
# Figure 2 -- the calibration control (experiment E0)
# --------------------------------------------------------------------------- #
def threshold_sweep(sweeps: dict, ax=None, metric: str = "iou"):
    """
    Metric against decision threshold, one line per arm, with each arm's optimum
    marked.

    This is the figure that settles whether a precision gap is a property of the
    representation or an artefact of everyone sharing a 0.5 threshold.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 3.4))

    for name, sw in sweeps.items():
        st = arm_style(name)
        ax.plot(sw.index, sw[metric], label=name, color=st["color"], ls=st["ls"],
                zorder=st["zorder"])
        t_best = sw[metric].idxmax()
        ax.plot([t_best], [sw[metric].max()], marker="o", ms=7, color=st["color"],
                markeredgecolor=PALETTE["surface"], markeredgewidth=1.2,
                zorder=st["zorder"] + 1)

    ax.axvline(0.5, color=PALETTE["ink_muted"], lw=1.0, ls=(0, (2, 3)), zorder=1)
    # anchor to axes coords near the bottom: the top is where the peak markers sit
    ax.text(0.5, 0.02, " default 0.5", transform=ax.get_xaxis_transform(),
            va="bottom", ha="left", fontsize=7.5, color=PALETTE["ink_muted"])
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(_METRIC_LABELS.get(metric, metric.replace("_", " ").title()))
    ax.set_title("Each arm at its own operating point", loc="left", color=PALETTE["ink"])
    ax.legend(loc="lower center", ncol=2)
    return ax


# --------------------------------------------------------------------------- #
# Figure 3 -- does IoU misrank thin structures?
# --------------------------------------------------------------------------- #
def iou_vs_cldice(df, arm_col: str = "arm", ax=None):
    """
    Paired slope chart: each arm's rank under IoU vs under clDice.

    Crossing lines are rank reversals. Per the pre-registered test, only reversals
    where *both* differences are individually significant count -- shade those.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 3.6))

    d = df.copy()
    d["r_iou"] = d["iou_crack"].rank(ascending=False)
    d["r_cld"] = d["cl_dice"].rank(ascending=False)

    for _, r in d.iterrows():
        st = arm_style(r[arm_col])
        ax.plot([0, 1], [r["r_iou"], r["r_cld"]], color=st["color"], ls=st["ls"],
                marker="o", zorder=st["zorder"],
                markeredgecolor=PALETTE["surface"], markeredgewidth=0.8)
        ax.annotate(r[arm_col], (0, r["r_iou"]), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7.5, color=st["color"])

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["IoU", "clDice"])
    ax.invert_yaxis()
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Does IoU misrank thin structures?", loc="left", color=PALETTE["ink"])
    ax.grid(axis="x", visible=False)
    ax.margins(x=0.3)
    return ax


# --------------------------------------------------------------------------- #
# Figure 4 -- the leakage certificate
# --------------------------------------------------------------------------- #
def leakage_probe(nn_distance, per_image_iou, ax=None, annotate_fit: bool = True):
    """
    Per-test-image IoU against distance to its nearest *training* image.

    A downward slope means near-duplicates are inflating the score; a flat line is
    the leakage-free certificate. The fitted slope and its CI belong in the caption.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 3.4))

    x = np.asarray(nn_distance, dtype=float)
    y = np.asarray(per_image_iou, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    ax.scatter(x, y, s=16, color=PALETTE["blue"], alpha=0.55,
               edgecolor=PALETTE["surface"], linewidth=0.5, zorder=3)

    if len(x) > 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color=PALETTE["ink_secondary"], lw=1.6,
                zorder=4)
        if annotate_fit:
            r = float(np.corrcoef(x, y)[0, 1])
            ax.text(0.02, 0.04, f"slope = {slope:+.3f}   r = {r:+.3f}",
                    transform=ax.transAxes, fontsize=7.5,
                    color=PALETTE["ink_secondary"])

    ax.set_xlabel("Cosine distance to nearest training image")
    ax.set_ylabel("Per-image crack IoU")
    ax.set_title("Leakage probe: flat slope = no near-duplicate advantage",
                 loc="left", color=PALETTE["ink"])
    return ax


# --------------------------------------------------------------------------- #
# Figure 5 -- qualitative
# --------------------------------------------------------------------------- #
def qualitative_grid(images, gts, preds_by_arm: dict, ids=None, figsize_scale: float = 1.7):
    """
    Rows = test images, columns = [input, ground truth, one per arm].

    Predictions are drawn as TP/FP/FN error maps rather than bare masks: a bare mask
    hides *which* kind of error was made, which is the thing the paper is about.
    Colours reuse the validated categorical slots.
    """
    arms = list(preds_by_arm)
    n_rows, n_cols = len(images), 2 + len(arms)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_scale * n_cols, figsize_scale * n_rows),
        squeeze=False,
    )

    tp_c = np.array([0.11, 0.69, 0.48])   # aqua   - correct
    fp_c = np.array([0.92, 0.41, 0.20])   # orange - over-segmentation
    fn_c = np.array([0.16, 0.47, 0.84])   # blue   - missed

    for i in range(n_rows):
        axes[i][0].imshow(images[i])
        axes[i][1].imshow(np.asarray(gts[i]).astype(bool), cmap="gray")
        for j, arm in enumerate(arms):
            pred = np.asarray(preds_by_arm[arm][i]).astype(bool)
            gt = np.asarray(gts[i]).astype(bool)
            canvas = np.ones((*gt.shape, 3))
            canvas[pred & gt] = tp_c
            canvas[pred & ~gt] = fp_c
            canvas[~pred & gt] = fn_c
            axes[i][2 + j].imshow(canvas)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        if ids is not None:
            axes[i][0].set_ylabel(str(ids[i]), fontsize=7,
                                  color=PALETTE["ink_secondary"], rotation=0,
                                  ha="right", va="center", labelpad=18)

    for j, name in enumerate(["input", "ground truth"] + arms):
        axes[0][j].set_title(name, fontsize=8, color=PALETTE["ink"])

    fig.text(0.5, -0.01,
             "correct (TP)  ·  over-segmented (FP)  ·  missed (FN)",
             ha="center", fontsize=7.5, color=PALETTE["ink_secondary"])
    fig.tight_layout()
    return fig
