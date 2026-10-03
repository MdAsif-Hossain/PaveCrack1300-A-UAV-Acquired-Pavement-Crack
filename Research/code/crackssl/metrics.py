"""
Segmentation metrics for thin-structure (crack) evaluation.

Design decisions are PINNED here because the council review flagged each as a place
where unstated choices silently change published numbers:

  * dataset-level vs image-level IoU are different quantities -> both are provided,
    and callers must say which they report.
  * clDice depends on the skeletonisation algorithm -> we pin
    ``skimage.morphology.skeletonize`` and record the skimage version in
    :data:`PROVENANCE`.
  * clDice is undefined when a mask is empty -> the convention is explicit in
    :func:`cl_dice` and the number of affected images is counted by the caller.
  * Thin-structure IoU is dominated by boundary thickness -> Boundary IoU is
    reported alongside, at a stated tolerance.

All functions take boolean / {0,1} arrays unless the name says ``prob``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import skimage
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

__all__ = [
    "PROVENANCE",
    "Confusion",
    "confusion",
    "iou",
    "dice",
    "precision",
    "recall",
    "average_precision",
    "cl_dice",
    "boundary_iou",
    "betti0_error",
    "sweep_thresholds",
    "best_threshold",
]

#: Recorded in every results file so a reviewer can reproduce the exact metric values.
PROVENANCE = {
    "skimage": skimage.__version__,
    "numpy": np.__version__,
    "skeletonizer": "skimage.morphology.skeletonize (default method)",
    "cldice_empty_convention": (
        "both empty -> 1.0 (perfect agreement); exactly one empty -> 0.0. "
        "Callers must report how many test images hit each case."
    ),
}

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# confusion matrix and the quantities derived from it
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Confusion:
    """Pixel counts for the positive (crack) class."""

    tp: int
    fp: int
    fn: int
    tn: int

    def __add__(self, other: "Confusion") -> "Confusion":
        return Confusion(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )

    @property
    def n_positive_gt(self) -> int:
        return self.tp + self.fn


def confusion(pred: np.ndarray, gt: np.ndarray) -> Confusion:
    """Pixel confusion counts for the crack class."""
    p = np.asarray(pred).astype(bool).ravel()
    g = np.asarray(gt).astype(bool).ravel()
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs gt {g.shape}")
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(p.size - tp - fp - fn)
    return Confusion(tp, fp, fn, tn)


def iou(c: Confusion) -> float:
    """Crack-class IoU. Returns nan when the class is absent from both."""
    denom = c.tp + c.fp + c.fn
    return float("nan") if denom == 0 else c.tp / denom


def dice(c: Confusion) -> float:
    denom = 2 * c.tp + c.fp + c.fn
    return float("nan") if denom == 0 else 2 * c.tp / denom


def precision(c: Confusion) -> float:
    denom = c.tp + c.fp
    return float("nan") if denom == 0 else c.tp / denom


def recall(c: Confusion) -> float:
    denom = c.tp + c.fn
    return float("nan") if denom == 0 else c.tp / denom


# --------------------------------------------------------------------------- #
# threshold-free
# --------------------------------------------------------------------------- #
def average_precision(prob: np.ndarray, gt: np.ndarray) -> float:
    """
    Area under the precision-recall curve, computed exactly over all distinct
    scores (no interpolation, no binning).

    This is the metric to quote when the claim is about the *representation*
    rather than about a particular operating point -- it is invariant to the
    decision threshold, which is exactly the confound raised against our
    earlier precision-collapse finding.
    """
    p = np.asarray(prob, dtype=np.float64).ravel()
    g = np.asarray(gt).astype(bool).ravel()
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: prob {p.shape} vs gt {g.shape}")
    n_pos = int(np.count_nonzero(g))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-p, kind="mergesort")
    g_sorted = g[order]
    p_sorted = p[order]

    tp_cum = np.cumsum(g_sorted)
    fp_cum = np.cumsum(~g_sorted)

    # keep only the last index of each run of equal scores: ties must be
    # resolved together or AP is inflated by score ordering within a tie.
    distinct = np.r_[np.nonzero(np.diff(p_sorted))[0], p_sorted.size - 1]
    tp_c = tp_cum[distinct]
    fp_c = fp_cum[distinct]

    prec = tp_c / np.maximum(tp_c + fp_c, _EPS)
    rec = tp_c / n_pos
    # step integral: sum over recall increments
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


# --------------------------------------------------------------------------- #
# topology / thin-structure aware
# --------------------------------------------------------------------------- #
def _skel(mask: np.ndarray) -> np.ndarray:
    return skeletonize(np.asarray(mask).astype(bool))


def cl_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    centerlineDice (clDice), Shit et al., CVPR 2021.

        T_prec = |skel(pred) & gt| / |skel(pred)|
        T_sens = |skel(gt) & pred| / |skel(gt)|
        clDice = 2 * T_prec * T_sens / (T_prec + T_sens)

    Empty-mask convention (PINNED, see :data:`PROVENANCE`):
      * both empty            -> 1.0
      * exactly one empty     -> 0.0

    Caveat worth stating in the paper: clDice is biased toward thicker
    structures -- a 1-2px centreline offset is fatal below ~4px width, which is
    precisely the crack regime. Report it *alongside* Boundary IoU, never alone.
    """
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    p_any, g_any = p.any(), g.any()
    if not p_any and not g_any:
        return 1.0
    if not p_any or not g_any:
        return 0.0

    sp, sg = _skel(p), _skel(g)
    n_sp, n_sg = int(sp.sum()), int(sg.sum())
    if n_sp == 0 or n_sg == 0:
        return 0.0

    t_prec = np.count_nonzero(sp & g) / n_sp
    t_sens = np.count_nonzero(sg & p) / n_sg
    if t_prec + t_sens == 0:
        return 0.0
    return float(2 * t_prec * t_sens / (t_prec + t_sens))


def boundary_iou(pred: np.ndarray, gt: np.ndarray, theta: int = 2) -> float:
    """
    Boundary IoU (Cheng et al., CVPR 2021) at pixel tolerance ``theta``.

    Restricts the IoU to a band of width ``theta`` inside each mask's boundary,
    so the score reflects contour agreement rather than interior area. For thin
    structures a crack narrower than 2*theta is entirely boundary, which is the
    point: it stops the metric being dominated by predicted thickness.
    """
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0

    def band(mask: np.ndarray) -> np.ndarray:
        # distance from each interior pixel to the nearest background pixel
        dist = ndi.distance_transform_edt(mask)
        return mask & (dist <= theta)

    bp, bg = band(p), band(g)
    inter = np.count_nonzero(bp & bg)
    union = np.count_nonzero(bp | bg)
    return float("nan") if union == 0 else inter / union


def betti0_error(pred: np.ndarray, gt: np.ndarray, connectivity: int = 2) -> int:
    """
    |beta0(pred) - beta0(gt)| -- absolute difference in the number of connected
    components. A proxy for topological correctness: a prediction that severs one
    crack into three fragments scores well on IoU but badly here.
    """
    structure = ndi.generate_binary_structure(2, connectivity)
    n_p = ndi.label(np.asarray(pred).astype(bool), structure=structure)[1]
    n_g = ndi.label(np.asarray(gt).astype(bool), structure=structure)[1]
    return int(abs(n_p - n_g))


# --------------------------------------------------------------------------- #
# threshold analysis -- the experiment that tests the calibration confound
# --------------------------------------------------------------------------- #
def sweep_thresholds(
    probs: list[np.ndarray] | np.ndarray,
    gts: list[np.ndarray] | np.ndarray,
    thresholds: np.ndarray | None = None,
    with_topology: bool = False,
) -> "object":
    """
    Pooled (dataset-level) precision / recall / IoU / Dice across a threshold sweep.

    ``probs`` and ``gts`` are per-image arrays. Counts are accumulated over the
    whole set before the ratio is taken, so this is *dataset-level* IoU -- state
    that in the paper; image-mean IoU is a different number.

    Returns a pandas DataFrame indexed by threshold.
    """
    import pandas as pd

    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

    probs = list(probs)
    gts = list(gts)
    rows = []
    for t in thresholds:
        total = Confusion(0, 0, 0, 0)
        cld: list[float] = []
        for pr, g in zip(probs, gts):
            pred = np.asarray(pr) >= t
            total = total + confusion(pred, g)
            if with_topology:
                cld.append(cl_dice(pred, g))
        row = {
            "threshold": float(t),
            "precision": precision(total),
            "recall": recall(total),
            "iou": iou(total),
            "dice": dice(total),
            "tp": total.tp,
            "fp": total.fp,
            "fn": total.fn,
        }
        if with_topology:
            row["cl_dice"] = float(np.nanmean(cld))
        rows.append(row)
    return pd.DataFrame(rows).set_index("threshold")


def best_threshold(sweep, metric: str = "iou") -> float:
    """Threshold maximising ``metric`` in a :func:`sweep_thresholds` result."""
    return float(sweep[metric].idxmax())
