"""
Split handling and the label-subset protocol.

The subset sampler is the scientifically load-bearing part of this file. Three
properties are required for a label-efficiency curve to mean anything, and each was
raised as a defect in the council review:

1. **Resampled per seed.** A fixed subset at 45 images makes the dominant source of
   variance -- *which* 45 images you drew -- invisible, so the error bars understate
   reality badly. The seed therefore indexes the subset AND the init jointly.
2. **Nested within a seed.** 5% is a subset of 10% is a subset of 25%. Without
   nesting, the curve's slope mixes "more labels" with "different labels", and
   non-monotone points become uninterpretable.
3. **Stratified by crack density.** Crack pixel fraction is heavy-tailed; an
   unstratified draw of 45 images routinely lands far from the pool's ~11% density.
   We stratify by quintile and guarantee no all-background subset.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Split", "load_split", "crack_density", "LabelBudget", "nested_subsets", "fingerprint"]


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    """Immutable record of the train/val/test partition, with a fingerprint."""

    train: list[str]
    val: list[str]
    test: list[str]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlaps = {
            "train/val": set(self.train) & set(self.val),
            "train/test": set(self.train) & set(self.test),
            "val/test": set(self.val) & set(self.test),
        }
        bad = {k: sorted(v)[:5] for k, v in overlaps.items() if v}
        if bad:
            raise ValueError(f"split overlap detected: {bad}")

    @property
    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def fingerprint(self) -> str:
        """Stable hash of the exact membership -- print it in every notebook."""
        return fingerprint(self.train, self.val, self.test)


def fingerprint(*id_lists: list[str]) -> str:
    h = hashlib.sha256()
    for ids in id_lists:
        h.update(b"|")
        for i in sorted(ids):
            h.update(str(i).encode())
            h.update(b",")
    return h.hexdigest()[:16]


def load_split(path: str | Path) -> Split:
    """Load the frozen Part-A split. Never re-splits; raises on overlap."""
    d = json.loads(Path(path).read_text())
    return Split(
        train=list(d["train"]),
        val=list(d["val"]),
        test=list(d["test"]),
        meta={k: v for k, v in d.items() if k not in {"train", "val", "test"}},
    )


# --------------------------------------------------------------------------- #
# crack density -- the stratification variable
# --------------------------------------------------------------------------- #
def crack_density(mask_paths: dict[str, str | Path], reader=None) -> dict[str, float]:
    """
    Fraction of crack pixels per image. ``reader`` takes a path and returns a 2-D
    array; defaults to PIL. Cache the result -- it is used by every seed.
    """
    if reader is None:
        from PIL import Image

        def reader(p):  # noqa: ANN001
            return np.array(Image.open(p))

    out: dict[str, float] = {}
    for sid, p in mask_paths.items():
        m = np.asarray(reader(p))
        out[sid] = float((m > 0).mean())
    return out


# --------------------------------------------------------------------------- #
# the label budget
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabelBudget:
    """One point on the label-efficiency curve."""

    fraction: float
    n_total: int          # labelled images drawn from the pool
    n_train: int          # of those, used for gradient updates
    n_val: int            # of those, held out for selection
    train_ids: list[str]
    val_ids: list[str]

    @property
    def label_cost(self) -> int:
        """
        The honest x-axis: EVERY image a human had to annotate, including the ones
        spent on model selection. Reporting only ``n_train`` understates the cost,
        which was a specific flaw in the earlier protocol.
        """
        return self.n_total


def nested_subsets(
    pool_ids: list[str],
    density: dict[str, float],
    fractions: list[float],
    seed: int,
    val_frac: float = 0.2,
    n_strata: int = 5,
    min_positive: int = 1,
) -> dict[float, LabelBudget]:
    """
    Draw nested, density-stratified label budgets for one seed.

    Returns ``{fraction: LabelBudget}``. Guarantees:

    * ``subset(f1) subset-of subset(f2)`` for ``f1 < f2`` (nesting)
    * each subset's density histogram approximates the pool's (stratification)
    * at least ``min_positive`` images contain crack pixels
    * validation is carved FROM the budget, never added on top -- so the x-axis is
      the true annotation cost

    The seed drives both the ordering and the stratum interleave, so different seeds
    give genuinely different subsets rather than prefixes of one ordering.
    """
    rng = np.random.default_rng(seed)
    ids = sorted(pool_ids)
    if not ids:
        raise ValueError("empty pool")
    missing = [i for i in ids if i not in density]
    if missing:
        raise KeyError(f"{len(missing)} ids missing from density map, e.g. {missing[:3]}")

    # --- stratify the pool by crack-density quintile -------------------------
    dens = np.array([density[i] for i in ids])
    edges = np.quantile(dens, np.linspace(0, 1, n_strata + 1))
    edges[-1] += 1e-9
    strata: list[list[str]] = [[] for _ in range(n_strata)]
    for sid, d in zip(ids, dens):
        k = int(np.clip(np.searchsorted(edges, d, side="right") - 1, 0, n_strata - 1))
        strata[k].append(sid)

    # --- build ONE priority ordering that is balanced across strata ----------
    # Nesting falls out of taking prefixes of this single ordering.
    for s in strata:
        rng.shuffle(s)
    order: list[str] = []
    cursors = [0] * n_strata
    stratum_cycle = list(range(n_strata))
    rng.shuffle(stratum_cycle)
    while len(order) < len(ids):
        progressed = False
        for k in stratum_cycle:
            if cursors[k] < len(strata[k]):
                order.append(strata[k][cursors[k]])
                cursors[k] += 1
                progressed = True
        if not progressed:  # pragma: no cover - defensive
            break

    # --- guarantee at least `min_positive` crack-bearing images up front -----
    positives = [i for i in order if density[i] > 0]
    if len(positives) < min_positive:
        raise ValueError("pool has too few crack-bearing images")
    head = positives[:min_positive]
    order = head + [i for i in order if i not in set(head)]

    # --- take prefixes -------------------------------------------------------
    out: dict[float, LabelBudget] = {}
    for f in sorted(fractions):
        n_total = max(min_positive + 1, int(round(f * len(ids))))
        n_total = min(n_total, len(ids))
        chosen = order[:n_total]
        n_val = max(1, int(round(val_frac * n_total))) if n_total > 1 else 0
        val_ids = chosen[-n_val:] if n_val else []
        train_ids = chosen[: n_total - n_val]
        out[f] = LabelBudget(
            fraction=f,
            n_total=n_total,
            n_train=len(train_ids),
            n_val=len(val_ids),
            train_ids=train_ids,
            val_ids=val_ids,
        )
    return out
