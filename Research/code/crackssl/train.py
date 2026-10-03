"""
Fixed-iteration training, evaluation and the saving contract.

Three decisions here are protocol, not taste, and each fixes a specific defect found
in the earlier study:

1. **Fixed ITERATIONS, not epochs.** With an epoch budget a 45-label run gets 20x
   fewer gradient updates than a 909-label run, so the label-efficiency curve would
   partly measure schedule length. :func:`train_arm` takes ``iterations``.
2. **Model selection on VALIDATION, never test.** Selecting on test is what made the
   previous results unusable. The test split is touched exactly once, in
   :func:`evaluate`, after training ends.
3. **Losses computed in float32 under AMP.** Half-precision overflowed in the earlier
   run; every reduction here is done on ``.float()`` tensors inside an
   ``autocast(enabled=False)`` block.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["TrainConfig", "dice_ce_loss", "train_arm", "predict_probs", "evaluate", "save_run"]


@dataclass
class TrainConfig:
    arm: str
    fraction: float
    seed: int
    iterations: int = 2000
    batch_size: int = 8
    enc_lr: float = 1e-5
    dec_lr: float = 1e-3
    weight_decay: float = 1e-2
    crack_weight: float = 4.63
    warmup_frac: float = 0.05
    eval_every: int = 200
    amp: bool = True
    img_size: int = 224
    split_fingerprint: str = ""
    lib_version: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor,
                 crack_weight: float = 4.63, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice + class-weighted cross-entropy for the crack class.

    Computed entirely in float32: under AMP the soft-Dice denominator sums over every
    pixel in the batch, which overflows fp16 at 224^2 x batch 8.
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        lg = logits.float()
        w = torch.tensor([1.0, crack_weight], device=lg.device, dtype=torch.float32)
        ce = F.cross_entropy(lg, target.long(), weight=w)

        prob = lg.softmax(1)[:, 1]
        tgt = (target == 1).float()
        inter = (prob * tgt).sum()
        denom = prob.sum() + tgt.sum()
        dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
        return ce + dice


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _infinite(loader):
    while True:
        yield from loader


def train_arm(model, train_loader, val_loader, cfg: TrainConfig, device: str = "cuda",
              log_fn=print):
    """
    Train for exactly ``cfg.iterations`` optimiser steps, selecting on validation
    crack-IoU. Returns ``(best_state, history)``.
    """
    from .metrics import Confusion, confusion, iou

    model.to(device)
    opt = torch.optim.AdamW(
        model.param_groups(cfg.enc_lr, cfg.dec_lr), weight_decay=cfg.weight_decay
    )
    warm = max(1, int(cfg.warmup_frac * cfg.iterations))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, cfg.iterations - warm))),
    )
    scaler = torch.amp.GradScaler(device, enabled=cfg.amp and device == "cuda")

    best = {"val_iou": -1.0, "state": None, "iteration": -1}
    history: list[dict] = []
    stream = _infinite(train_loader)
    t0 = time.time()

    for it in range(1, cfg.iterations + 1):
        model.train()
        x, y = next(stream)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device, enabled=cfg.amp and device == "cuda"):
            logits = model(x)
        loss = dice_ce_loss(logits, y, cfg.crack_weight)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], 1.0
        )
        scaler.step(opt)
        scaler.update()
        sched.step()

        if it % cfg.eval_every == 0 or it == cfg.iterations:
            model.eval()
            tot = Confusion(0, 0, 0, 0)
            vloss = 0.0
            n = 0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    with torch.autocast(device, enabled=cfg.amp and device == "cuda"):
                        lv = model(xv)
                    vloss += float(dice_ce_loss(lv, yv, cfg.crack_weight))
                    n += 1
                    pred = lv.float().argmax(1).cpu().numpy()
                    tot = tot + confusion(pred == 1, yv.cpu().numpy() == 1)
            v_iou = iou(tot)
            history.append({
                "iteration": it, "train_loss": float(loss.detach()),
                "val_loss": vloss / max(n, 1), "val_crack_iou": v_iou,
                "lr_decoder": opt.param_groups[-1]["lr"],
                "minutes": (time.time() - t0) / 60,
            })
            log_fn(f"  it {it:>5}/{cfg.iterations}  loss {float(loss.detach()):.4f}  "
                   f"val crack-IoU {v_iou:.4f}" + ("  *" if v_iou > best["val_iou"] else ""))
            if np.isfinite(v_iou) and v_iou > best["val_iou"]:
                best = {
                    "val_iou": float(v_iou), "iteration": it,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }

    if best["state"] is None:                      # pragma: no cover - defensive
        best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best, history


# --------------------------------------------------------------------------- #
# inference and evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_probs(model, loader, device: str = "cuda", amp: bool = True) -> np.ndarray:
    """
    Crack-class probabilities for every image in ``loader``, as float16 ``[N,H,W]``.

    Saving probabilities rather than argmax masks is what makes threshold sweeps,
    average precision and calibration analysis possible later at zero GPU cost. The
    earlier study saved argmax only, which is why none of it could be re-analysed.
    """
    model.to(device).eval()
    out = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        with torch.autocast(device, enabled=amp and device == "cuda"):
            logits = model(x)
        out.append(logits.float().softmax(1)[:, 1].cpu().numpy().astype(np.float16))
    return np.concatenate(out, 0)


def tune_threshold(val_probs: np.ndarray, val_masks: np.ndarray,
                   grid: np.ndarray | None = None) -> tuple[float, "object"]:
    """
    Pick the decision threshold that maximises crack-IoU **on validation**.

    Without this, every arm is forced through a shared 0.5 cut and any difference in
    precision is confounded with calibration -- the objection raised against our
    earlier precision-collapse finding.
    """
    from .metrics import sweep_thresholds, best_threshold

    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.025), 3)
    sweep = sweep_thresholds(list(val_probs), list(val_masks), thresholds=grid)
    return best_threshold(sweep, "iou"), sweep


def evaluate(test_probs: np.ndarray, test_masks: np.ndarray, threshold: float,
             image_ids: list[str] | None = None, with_topology: bool = True):
    """
    Full metric set on the test split, at both the tuned threshold and 0.5, plus
    threshold-free average precision.

    Returns ``(summary_dict, per_image_dataframe)``.
    """
    import pandas as pd

    from .metrics import (Confusion, average_precision, betti0_error, boundary_iou,
                          cl_dice, confusion, dice, iou, precision, recall)

    ids = image_ids or [f"img_{i:04d}" for i in range(len(test_probs))]
    rows = []
    pooled_tuned = Confusion(0, 0, 0, 0)
    pooled_half = Confusion(0, 0, 0, 0)
    n_both_empty = n_one_empty = 0

    for sid, pr, gt in zip(ids, test_probs, test_masks):
        pr = np.asarray(pr, dtype=np.float32)
        gt = np.asarray(gt).astype(bool)
        pred_t = pr >= threshold
        pred_h = pr >= 0.5

        c_t = confusion(pred_t, gt)
        pooled_tuned = pooled_tuned + c_t
        pooled_half = pooled_half + confusion(pred_h, gt)

        row = {
            "image_id": sid,
            "gt_crack_fraction": float(gt.mean()),
            "iou": iou(c_t), "dice": dice(c_t),
            "precision": precision(c_t), "recall": recall(c_t),
            "ap": average_precision(pr, gt),
        }
        if with_topology:
            row["cl_dice"] = cl_dice(pred_t, gt)
            row["boundary_iou"] = boundary_iou(pred_t, gt, theta=2)
            row["betti0_error"] = betti0_error(pred_t, gt)
        if not pred_t.any() and not gt.any():
            n_both_empty += 1
        elif pred_t.any() != gt.any():
            n_one_empty += 1
        rows.append(row)

    per_image = pd.DataFrame(rows)
    summary = {
        "threshold_tuned": float(threshold),
        # dataset-level: counts pooled before the ratio
        "iou_dataset": iou(pooled_tuned),
        "dice_dataset": dice(pooled_tuned),
        "precision_dataset": precision(pooled_tuned),
        "recall_dataset": recall(pooled_tuned),
        # the same quantity at the naive shared cut, for the calibration comparison
        "iou_dataset_at_0.5": iou(pooled_half),
        "precision_dataset_at_0.5": precision(pooled_half),
        "recall_dataset_at_0.5": recall(pooled_half),
        # image-level: mean of per-image ratios (a DIFFERENT number -- state which you report)
        "iou_image_mean": float(np.nanmean(per_image["iou"])),
        "ap_image_mean": float(np.nanmean(per_image["ap"])),
        "n_images": len(per_image),
        "n_both_empty": n_both_empty,
        "n_exactly_one_empty": n_one_empty,
    }
    if with_topology:
        summary["cl_dice_image_mean"] = float(np.nanmean(per_image["cl_dice"]))
        summary["boundary_iou_image_mean"] = float(np.nanmean(per_image["boundary_iou"]))
        summary["betti0_error_image_mean"] = float(np.nanmean(per_image["betti0_error"]))
    return summary, per_image


# --------------------------------------------------------------------------- #
# the saving contract
# --------------------------------------------------------------------------- #
def save_run(out_dir: str | Path, cfg: TrainConfig, best: dict, history: list[dict],
             summary: dict, per_image, test_probs: np.ndarray,
             val_sweep=None, master_csv: str | Path | None = None,
             save_model: bool = True) -> Path:
    """
    Write the required artifacts, then append one row to the master table.

    ``save_model`` exists for a hard practical reason: a ViT-B checkpoint is ~330 MB,
    and the full matrix (175 runs) would be ~58 GB against Kaggle's ~20 GB output
    cap. Probabilities are what every later analysis actually needs (3.4 GB for the
    whole matrix), so the convention is to keep checkpoints for seed 0 only and rely
    on ``probs_test.npy`` elsewhere. Set it to True whenever you may want to re-run
    inference differently for that cell.
    """
    import pandas as pd

    run_id = f"{cfg.arm}__f{cfg.fraction:g}__s{cfg.seed}"
    d = Path(out_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)

    if save_model:
        torch.save(best["state"], d / "model_best.pt")                   # 1 (optional)
    np.save(d / "probs_test.npy", test_probs.astype(np.float16))         # 2 (always)
    per_image.to_csv(d / "per_image_metrics.csv", index=False)           # 3
    (d / "run_config.json").write_text(json.dumps(                       # 4
        {**asdict(cfg), "best_val_iou": best["val_iou"],
         "best_iteration": best["iteration"], "run_id": run_id}, indent=1))
    pd.DataFrame(history).to_csv(d / "history.csv", index=False)         # 5
    if val_sweep is not None:
        val_sweep.to_csv(d / "val_threshold_sweep.csv")
    (d / "summary.json").write_text(json.dumps(summary, indent=1))

    row = {"run_id": run_id, "arm": cfg.arm, "fraction": cfg.fraction,
           "seed": cfg.seed, "iterations": cfg.iterations,
           "n_labels": cfg.extra.get("n_labels"),
           "split_fingerprint": cfg.split_fingerprint,
           "best_val_iou": best["val_iou"], "has_checkpoint": bool(save_model),
           **summary}
    if master_csv is not None:                                           # 6
        mp = Path(master_csv)
        df = pd.DataFrame([row])
        df.to_csv(mp, mode="a", header=not mp.exists(), index=False)
    return d
