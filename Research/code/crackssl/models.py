"""
Segmentation arms for the label-efficiency study.

One decoder (ASPP) is held fixed across every arm, so the only thing that varies is
how the encoder was initialised. That is the whole point of the controlled design:
if the decoder differed too, a difference in the curve would be uninterpretable.

Arms
----
``dinov2_ft``      DINOv2-base ViT-B/14, LVD-142M self-supervised, fine-tuned
``dinov2_frozen``  the same encoder, frozen (linear-probe regime)
``mae_ft``         MAE ViT-B/16, ImageNet-1k self-supervised, fine-tuned
``imagenet_sup``   ViT-B/16 supervised on ImageNet-1k -- the baseline reviewers ask for first
``random_init``    ViT-B/16 random weights -- the floor

Caveat to state in the paper: DINOv2 uses patch 14 (16x16 token grid at 224) while the
others use patch 16 (14x14 grid). The ASPP head is convolutional so it consumes either,
but the grids are not identical and that is a confound we disclose rather than hide.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ArmSpec", "ARMS", "ASPPHead", "SegModel", "build_arm"]


@dataclass(frozen=True)
class ArmSpec:
    name: str
    hf_id: str | None          # None -> randomly initialised
    frozen: bool
    pretrained: bool
    family: str                # for the paper's taxonomy column
    note: str


ARMS: dict[str, ArmSpec] = {
    "dinov2_ft": ArmSpec(
        "dinov2_ft", "facebook/dinov2-base", False, True,
        "self-distillation (foundation)",
        "LVD-142M, distilled from ViT-g; a foundation-scale upper bound, not an IN-1k peer",
    ),
    "dinov2_frozen": ArmSpec(
        "dinov2_frozen", "facebook/dinov2-base", True, True,
        "self-distillation (foundation)",
        "same weights as dinov2_ft; isolates adaptation from representation",
    ),
    "mae_ft": ArmSpec(
        "mae_ft", "facebook/vit-mae-base", False, True,
        "masked reconstruction",
        "ImageNet-1k, 1600 epochs",
    ),
    "imagenet_sup": ArmSpec(
        "imagenet_sup", "google/vit-base-patch16-224", False, True,
        "supervised transfer",
        "the first baseline a reviewer asks for",
    ),
    "random_init": ArmSpec(
        "random_init", None, False, False,
        "none",
        "performance floor; isolates what pretraining actually buys",
    ),
}


# --------------------------------------------------------------------------- #
# decoder -- identical for every arm
# --------------------------------------------------------------------------- #
class ASPPHead(nn.Module):
    """
    DeepLabV3-style Atrous Spatial Pyramid Pooling.

    Dilation rates are small (1, 3, 6, 9) because the token grid is only 14-16 px
    across; the standard (6, 12, 18) would exceed the grid extent and degenerate into
    1x1 convolutions.
    """

    #: GroupNorm, not BatchNorm, for two reasons -- one correctness, one scientific:
    #:  * the global-pooling branch emits [B, mid, 1, 1]; BatchNorm2d raises on a
    #:    batch of 1 ("expected more than 1 value per channel"), which would crash
    #:    any single-image evaluation.
    #:  * batch statistics estimated from 8 images over a 45-image training set are
    #:    noisy, and that noise would vary systematically along the label-efficiency
    #:    curve -- i.e. it would confound the very axis the study measures.
    #: GroupNorm is batch-independent, so the normalisation is identical at every
    #: label fraction.
    _GROUPS = 32

    def __init__(self, in_ch: int, n_classes: int = 2, mid: int = 256,
                 rates: tuple[int, ...] = (1, 3, 6, 9), dropout: float = 0.1):
        super().__init__()

        def norm(ch: int) -> nn.Module:
            return nn.GroupNorm(min(self._GROUPS, ch), ch)

        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_ch, mid, 3 if r > 1 else 1,
                          padding=r if r > 1 else 0, dilation=r, bias=False),
                norm(mid), nn.ReLU(inplace=True),
            )
            for r in rates
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid, 1, bias=False),
            norm(mid), nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid * (len(rates) + 1), mid, 1, bias=False),
            norm(mid), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(mid, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        feats.append(F.interpolate(self.pool(x), size=hw, mode="bilinear", align_corners=False))
        return self.classifier(self.project(torch.cat(feats, dim=1)))


# --------------------------------------------------------------------------- #
# encoder + token->grid adapter + decoder
# --------------------------------------------------------------------------- #
class SegModel(nn.Module):
    """
    ViT encoder -> token-to-grid adapter -> ASPP -> bilinear upsample to input size.

    The adapter is the part that makes a sequence model usable by a convolutional
    decoder: we drop the CLS token and fold the remaining N patch tokens back onto
    the sqrt(N) x sqrt(N) grid they came from. Without it the decoder would receive a
    1-D sequence and spatial convolution would be meaningless.
    """

    def __init__(self, encoder: nn.Module, embed_dim: int, n_classes: int = 2,
                 frozen: bool = False, n_prefix_tokens: int = 1):
        super().__init__()
        self.encoder = encoder
        self.head = ASPPHead(embed_dim, n_classes)
        self.frozen = frozen
        self.n_prefix_tokens = n_prefix_tokens

        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def _tokens_to_grid(self, tok: torch.Tensor) -> torch.Tensor:
        b, n, d = tok.shape
        side = int(round(n ** 0.5))
        if side * side != n:
            raise RuntimeError(
                f"{n} patch tokens is not a square grid -- check n_prefix_tokens "
                f"(currently {self.n_prefix_tokens}) for this encoder."
            )
        return tok.transpose(1, 2).reshape(b, d, side, side)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        if self.frozen:
            with torch.no_grad():
                out = self.encoder(pixel_values=x)
        else:
            out = self.encoder(pixel_values=x)

        tok = out.last_hidden_state[:, self.n_prefix_tokens:, :]
        logits = self.head(self._tokens_to_grid(tok))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)

    def param_groups(self, enc_lr: float, dec_lr: float) -> list[dict]:
        """
        Differential learning rates: a low LR on the pretrained encoder adapts the
        representation without erasing it; a high LR on the freshly initialised head.
        For a frozen arm the encoder group is empty, which AdamW handles fine.
        """
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        dec = list(self.head.parameters())
        groups = [{"params": dec, "lr": dec_lr, "name": "decoder"}]
        if enc:
            groups.insert(0, {"params": enc, "lr": enc_lr, "name": "encoder"})
        return groups


def build_arm(arm: str, n_classes: int = 2, img_size: int = 224) -> tuple[SegModel, ArmSpec]:
    """Construct one arm. Requires `transformers`; downloads the checkpoint on first use."""
    from transformers import AutoConfig, AutoModel, ViTMAEModel

    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    spec = ARMS[arm]

    if spec.hf_id is None:
        cfg = AutoConfig.from_pretrained("google/vit-base-patch16-224", image_size=img_size)
        encoder = AutoModel.from_config(cfg)
        dim, prefix = cfg.hidden_size, 1
    elif "vit-mae" in spec.hf_id:
        # MAE must run with mask_ratio=0 at transfer time: masking is a pretraining
        # objective, not an inference-time behaviour.
        encoder = ViTMAEModel.from_pretrained(spec.hf_id, mask_ratio=0.0)
        dim, prefix = encoder.config.hidden_size, 1
    else:
        encoder = AutoModel.from_pretrained(spec.hf_id)
        dim = encoder.config.hidden_size
        # DINOv2 emits CLS + (optional) register tokens before the patch tokens
        prefix = 1 + int(getattr(encoder.config, "num_register_tokens", 0) or 0)

    model = SegModel(encoder, dim, n_classes, frozen=spec.frozen, n_prefix_tokens=prefix)
    return model, spec
