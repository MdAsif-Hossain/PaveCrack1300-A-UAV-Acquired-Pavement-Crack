# 🎓 Viva Questions — Answered in Plain English

This document answers **every** question from Section 6 of the assignment (`Sec 4 - Assignment Part A.pdf`) —
both the **coding questions (§6.1)** and the **theory questions (§6.2)** — in simple, clear language, tied to
what we actually did in the notebooks.

> How to read this: each answer is short and self-contained. Where a question is about our code, we point to the
> exact choice we made and *why*.

---

## Part A — Coding Questions (§6.1)

### A.1 Data Pipeline (Notebooks 0–1)

**Q: How did you load and parse the mask format for your dataset?**
Our masks are single-channel PNGs where a pixel is `255` for crack and `0` for background. We open each mask with
PIL (`np.array(Image.open(path))`) and convert it to class IDs with one line: `(mask > 0).astype("uint8")` — this
maps `255 → 1` (crack) and `0 → 0` (background). We verified (by reading a sample of masks) that the only pixel
values present are `{0, 255}`, so this remap is exact.

**Q: Walk through your train/val/test split. How did you verify no image appears in two splits? What seed, and why?**
The dataset gives no "which photo did this patch come from" ID, and patches from the same road look alike, so a naive
random split could leak. We do two things: (1) build **groups** by perceptual hashing — any near-duplicate images are
forced into the same split; (2) split the *groups* 70/15/15, **stratified by crack density** so each split has a
similar crack distribution. We fix `seed = 42` (via `np.random.RandomState(42)`) so the split is reproducible. We
**assert** the three sample-ID sets have no intersection (`train & val`, `train & test`, `val & test` are all empty)
and that their sizes sum to 1,300 — and we print "LEAKAGE CHECK PASSED". The split is saved once to `split.json` and
reused by all three model notebooks.

**Q: Explain each Albumentations transform — what it does, why you included it, and its probability.**
- `HorizontalFlip` / `VerticalFlip` / `RandomRotate90` (p=0.5 each): mirror/rotate the patch. Because the view is
  top-down, there is no "correct" orientation, so these are safe and multiply our effective data ~8×.
- `ShiftScaleRotate` (p=0.5): small translate/zoom/rotate → the model learns position- and scale-invariance.
- `RandomBrightnessContrast` (p=0.5) and `RandomGamma` (p=0.3): change lighting/tone → robustness to different flights
  and sun angles.
- `GaussNoise` (p=0.2) and `GaussianBlur` (p=0.2): mild sensor noise and motion/defocus blur from a moving drone.
- `Normalize` + `ToTensorV2` (always): ImageNet mean/std (because backbones are ImageNet-pretrained) and conversion to
  a tensor.
Probabilities are kept modest so thin cracks are never destroyed.

**Q: How does your `__getitem__` ensure the image and mask get the *same* random transform every call?**
We pass both into **one** Albumentations call: `self.augment(image=img, mask=msk)`. Albumentations applies each
*geometric* transform with the *same* sampled parameters to both the image and the mask, and applies *photometric*
transforms (brightness, blur, noise) to the **image only**. Because it is a single call, there is no way for the image
and mask to get different geometry — they stay pixel-aligned.

**Q: What does your sanity-check visualization confirm, and what would a bug look like?**
It draws real training samples *after augmentation* as image | mask | red-overlay. It confirms three things: (1) the
crack in the image lines up with the mask even after flips/rotations, (2) the class→colour mapping is consistent, and
(3) augmentations didn't wipe out thin cracks. A bug would show the red overlay *not* sitting on the visible crack
(image and mask desynchronised) or a mask with grey/blurred values (wrong interpolation on the mask).

### A.2 DeepLabV3 (Notebook 2)

**Q: Which torchvision constructor did you use and why? What does the backbone choice trade off?**
`deeplabv3_resnet50(weights=DEFAULT, aux_loss=True)` — COCO-pretrained. ResNet-50 is the "balanced default": more
accurate than MobileNetV3, cheaper than ResNet-101, and fits a Kaggle T4 at 512×512. Bigger backbone = more accuracy
but more memory/time; smaller = faster but weaker on thin structures.

**Q: How did you adapt the pretrained classifier head for your class count?**
The pretrained head predicts 21 COCO classes. We replace the final 1×1 conv of both the main and auxiliary heads with
a new `nn.Conv2d(256, 2, 1)` (2 = background, crack). Only those two small layers are re-initialized; everything else
keeps its pretrained weights.

**Q: What is the ASPP module and why does it help multi-scale understanding?**
ASPP (Atrous Spatial Pyramid Pooling) runs several **dilated** convolutions *in parallel* at different dilation rates,
plus a global image-pooling branch. Each rate "sees" a different amount of context, so the network captures both fine
hairline cracks and wide alligator cracking in one shot — exactly the multi-scale nature of pavement cracks.

**Q: What loss did you use? If you weighted classes, how were the weights computed?**
`Dice + class-weighted CrossEntropy`. The CE class weights come from the **pixel** histogram via median-frequency
balancing: `weight_c = median(freqs) / freq_c`, which gives crack ≈ 4.6 (we cap it to avoid gradient spikes). Dice
handles the imbalance directly by optimizing overlap; DeepLab also adds `0.4 ×` the same loss on the auxiliary head.

**Q: What optimizer and LR schedule, and what does the schedule do each step?**
AdamW (lr 1e-4, weight-decay 1e-2) with `SequentialLR`: a **LinearLR warmup** for the first 3 epochs (LR ramps from
0.1× up to the base LR), then **CosineAnnealing** (LR decays smoothly along a cosine to near zero). Warmup stabilizes
early training from the pretrained weights; cosine decay lets it settle into a good minimum.

**Q: Walk through your training loop — how do you switch `model.train()` / `model.eval()`, and why does it matter?**
In each epoch we call `model.train()` before the training loop and `model.eval()` (inside `torch.no_grad()`) for
validation/test. `train()` enables BatchNorm to update running stats and dropout to be active; `eval()` freezes BN to
its learned stats and disables dropout. Getting this wrong makes validation numbers noisy and wrong.

### A.3 SegFormer-B0 (Notebook 3)

**Q: Which pretrained checkpoint did you load and why that domain?**
`nvidia/segformer-b0-finetuned-ade-512-512` (ADE20K, general scenes). ADE20K is broad and its head transfers well as a
starting point; it is also natively trained at 512×512, matching our patch size.

**Q: Explain `ignore_mismatched_sizes=True` — which layers are re-initialized and why?**
The ADE20K head predicts 150 classes; we need 2. That flag tells HuggingFace to **keep** every weight that matches and
**re-initialize only the final classifier** (`decode_head.classifier`, shape 150→2). The load report confirms exactly
those weights are reinit; the whole MiT encoder and decoder are kept.

**Q: What is the MiT-B0 encoder and how does it differ from a standard ViT?**
MiT (Mix Transformer) is a *hierarchical* transformer: it produces feature maps at 4 shrinking resolutions (like a CNN),
uses **overlapping** patch embeddings (so patch borders aren't lost), uses **efficient self-attention** (it reduces the
key/value spatial size so attention is cheaper), and has **no positional embeddings** (it uses a small conv inside the
FFN instead, so any input size works). A standard ViT is single-scale, non-overlapping patches, full attention, and
fixed positional embeddings.

**Q: What is the All-MLP decoder and how does it fuse multi-stage features?**
It is a lightweight head: each of the 4 encoder feature maps is passed through a simple **linear (MLP)** layer to a
common channel size, upsampled to the same resolution, **concatenated**, and fused by another MLP to produce the class
logits. No heavy convolutions — it relies on the encoder already carrying strong global context.

**Q: How did you handle the HuggingFace loss vs computing your own?**
We compute **our own** Dice+CE loss on the logits (we do *not* pass `labels=` to the model). This is deliberate: it
makes the loss byte-identical to the DeepLab notebook, keeping the benchmark fair.

**Q: How did you resize/pad images, and was that in the model or the dataloader?**
Our images are already 512×512, so no resizing is needed in the dataloader. SegFormer internally outputs logits at
**¼ resolution** (128×128); we upsample them back to 512 with `F.interpolate(..., mode="bilinear")` in our code (not
inside the model) before computing loss/metrics.

### A.4 YOLOv26-Sem (Notebook 4)

**Q: What is the difference between a `-sem` and a `-seg` checkpoint, and why does the wrong one change the task?**
`-seg` is **instance** segmentation (separate mask per detected object, with boxes). `-sem` is **semantic**
segmentation (one class label per pixel, no instances). Ultralytics picks the task from the checkpoint name, so loading
`-seg` would train object masks with boxes — the wrong problem entirely. We use `yolo26s-sem.pt`.

**Q: What does your `data.yaml` contain? Walk through each field.**
`path` (dataset root), `train`/`val`/`test` (image folders relative to root), `masks_dir` (the folder holding the
matching masks), and `names` (the class-ID → name map: `0: background, 1: crack`). YOLO finds each image's mask by
swapping the `images` part of the path for `masks_dir` and matching the filename stem.

**Q: What mask format does YOLOv26-Sem expect, and how did you verify it?**
PNG masks where **pixel value = class ID** and **255 = ignore**. Our source crack value is 255, which would be read as
"ignore", so we **remap 255 → 1** when exporting. We then assert the exported masks contain only `{0, 1}` before
training (shown in the conversion-check figure).

**Q: What does `imgsz` control and how did you pick it?**
`imgsz` is the resolution YOLO resizes inputs to for training/inference. We set `imgsz=512` = the native patch size, so
we neither upscale (wasting compute) nor downscale (losing thin cracks).

**Q: How do you read back `result.semantic_mask.data` and turn it into mIoU?**
After `model.predict(image)`, `result.semantic_mask.data` is a dense class map (H×W of class IDs). We resize it to the
ground-truth size if needed, then feed prediction+GT into the **same `ConfMat`** used for the other two models, which
yields mIoU, per-class IoU, Dice, pixel accuracy, etc.

**Q: What optimizer/LR schedule does Ultralytics use by default, and did you override anything?**
Ultralytics auto-selects the optimizer (SGD/AdamW) and uses its own LR schedule (`lr0=0.01` with warmup + decay). We
kept those defaults but set `fliplr=0.5, flipud=0.5` (top-down flips) and turned **HSV colour jitter off** (asphalt is
near-grayscale), plus `seed=42`.

### A.5 Error Analysis & Comparison (Notebook 4, Task H)

**Q: Which model got the best mIoU and why do you think it won here?**
**DeepLabV3-ResNet50 (mIoU 0.8383).** Its ASPP captures multi-scale context and its ResNet-50 backbone has the most
capacity to model thin, branching cracks — a good fit for this data.

**Q: Which class was hardest across all three models, and why?**
**Crack.** It is the minority class (~11% of pixels), often only 1–3 pixels wide, low-contrast, and has ambiguous
boundaries — so it's intrinsically hard to segment precisely.

**Q: Do the three models fail on the same images or different ones? What does that tell you?**
The **same** ones. Their per-image crack IoU correlates **0.89–0.94**, and **12 of the 20 hardest images are shared by
all three**. That means the residual error is driven by the *data* (genuinely hard images), not by any single model's
architecture — the three inductive biases don't disagree much on what's hard.

**Q: If you had to improve the worst model, what would you change?**
For SegFormer-B0 (lowest here): (1) move to a bigger encoder (B2) or train at higher resolution to recover thin-crack
detail, (2) add a **boundary-aware / Tversky** loss term to push recall on thin structures, and (3) train longer with a
lower final LR. Data-side: the shared-failure finding says better/denser labels on the hard images would help *all*
models.

---

## Part B — Theory Questions (§6.2)

### B.1 Semantic-Segmentation Fundamentals

**Semantic vs detection vs instance vs panoptic?** Object **detection** draws boxes around things. **Semantic**
segmentation labels *every pixel* with a class but does not separate individual objects (all cracks are just "crack").
**Instance** segmentation gives a separate mask per object. **Panoptic** = semantic + instance combined (every pixel
gets a class, and countable objects also get instance IDs). Ours is **semantic**.

**IoU and why mIoU beats pixel accuracy on imbalanced data.** IoU (Jaccard) for a class = `TP / (TP + FP + FN)` — the
overlap between prediction and truth divided by their union. **mIoU** averages IoU over classes. Pixel accuracy is
misleading here: a model that predicts "background everywhere" scores ~89% pixel accuracy on our data while completely
missing every crack; its crack IoU (and thus mIoU) would be near 0, which correctly exposes the failure.

**Dice coefficient and its link to F1 and IoU.** `Dice = 2·TP / (2·TP + FP + FN)`. Dice is exactly the **F1 score** at
the pixel level (harmonic mean of precision and recall). Dice and IoU are monotonically related:
`IoU = Dice / (2 − Dice)`, so ranking by one matches ranking by the other, but Dice gives more weight to overlap.

**Pixel-level vs image-level confusion matrix.** A *pixel-level* matrix counts every pixel's predicted-vs-true class, so
one image contributes ~262k entries and boundary errors show up. An *image-level* matrix only asks "did the image
contain crack, yes/no" — it hides where and how much the model got wrong. Segmentation needs the pixel-level version.

**The "ignore index" (255).** Some pixels are unlabeled or don't-care (borders, ambiguous regions). Marking them with a
special value like 255 tells the loss to **skip** them, so the model isn't penalized or rewarded for guessing on pixels
that have no reliable label. (Our data has no ignore pixels, but YOLO uses this convention.)

### B.2 Encoder–Decoder Architectures

**Encoder–decoder structure.** The **encoder** downsamples the image into small, rich feature maps that capture *what*
is present (semantics) but lose spatial detail. The **decoder** upsamples those features back to full resolution,
recovering *where* each thing is, to produce a per-pixel map.

**Skip connections (U-Net).** They copy high-resolution features from early encoder layers directly to the matching
decoder layer. This restores fine spatial detail (edges, thin structures) that was lost during downsampling, which is
critical for hairline cracks.

**Atrous (dilated) convolution.** A convolution with gaps between its sampled pixels (dilation rate `r` inserts `r−1`
holes). It enlarges the **receptive field** (how much context a filter sees) *without* adding parameters or lowering
resolution — you see more context for free.

**ASPP and why parallel dilation rates.** ASPP applies several atrous convs at different rates at once, then combines
them. Different rates capture objects of different sizes simultaneously, so one module handles both narrow and wide
cracks.

**Output stride in DeepLabV3.** Output stride = input size ÷ final feature-map size (e.g., 16 or 8). A **smaller**
output stride (8) keeps larger, more detailed feature maps → sharper segmentation, but costs more memory and compute; a
larger one is cheaper but coarser.

### B.3 Transformer-Based Segmentation (SegFormer)

**Vision Transformer (ViT).** It cuts the image into fixed patches (e.g., 16×16), flattens each into a vector ("token"),
adds positional embeddings, and processes the sequence with transformer self-attention — every patch can attend to
every other patch, giving global context.

**MiT vs standard ViT.** MiT (used by SegFormer) is **hierarchical** (4 scales, like a CNN), uses **overlapping** patch
embeddings (no lost borders), **efficient self-attention** (downsamples keys/values to cut cost), and **no positional
embeddings** (a conv inside the FFN provides position info, so any resolution works). Standard ViT is single-scale,
non-overlapping, full (expensive) attention, with fixed positional embeddings.

**Why an all-MLP decoder instead of a heavy CNN decoder.** Because MiT's attention already provides large receptive
fields at every stage, the decoder doesn't need heavy convolutions to gather context. A few MLP layers to unify,
upsample and fuse the 4 feature scales are enough — making SegFormer small, fast, and accurate.

**Multi-head self-attention on patches; what the attention map shows.** Each patch produces query/key/value vectors;
attention weights every patch by how relevant it is to the current patch, and multiple "heads" learn different relations
in parallel. An attention map visualizes, for a chosen patch, which other patches it "looks at" — e.g., a patch on a
crack attending along the crack's length.

**What "B0" means; how B2/B5 differ.** B0 is the smallest SegFormer size (fewest layers/channels — ~3.7 M params).
B2, B5 are progressively larger: more capacity and accuracy, but more memory and slower. The encoder size is fixed at
B0 by this assignment.

### B.4 YOLO-Based Semantic Segmentation

**How YOLO (a detector) is adapted for semantic segmentation.** Instead of a head that outputs boxes + class scores, the
`-sem` variant adds a segmentation head that outputs a **per-pixel class map** at input resolution, trained with a
pixel-wise loss — so the same fast backbone/neck now produces dense labels rather than boxes.

**C2f block and gradient flow.** C2f (Cross-Stage-Partial with 2 convs) splits features into two paths — one processed
by several bottleneck blocks, one passed through — then concatenates. The multiple shortcut paths give gradients more
routes back, improving flow and letting the network go deeper without vanishing gradients, at lower compute.

**YOLO neck (PAN/FPN) and why multi-scale fusion matters.** The neck (Feature Pyramid + Path Aggregation Network) mixes
features top-down *and* bottom-up so every scale carries both fine detail and high-level semantics. Segmentation needs
this because cracks appear at many widths — fusing scales lets one map handle all of them.

**Why `imgsz=1024` by default, and the trade-off of a smaller value.** YOLO26-Sem was trained on Cityscapes at 1024,
which suits large road scenes. A smaller `imgsz` is faster and uses less memory but blurs/erases very thin structures.
For us the native patch is 512, so we fine-tune at **512** — matching the data rather than upscaling.

**`-sem` vs `-seg` output and why metrics differ.** `-sem` outputs one class label per pixel → evaluated with mIoU/Dice
(pixel overlap). `-seg` outputs per-instance masks + boxes → evaluated with mask **mAP** (instance detection quality).
Different outputs need different metrics.

### B.5 Training & Optimization

**Cross-entropy for segmentation (pixel-wise).** For each pixel it is `−Σ_c y_c · log(p_c)` (true class one-hot `y`,
predicted softmax probabilities `p`); the loss is averaged over all pixels. It pushes the predicted probability of the
true class toward 1.

**Focal loss — when and why.** Focal loss = CE scaled by `(1 − p_true)^γ`, which **down-weights easy, confident pixels**
so training focuses on hard/rare ones. It helps under strong imbalance (e.g., tiny foreground) where plain CE is
dominated by easy background.

**Class weights from pixel frequency.** Compute each class's pixel fraction `freq_c`; a common formula is
`weight_c = median(freqs) / freq_c` (median-frequency balancing) or `weight_c = 1 / freq_c` (inverse frequency). Rarer
classes get larger weights. We used median-frequency balancing (crack ≈ 4.6).

**SGD+momentum vs Adam vs AdamW; weight decay.** SGD+momentum is simple and generalizes well but needs careful LR
tuning. Adam adapts the LR per-parameter (fast, forgiving). **AdamW** is Adam with **decoupled weight decay** — it
shrinks weights correctly (not tangled into the gradient), which regularizes better for fine-tuning. Weight decay
penalizes large weights to reduce overfitting. We used AdamW.

**Warmup + cosine annealing.** *Warmup* starts the LR small and ramps it up over the first few epochs, so the pretrained
weights aren't wrecked by a big early step. *Cosine annealing* then decays the LR smoothly along a cosine curve to near
zero, letting the model settle. The curve rises for a few epochs, then falls like the right half of a cosine.

**Transfer learning and why it beats random init.** Transfer learning starts from weights trained on a huge dataset
(ImageNet/COCO), so the model already knows edges, textures, shapes. Fine-tuning adapts that to our task with far less
data and time — random initialization would need much more data to relearn basic features and usually ends up worse.

**Overfitting — detect and mitigate.** Overfitting = the model memorizes training data. You spot it when **training loss
keeps dropping but validation loss rises** (or val mIoU plateaus/drops) — a widening gap. Three mitigations: (1) more
**data augmentation**, (2) **regularization** (weight decay, dropout, early stopping), (3) a **smaller model** or
transfer learning. Our curves show no such gap → no overfitting.

### B.6 Data & Augmentation

**Data leakage in a segmentation benchmark.** Leakage is when information from test leaks into training, inflating the
score. Concrete example: **video frames or overlapping/tiled crops of the same scene** split randomly — near-identical
images end up in both train and test, so the model "recognizes" test images. Our patches came from shared source photos,
which is exactly this risk; we grouped near-duplicates to prevent it.

**Why the mask gets the same *geometric* transform but not colour jitter.** If you flip/rotate the image you must
flip/rotate the mask the same way, or the labels no longer line up. But colour changes (brightness, blur, noise) don't
move anything — and a mask is class IDs, not a picture, so brightening it would corrupt the labels. So geometry:
image+mask together; photometry: image only.

**Random hflip, crop, colour jitter, gaussian blur — safe on masks?** Hflip and crop are **geometric** → apply to both
image and mask (with nearest-neighbour on the mask so labels stay integer). Colour jitter and gaussian blur are
**photometric** → image **only**; applying them to a mask would destroy the class IDs.

**What is Albumentations and how does it guarantee synchronized transforms?** Albumentations is a fast image-augmentation
library with first-class mask support. You declare targets (`image=`, `mask=`) in one `Compose`; it samples each
transform's random parameters once and applies the geometric ones identically to image and mask, while automatically
skipping photometric ones on the mask. One call = guaranteed sync.

**The three split roles; why test is touched once.** **Train** fits the weights; **validation** tunes choices
(hyperparameters, early stopping, checkpoint selection); **test** is the final, untouched exam. If you look at test more
than once and adjust anything based on it, you start fitting to it — the score stops predicting real-world performance.
So test is evaluated exactly once, at the very end.

---

<p align="center"><sub>CSE 348 — Digital Image Processing · PaveCrack1300 Segmentation Benchmark</sub></p>
