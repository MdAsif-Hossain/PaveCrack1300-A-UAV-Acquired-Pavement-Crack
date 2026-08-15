# Assignment 2 (Part B) — Kaggle Run Guide

Everything needed to run the 6 notebooks: what to attach, GPU settings, order, and what each one produces.
**Order matters** — later notebooks consume earlier notebooks' *committed* outputs.

---

## 0. What you need before starting

Two things from **Assignment 1**, already on your Kaggle account (you committed them for Part A):

| Needed | Which Part-A notebook produced it | Contains |
|---|---|---|
| **PARTA-SPLIT** | `eda-and-data-prep` (Part-A NB0) | `split.json` ← the leakage-safe split |
| **PARTA-RESULTS** | `seg-deeplabv3` (Part-A NB1) | `results.json` ← supervised baseline, mIoU 0.8383 |

> In your earlier run the split dataset appeared as `.../nb0-eda-and-data-preparation-data/split.json` — that's
> the one to attach. No Part-A model checkpoints are needed.

Plus the raw dataset:

| **RAW** | Kaggle dataset `mantashamahi/pavecrack1300-a-uav-acquired-pavement-crack` |
|---|---|

---

## 1. Run order at a glance

| # | Notebook | Attach these inputs | GPU | Internet | ~Time |
|---|---|---|:---:|:---:|---|
| 0 | `0_data-prep` | RAW + **PARTA-SPLIT** | ❌ CPU | On | ~2 min |
| 1 | `1_simclr` | RAW + **PARTB-ROLES** | ✅ T4 | On | ~1–1.5 h |
| 2 | `2_byol` | RAW + **PARTB-ROLES** | ✅ T4 | On | ~1 h |
| 3 | `3_mae` | RAW + **PARTB-ROLES** | ✅ T4 | On | ~1 h |
| 4 | `4_dinov2` | RAW + **PARTB-ROLES** | ✅ T4 | ✅ **On (required)** | ~1.5–3 h |
| 5 | `5_final-comparison` | outputs of NB1+NB2+NB3+NB4 + **PARTA-RESULTS** | ❌ CPU | On | ~2 min |

**PARTB-ROLES** = the committed output of step 0 (contains `partB_roles.json`).

Total ≈ **4–6 GPU-hours**, inside Kaggle's ~30 h/week quota. Notebooks 1–4 are independent — run them in
**parallel sessions** to save wall-clock time.

---

## 2. Step-by-step

### Step 0 — Data prep (do this first, everything depends on it)
1. New notebook → **File → Import Notebook** → upload `notebooks/0_data-prep.ipynb`
2. Title it `pavecrack1300-partb-data-prep`
3. **+ Add Input** → add **RAW** *and* **PARTA-SPLIT**
4. *Settings*: Accelerator **None (CPU)** — saves GPU quota
5. **Run All**. Expected output:
   ```
   Part-A counts: {'train': 909, 'val': 195, 'test': 196} | seed: 42
   UNLABELLED pretrain <- Part-A train : 909 images
   LABELLED  fine-tune <- Part-A val   : 195 images
   MONITOR + EVAL      <- Part-A test  : 196 images
   label-efficiency ratio = 195/909 = 21.5%
   ```
6. **Save Version → Save & Run All (Commit)** ← *this is what makes its output attachable*

### Steps 1–4 — The four SSL methods
For each of `1_simclr`, `2_byol`, `3_mae`, `4_dinov2`:
1. Import the notebook; title it `pavecrack1300-partb-<method>`
2. **+ Add Input** → **RAW** *and* **PARTB-ROLES** (step 0's committed output)
3. *Settings*: Accelerator **GPU T4 x2**, Internet **On**
   - `4_dinov2` **must** have Internet on — it downloads `facebook/dinov2-small`
4. **Run All**, then **Commit**

**Early sanity check (~2 min in):** the SSL loss should print and decrease:
```
  SSL ep 01 NT-Xent 5.9xxx
  SSL ep 05 NT-Xent 5.2xxx     <- going down = healthy
Task B pretraining path: from scratch on the Part-A train split ...
wall-clock: X.X min total, 0.XX min/epoch
```
If `min/epoch` is under 10, you're fine to let it run to completion in one notebook (that's the assignment's
threshold for splitting into two notebooks).

### Step 5 — Final comparison
1. Import `5_final-comparison`; title it `pavecrack1300-partb-comparison`
2. **+ Add Input** → the **committed outputs of all four method notebooks** *and* **PARTA-RESULTS**
3. CPU is fine. **Run All** → **Commit**
4. It merges everything and prints the label-efficiency table + charts.

---

## 3. What each notebook produces

| Notebook | Outputs |
|---|---|
| `0_data-prep` | `partB_roles.json`, Task-A sanity grid |
| each method | pretext-loss curve · fine-tune + monitoring curves · confusion matrix (counts + normalized) · per-image IoU histogram · worst-image grid · **error maps (TP/FP/FN)** · `results_partB.json` · `{method}_qual_preds.npz` |
| `3_mae` extra | MAE reconstruction figure |
| `1_simclr` extra | frozen-vs-fine-tuned ablation chart (2 result rows) |
| `5_final-comparison` | styled **label-efficiency table** + CSV · bar chart vs Part-A baseline · **radar + box plot** · **4-method qualitative grid** · **failure-overlap chart** |

---

## 4. Troubleshooting

**`CUDA out of memory`** — the only likely resource issue. In the SSL pretraining cell, lower the batch size:
- SimCLR / BYOL: `batch_size=64` → `32`
- MAE: `batch_size=48` → `24`
- DINOv2: `batch_size=32` → `16`
Nothing else needs changing.

**Two bugs already fixed** (don't be alarmed if you saw them in an earlier attempt):
- `value cannot be converted to type c10::Half without overflow` → fixed (losses now computed in float32)
- `Only Tensors created explicitly by the user ... deepcopy` in DINOv2 → fixed (teacher built via `load_state_dict`)

Make sure you're running the **current** files — the SimCLR pretraining cell should contain
`torch.autocast(device_type="cuda", enabled=False)`, not `-9e15`.

**A figure fails** — all plotting is wrapped in `try/except`; it prints e.g. `error-map figure skipped: ...` and
training results are still saved. Not fatal.

**`assert len(results) >= 3`** style errors in NB5 → you forgot to attach one of the method outputs.

---

## 5. Submission (PDF §5)

Set **all 6 notebooks to Public**, then post to Google Classroom:
```
Group: <group>            Section: <3 or 4>
Dataset: PaveCrack1300 (same as Part A)
Data Prep: <link>
SimCLR: <link>   BYOL: <link>   MAE: <link>   DINOv2: <link>
Final Comparison: <link>
Report PDF: <attached>
```
Also fill the header placeholders in every notebook (members 2–4, instructor, group, section).

**Deadlines — Section 3: 17.08.2026 · Section 4: 18.08.2026.**
