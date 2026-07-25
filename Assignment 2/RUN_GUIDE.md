# Part B — Kaggle Run Guide

How to run all 6 notebooks, which datasets to attach, and GPU settings. **Order matters** — later notebooks
consume earlier notebooks' committed outputs.

## The datasets you'll be attaching (Kaggle **+ Add Input**)

| Short name | What it is | Where it comes from |
|---|---|---|
| **RAW** | PaveCrack1300 images + masks | Kaggle dataset `mantashamahi/pavecrack1300-a-uav-acquired-pavement-crack` |
| **PARTA-SPLIT** | `split.json` (the leakage-safe split) | your committed **Part-A NB0** (`eda-and-data-prep`) output |
| **PARTB-ROLES** | `partB_roles.json` (role mapping) | your committed **Part-B NB0** (`0_data-prep`) output |
| **PARTA-RESULTS** | Part-A `results.json` (supervised baseline) | your committed **Part-A NB1** (`seg-deeplabv3`) output |
| **PARTB-\<method\>** | each method's `results_partB.json` | that method notebook's committed output |

## Run order, inputs, and GPU

| # | Notebook | Attach (Inputs) | GPU | Internet | ~Time | After run |
|---|---|---|---|---|---|---|
| 0 | `0_data-prep` | RAW + PARTA-SPLIT | ❌ CPU | off | ~2 min | Commit ✓ (done) |
| 1 | `1_simclr` | RAW + PARTB-ROLES | ✅ **T4** | on* | ~1–1.5 h | Commit |
| 2 | `2_byol` | RAW + PARTB-ROLES | ✅ **T4** | on* | ~1 h | Commit |
| 3 | `3_mae` | RAW + PARTB-ROLES | ✅ **T4** | on* | ~1 h | Commit |
| 4 | `4_dinov2` | RAW + PARTB-ROLES | ✅ **T4** | ✅ **on** | ~1.5–3 h | Commit |
| 5 | `5_final-comparison` | PARTB-simclr + PARTB-byol + PARTB-mae + PARTB-dinov2 + PARTA-RESULTS | ❌ CPU | off | ~2 min | Commit |

\* **Internet:** SimCLR/BYOL/MAE don't download model weights (encoders start from random init per the assignment),
but leave Internet **on** anyway so the `pip install albumentations` fallback works if Kaggle's image lacks it.
**DINOv2 requires Internet on** — it downloads the official `facebook/dinov2-small` checkpoint (the recommended
"continue-pretraining" path) and upgrades `transformers`.

### GPU choice
- Use **GPU T4 x2** (or single T4 / P100 — the code uses one GPU, `cuda:0`). Turn it **On** in the notebook's
  *Settings → Accelerator* for notebooks 1–4. Notebooks 0 and 5 are CPU-only (save your GPU quota).
- Kaggle gives ~30 GPU-hours/week. Budget: the four method notebooks total roughly **4–6 GPU-hours**, well within quota.
- Each notebook finishes inside one session (all < 9 h), so **no pretrain/downstream split is needed**.

## Step-by-step for every method notebook (1–4)
1. **+ Add Input** → attach **RAW** and **PARTB-ROLES**.
2. *Settings* → **Accelerator = GPU T4**, **Internet = On**.
3. **Run All**. Watch the first SSL epoch print a per-epoch time — if it ever exceeds ~10 min/epoch you'd split
   that method, but at 224² none should.
4. **Save Version → Save & Run All (Commit)**. Name it clearly, e.g. `pavecrack1300-partb-simclr`.
5. Its output now carries `results_partB.json` + figures for the final notebook.

## Final comparison (5)
Attach the **four committed method outputs** + your **Part-A NB1 output** (for the supervised `results.json`
baseline). Run on CPU. It merges everything into the label-efficiency table + chart.

## What each notebook produces (for the report)
- **0**: `partB_roles.json`, sanity grid.
- **1–4**: SSL pretrain-loss curve, fine-tune/monitor curves, confusion matrix, worst-image grid, and a row in
  `results_partB.json`. SimCLR also runs the **frozen-vs-fine-tuned ablation** (two rows: `SimCLR-frozen`, `SimCLR`).
- **5**: `partB_label_efficiency.csv` + chart.

## Submission (per the PDF §5)
6 public Kaggle links + the LaTeX report PDF, one line per notebook:
```
Data Prep: <link>
SimCLR:  <link>   BYOL: <link>   MAE: <link>   DINOv2: <link>
Final Comparison: <link>
Report PDF: <attached>
```
Deadline: **Sec 3 — 17.08.2026** / **Sec 4 — 18.08.2026**.
