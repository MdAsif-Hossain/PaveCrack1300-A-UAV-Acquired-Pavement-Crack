# Part B Report — how to compile

`report.tex` is a **two-column IEEE conference paper** (the format the assignment mandates).

## Compile

**Easiest — Overleaf:**
1. New Project → Upload Project → upload `report.tex` **and** the `figures/` folder
   (put `figures/` one level above `report.tex`, or change `\graphicspath` to `{{figures/}}`)
2. Set compiler to **pdfLaTeX** → Recompile

**Locally:**
```bash
cd report
pdflatex report.tex
pdflatex report.tex     # run twice so \ref and citations resolve
```
`IEEEtran.cls` ships with TeX Live / MiKTeX. If it's missing: `tlmgr install ieeetran`.

## Before you submit — fill these in
Search the `.tex` for `$\langle$` and replace:
- `$\langle$Member 2/3/4$\rangle$` + their `$\langle$ID$\rangle$`
- `Group $\langle$number$\rangle$`, `Section $\langle$3/4$\rangle$`
- `Instructor: $\langle$fill instructor$\rangle$`

## Structure (maps to the assignment's Section 6)

| Report section | Assignment requirement |
|---|---|
| Title / authors / abstract | 6.1 |
| Introduction | 6.2 |
| Dataset and Split | 6.3 (incl. both caveats) |
| Methodology | 6.4 (4 pretext tasks + ViT adapter) |
| Fine-tuning Setup | 6.5 (frozen-vs-FT, hyperparameter table, curves) |
| Results | 6.6 (full metric table + chart) |
| Error Analysis | 6.7 |
| Label-Efficiency Discussion | 6.8 |
| Insights | 6.9 (mandatory) |
| Conclusion + References | 6.10 |

No Literature Review — correctly omitted, per the assignment.

## Figures used
9 figures, all in `../figures/` (extracted from your executed notebooks):
`simclr/byol/mae/dinov2_pretrain_loss.png`, `dinov2_curves.png`,
`dinov2_confusion.png`, `simclr_confusion.png`, `dinov2_worst.png`,
`final_label_efficiency.png`.

21 figures were extracted in total — swap in any others you prefer.

## ⚠️ Numbers to re-check before submitting
The Part-A baseline is quoted as **DeepLabV3-ResNet50, mIoU 0.8383** (from your Assignment 1
notebooks). Your Kaggle `results.json` attached to NB5 instead reported **YOLO26s-Sem at 0.8319**.
Confirm which Part-A run is the one you submitted and make the report match; if it changes, the
"% of supervised" column and the 85.0% headline shift slightly.
