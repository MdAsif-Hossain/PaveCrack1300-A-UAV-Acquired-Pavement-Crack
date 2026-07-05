# Extract every embedded PNG figure from the 5 executed Kaggle output notebooks into figures/
# so the README can reference them as files. Names are derived from each cell's savefig() target.
import json, re, base64
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "figures"; FIG.mkdir(exist_ok=True)

NBS = [
    ("eda.ipynb",                                  "probe", None),
    ("nb0-eda-data-preparation.ipynb",             "nb0",   None),
    ("nb1-deeplabv3-resnet50.ipynb",               "nb1",   "DeepLabV3-ResNet50"),
    ("nb2-segformer-b0.ipynb",                     "nb2",   "SegFormer-B0"),
    ("nb3-yolov26-semantic-final-comparison.ipynb","nb3",   "YOLO26s-Sem"),
]

def targets(src, model):
    hits = re.findall(r'savefig\(\s*WORK\s*/\s*f?(["\'])(.*?\.png)\1', src)
    out = []
    for _q, n in hits:
        if model:
            n = n.replace("{CONFIG['model']}", model).replace('{CONFIG["model"]}', model)
        n = re.sub(r"[{}\[\]'\"]", "", n)
        if model and n.startswith(model + "_"):
            n = n[len(model) + 1:]                     # strip long model prefix
        out.append(n)
    return out

manifest = {}
for fname, prefix, model in NBS:
    p = REPO / fname
    if not p.exists():
        print("MISSING", fname); continue
    nb = json.load(open(p, encoding="utf-8"))
    seq = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code": continue
        src = "".join(c["source"]); tg = targets(src, model)
        imgs = [o for o in c.get("outputs", [])
                if o.get("output_type") in ("display_data", "execute_result")
                and "image/png" in o.get("data", {})]
        for k, o in enumerate(imgs):
            if k < len(tg):
                base = tg[k]
            elif "results.png" in src or "training curves" in src.lower():
                base = "yolo_training_curves.png"
            else:
                base = f"fig{seq}.png"
            name = f"{prefix}_{base}"
            (FIG / name).write_bytes(base64.b64decode(o["data"]["image/png"]))
            manifest.setdefault(prefix, []).append(name)
            seq += 1

total = sum(len(v) for v in manifest.values())
# clean stray generic names from the previous run
for f in FIG.glob("*cell_fig*.png"): f.unlink()
print(f"extracted {total} figures into {FIG}")
for pfx, names in manifest.items():
    print(f"\n[{pfx}]")
    for n in names: print("   ", n)
