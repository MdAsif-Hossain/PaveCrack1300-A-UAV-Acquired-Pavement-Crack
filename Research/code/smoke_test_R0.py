"""
Smoke-test R0 against a synthetic Kaggle-shaped input tree.

Runs every R0 code cell except the DINOv2 embedding one (which needs a model download),
so path discovery, the split guard, density, the dHash audit and the budget preview are
all exercised before anyone spends a Kaggle session on them.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

NB = HERE / "notebooks" / "R0_prepare.ipynb"
N_TRAIN, N_VAL, N_TEST = 40, 10, 10


def build_fake_input(root: Path) -> Path:
    """Mimic /kaggle/input: a dataset folder with images/ + masks/, and a split folder."""
    rng = np.random.default_rng(0)
    ds = root / "pavecrack1300"
    (ds / "images").mkdir(parents=True)
    (ds / "masks").mkdir(parents=True)

    ids = [f"crack_{i:04d}" for i in range(N_TRAIN + N_VAL + N_TEST)]
    for k, sid in enumerate(ids):
        img = rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)
        m = np.zeros((64, 64), np.uint8)
        if k % 7 != 0:                      # ~1 in 7 has no crack
            r = rng.integers(5, 58)
            m[r : r + rng.integers(1, 4), 5:60] = 255
            img[m > 0] = 30                 # make the crack visible so dHash differs
        Image.fromarray(img).save(ds / "images" / f"{sid}.jpg")
        Image.fromarray(m).save(ds / "masks" / f"{sid}.png")

    # two deliberate near-duplicates ACROSS splits -> the audit must notice
    dup_src, dup_dst = ids[0], ids[N_TRAIN + 1]          # train -> val
    shutil.copy(ds / "images" / f"{dup_src}.jpg", ds / "images" / f"{dup_dst}.jpg")

    split = root / "parta-split"
    split.mkdir()
    (split / "split.json").write_text(json.dumps({
        "train": ids[:N_TRAIN],
        "val": ids[N_TRAIN:N_TRAIN + N_VAL],
        "test": ids[N_TRAIN + N_VAL:],
        "seed": 42,
    }))
    return ds


def cell_sources(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="r0smoke_"))
    work = tmp / "working"
    work.mkdir()
    build_fake_input(tmp / "input")

    srcs = cell_sources(NB)
    ns: dict = {"display": lambda x: print(x)}
    failures = []

    for i, src in enumerate(srcs):
        # redirect the Kaggle paths and the library import at the source level
        src = (src
               .replace('sys.path.insert(0, "/kaggle/input/crackssl-lib")',
                        f'sys.path.insert(0, r"{HERE}")')
               .replace('Path("/kaggle/working")', f'Path(r"{work}")')
               .replace('Path("/kaggle/input")', f'Path(r"{tmp / "input"}")'))

        if "AutoModel" in src or "AutoImageProcessor" in src:
            print(f"cell {i}: SKIPPED (needs DINOv2 download)")
            # fabricate what the skipped cell would have produced
            import pandas as pd
            split = ns["split"]
            rng = np.random.default_rng(1)
            ns["probe"] = pd.DataFrame({
                "test_id": split.test,
                "nearest_train_id": split.train[: len(split.test)],
                "cosine_similarity": rng.uniform(0.5, 0.99, len(split.test)),
                "cosine_distance": rng.uniform(0.01, 0.5, len(split.test)),
                "test_crack_fraction": [ns["density"][i] for i in split.test],
            })
            continue

        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
            print(f"cell {i}: OK")
        except Exception as e:                        # noqa: BLE001
            failures.append((i, type(e).__name__, str(e)[:200]))
            print(f"cell {i}: FAILED  {type(e).__name__}: {str(e)[:200]}")

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} cell(s) failed:")
        for i, k, m in failures:
            print(f"  cell {i}: {k}: {m}")
        return 1

    print("ALL CELLS PASSED")
    print("outputs written:", sorted(p.name for p in work.glob("*") if p.is_file()))
    audit = ns.get("audit")
    if audit is not None:
        print("\nleakage audit (the planted cross-split duplicate should appear):")
        print(audit.to_string(index=False))
    print("\nfingerprint:", ns.get("FINGERPRINT"))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
