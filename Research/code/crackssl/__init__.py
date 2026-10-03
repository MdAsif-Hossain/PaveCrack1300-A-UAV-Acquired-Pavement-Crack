"""Label-efficiency study for thin-structure (crack) segmentation."""

from . import data, metrics, viz  # noqa: F401

__version__ = "0.2.0"
__all__ = ["data", "metrics", "viz", "models", "train"]


def __getattr__(name):
    """Lazily expose torch-dependent modules so CPU-only notebooks import fast."""
    if name in {"models", "train"}:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
