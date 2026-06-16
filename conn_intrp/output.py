"""
Output management, serialization, and checkpointing.

Provides timestamped run directories, JSON serialization with tensor
support, and checkpoint save/load for resumable experiment runs.

Example::

    >>> from conn_intrp.output import make_run_dir, save_json, save_checkpoint
    >>> run_dir = make_run_dir("outputs", "smolvlm2", "ablation")
    >>> save_json(run_dir / "metadata.json", {"model": "smolvlm2"})
    >>> save_checkpoint(run_dir, "table_list", {"coefficients": tensor})

Main Functions:
    make_run_dir: Create a timestamped output directory.
    save_json: Serialize data to JSON with tensor/ndarray support.
    save_checkpoint: Save a dict of artifacts as a ``.pt`` file.
    load_checkpoint: Load a checkpoint if it exists.
    get_completed_categories: List categories with existing checkpoints.
"""

import datetime
import json
import torch
import numpy as np
from pathlib import Path
from contextlib import contextmanager


def make_run_dir(
    base: str | Path, model_name: str, method: str, tag: str | None = None
) -> Path:
    """
    Create a timestamped output directory.

    :param base: Parent directory for all runs.
    :type base: str | Path
    :param model_name: Short model identifier (e.g. ``"smolvlm2"``).
    :type model_name: str
    :param method: Pipeline phase (e.g. ``"dm"``, ``"ablation"``).
    :type method: str
    :param tag: Optional suffix for the directory name.
    :type tag: str | None
    :returns: Path to the created directory.
    :rtype: Path
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{model_name}_{method}_{ts}"
    if tag:
        name += f"_{tag}"
    run_dir = Path(base) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: str | Path, data: dict) -> None:
    """
    Serialize *data* to JSON, converting tensors and ndarrays automatically.

    :param path: Destination file path.
    :type path: str | Path
    :param data: Data to serialize.
    :type data: dict
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_make_serializable(data), f, indent=2)


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def save_checkpoint(run_dir: str | Path, name: str, artifacts: dict) -> None:
    """
    Save experiment artifacts as a ``.pt`` checkpoint.

    :param run_dir: Run output directory.
    :type run_dir: str | Path
    :param name: Category or stage name (used as filename stem).
    :type name: str
    :param artifacts: Dict of tensors/data to persist.
    :type artifacts: dict
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(artifacts, ckpt_dir / f"{name}.pt")


def load_checkpoint(run_dir: str | Path, name: str) -> dict | None:
    """
    Load a checkpoint if it exists, otherwise return ``None``.

    :param run_dir: Run output directory.
    :type run_dir: str | Path
    :param name: Category or stage name.
    :type name: str
    :returns: Loaded artifacts dict, or ``None``.
    :rtype: dict | None
    """
    path = Path(run_dir) / "checkpoints" / f"{name}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    return None


def get_completed_categories(run_dir: str | Path) -> set[str]:
    """
    Return the set of category names that have existing checkpoints.

    :param run_dir: Run output directory.
    :type run_dir: str | Path
    :returns: Set of completed category name strings.
    :rtype: set[str]
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return set()
    return {p.stem for p in ckpt_dir.glob("*.pt")}


@contextmanager
def track_mem(label: str):
    """
    Context manager that prints peak GPU memory for a labelled block.

    :param label: Description printed alongside the peak memory.
    :type label: str
    """
    torch.cuda.reset_peak_memory_stats()
    yield
    print(f"{label}: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB peak")
