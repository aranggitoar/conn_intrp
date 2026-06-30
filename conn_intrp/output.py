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
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def fs_safe(name: str) -> str:
    """Replace path separators so *name* is safe as a single path component.

    :param name: Raw name that may contain ``/`` characters.
    :type name: str
    :returns: Sanitised string safe for use as a filename component.
    :rtype: str
    """
    return name.replace("/", "_")


def make_run_dir(
    base: str | Path,
    model_name: str,
    method: str,
    tag: str | None = None,
    *,
    resume: bool = False,
) -> Path:
    """
    Create a timestamped output directory, or resume the latest one.

    When *resume* is ``True``, returns the most recent existing directory
    that matches ``{model_name}_{method}_*`` under *base*. Falls back to
    creating a new directory if none exists.

    :param base: Parent directory for all runs.
    :type base: str | Path
    :param model_name: Short model identifier (e.g. ``"smolvlm2"``).
    :type model_name: str
    :param method: Pipeline phase (e.g. ``"dm"``, ``"ablation"``).
    :type method: str
    :param tag: Optional suffix for the directory name.
    :type tag: str | None
    :param resume: If ``True``, reuse the latest matching run directory.
    :type resume: bool
    :returns: Path to the created or resumed directory.
    :rtype: Path
    """
    if resume:
        existing = find_latest_run(base, model_name, method)
        if existing is not None:
            print(f"Resuming run: {existing}")
            return existing

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{model_name}_{method}_{ts}"
    if tag:
        name += f"_{tag}"
    run_dir = Path(base) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_latest_run(
    base: str | Path,
    model_name: str,
    method: str,
) -> Path | None:
    """
    Find the most recent run directory for a given model and method.

    Scans *base* for directories matching ``{model_name}_{method}_*``
    and returns the latest by timestamp. Returns ``None`` if no match.

    :param base: Parent directory for all runs.
    :type base: str | Path
    :param model_name: Short model identifier (e.g. ``"smolvlm2"``).
    :type model_name: str
    :param method: Pipeline phase (e.g. ``"dm"``, ``"ablation"``).
    :type method: str
    :returns: Path to the latest matching run directory, or ``None``.
    :rtype: Path | None
    """
    base = Path(base)
    if not base.exists():
        return None
    prefix = f"{model_name}_{method}_"
    candidates = sorted(
        (d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix)),
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def update_metadata(run_dir: str | Path, data: dict) -> None:
    """
    Create or update ``metadata.json`` in *run_dir*.

    On first call, writes *data* as-is.  On subsequent calls, merges
    *data* into the existing file (top-level keys are overwritten).

    :param run_dir: Run output directory.
    :type run_dir: str | Path
    :param data: Metadata fields to write or update.
    :type data: dict
    """
    path = Path(run_dir) / "metadata.json"
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
        existing.update(_make_serializable(data))
        data = existing
    save_json(path, data)


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
    """Recursively convert non-JSON-serializable types to JSON-safe equivalents."""
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
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = ckpt_dir / f"{fs_safe(name)}.pt"
    tmp = final.with_suffix(".pt.tmp")
    torch.save(artifacts, tmp)
    tmp.rename(final)


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
    path = Path(run_dir) / "checkpoints" / f"{fs_safe(name)}.pt"
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
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    yield
    if torch.cuda.is_available():
        print(f"{label}: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB peak")
