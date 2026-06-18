"""
Analysis utilities for directional masking results.

Loads saved masks from a run directory and provides composable
functions for Colab exploration: summary tables, survivor indices,
cross-category overlap matrices, and distribution breakdowns.

Example::

    >>> from conn_intrp.dm_analysis import load_dm_masks, summary_table, overlap_matrix
    >>> masks = load_dm_masks("outputs/internvl3_5_dm_20260617_133647")
    >>> summary_table(masks)
    >>> overlap_matrix(masks, "linear_2")

Main Functions:
    load_dm_masks: Load all mask tensors from a run directory.
    summary_table: Per-layer, per-category stats as a DataFrame.
    survivors: Direction indices above a threshold.
    overlap_matrix: Category × category survivor overlap counts.
    jaccard_matrix: Category × category Jaccard similarity.
    distribution: Bucket breakdown of mask weights.
"""

import json
import torch
import pandas as pd
from pathlib import Path

from .output import fs_safe


def load_dm_masks(run_dir: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    """
    Load all mask ``.pt`` files from *run_dir*.

    :param run_dir: Path to a DM run output directory.
    :type run_dir: str | Path
    :returns: ``{layer_name: {category_name: mask_tensor}}``.
    :rtype: dict[str, dict[str, torch.Tensor]]
    """
    run_dir = Path(run_dir)
    layer_names = _read_layer_names(run_dir)
    # sort so longest layer name is tried first (linear_1 before linear)
    layer_names.sort(key=len, reverse=True)
    masks: dict[str, dict[str, torch.Tensor]] = {}
    for pt in sorted(run_dir.glob("mask_*.pt")):
        rest = pt.stem.removeprefix("mask_")  # e.g. linear_2_figure_diagram
        layer, cat = _split_by_layers(rest, layer_names)
        masks.setdefault(layer, {})[cat] = torch.load(pt, weights_only=True)
    return masks


def _read_layer_names(run_dir: Path) -> list[str]:
    meta = run_dir / "metadata.json"
    if meta.exists():
        with open(meta) as f:
            return json.load(f).get("layers", [])
    # fallback: infer from filenames
    names = set()
    for pt in run_dir.glob("mask_*.pt"):
        rest = pt.stem.removeprefix("mask_")
        for n in (2, 1):
            candidate = "_".join(rest.split("_")[:n])
            names.add(candidate)
    return list(names)


def _split_by_layers(
    rest: str, layer_names: list[str],
) -> tuple[str, str]:
    for ln in layer_names:
        prefix = ln + "_"
        if rest.startswith(prefix):
            return ln, rest[len(prefix):]
    return rest.split("_", 1)[0], rest.split("_", 1)[1]


def summary_table(masks: dict[str, dict[str, torch.Tensor]],
                  threshold: float = 0.5) -> pd.DataFrame:
    """
    One row per (layer, category) with survivor/dead counts and mean KL-relevant stats.

    :param masks: Output of :func:`load_dm_masks`.
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param threshold: Survivor threshold.
    :type threshold: float
    :returns: DataFrame with columns: layer, category, n_dirs, survivors,
        dead, mid, mean_weight, median_weight.
    :rtype: pd.DataFrame
    """
    rows = []
    for layer, cats in masks.items():
        for cat, m in cats.items():
            n = m.numel()
            surv = (m > threshold).sum().item()
            dead = (m < 0.05).sum().item()
            rows.append(dict(
                layer=layer, category=cat, n_dirs=n,
                survivors=surv, dead=dead, mid=n - surv - dead,
                mean_weight=m.mean().item(),
                median_weight=m.median().item(),
            ))
    return pd.DataFrame(rows)


def survivors(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str, category: str,
    threshold: float = 0.5,
) -> list[int]:
    """
    Sorted direction indices whose mask weight exceeds *threshold*.

    :param masks: Output of :func:`load_dm_masks`.
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name (e.g. ``"linear_2"``).
    :type layer: str
    :param category: Category name (e.g. ``"figure_diagram"``).
    :type category: str
    :param threshold: Weight threshold.
    :type threshold: float
    :returns: Sorted list of direction indices.
    :rtype: list[int]
    """
    m = masks[layer][category]
    return torch.where(m > threshold)[0].tolist()


def ranked_directions(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str, category: str,
    top_k: int | None = None,
) -> pd.DataFrame:
    """
    Directions sorted by mask weight (descending).

    :param masks: Output of :func:`load_dm_masks`.
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name.
    :type layer: str
    :param category: Category name.
    :type category: str
    :param top_k: If given, only return the top *k* directions.
    :type top_k: int | None
    :returns: DataFrame with columns: direction, weight.
    :rtype: pd.DataFrame
    """
    m = masks[layer][category]
    idx = m.argsort(descending=True)
    if top_k is not None:
        idx = idx[:top_k]
    return pd.DataFrame({
        "direction": idx.tolist(),
        "weight": m[idx].tolist(),
    })


def distribution(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str, category: str,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    """
    Bucket breakdown of mask weights.

    :param masks: Output of :func:`load_dm_masks`.
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name.
    :type layer: str
    :param category: Category name.
    :type category: str
    :param bins: Bin edges. Defaults to ``[0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.01]``.
    :type bins: list[float] | None
    :returns: DataFrame with columns: range, count.
    :rtype: pd.DataFrame
    """
    if bins is None:
        bins = [0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.01]
    m = masks[layer][category]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        label = f"{lo:.2f}-{hi:.2f}"
        count = ((m >= lo) & (m < hi)).sum().item()
        rows.append(dict(range=label, count=count))
    return pd.DataFrame(rows)


def overlap_matrix(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Category × category matrix of survivor overlap (intersection size).

    :param masks: Output of :func:`load_dm_masks`.
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name.
    :type layer: str
    :param threshold: Survivor threshold.
    :type threshold: float
    :returns: Square DataFrame, rows and columns are categories.
    :rtype: pd.DataFrame
    """
    cats = list(masks[layer].keys())
    surv = {c: set(torch.where(masks[layer][c] > threshold)[0].tolist())
            for c in cats}
    data = [[len(surv[a] & surv[b]) for b in cats] for a in cats]
    return pd.DataFrame(data, index=cats, columns=cats)


def jaccard_matrix(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Category × category Jaccard similarity of survivors.

    :param masks: Output of :func:`load_dm_masks`.
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name.
    :type layer: str
    :param threshold: Survivor threshold.
    :type threshold: float
    :returns: Square DataFrame with Jaccard values (0–1).
    :rtype: pd.DataFrame
    """
    cats = list(masks[layer].keys())
    surv = {c: set(torch.where(masks[layer][c] > threshold)[0].tolist())
            for c in cats}
    data = []
    for a in cats:
        row = []
        for b in cats:
            union = len(surv[a] | surv[b])
            row.append(len(surv[a] & surv[b]) / union if union else 1.0)
        data.append(row)
    return pd.DataFrame(data, index=cats, columns=cats)


def direction_profile(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str,
    S: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Per-direction stats across all categories, optionally with singular values.

    Returns one row per direction.  Boolean columns indicate survival in
    each category; ``n_survived`` counts them.  When *S* is provided,
    ``sv_magnitude`` is included so you can correlate importance with
    SVD rank.

    :param masks: Output of :func:`load_dm_masks`.
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name.
    :type layer: str
    :param S: Singular value vector for this layer (e.g. ``adapter.svd_layers[i].S``).
    :type S: torch.Tensor | None
    :param threshold: Survivor threshold.
    :type threshold: float
    :returns: DataFrame with columns: direction, [sv_magnitude,]
        n_survived, mean_weight, then one bool column per category.
    :rtype: pd.DataFrame
    """
    cats = list(masks[layer].keys())
    n_dirs = masks[layer][cats[0]].numel()
    weights = torch.stack([masks[layer][c] for c in cats])  # (n_cats, n_dirs)
    survived = weights > threshold  # (n_cats, n_dirs)

    data = {"direction": list(range(n_dirs))}
    if S is not None:
        data["sv_magnitude"] = S.detach().cpu().float().tolist()
    data["n_survived"] = survived.sum(dim=0).tolist()
    data["mean_weight"] = weights.mean(dim=0).tolist()
    for i, cat in enumerate(cats):
        data[cat] = survived[i].tolist()
    return pd.DataFrame(data)


def compare_categories(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str, cat_a: str, cat_b: str,
    threshold: float = 0.5,
) -> dict[str, list[int]]:
    """
    Compare survivors between two categories on the same layer.

    :param masks: Output of :func:`load_dm_masks`.
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name.
    :type layer: str
    :param cat_a: First category.
    :type cat_a: str
    :param cat_b: Second category.
    :type cat_b: str
    :param threshold: Survivor threshold.
    :type threshold: float
    :returns: Dict with keys ``shared``, ``only_a``, ``only_b`` — sorted
        lists of direction indices.
    :rtype: dict[str, list[int]]
    """
    sa = set(torch.where(masks[layer][cat_a] > threshold)[0].tolist())
    sb = set(torch.where(masks[layer][cat_b] > threshold)[0].tolist())
    return {
        "shared": sorted(sa & sb),
        f"only_{cat_a}": sorted(sa - sb),
        f"only_{cat_b}": sorted(sb - sa),
    }
