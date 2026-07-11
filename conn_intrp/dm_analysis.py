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
    load_dm_masks: Load mask tensors from a DM run directory
    summary_table: Per-layer, per-category survivor counts and weight statistics
    survivors: Direction indices above a weight threshold
    gap_survivors: Direction indices where one category leads by a minimum gap
    random_baseline_masks: Binary random masks matching evaluate_dm_baselines RNG
    random_continuous_masks: Continuous uniform [0, 1] random masks for similarity baselines
    overlap_matrix: Category x category survivor intersection counts
    jaccard_matrix: Category x category Jaccard similarity of survivor sets
    distribution: Bucket breakdown of mask weight distribution
    mask_baseline_table: Optimized vs random mask KL across thresholds
"""

import json
from pathlib import Path

import pandas as pd
import torch


def load_dm_masks(run_dir: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    """
    Load all mask ``.pt`` files from a DM run directory.

    :param run_dir: Path to a DM run output directory
    :type run_dir: str | Path
    :returns: ``{layer_name: {category_name: mask_tensor}}``
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
        if not torch.cuda.is_available():
            masks.setdefault(layer, {})[cat] = torch.load(
                pt, weights_only=True, map_location=torch.device("cpu")
            )
        else:
            masks.setdefault(layer, {})[cat] = torch.load(pt, weights_only=True)
    return masks


def _read_layer_names(run_dir: Path) -> list[str]:
    """Read layer names from metadata.json, or infer from mask filenames."""
    meta = run_dir / "metadata.json"
    if meta.exists():
        with open(meta) as f:
            return json.load(f).get("layers", [])
    names = set()
    for pt in run_dir.glob("mask_*.pt"):
        rest = pt.stem.removeprefix("mask_")
        for n in (2, 1):
            candidate = "_".join(rest.split("_")[:n])
            names.add(candidate)
    return list(names)


def _split_by_layers(
    rest: str,
    layer_names: list[str],
) -> tuple[str, str]:
    """Split a mask filename stem into (layer_name, category) using longest-prefix match."""
    for ln in layer_names:
        prefix = ln + "_"
        if rest.startswith(prefix):
            return ln, rest[len(prefix) :]
    parts = rest.split("_", 1)
    return parts[0], parts[1]


def summary_table(
    masks: dict[str, dict[str, torch.Tensor]], threshold: float = 0.5
) -> pd.DataFrame:
    """
    One row per (layer, category) with survivor/dead counts and weight stats.

    Directions below 0.05 are counted as dead.

    :param masks: Loaded DM masks, ``{layer: {category: tensor}}``
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param threshold: Survivor threshold
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
            rows.append(
                dict(
                    layer=layer,
                    category=cat,
                    n_dirs=n,
                    survivors=surv,
                    dead=dead,
                    mid=n - surv - dead,
                    mean_weight=m.mean().item(),
                    median_weight=m.median().item(),
                )
            )
    return pd.DataFrame(rows)


def survivors(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str,
    category: str,
    threshold: float = 0.5,
) -> list[int]:
    """
    Sorted direction indices whose mask weight exceeds *threshold*.

    :param masks: Loaded DM masks, ``{layer: {category: tensor}}``
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name (e.g. ``"linear_2"``)
    :type layer: str
    :param category: Category name (e.g. ``"figure_diagram"``)
    :type category: str
    :param threshold: Weight threshold
    :type threshold: float
    :returns: Sorted list of direction indices
    :rtype: list[int]
    """
    m = masks[layer][category]
    return torch.where(m > threshold)[0].tolist()


def gap_survivors(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str,
    category: str,
    min_gap: float = 0.1,
) -> list[tuple[int, float]]:
    """
    Direction indices where *category* has the highest mask weight
    by at least *min_gap* over every other category.

    Returns ``(direction_index, gap)`` pairs sorted by gap descending.

    :param masks: Loaded DM masks, ``{layer: {category: tensor}}``
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name
    :type layer: str
    :param category: Category name
    :type category: str
    :param min_gap: Minimum weight difference vs. the next-best category
    :type min_gap: float
    :returns: List of ``(direction_index, gap)`` sorted by gap descending
    :rtype: list[tuple[int, float]]
    """
    cats = list(masks[layer].keys())
    all_w = torch.stack([masks[layer][c] for c in cats])  # (n_cats, n_dirs)
    ci = cats.index(category)
    cat_w = all_w[ci]
    other_max = all_w[[j for j in range(len(cats)) if j != ci]].max(dim=0).values
    gap = cat_w - other_max
    hit = torch.where(gap > min_gap)[0]
    order = gap[hit].argsort(descending=True)
    return [(hit[i].item(), gap[hit[i]].item()) for i in order]


def ranked_directions(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str,
    category: str,
    top_k: int | None = None,
) -> pd.DataFrame:
    """
    Directions sorted by mask weight (descending).

    :param masks: Loaded DM masks, ``{layer: {category: tensor}}``
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name
    :type layer: str
    :param category: Category name
    :type category: str
    :param top_k: If given, only return the top *k* directions
    :type top_k: int | None
    :returns: DataFrame with columns: direction, weight
    :rtype: pd.DataFrame
    """
    m = masks[layer][category]
    idx = m.argsort(descending=True)
    if top_k is not None:
        idx = idx[:top_k]
    return pd.DataFrame(
        {
            "direction": idx.tolist(),
            "weight": m[idx].tolist(),
        }
    )


def distribution(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str,
    category: str,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    """
    Bucket breakdown of mask weights.

    :param masks: Loaded DM masks, ``{layer: {category: tensor}}``
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name
    :type layer: str
    :param category: Category name
    :type category: str
    :param bins: Bin edges. Defaults to ``[0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.01]``
    :type bins: list[float] | None
    :returns: DataFrame with columns: range, count
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

    :param masks: Loaded DM masks, ``{layer: {category: tensor}}``
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name
    :type layer: str
    :param threshold: Survivor threshold
    :type threshold: float
    :returns: Square DataFrame, rows and columns are categories
    :rtype: pd.DataFrame
    """
    cats = list(masks[layer].keys())
    surv = {c: set(torch.where(masks[layer][c] > threshold)[0].tolist()) for c in cats}
    data = [[len(surv[a] & surv[b]) for b in cats] for a in cats]
    return pd.DataFrame(data, index=cats, columns=cats)


def jaccard_matrix(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Category × category Jaccard similarity of survivors.

    :param masks: Loaded DM masks, ``{layer: {category: tensor}}``
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name
    :type layer: str
    :param threshold: Survivor threshold
    :type threshold: float
    :returns: Square DataFrame with Jaccard values (0–1)
    :rtype: pd.DataFrame
    """
    cats = list(masks[layer].keys())
    surv = {c: set(torch.where(masks[layer][c] > threshold)[0].tolist()) for c in cats}
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
    singular_values: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Per-direction stats across all categories, optionally with singular values.

    Returns one row per direction.  Boolean columns indicate survival in
    each category; ``n_survived`` counts them.  When *singular_values* is
    provided, ``sv_magnitude`` is included so you can correlate importance
    with SVD rank.

    :param masks: Loaded DM masks, ``{layer: {category: tensor}}``
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name
    :type layer: str
    :param singular_values: Singular value vector for this layer
    :type singular_values: torch.Tensor | None
    :param threshold: Survivor threshold
    :type threshold: float
    :returns: DataFrame with columns: direction, [sv_magnitude,]
        n_survived, mean_weight, then one bool column per category
    :rtype: pd.DataFrame
    """
    cats = list(masks[layer].keys())
    n_dirs = masks[layer][cats[0]].numel()
    weights = torch.stack([masks[layer][c] for c in cats])  # (n_cats, n_dirs)
    survived = weights > threshold  # (n_cats, n_dirs)

    data = {"direction": list(range(n_dirs))}
    if singular_values is not None:
        data["sv_magnitude"] = singular_values.detach().cpu().float().tolist()
    data["n_survived"] = survived.sum(dim=0).tolist()
    data["mean_weight"] = weights.mean(dim=0).tolist()
    for i, cat in enumerate(cats):
        data[cat] = survived[i].tolist()
    return pd.DataFrame(data)


def random_baseline_masks(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str,
    category: str,
    threshold: float,
    n_seeds: int = 3,
) -> list[torch.Tensor]:
    """
    Generate binary random masks matching the RNG used by evaluate_dm_baselines.

    Each mask has the same number of survivors as the optimized mask at
    *threshold*, with random positions selected via seeded torch.randperm.

    :param masks: Loaded DM masks, ``{layer: {category: tensor}}``
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name
    :type layer: str
    :param category: Category name
    :type category: str
    :param threshold: Binarisation threshold
    :type threshold: float
    :param n_seeds: Number of seeds to generate
    :type n_seeds: int
    :returns: Binary mask tensors, one per seed
    :rtype: list[torch.Tensor]
    """
    m = masks[layer][category]
    n_surv = int((m > threshold).sum().item())
    n_dirs = m.numel()
    out = []
    for seed in range(n_seeds):
        gen = torch.Generator().manual_seed(seed)
        rm = torch.zeros(n_dirs)
        if n_surv > 0:
            idx = torch.randperm(n_dirs, generator=gen)[:n_surv]
            rm[idx] = 1.0
        out.append(rm)
    return out


def random_continuous_masks(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str,
    n_seeds: int = 3,
    threshold: float | None = None,
) -> list[dict[str, torch.Tensor]]:
    """
    Generate continuous uniform [0, 1] random masks for all categories.

    When *threshold* is provided, each mask only covers surviving
    directions (weight > threshold), with length equal to the survivor
    count for that category. Otherwise each mask has the full direction count.

    :param masks: Loaded DM masks, ``{layer: {category: tensor}}``
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name
    :type layer: str
    :param n_seeds: Number of random seeds to generate
    :type n_seeds: int
    :param threshold: Survivor threshold for truncating mask length
    :type threshold: float | None
    :returns: List of ``{category: mask_tensor}`` dicts, one per seed
    :rtype: list[dict[str, torch.Tensor]]
    """
    cats = list(masks[layer].keys())
    n_dirs = masks[layer][cats[0]].numel()
    out = []
    for seed in range(n_seeds):
        gen = torch.Generator().manual_seed(seed)
        seed_masks = {}
        for cat in cats:
            if threshold is not None:
                length = int((masks[layer][cat] > threshold).sum().item())
            else:
                length = n_dirs
            seed_masks[cat] = torch.rand(length, generator=gen)
        out.append(seed_masks)
    return out


def random_mask(
    n_dirs: int,
    n_survivors: int,
    seed: int = 0,
) -> torch.Tensor:
    """
    Generate a soft binary mask (0.01/0.99) with *n_survivors* random directions on.

    Surviving directions get weight 0.99 (matching optimized-mask init),
    the rest get 0.01.

    :param n_dirs: Total number of directions
    :type n_dirs: int
    :param n_survivors: How many directions to mark as surviving
    :type n_survivors: int
    :param seed: RNG seed for reproducibility
    :type seed: int
    :returns: Mask tensor of shape ``(n_dirs,)``
    :rtype: torch.Tensor
    """
    gen = torch.Generator().manual_seed(seed)
    mask = torch.full((n_dirs,), 0.01)
    indices = torch.randperm(n_dirs, generator=gen)[:n_survivors]
    mask[indices] = 0.99
    return mask


def mask_agreement(
    masks_a: dict[str, dict[str, torch.Tensor]],
    masks_b: dict[str, dict[str, torch.Tensor]],
    layer: str,
    category: str,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """
    Compare two runs' masks for the same layer and category.

    Returns Jaccard similarity of survivor sets, Pearson correlation
    of raw weights, and set-difference counts.

    :param masks_a: First run's masks, ``{layer: {category: tensor}}``
    :type masks_a: dict[str, dict[str, torch.Tensor]]
    :param masks_b: Second run's masks, ``{layer: {category: tensor}}``
    :type masks_b: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name
    :type layer: str
    :param category: Category name
    :type category: str
    :param threshold: Survivor threshold
    :type threshold: float
    :returns: Dict with ``jaccard``, ``weight_correlation``, ``shared``,
        ``only_a``, ``only_b``, ``survivors_a``, ``survivors_b``.
    :rtype: dict[str, float | int]
    """
    ma = masks_a[layer][category]
    mb = masks_b[layer][category]
    sa = set(torch.where(ma > threshold)[0].tolist())
    sb = set(torch.where(mb > threshold)[0].tolist())
    union = len(sa | sb)
    jaccard = len(sa & sb) / union if union else 1.0
    correlation = torch.corrcoef(torch.stack([ma.float(), mb.float()]))[0, 1].item()
    return {
        "jaccard": jaccard,
        "weight_correlation": correlation,
        "shared": len(sa & sb),
        "only_a": len(sa - sb),
        "only_b": len(sb - sa),
        "survivors_a": len(sa),
        "survivors_b": len(sb),
    }


def compare_categories(
    masks: dict[str, dict[str, torch.Tensor]],
    layer: str,
    cat_a: str,
    cat_b: str,
    threshold: float = 0.5,
) -> dict[str, list[int]]:
    """
    Compare survivors between two categories on the same layer.

    :param masks: Loaded DM masks, ``{layer: {category: tensor}}``
    :type masks: dict[str, dict[str, torch.Tensor]]
    :param layer: Layer name
    :type layer: str
    :param cat_a: First category
    :type cat_a: str
    :param cat_b: Second category
    :type cat_b: str
    :param threshold: Survivor threshold
    :type threshold: float
    :returns: Dict with keys ``shared``, ``only_{cat_a}``, ``only_{cat_b}`` —
        sorted lists of direction indices
    :rtype: dict[str, list[int]]
    """
    sa = set(torch.where(masks[layer][cat_a] > threshold)[0].tolist())
    sb = set(torch.where(masks[layer][cat_b] > threshold)[0].tolist())
    return {
        "shared": sorted(sa & sb),
        f"only_{cat_a}": sorted(sa - sb),
        f"only_{cat_b}": sorted(sb - sa),
    }


def mask_baseline_table(
    run_dir: str | Path,
) -> pd.DataFrame:
    """
    Optimized vs random mask KL across thresholds.

    Loads ``baseline_kl.json`` from a DM run directory and returns a
    summary DataFrame comparing the optimized mask KL to random mask
    KL at each binarisation threshold.

    :param run_dir: DM run output directory
    :type run_dir: str | Path
    :returns: DataFrame with columns: category, layer, threshold,
        n_survivors, optimized_kl, random_kl, ratio
    :rtype: pd.DataFrame
    """
    run_dir = Path(run_dir)
    with open(run_dir / "baseline_kl.json") as f:
        data = json.load(f)

    rows = []
    for cat, cat_data in sorted(data.items()):
        for thr_key, thr_data in sorted(cat_data.items()):
            if thr_key == "continuous":
                for layer, kl in thr_data["optimized_kl"].items():
                    rows.append({
                        "category": cat,
                        "layer": layer,
                        "threshold": "continuous",
                        "n_survivors": None,
                        "optimized_kl": kl,
                        "random_kl": None,
                        "ratio": None,
                    })
                continue
            for layer in thr_data.get("optimized_kl", {}):
                opt_kl = thr_data["optimized_kl"][layer]
                rand_kl = thr_data["random_kl_mean"][layer]
                n_surv = thr_data["survivors"][layer]
                ratio = opt_kl / rand_kl if rand_kl > 0 else float("inf")
                rows.append({
                    "category": cat,
                    "layer": layer,
                    "threshold": float(thr_key),
                    "n_survivors": n_surv,
                    "optimized_kl": opt_kl,
                    "random_kl": rand_kl,
                    "ratio": ratio,
                })

    return pd.DataFrame(rows)
