"""
Analysis utilities for spatial probe results.

Loads saved probe projections and provides per-direction selectivity
metrics, cross-reference with ablation importance, and inter-direction
spatial clustering.

Example::

    >>> from conn_intrp import load_probe, probe_selectivity_table
    >>> probe = load_probe("outputs/internvl3_5_probe_20260628_164227")
    >>> probe_selectivity_table(probe)

Main Functions:
    load_probe: Load probe projections from a run directory
    probe_selectivity_table: Per-direction spatial selectivity summary
    probe_ablation_cross: Cross-reference spatial selectivity with ablation KL
    probe_direction_clusters: Pairwise spatial similarity and cluster assignment
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_probe(run_dir: str | Path) -> dict:
    """
    Load spatial probe results from a run directory.

    :param run_dir: Path to the probe output directory
    :type run_dir: str | Path
    :returns: ``{category: {"meta": dict, "projections": {layer: Tensor}}}``
        where projections have shape ``(n_images, n_patches, n_dirs)``
    :rtype: dict
    """
    run_dir = Path(run_dir)
    result = {}
    for cat_dir in sorted(run_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        meta_path = cat_dir / "probe_meta.json"
        proj_path = cat_dir / "probe_projections.pt"
        if not meta_path.exists() or not proj_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        projections = torch.load(proj_path, map_location="cpu", weights_only=True)
        result[cat_dir.name] = {"meta": meta, "projections": projections}
    return result


def probe_selectivity_table(
    probe: dict,
    *,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """
    Per-direction spatial selectivity summary.

    :param probe: Loaded probe results from :func:`load_probe`
    :type probe: dict
    :param categories: Subset of categories to include (default: all)
    :type categories: list[str] | None
    :returns: DataFrame with columns: category, layer, direction,
        spatial_cv, entropy, per_image_r, peak_row, peak_col, center_bias
    :rtype: pd.DataFrame
    """
    rows = []
    cats = sorted(categories or probe.keys())

    for cat in cats:
        if cat not in probe:
            continue
        meta = probe[cat]["meta"]
        projections = probe[cat]["projections"]
        grid_size = meta["grid_size"]
        n_patches = grid_size * grid_size

        center_start = grid_size // 4
        center_end = grid_size - center_start
        center_mask = np.zeros(n_patches, dtype=bool)
        for r in range(center_start, center_end):
            for c in range(center_start, center_end):
                center_mask[r * grid_size + c] = True

        for layer_name, proj in projections.items():
            data = proj.numpy()
            dir_indices = meta["layers"][layer_name]["directions"]

            for i, d_idx in enumerate(dir_indices):
                img_heatmaps = data[:, :, i]
                mean_hm = img_heatmaps.mean(axis=0)
                abs_mean = np.abs(mean_hm)

                cv = abs_mean.std() / (abs_mean.mean() + 1e-10)

                p = abs_mean / (abs_mean.sum() + 1e-10)
                raw_entropy = -(p * np.log(p + 1e-10)).sum()
                max_entropy = np.log(n_patches)
                entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

                corrs = []
                for img_idx in range(data.shape[0]):
                    hm = img_heatmaps[img_idx]
                    if hm.std() > 1e-10 and mean_hm.std() > 1e-10:
                        corrs.append(np.corrcoef(hm, mean_hm)[0, 1])
                per_image_r = float(np.mean(corrs)) if corrs else 0.0

                peak_idx = int(np.argmax(abs_mean))
                peak_row = peak_idx // grid_size
                peak_col = peak_idx % grid_size

                center_act = abs_mean[center_mask].mean()
                all_act = abs_mean.mean()
                center_bias = center_act / (all_act + 1e-10)

                rows.append({
                    "category": cat,
                    "layer": layer_name,
                    "direction": d_idx,
                    "spatial_cv": cv,
                    "entropy": entropy,
                    "per_image_r": per_image_r,
                    "peak_row": peak_row,
                    "peak_col": peak_col,
                    "center_bias": center_bias,
                })

    return pd.DataFrame(rows)


def probe_ablation_cross(
    probe: dict,
    abl: dict,
    *,
    baseline: str = "cat",
) -> pd.DataFrame:
    """
    Cross-reference spatial selectivity with individual ablation KL.

    For each probed direction that also has individual ablation data,
    reports spatial metrics alongside KL.  Includes a per-layer Spearman
    rank correlation between KL and spatial CV (same value for every
    direction in a layer, summarising the layer-level relationship).

    :param probe: Loaded probe results from :func:`load_probe`
    :type probe: dict
    :param abl: Loaded ablation results from
        :func:`~conn_intrp.ablation_analysis.load_ablation`
    :type abl: dict
    :param baseline: Ablation baseline for KL (``"cat"``, ``"global"``,
        ``"zero"``, ``"rand"``)
    :type baseline: str
    :returns: DataFrame with columns: category, layer, direction,
        indiv_kl, spatial_cv, per_image_r, spearman_kl_cv
    :rtype: pd.DataFrame
    """
    kl_key = f"kl_div_{baseline}"
    rows = []

    for cat in sorted(probe.keys()):
        if cat not in abl:
            continue
        meta = probe[cat]["meta"]
        projections = probe[cat]["projections"]
        dl = abl[cat].get("delta_logits", {})

        for layer_name, proj in projections.items():
            data = proj.numpy()
            dir_indices = meta["layers"][layer_name]["directions"]
            layer_dl = dl if layer_name == "proj" else dl.get(layer_name, {})

            layer_rows = []
            for i, d_idx in enumerate(dir_indices):
                d_data = layer_dl.get(d_idx, {})
                kl_vals = d_data.get(kl_key)
                if kl_vals is None:
                    continue
                if isinstance(kl_vals, torch.Tensor):
                    mean_kl = kl_vals.mean().item()
                else:
                    mean_kl = float(np.mean(kl_vals))

                img_heatmaps = data[:, :, i]
                mean_hm = img_heatmaps.mean(axis=0)
                abs_mean = np.abs(mean_hm)
                cv = abs_mean.std() / (abs_mean.mean() + 1e-10)

                corrs = []
                for img_idx in range(data.shape[0]):
                    hm = img_heatmaps[img_idx]
                    if hm.std() > 1e-10 and mean_hm.std() > 1e-10:
                        corrs.append(np.corrcoef(hm, mean_hm)[0, 1])
                per_image_r = float(np.mean(corrs)) if corrs else 0.0

                layer_rows.append({
                    "category": cat,
                    "layer": layer_name,
                    "direction": d_idx,
                    "indiv_kl": mean_kl,
                    "spatial_cv": cv,
                    "per_image_r": per_image_r,
                })

            spearman = float("nan")
            matched = [r for r in layer_rows if r["indiv_kl"] > 0]
            if len(matched) > 3:
                kls = np.array([r["indiv_kl"] for r in matched])
                cvs = np.array([r["spatial_cv"] for r in matched])
                spearman = _spearman(kls, cvs)

            for r in layer_rows:
                r["spearman_kl_cv"] = spearman
                rows.append(r)

    return pd.DataFrame(rows)


def probe_direction_clusters(
    probe: dict,
    *,
    threshold: float = 0.7,
    categories: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise spatial similarity and cluster assignment for directions.

    Computes cosine similarity between directions' mean spatial heatmaps
    (averaged across categories to exploit the universal-pattern finding),
    then assigns cluster labels via connected components at the given
    similarity threshold.

    :param probe: Loaded probe results from :func:`load_probe`
    :type probe: dict
    :param threshold: Cosine similarity threshold for same-cluster
        membership (default 0.7)
    :type threshold: float
    :param categories: Subset of categories to average over (default: all)
    :type categories: list[str] | None
    :returns: ``(assignments, similarities)`` — assignments has columns:
        layer, direction, cluster, cluster_size, mean_intra_sim;
        similarities has columns: layer, dir_a, dir_b, cosine_sim
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    cats = sorted(c for c in (categories or probe.keys()) if c in probe)

    layer_accum: dict[str, dict[int, list[np.ndarray]]] = {}
    layer_dir_order: dict[str, list[int]] = {}

    for cat in cats:
        meta = probe[cat]["meta"]
        projections = probe[cat]["projections"]
        for layer_name, proj in projections.items():
            data = proj.numpy()
            dir_indices = meta["layers"][layer_name]["directions"]
            if layer_name not in layer_accum:
                layer_accum[layer_name] = {}
                layer_dir_order[layer_name] = dir_indices
            for i, d_idx in enumerate(dir_indices):
                mean_hm = data[:, :, i].mean(axis=0)
                if d_idx not in layer_accum[layer_name]:
                    layer_accum[layer_name][d_idx] = []
                layer_accum[layer_name][d_idx].append(mean_hm)

    assign_rows = []
    sim_rows = []

    for layer_name in sorted(layer_accum.keys()):
        dir_hms = layer_accum[layer_name]
        dir_indices = layer_dir_order[layer_name]
        n = len(dir_indices)

        grand_means = np.stack([
            np.mean(dir_hms[d], axis=0) for d in dir_indices
        ])

        norms = np.linalg.norm(grand_means, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = grand_means / norms
        sim_matrix = normed @ normed.T

        for i in range(n):
            for j in range(i + 1, n):
                sim_rows.append({
                    "layer": layer_name,
                    "dir_a": dir_indices[i],
                    "dir_b": dir_indices[j],
                    "cosine_sim": float(sim_matrix[i, j]),
                })

        adj = sim_matrix >= threshold
        clusters = _connected_components(adj)

        for i, d_idx in enumerate(dir_indices):
            c = clusters[i]
            members = [j for j in range(n) if clusters[j] == c]
            cluster_size = len(members)
            if cluster_size > 1:
                intra_sims = [float(sim_matrix[i, j]) for j in members if j != i]
                mean_intra = float(np.mean(intra_sims))
            else:
                mean_intra = 1.0

            assign_rows.append({
                "layer": layer_name,
                "direction": d_idx,
                "cluster": c,
                "cluster_size": cluster_size,
                "mean_intra_sim": mean_intra,
            })

    return pd.DataFrame(assign_rows), pd.DataFrame(sim_rows)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without scipy."""
    n = len(a)
    rank_a = np.empty(n)
    rank_a[np.argsort(a)] = np.arange(n)
    rank_b = np.empty(n)
    rank_b[np.argsort(b)] = np.arange(n)
    d = rank_a - rank_b
    return float(1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1)))


def _connected_components(adj: np.ndarray) -> list[int]:
    """Connected component labels from a boolean adjacency matrix."""
    n = adj.shape[0]
    labels = [-1] * n
    current = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        stack = [start]
        while stack:
            node = stack.pop()
            if labels[node] >= 0:
                continue
            labels[node] = current
            for neighbor in range(n):
                if adj[node, neighbor] and labels[neighbor] < 0:
                    stack.append(neighbor)
        current += 1
    return labels
