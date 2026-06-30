"""
Analysis utilities for ablation results.

Loads saved ablation outputs and provides composable functions for
summary statistics, cross-model comparison, and chart-ready DataFrames.

Example::

    >>> from conn_intrp.ablation_analysis import load_ablation, joint_kl_table, cumulative_kl
    >>> abl = load_ablation("outputs/internvl3_5_ablation_20260624_025712")
    >>> joint_kl_table(abl, threshold=0.7)
    >>> cumulative_kl(abl, "figure_diagram")

Main Functions:
    load_ablation: Load all ablation results from a run directory.
    joint_kl_table: Joint ablation KL summary across categories/thresholds.
    baseline_comparison: Zero vs mean vs random KL per category.
    cumulative_kl: Per-direction KL sorted by DM weight (coding regime curve).
    gold_prob_summary: Gold log-prob change distribution per category.
    topk_botk_summary: Aggregated top-K/bottom-K token shifts per category.
    super_additivity: Joint vs sum-of-individual KL ratios (MLP connectors).
    most_changed_directions: Directions with largest individual KL per category.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_ablation(run_dir: str | Path) -> dict:
    """
    Load all ablation results from a run directory.

    :param run_dir: Path to the ablation output directory.
    :returns: ``{category: {"anls": dict, "joint": dict, "delta_logits": dict,
        "joint_delta_logits": dict}}``
    """
    run_dir = Path(run_dir)
    result = {}
    for cat_dir in sorted(run_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        entry = {}
        anls_path = cat_dir / "anls_summary.json"
        if anls_path.exists():
            with open(anls_path) as f:
                entry["anls"] = json.load(f)
        joint_path = cat_dir / "joint_anls_summary.json"
        if joint_path.exists():
            with open(joint_path) as f:
                entry["joint"] = json.load(f)
        dl_path = cat_dir / "delta_logits.pt"
        if dl_path.exists():
            entry["delta_logits"] = torch.load(dl_path, map_location="cpu", weights_only=True)
        jdl_path = cat_dir / "joint_delta_logits.pt"
        if jdl_path.exists():
            entry["joint_delta_logits"] = torch.load(jdl_path, map_location="cpu", weights_only=True)
        result[cat] = entry
    return result


def _layer_names(abl: dict) -> list[str]:
    cat = next(iter(abl))
    dirs = abl[cat]["anls"]["directions"]
    if isinstance(dirs, dict):
        return list(dirs.keys())
    return ["proj"]


def _direction_list(abl: dict, category: str) -> dict[str, list[int]]:
    dirs = abl[category]["anls"]["directions"]
    if isinstance(dirs, dict):
        return dirs
    return {"proj": dirs}


def joint_kl_table(
    abl: dict,
    *,
    threshold: float = 0.7,
    baseline: str = "cat",
) -> pd.DataFrame:
    """
    Joint ablation KL summary: active vs random per category.

    :param abl: Output of :func:`load_ablation`.
    :param threshold: Binarisation threshold.
    :param baseline: Ablation baseline (``"cat"``, ``"global"``, ``"zero"``, ``"rand"``).
    :returns: DataFrame with columns: category, layer, n_dirs, active_kl,
        random_kl, ratio.
    """
    rows = []
    active_key = f"active_{threshold}"
    random_key = f"random_{threshold}"
    for cat, data in sorted(abl.items()):
        joint = data.get("joint", {})
        sets = joint.get("sets", {})
        if active_key not in sets or random_key not in sets:
            continue
        for layer in sets[active_key]:
            act = sets[active_key][layer]
            rnd = sets[random_key][layer]
            kl_key = f"kl_{baseline}"
            act_kl = np.mean(act[kl_key])
            rnd_kl = np.mean(rnd[kl_key])
            ratio = act_kl / rnd_kl if rnd_kl > 0 else float("inf")
            rows.append({
                "category": cat,
                "layer": layer,
                "n_dirs": act["n_directions"],
                "active_kl": act_kl,
                "random_kl": rnd_kl,
                "ratio": ratio,
            })
    return pd.DataFrame(rows)


def baseline_comparison(
    abl: dict,
    *,
    threshold: float = 0.7,
    set_type: str = "active",
) -> pd.DataFrame:
    """
    Compare KL across ablation methods (zero, cat-mean, global-mean, random).

    :param abl: Output of :func:`load_ablation`.
    :param threshold: Binarisation threshold.
    :param set_type: ``"active"`` or ``"random"``.
    :returns: DataFrame with columns: category, layer, kl_zero, kl_cat,
        kl_global, kl_rand.
    """
    rows = []
    set_key = f"{set_type}_{threshold}"
    for cat, data in sorted(abl.items()):
        sets = data.get("joint", {}).get("sets", {})
        if set_key not in sets:
            continue
        for layer in sets[set_key]:
            s = sets[set_key][layer]
            rows.append({
                "category": cat,
                "layer": layer,
                "n_dirs": s["n_directions"],
                "kl_zero": np.mean(s["kl_zero"]),
                "kl_cat": np.mean(s["kl_cat"]),
                "kl_global": np.mean(s["kl_global"]),
                "kl_rand": np.mean(s["kl_rand"]),
            })
    return pd.DataFrame(rows)


def cumulative_kl(
    abl: dict,
    category: str,
    *,
    baseline: str = "cat",
    dm_masks: dict | None = None,
) -> pd.DataFrame:
    """
    Per-direction mean KL sorted by DM mask weight descending.

    When *dm_masks* is provided, directions are sorted by their mask
    weight for this category. Otherwise sorted by KL descending.

    :param abl: Output of :func:`load_ablation`.
    :param category: Category name.
    :param baseline: Ablation baseline.
    :param dm_masks: Output of ``dm_analysis.load_dm_masks``, optional.
    :returns: DataFrame with columns: layer, direction, kl, mask_weight,
        cumulative_kl, cumulative_frac.
    """
    dl = abl[category].get("delta_logits", {})
    layers = _layer_names(abl)
    dir_list = _direction_list(abl, category)
    kl_key = f"kl_div_{baseline}"

    rows = []
    for layer in layers:
        if layer == "proj":
            layer_dl = dl
        else:
            layer_dl = dl.get(layer, {})

        for d_idx in dir_list[layer]:
            d_data = layer_dl.get(d_idx, {})
            kl_vals = d_data.get(kl_key)
            if kl_vals is None:
                continue
            if isinstance(kl_vals, torch.Tensor):
                mean_kl = kl_vals.mean().item()
            else:
                mean_kl = np.mean(kl_vals)

            mask_w = 0.0
            if dm_masks is not None:
                layer_masks = dm_masks.get(layer, {})
                cat_mask = layer_masks.get(category)
                if cat_mask is not None and d_idx < len(cat_mask):
                    mask_w = cat_mask[d_idx].item()

            rows.append({
                "layer": layer,
                "direction": d_idx,
                "kl": mean_kl,
                "mask_weight": mask_w,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    sort_col = "mask_weight" if dm_masks is not None else "kl"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df["cumulative_kl"] = df["kl"].cumsum()
    total = df["cumulative_kl"].iloc[-1]
    df["cumulative_frac"] = df["cumulative_kl"] / total if total > 0 else 0.0
    return df


def gold_prob_summary(
    abl: dict,
    *,
    baseline: str = "cat",
    level: str = "joint",
    threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Gold log-prob change distribution per category.

    :param abl: Output of :func:`load_ablation`.
    :param baseline: Ablation baseline.
    :param level: ``"joint"`` for joint ablation, ``"individual"`` for
        per-direction (returns median across directions).
    :param threshold: Binarisation threshold (only for ``level="joint"``).
    :returns: DataFrame with columns: category, layer, mean, median, std,
        q25, q75, min, max, n_images.
    """
    rows = []
    key = f"delta_gold_prob_{baseline}"

    for cat, data in sorted(abl.items()):
        if level == "joint":
            set_key = f"active_{threshold}"
            sets = data.get("joint", {}).get("sets", {})
            if set_key not in sets:
                continue
            for layer in sets[set_key]:
                vals = np.array(sets[set_key][layer][key])
                rows.append(_dist_row(cat, layer, vals))
        else:
            dl = data.get("delta_logits", {})
            layers = _layer_names(abl)
            dir_list = _direction_list(abl, cat)
            for layer in layers:
                layer_dl = dl if layer == "proj" else dl.get(layer, {})
                all_medians = []
                for d_idx in dir_list[layer]:
                    d_data = layer_dl.get(d_idx, {})
                    vals = d_data.get(key)
                    if vals is None:
                        continue
                    if isinstance(vals, torch.Tensor):
                        all_medians.append(vals.median().item())
                    else:
                        all_medians.append(np.median(vals))
                if all_medians:
                    arr = np.array(all_medians)
                    rows.append(_dist_row(cat, layer, arr))

    return pd.DataFrame(rows)


def _dist_row(cat: str, layer: str, vals: np.ndarray) -> dict:
    return {
        "category": cat,
        "layer": layer,
        "mean": vals.mean(),
        "median": np.median(vals),
        "std": vals.std(),
        "q25": np.percentile(vals, 25),
        "q75": np.percentile(vals, 75),
        "min": vals.min(),
        "max": vals.max(),
        "n": len(vals),
    }


def topk_botk_summary(
    abl: dict,
    category: str,
    *,
    baseline: str = "cat",
    level: str = "joint",
    threshold: float = 0.7,
    k: int = 10,
    tokenizer=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregated top-K and bottom-K token shifts for a category.

    Counts how often each token appears in the top/bottom-K across images,
    with mean logit delta and mean probability change.

    :param abl: Output of :func:`load_ablation`.
    :param category: Category name.
    :param baseline: Ablation baseline.
    :param level: ``"joint"`` or ``"individual"``.
    :param threshold: Binarisation threshold (joint only).
    :param k: Number of top/bottom tokens to aggregate over.
    :param tokenizer: HuggingFace tokenizer for decoding token IDs.
    :returns: ``(topk_df, botk_df)`` each with columns: token_id, token,
        count, mean_delta, mean_prob_orig, mean_prob_ablated.
    """
    topk_key = f"topk_{baseline}"
    botk_key = f"botk_{baseline}"

    def _aggregate(tensor: torch.Tensor) -> pd.DataFrame:
        # (n_images, 4, K) -> channels: token_idx, logit_delta, prob_orig, prob_ablated
        n_images = tensor.shape[0]
        K = min(k, tensor.shape[2])
        token_ids = tensor[:, 0, :K].long().reshape(-1).tolist()
        deltas = tensor[:, 1, :K].reshape(-1).tolist()
        prob_orig = tensor[:, 2, :K].reshape(-1).tolist()
        prob_abl = tensor[:, 3, :K].reshape(-1).tolist()

        from collections import defaultdict
        acc = defaultdict(lambda: {"count": 0, "delta_sum": 0.0, "po_sum": 0.0, "pa_sum": 0.0})
        for tid, d, po, pa in zip(token_ids, deltas, prob_orig, prob_abl):
            acc[tid]["count"] += 1
            acc[tid]["delta_sum"] += d
            acc[tid]["po_sum"] += po
            acc[tid]["pa_sum"] += pa

        rows = []
        for tid, v in sorted(acc.items(), key=lambda x: -x[1]["count"]):
            token_str = tokenizer.decode([tid]) if tokenizer else str(tid)
            rows.append({
                "token_id": tid,
                "token": token_str,
                "count": v["count"],
                "frequency": v["count"] / n_images,
                "mean_delta": v["delta_sum"] / v["count"],
                "mean_prob_orig": v["po_sum"] / v["count"],
                "mean_prob_ablated": v["pa_sum"] / v["count"],
            })
        return pd.DataFrame(rows)

    data = abl[category]

    if level == "joint":
        set_key = f"active_{threshold}"
        sets = data["joint"]["sets"]
        layers = list(sets[set_key].keys())
        # Use last layer (most downstream)
        layer = layers[-1]
        s = sets[set_key][layer]
        jdl = data["joint_delta_logits"][set_key][layer]
        top_df = _aggregate(jdl[topk_key])
        bot_df = _aggregate(jdl[botk_key])
    else:
        dl = data["delta_logits"]
        layers = _layer_names(abl)
        dir_list = _direction_list(abl, category)
        all_top = []
        all_bot = []
        for layer in layers:
            layer_dl = dl if layer == "proj" else dl.get(layer, {})
            for d_idx in dir_list[layer]:
                d_data = layer_dl.get(d_idx, {})
                if topk_key in d_data:
                    all_top.append(d_data[topk_key])
                if botk_key in d_data:
                    all_bot.append(d_data[botk_key])
        if all_top:
            top_df = _aggregate(torch.cat(all_top, dim=0))
        else:
            top_df = pd.DataFrame()
        if all_bot:
            bot_df = _aggregate(torch.cat(all_bot, dim=0))
        else:
            bot_df = pd.DataFrame()

    return top_df, bot_df


def super_additivity(
    abl: dict,
    *,
    threshold: float = 0.7,
    baseline: str = "cat",
) -> pd.DataFrame:
    """
    Joint KL vs sum-of-individual KL ratios.

    For each category and layer, compares the mean KL from joint ablation
    to the sum of per-direction mean KLs for the matched direction set.

    :param abl: Output of :func:`load_ablation`.
    :param threshold: Binarisation threshold.
    :param baseline: Ablation baseline.
    :returns: DataFrame with columns: category, layer, joint_kl,
        sum_individual_kl, ratio, n_joint_dirs, n_matched_dirs.
    """
    kl_key = f"kl_{baseline}"
    kl_div_key = f"kl_div_{baseline}"
    set_key = f"active_{threshold}"

    rows = []
    for cat, data in sorted(abl.items()):
        sets = data.get("joint", {}).get("sets", {})
        if set_key not in sets:
            continue

        dl = data.get("delta_logits", {})
        dir_list = _direction_list(abl, cat)

        for layer in sets[set_key]:
            act = sets[set_key][layer]
            joint_kl = np.mean(act[kl_key])
            joint_dirs = set(act["directions"])
            indiv_dirs = set(dir_list.get(layer, []))
            matched = joint_dirs & indiv_dirs

            sum_kl = 0.0
            layer_dl = dl if layer == "proj" else dl.get(layer, {})
            for d_idx in matched:
                d_data = layer_dl.get(d_idx, {})
                vals = d_data.get(kl_div_key)
                if vals is None:
                    continue
                if isinstance(vals, torch.Tensor):
                    sum_kl += vals.mean().item()
                else:
                    sum_kl += np.mean(vals)

            ratio = joint_kl / sum_kl if sum_kl > 0 else float("inf")
            rows.append({
                "category": cat,
                "layer": layer,
                "joint_kl": joint_kl,
                "sum_individual_kl": sum_kl,
                "ratio": ratio,
                "n_joint_dirs": len(joint_dirs),
                "n_matched_dirs": len(matched),
            })

    return pd.DataFrame(rows)


def most_changed_directions(
    abl: dict,
    *,
    baseline: str = "cat",
    n: int = 5,
) -> pd.DataFrame:
    """
    Directions with the largest individual mean KL, per category.

    :param abl: Output of :func:`load_ablation`.
    :param baseline: Ablation baseline.
    :param n: Number of top directions per category per layer.
    :returns: DataFrame with columns: category, layer, direction, mean_kl, rank.
    """
    kl_key = f"kl_div_{baseline}"
    rows = []

    for cat, data in sorted(abl.items()):
        dl = data.get("delta_logits", {})
        dir_list = _direction_list(abl, cat)
        layers = _layer_names(abl)

        for layer in layers:
            layer_dl = dl if layer == "proj" else dl.get(layer, {})
            dir_kls = []
            for d_idx in dir_list[layer]:
                d_data = layer_dl.get(d_idx, {})
                vals = d_data.get(kl_key)
                if vals is None:
                    continue
                if isinstance(vals, torch.Tensor):
                    mean_kl = vals.mean().item()
                else:
                    mean_kl = np.mean(vals)
                dir_kls.append((d_idx, mean_kl))

            dir_kls.sort(key=lambda x: -x[1])
            for rank, (d_idx, kl) in enumerate(dir_kls[:n], 1):
                rows.append({
                    "category": cat,
                    "layer": layer,
                    "direction": d_idx,
                    "mean_kl": kl,
                    "rank": rank,
                })

    return pd.DataFrame(rows)
