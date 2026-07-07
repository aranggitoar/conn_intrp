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
    load_ablation: Load ablation results from a run directory
    joint_kl_table: Active vs random KL summary across categories and thresholds
    baseline_comparison: Zero vs cat-mean vs global-mean vs random KL per category
    cumulative_kl: Per-direction KL sorted by DM weight for coding-regime comparison
    gold_prob_summary: Gold log-prob change distribution per category
    topk_botk_summary: Aggregated top-K/bottom-K token shifts per category
    anls_summary: ANLS change per direction or per category
    super_additivity: Joint vs sum-of-individual KL ratios
    most_changed_directions: Directions with largest individual KL per category
    kl_budget: Active-set KL as fraction of total-layer KL budget
    delta_to_prob_change: Convert log-prob deltas to multiplier or percentage
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_ablation(run_dir: str | Path) -> dict:
    """
    Load all ablation results from a run directory.

    :param run_dir: Path to the ablation output directory
    :type run_dir: str | Path
    :returns: ``{category: {"anls": dict, "joint": dict, "delta_logits": dict,
        "joint_delta_logits": dict, "total": dict}}``
    :rtype: dict
    """
    run_dir = Path(run_dir)
    result = {}
    for cat_dir in sorted(run_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if not (cat_dir / "anls_summary.json").exists():
            continue
        cat = cat_dir.name
        entry = {}
        with open(cat_dir / "anls_summary.json") as f:
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
        total_path = cat_dir / "total_ablation.json"
        if total_path.exists():
            with open(total_path) as f:
                entry["total"] = json.load(f)
        result[cat] = entry
    return result


def _layer_names(abl: dict) -> list[str]:
    """Infer layer names from the first category's direction structure."""
    cat = next(iter(abl))
    dirs = abl[cat]["anls"]["directions"]
    if isinstance(dirs, dict):
        return list(dirs.keys())
    return ["proj"]


def _direction_list(abl: dict, category: str) -> dict[str, list[int]]:
    """Return ``{layer: [dir_indices]}`` for a category, wrapping flat lists under ``"proj"``."""
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

    :param abl: Loaded ablation results, ``{category: {anls, joint, ...}}``
    :type abl: dict
    :param threshold: Binarisation threshold
    :type threshold: float
    :param baseline: Ablation baseline (``"cat"``, ``"global"``, ``"zero"``, ``"rand"``)
    :type baseline: str
    :returns: DataFrame with columns: category, layer, n_dirs, active_kl,
        random_kl, ratio
    :rtype: pd.DataFrame
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

    :param abl: Loaded ablation results, ``{category: {anls, joint, ...}}``
    :type abl: dict
    :param threshold: Binarisation threshold
    :type threshold: float
    :param set_type: ``"active"`` or ``"random"``
    :type set_type: str
    :returns: DataFrame with columns: category, layer, kl_zero, kl_cat,
        kl_global, kl_rand
    :rtype: pd.DataFrame
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

    :param abl: Loaded ablation results, ``{category: {anls, joint, ...}}``
    :type abl: dict
    :param category: Category name
    :type category: str
    :param baseline: Ablation baseline
    :type baseline: str
    :param dm_masks: DM mask weights, ``{layer: {category: tensor}}``, optional
    :type dm_masks: dict | None
    :returns: DataFrame with columns: layer, direction, kl, mask_weight,
        cumulative_kl, cumulative_frac
    :rtype: pd.DataFrame
    """
    dl = abl[category].get("delta_logits", {})
    layers = _layer_names(abl)
    dir_list = _direction_list(abl, category)
    kl_key = f"kl_div_{baseline}"

    rows = []
    for layer in layers:
        layer_dl = _get_layer_dl(dl, layer)

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
    group_by: str = "category",
    representation: str = "multiplier",
) -> pd.DataFrame:
    """
    Gold log-prob change distribution.

    By default reports probability multipliers (e.g. 0.6 means gold
    token probability became 60% of original after ablation).  Set
    *representation* to ``"nats"`` for raw log-prob deltas or ``"pct"``
    for percentage change.

    :param abl: Loaded ablation results, ``{category: {anls, joint, ...}}``
    :type abl: dict
    :param baseline: Ablation baseline
    :type baseline: str
    :param level: ``"joint"`` for joint ablation, ``"individual"`` for
        per-direction results
    :type level: str
    :param threshold: Binarisation threshold (only for ``level="joint"``)
    :type threshold: float
    :param group_by: ``"category"`` (default) summarises across
        directions; ``"direction"`` returns one row per direction
        with distribution stats across images.  Only applies to
        ``level="individual"``.
    :type group_by: str
    :param representation: ``"multiplier"`` (default) — probability
        ratio after/before; ``"pct"`` — percentage change;
        ``"nats"`` — raw log-prob delta
    :type representation: str
    :returns: DataFrame with columns: category, layer, mean, median, std,
        q25, q75, min, max, n_images (plus direction when
        ``group_by="direction"``)
    :rtype: pd.DataFrame
    """
    rows = []
    key = f"delta_gold_prob_{baseline}"

    def _convert(vals):
        if representation != "nats":
            return delta_to_prob_change(vals, mode=representation)
        return vals

    for cat, data in sorted(abl.items()):
        if level == "joint":
            set_key = f"active_{threshold}"
            sets = data.get("joint", {}).get("sets", {})
            if set_key not in sets:
                continue
            for layer in sets[set_key]:
                vals = np.array(sets[set_key][layer][key])
                rows.append(_dist_row(cat, layer, _convert(vals)))
        elif group_by == "direction":
            dl = data.get("delta_logits", {})
            layers = _layer_names(abl)
            dir_list = _direction_list(abl, cat)
            for layer in layers:
                layer_dl = _get_layer_dl(dl, layer)
                for d_idx in dir_list[layer]:
                    d_data = layer_dl.get(d_idx, {})
                    vals = d_data.get(key)
                    if vals is None:
                        continue
                    if isinstance(vals, torch.Tensor):
                        vals = vals.numpy()
                    else:
                        vals = np.array(vals)
                    row = _dist_row(cat, layer, _convert(vals))
                    row["direction"] = d_idx
                    rows.append(row)
        else:
            dl = data.get("delta_logits", {})
            layers = _layer_names(abl)
            dir_list = _direction_list(abl, cat)
            for layer in layers:
                layer_dl = _get_layer_dl(dl, layer)
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
                    rows.append(_dist_row(cat, layer, _convert(arr)))

    return pd.DataFrame(rows)


def anls_summary(
    abl: dict,
    *,
    baseline: str = "cat",
    level: str = "joint",
    threshold: float = 0.7,
    group_by: str = "category",
) -> pd.DataFrame:
    """
    ANLS change summary from ablation results.

    :param abl: Loaded ablation results
    :type abl: dict
    :param baseline: Ablation baseline (``"cat"``, ``"zero"``, ``"global"``, ``"rand"``)
    :type baseline: str
    :param level: ``"joint"`` or ``"individual"``
    :type level: str
    :param threshold: Binarisation threshold (joint only)
    :type threshold: float
    :param group_by: ``"category"`` summarises per-direction deltas
        as a distribution; ``"direction"`` returns one row per direction.
        Only applies to ``level="individual"``.
    :type group_by: str
    :returns: DataFrame with ANLS original, ablated, and delta columns
    :rtype: pd.DataFrame
    """
    _baseline_map = {
        "cat": "anls_ablated_per_category_mean",
        "zero": "anls_ablated_zero_mean",
        "global": "anls_ablated_global_mean",
        "rand": "anls_ablated_random",
    }
    rows = []

    for cat, data in sorted(abl.items()):
        anls = data.get("anls", {})
        orig = anls.get("anls_original", float("nan"))

        if level == "joint":
            set_key = f"active_{threshold}"
            sets = data.get("joint", {}).get("sets", {})
            if set_key not in sets:
                continue
            for layer in sets[set_key]:
                ablated = sets[set_key][layer].get(f"anls_{baseline}", float("nan"))
                rows.append({
                    "category": cat,
                    "layer": layer,
                    "n_dirs": sets[set_key][layer].get("n_directions", 0),
                    "anls_original": orig,
                    "anls_ablated": ablated,
                    "delta": orig - ablated,
                })
        else:
            abl_key = _baseline_map.get(baseline, _baseline_map["cat"])
            abl_vals_by_layer = anls.get(abl_key, {})
            dirs_by_layer = anls.get("directions", {})
            if isinstance(dirs_by_layer, list):
                dirs_by_layer = {"proj": dirs_by_layer}
                abl_vals_by_layer = {"proj": abl_vals_by_layer}

            for layer in dirs_by_layer:
                dir_list = dirs_by_layer[layer]
                abl_vals = abl_vals_by_layer.get(layer, [])
                if not abl_vals:
                    continue

                if group_by == "direction":
                    for i, d_idx in enumerate(dir_list):
                        ablated = abl_vals[i]
                        rows.append({
                            "category": cat,
                            "layer": layer,
                            "direction": d_idx,
                            "anls_original": orig,
                            "anls_ablated": ablated,
                            "delta": orig - ablated,
                        })
                else:
                    deltas = np.array([orig - v for v in abl_vals])
                    rows.append({
                        "category": cat,
                        "layer": layer,
                        "anls_original": orig,
                        "mean_delta": deltas.mean(),
                        "median_delta": np.median(deltas),
                        "std_delta": deltas.std(),
                        "max_delta": deltas.max(),
                        "min_delta": deltas.min(),
                        "n_dirs": len(deltas),
                    })

    return pd.DataFrame(rows)


def delta_to_prob_change(
    delta: np.ndarray | float,
    mode: str = "multiplier",
) -> np.ndarray | float:
    """
    Convert log-prob deltas to interpretable probability change.

    Delta is ``lp_orig - lp_ablated`` (positive means ablation hurt).

    :param delta: Raw delta in nats (scalar or array)
    :type delta: np.ndarray | float
    :param mode: ``"multiplier"`` — probability ratio (0.6 means
        probability became 60% of original; 2.0 means doubled).
        ``"pct"`` — percentage change (−40% means decreased by 40%;
        +100% means doubled).  Asymmetric: decreases cap at −100%,
        increases are unbounded.
    :type mode: str
    :returns: Converted values, same shape as *delta*
    :rtype: np.ndarray | float
    """
    ratio = np.exp(-np.asarray(delta))
    if mode == "multiplier":
        return ratio
    return (ratio - 1) * 100


def _get_layer_dl(dl: dict, layer: str) -> dict:
    """Return the delta-logits sub-dict for a given layer."""
    if layer in dl and isinstance(dl[layer], dict):
        return dl[layer]
    return dl


def _dist_row(cat: str, layer: str, vals: np.ndarray) -> dict:
    """Build a distribution summary dict for one category-layer pair."""
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
        "n_images": len(vals),
    }


def topk_botk_summary(
    abl: dict,
    category: str,
    *,
    baseline: str = "cat",
    level: str = "joint",
    threshold: float = 0.7,
    k: int = 10,
    group_by: str = "category",
    tokenizer=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregated top-K and bottom-K token shifts for a category.

    Counts how often each token appears in the top/bottom-K across images,
    with mean logit delta and mean probability change.

    :param abl: Loaded ablation results, ``{category: {anls, joint, ...}}``
    :type abl: dict
    :param category: Category name
    :type category: str
    :param baseline: Ablation baseline
    :type baseline: str
    :param level: ``"joint"`` or ``"individual"``
    :type level: str
    :param threshold: Binarisation threshold (joint only)
    :type threshold: float
    :param k: Number of top/bottom tokens to aggregate over
    :type k: int
    :param group_by: ``"category"`` (default) pools all directions;
        ``"direction"`` returns per-direction rows with extra
        ``layer`` and ``direction`` columns.  Only applies to
        ``level="individual"``.
    :type group_by: str
    :param tokenizer: HuggingFace tokenizer for decoding token IDs
    :returns: ``(topk_df, botk_df)`` each with columns: token_id, token,
        count, frequency, mean_delta, mean_prob_orig, mean_prob_ablated
        (plus layer, direction when ``group_by="direction"``)
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    topk_key = f"topk_{baseline}"
    botk_key = f"botk_{baseline}"

    def _aggregate(tensor: torch.Tensor, prefix: dict | None = None) -> pd.DataFrame:
        n_images = tensor.shape[0]
        K = min(k, tensor.shape[2])
        token_ids = tensor[:, 0, :K].long().reshape(-1).tolist()
        deltas = tensor[:, 1, :K].reshape(-1).tolist()
        prob_orig = tensor[:, 2, :K].reshape(-1).tolist()
        prob_abl = tensor[:, 3, :K].reshape(-1).tolist()

        acc = defaultdict(lambda: {"count": 0, "delta_sum": 0.0, "po_sum": 0.0, "pa_sum": 0.0})
        for tid, d, po, pa in zip(token_ids, deltas, prob_orig, prob_abl):
            acc[tid]["count"] += 1
            acc[tid]["delta_sum"] += d
            acc[tid]["po_sum"] += po
            acc[tid]["pa_sum"] += pa

        rows = []
        for tid, v in sorted(acc.items(), key=lambda x: -x[1]["count"]):
            token_str = tokenizer.decode([tid]) if tokenizer else str(tid)
            row = {}
            if prefix:
                row.update(prefix)
            row.update({
                "token_id": tid,
                "token": token_str,
                "count": v["count"],
                "frequency": v["count"] / n_images,
                "mean_delta": v["delta_sum"] / v["count"],
                "mean_prob_orig": v["po_sum"] / v["count"],
                "mean_prob_ablated": v["pa_sum"] / v["count"],
            })
            rows.append(row)
        return pd.DataFrame(rows)

    data = abl[category]

    if level == "joint":
        set_key = f"active_{threshold}"
        sets = data["joint"]["sets"]
        layers = list(sets[set_key].keys())
        layer = layers[-1]
        jdl = data["joint_delta_logits"][set_key][layer]
        top_df = _aggregate(jdl[topk_key])
        bot_df = _aggregate(jdl[botk_key])
    elif group_by == "direction":
        dl = data["delta_logits"]
        layers = _layer_names(abl)
        dir_list = _direction_list(abl, category)
        top_parts = []
        bot_parts = []
        for layer in layers:
            layer_dl = dl if layer == "proj" else dl.get(layer, {})
            for d_idx in dir_list[layer]:
                d_data = layer_dl.get(d_idx, {})
                prefix = {"layer": layer, "direction": d_idx}
                if topk_key in d_data:
                    top_parts.append(_aggregate(d_data[topk_key], prefix))
                if botk_key in d_data:
                    bot_parts.append(_aggregate(d_data[botk_key], prefix))
        top_df = pd.concat(top_parts, ignore_index=True) if top_parts else pd.DataFrame()
        bot_df = pd.concat(bot_parts, ignore_index=True) if bot_parts else pd.DataFrame()
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

    :param abl: Loaded ablation results, ``{category: {anls, joint, ...}}``
    :type abl: dict
    :param threshold: Binarisation threshold
    :type threshold: float
    :param baseline: Ablation baseline
    :type baseline: str
    :returns: DataFrame with columns: category, layer, joint_kl,
        sum_individual_kl, ratio, n_joint_dirs, n_matched_dirs
    :rtype: pd.DataFrame
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

    :param abl: Loaded ablation results, ``{category: {anls, joint, ...}}``
    :type abl: dict
    :param baseline: Ablation baseline
    :type baseline: str
    :param n: Number of top directions per category per layer
    :type n: int
    :returns: DataFrame with columns: category, layer, direction, mean_kl, rank
    :rtype: pd.DataFrame
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


def kl_budget(
    abl: dict,
    *,
    threshold: float = 0.7,
    baseline: str = "cat",
    budget_baseline: str = "global",
    level: str = "joint",
) -> pd.DataFrame:
    """
    KL as a fraction of total-layer KL budget.

    Requires ``total_ablation.json`` in the run directory (from
    :func:`~conn_intrp.ablation.run_total_ablation`).

    At ``level="joint"``, reports the active-set KL (from joint
    ablation) as a fraction of total-layer KL, with per-image ratio
    distribution.  At ``level="individual"``, reports each direction's
    individual ablation KL as a fraction of total-layer KL.

    :param abl: Loaded ablation results from :func:`load_ablation`
    :type abl: dict
    :param threshold: Binarisation threshold for the active set
        (only for ``level="joint"``)
    :type threshold: float
    :param baseline: Ablation baseline for the numerator KL
    :type baseline: str
    :param budget_baseline: Which total ablation to use as budget
        denominator — ``"global"`` (recommended, no OOD spoofing) or
        ``"zero"``
    :type budget_baseline: str
    :param level: ``"joint"`` for active-set KL budget, ``"individual"``
        for per-direction KL budget
    :type level: str
    :returns: DataFrame with budget fraction columns.  ``level="joint"``
        includes efficiency and per-image distribution;
        ``level="individual"`` includes direction column.
    :rtype: pd.DataFrame
    """
    budget_key = f"kl_{budget_baseline}"
    rows = []

    if level == "individual":
        kl_div_key = f"kl_div_{baseline}"
        for cat, data in sorted(abl.items()):
            total = data.get("total")
            if total is None:
                continue
            dl = data.get("delta_logits", {})
            layers = _layer_names(abl)
            dir_list = _direction_list(abl, cat)

            for layer in layers:
                if layer not in total.get("layers", {}):
                    continue
                budget_kl = np.array(total["layers"][layer][budget_key])
                budget_mean = budget_kl.mean()
                n_total = total["n_directions"][layer]
                layer_dl = _get_layer_dl(dl, layer)

                for d_idx in dir_list[layer]:
                    d_data = layer_dl.get(d_idx, {})
                    kl_vals = d_data.get(kl_div_key)
                    if kl_vals is None:
                        continue
                    if isinstance(kl_vals, torch.Tensor):
                        dir_kl = np.array(kl_vals.numpy())
                    else:
                        dir_kl = np.array(kl_vals)

                    dir_mean = dir_kl.mean()
                    budget_frac = dir_mean / budget_mean if budget_mean > 0 else float("inf")

                    safe = budget_kl > 1e-8
                    per_image = dir_kl[safe] / budget_kl[safe]

                    rows.append({
                        "category": cat,
                        "layer": layer,
                        "direction": d_idx,
                        "n_total": n_total,
                        "dir_kl": dir_mean,
                        "budget_kl": budget_mean,
                        "budget_frac": budget_frac,
                        "per_image_median": np.median(per_image) if len(per_image) > 0 else float("nan"),
                        "per_image_q25": np.percentile(per_image, 25) if len(per_image) > 0 else float("nan"),
                        "per_image_q75": np.percentile(per_image, 75) if len(per_image) > 0 else float("nan"),
                    })
    else:
        set_key = f"active_{threshold}"
        kl_key = f"kl_{baseline}"

        for cat, data in sorted(abl.items()):
            total = data.get("total")
            sets = data.get("joint", {}).get("sets", {})
            if total is None or set_key not in sets:
                continue

            for layer in sets[set_key]:
                if layer not in total.get("layers", {}):
                    continue

                act = sets[set_key][layer]
                act_kl = np.array(act[kl_key])
                budget_kl = np.array(total["layers"][layer][budget_key])
                zero_kl = np.array(total["layers"][layer]["kl_zero"])
                global_kl = np.array(total["layers"][layer]["kl_global"])

                n_active = act["n_directions"]
                n_total = total["n_directions"][layer]
                dir_frac = n_active / n_total

                act_mean = act_kl.mean()
                budget_mean = budget_kl.mean()
                budget_frac = act_mean / budget_mean if budget_mean > 0 else float("inf")

                safe = budget_kl > 1e-8
                per_image = act_kl[safe] / budget_kl[safe]

                rows.append({
                    "category": cat,
                    "layer": layer,
                    "n_active": n_active,
                    "n_total": n_total,
                    "dir_frac": dir_frac,
                    "active_kl": act_mean,
                    "budget_kl": budget_mean,
                    "budget_frac": budget_frac,
                    "efficiency": budget_frac / dir_frac if dir_frac > 0 else float("inf"),
                    "spoofing_ratio": zero_kl.mean() / global_kl.mean() if global_kl.mean() > 0 else float("inf"),
                    "per_image_median": np.median(per_image) if len(per_image) > 0 else float("nan"),
                    "per_image_q25": np.percentile(per_image, 25) if len(per_image) > 0 else float("nan"),
                    "per_image_q75": np.percentile(per_image, 75) if len(per_image) > 0 else float("nan"),
                })

    return pd.DataFrame(rows)
