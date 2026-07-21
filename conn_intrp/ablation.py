"""
Mean ablation of SVD directions (Phase 2).

Two-stage pipeline:

1. :func:`compute_category_means` — extract SVD coefficients and compute
   per-category and global mean vectors.
2. :func:`run_ablation` — replace each direction with its mean, reconstruct,
   generate, score ANLS, and extract Δlogit artifacts.

Baselines: per-category mean, global mean, zero, random.

Example::

    >>> from conn_intrp import compute_category_means, run_ablation
    >>> cat_means, global_mean = compute_category_means(
    ...     adapter, categorized, batch_size=4,
    ...     image_base_path=img_path, run_dir=coeff_dir)
    >>> run_ablation(adapter, categorized, cat_means, global_mean,
    ...     directions_to_ablate={"linear_1": [3, 7], "linear_2": [23, 70]},
    ...     batch_size=4, K=15,
    ...     image_base_path=img_path, run_dir=run_dir,
    ...     coefficients_dir=coeff_dir)

Main Functions:
    compute_category_means: SVD coefficients + per-category/global means.
    load_category_coefficients: Load one category's coefficients from checkpoint.
    run_ablation: Per-direction ablation with ANLS scoring and Δlogit extraction.
    run_joint_ablation: Multi-direction set ablation for validating DM masks.
    run_total_ablation: Zero/global-mean ablation of all directions for total KL budget.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .data import best_anls
from .models.base import ModelAdapter
from .output import (
    fs_safe,
    get_completed_categories,
    load_checkpoint,
    save_checkpoint,
    save_json,
    update_metadata,
)

_BASELINE_KEYS = ("cat", "global", "zero", "rand")

_DELTA_SCHEMA = {
    "_note": "* is one of: cat, global, zero, rand (ablation baseline). "
    "Keyed by {layer: {dir_idx: {field: tensor}}}",
    "signed_mean_*": "(vocab_size,) mean logit delta across images",
    "abs_mean_*": "(vocab_size,) mean |logit delta| across images",
    "topk_*": "(n_images, 4, K) top-K positive logit shifts; "
    "channels: [token_idx, logit_delta, prob_orig, prob_ablated]",
    "botk_*": "(n_images, 4, K) top-K negative logit shifts; "
    "channels: [token_idx, logit_delta, prob_orig, prob_ablated]",
    "kl_div_*": "(n_images,) per-image KL divergence",
    "delta_gold_prob_*": "(n_images,) per-image gold token log-prob change",
}

_JOINT_DELTA_SCHEMA = {
    "_note": "* is one of: cat, global, zero, rand (ablation baseline). "
    "Keyed by {set_label: {layer: {field: tensor}}}",
    "signed_mean_*": "(vocab_size,) mean logit delta across images",
    "abs_mean_*": "(vocab_size,) mean |logit delta| across images",
    "topk_*": "(n_images, 4, K) top-K positive logit shifts; "
    "channels: [token_idx, logit_delta, prob_orig, prob_ablated]",
    "botk_*": "(n_images, 4, K) top-K negative logit shifts; "
    "channels: [token_idx, logit_delta, prob_orig, prob_ablated]",
}

_ANLS_SUMMARY_SCHEMA = {
    "_note": "Per-direction single ablation results. "
    "anls_ablated_* is {layer: [per_dir_mean_anls]}. "
    "* is one of: per_category_mean, global_mean, zero_mean, random",
    "category": "str — category name",
    "n_samples": "int — number of images evaluated",
    "directions": "{layer: [dir_indices]} — directions ablated per layer",
    "anls_original": "float — mean ANLS without ablation",
    "anls_ablated_*": "{layer: [float]} — per-direction mean ANLS",
}

_JOINT_ANLS_SUMMARY_SCHEMA = {
    "_note": "Joint ablation results. * suffix is one of: cat, global, zero, rand. "
    "Keyed by sets → {set_label} → {layer}",
    "category": "str — category name",
    "n_samples": "int — number of images evaluated",
    "anls_original": "float — mean ANLS without ablation",
    "sets.{set_label}.{layer}.directions": "list[int] — direction indices in this set",
    "sets.{set_label}.{layer}.n_directions": "int — count of directions",
    "sets.{set_label}.{layer}.anls_*": "float — mean ANLS across images",
    "sets.{set_label}.{layer}.kl_*": "list[float] — per-image KL divergence (length = n_samples)",
    "sets.{set_label}.{layer}.delta_gold_prob_*": "list[float] — per-image gold log-prob change",
}

_TOTAL_ABLATION_SCHEMA = {
    "_note": "Total ablation of all directions in a layer. "
    "Gives the KL budget (upper bound) for interpreting partial ablation results",
    "category": "str — category name",
    "n_samples": "int — number of images evaluated",
    "n_directions": "{layer: int} — total direction count per layer",
    "anls_original": "float — mean ANLS without ablation",
    "layers.{layer}.kl_zero": "list[float] — per-image KL from zero ablation",
    "layers.{layer}.kl_global": "list[float] — per-image KL from global-mean ablation",
    "layers.{layer}.anls_zero": "float — mean ANLS after zero ablation",
    "layers.{layer}.anls_global": "float — mean ANLS after global-mean ablation",
    "layers.{layer}.delta_gold_prob_zero": "list[float] — per-image gold log-prob change (zero)",
    "layers.{layer}.delta_gold_prob_global": "list[float] — per-image gold log-prob change (global)",
}


def _make_accumulators(n_tasks: int, vocab_size: int) -> dict:
    """Create per-baseline accumulators for ablation metrics."""
    return {
        "nls": [[] for _ in range(n_tasks)],
        "delta_sum": [torch.zeros(vocab_size) for _ in range(n_tasks)],
        "delta_abs_sum": [torch.zeros(vocab_size) for _ in range(n_tasks)],
        "topk": [[] for _ in range(n_tasks)],
        "botk": [[] for _ in range(n_tasks)],
        "kl_div": [[] for _ in range(n_tasks)],
        "delta_gold_prob": [[] for _ in range(n_tasks)],
    }


def _load_ckpt_compat(ckpt: dict, component_name: str) -> tuple[
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor] | None,
]:
    """Normalize checkpoint format across versions, returning (coefficients, a_star, global_mean_contrib)."""
    raw_coeff = ckpt.get("coefficients")
    raw_astar = ckpt["a_star"]
    if isinstance(raw_coeff, torch.Tensor):
        raw_coeff = {component_name: raw_coeff}
    if isinstance(raw_astar, torch.Tensor):
        raw_astar = {component_name: raw_astar}
    raw_contrib = ckpt.get("global_mean_contrib")
    if isinstance(raw_contrib, torch.Tensor):
        raw_contrib = {component_name: raw_contrib}
    return raw_coeff, raw_astar, raw_contrib


def load_all_coefficients(
    run_dir: Path,
    category_names: list[str],
    component_name: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Load all categories' per-layer SVD coefficients from checkpoints.

    :param run_dir: Directory containing category checkpoints
    :type run_dir: Path
    :param category_names: Category names to load
    :type category_names: list[str]
    :param component_name: Adapter's ``component_name`` for old-format compat
    :type component_name: str
    :returns: ``{category: {layer_name: Tensor}}`` on CPU
    :rtype: dict[str, dict[str, torch.Tensor]]
    """
    return {
        name: load_category_coefficients(run_dir, name, component_name) for name in category_names
    }


def load_category_coefficients(
    run_dir: Path,
    category_name: str,
    component_name: str,
) -> dict[str, torch.Tensor]:
    """
    Load one category's per-layer SVD coefficients from checkpoint.

    :param run_dir: Directory containing category checkpoints
    :type run_dir: Path
    :param category_name: Category name (will be fs-safe'd)
    :type category_name: str
    :param component_name: Adapter's ``component_name`` for old-format compat
    :type component_name: str
    :returns: ``{layer_name: Tensor}`` on CPU
    :rtype: dict[str, torch.Tensor]
    """
    ckpt = load_checkpoint(run_dir, category_name)
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint for "{category_name}" in {run_dir}')
    raw_coeff, _, _ = _load_ckpt_compat(ckpt, component_name)
    if raw_coeff is None:
        raise ValueError(
            f'Checkpoint for "{category_name}" has no coefficients '
            f"(saved with save_coefficients=False?)"
        )
    return raw_coeff


def compute_category_means(
    adapter: ModelAdapter,
    data_categorized: dict[str, list],
    *,
    batch_size: int,
    image_base_path: Path,
    run_dir: Path,
    save_coefficients: bool = True,
    rerun: bool = False,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, torch.Tensor],
]:
    """
    Compute per-layer SVD coefficients and per-category/global mean vectors.

    Images appearing in multiple categories are counted once for the
    global mean (tracked via a ``seen_images`` set). Supports resuming
    from checkpoints (including old single-layer format).

    When *save_coefficients* is True (default), per-category coefficient
    tensors are included in checkpoints for later use by
    :func:`run_ablation`. Set to False to save only means (smaller
    checkpoints). Use :func:`load_category_coefficients` or
    :func:`load_all_coefficients` to load coefficients on demand.

    :param adapter: Model adapter instance
    :type adapter: ModelAdapter
    :param data_categorized: Map of category name to list of data dicts
    :type data_categorized: dict[str, list]
    :param batch_size: Number of images per forward pass
    :type batch_size: int
    :param image_base_path: Root directory for image files
    :type image_base_path: Path
    :param run_dir: Output directory for this run
    :type run_dir: Path
    :param save_coefficients: Whether to include raw coefficient tensors
        in checkpoints. Default True
    :type save_coefficients: bool
    :param rerun: Ignore existing checkpoints and recompute everything
    :type rerun: bool
    :returns: ``(per_category_a_star, global_a_star)`` where means are
        ``{cat: {layer: Tensor}}`` / ``{layer: Tensor}`` on CUDA
    :rtype: tuple[dict[str, dict[str, torch.Tensor]],
        dict[str, torch.Tensor]]
    """
    layers = adapter.svd_layers
    layer_names = [l.name for l in layers]
    completed = set() if rerun else get_completed_categories(run_dir)

    per_category_a_star: dict[str, dict[str, torch.Tensor]] = {}

    global_mean_sum = {l.name: torch.zeros(l.n_dirs, dtype=torch.float32) for l in layers}
    n_unique_images = 0
    seen_images: set[str] = set()

    for name, data in tqdm(data_categorized.items(), desc="Computing category means"):
        if fs_safe(name) in completed:
            ckpt = load_checkpoint(run_dir, name)
            if ckpt is not None and "a_star" in ckpt:
                raw_coeff, raw_astar, raw_contrib = _load_ckpt_compat(ckpt, adapter.component_name)
                missing = set(layer_names) - set(raw_astar.keys())
                if not missing:
                    per_category_a_star[name] = {
                        ln: raw_astar[ln].to(dtype=torch.float32, device="cuda")
                        for ln in raw_astar
                    }
                    for img_id in ckpt.get("new_image_ids", []):
                        seen_images.add(img_id)
                    if raw_contrib is not None:
                        for ln in raw_contrib:
                            if ln in global_mean_sum:
                                global_mean_sum[ln] += raw_contrib[ln]
                    n_unique_images += ckpt.get("n_new_images", 0)
                    print(f'  Loaded checkpoint for "{name}"')
                    continue
                print(f'  Checkpoint for "{name}" missing layers {missing}' f" — computing")
                new_image_set = set(ckpt.get("new_image_ids", []))
                for img_id in new_image_set:
                    seen_images.add(img_id)
                if raw_contrib is not None:
                    for ln in raw_contrib:
                        if ln in global_mean_sum:
                            global_mean_sum[ln] += raw_contrib[ln]
                n_unique_images += ckpt.get("n_new_images", 0)

                missing_svd = [l for l in layers if l.name in missing]
                cat_coeff_new = {
                    l.name: torch.empty(len(data), adapter.n_patches, l.n_dirs, dtype=torch.float16)
                    for l in missing_svd
                }
                cat_contrib_new = {
                    l.name: torch.zeros(l.n_dirs, dtype=torch.float32) for l in missing_svd
                }
                cat_mean_sum_new = {
                    l.name: torch.zeros(l.n_dirs, dtype=torch.float32) for l in missing_svd
                }
                n_patches_seen_new = 0

                for i in tqdm(
                    range(0, len(data), batch_size),
                    total=math.ceil(len(data) / batch_size),
                    desc=f'"{name}" (missing layers)',
                ):
                    batch = data[i : i + batch_size]
                    inputs = adapter.preprocess(batch, image_base_path)
                    coefficients = adapter.compute_coefficients_per_layer(inputs)
                    actual = next(iter(coefficients.values())).shape[0]
                    for ln in missing:
                        cat_coeff_new[ln][i : i + actual] = coefficients[ln].cpu()
                        cat_mean_sum_new[ln] += coefficients[ln].sum(dim=(0, 1)).cpu().float()
                    n_patches_seen_new += actual * adapter.n_patches
                    for j, datum in enumerate(batch):
                        if datum["image"] in new_image_set:
                            for ln in missing:
                                cat_contrib_new[ln] += coefficients[ln][j].mean(dim=0).cpu().float()

                merged_astar = dict(raw_astar)
                merged_contrib = dict(raw_contrib) if raw_contrib else {}
                for ln in missing:
                    merged_astar[ln] = cat_mean_sum_new[ln] / n_patches_seen_new
                    merged_contrib[ln] = cat_contrib_new[ln]
                    global_mean_sum[ln] += cat_contrib_new[ln]

                per_category_a_star[name] = {
                    ln: merged_astar[ln].to(dtype=torch.float32, device="cuda")
                    for ln in layer_names
                }
                ckpt_data = {
                    "a_star": {ln: merged_astar[ln] for ln in layer_names},
                    "new_image_ids": ckpt.get("new_image_ids", []),
                    "global_mean_contrib": merged_contrib,
                    "n_new_images": ckpt.get("n_new_images", 0),
                }
                if save_coefficients:
                    merged_coeff = dict(raw_coeff) if raw_coeff else {}
                    merged_coeff.update(cat_coeff_new)
                    ckpt_data["coefficients"] = merged_coeff
                save_checkpoint(run_dir, name, ckpt_data)
                continue

        length = len(data)
        cat_coefficients = {
            l.name: torch.empty(length, adapter.n_patches, l.n_dirs, dtype=torch.float16)
            for l in layers
        }
        cat_global_contrib = {l.name: torch.zeros(l.n_dirs, dtype=torch.float32) for l in layers}
        cat_mean_sum = {l.name: torch.zeros(l.n_dirs, dtype=torch.float32) for l in layers}
        cat_new_images: list[str] = []
        n_patches_seen = 0

        for i in tqdm(
            range(0, length, batch_size),
            total=math.ceil(length / batch_size),
            desc=f'"{name}"',
        ):
            batch = data[i : i + batch_size]
            inputs = adapter.preprocess(batch, image_base_path)
            coefficients = adapter.compute_coefficients_per_layer(inputs)
            actual = next(iter(coefficients.values())).shape[0]

            for ln in layer_names:
                cat_coefficients[ln][i : i + actual] = coefficients[ln].cpu()
                cat_mean_sum[ln] += coefficients[ln].sum(dim=(0, 1)).cpu().float()

            n_patches_seen += actual * adapter.n_patches

            for j, datum in enumerate(batch):
                img_id = datum["image"]
                if img_id not in seen_images:
                    seen_images.add(img_id)
                    cat_new_images.append(img_id)
                    for ln in layer_names:
                        cat_global_contrib[ln] += coefficients[ln][j].mean(dim=0).cpu().float()

        cat_a_star = {
            ln: (cat_mean_sum[ln] / n_patches_seen).to(device="cuda")
            for ln in layer_names
        }
        per_category_a_star[name] = cat_a_star
        for ln in layer_names:
            global_mean_sum[ln] += cat_global_contrib[ln]
        n_unique_images += len(cat_new_images)

        ckpt_data = {
            "a_star": {ln: cat_a_star[ln].cpu() for ln in layer_names},
            "new_image_ids": cat_new_images,
            "global_mean_contrib": cat_global_contrib,
            "n_new_images": len(cat_new_images),
        }
        if save_coefficients:
            ckpt_data["coefficients"] = cat_coefficients
        save_checkpoint(run_dir, name, ckpt_data)

    global_a_star = {
        ln: (global_mean_sum[ln] / n_unique_images).to(device="cuda")
        for ln in layer_names
    }

    save_json(
        run_dir / "means.json",
        {
            "n_unique_images": n_unique_images,
            "per_category_sizes": {n: len(d) for n, d in data_categorized.items()},
        },
    )

    return per_category_a_star, global_a_star


def run_ablation(
    adapter: ModelAdapter,
    data_categorized: dict[str, list],
    per_category_a_star: dict[str, dict[str, torch.Tensor]],
    global_a_star: dict[str, torch.Tensor],
    *,
    directions_to_ablate: dict[str, list[int]],
    batch_size: int,
    K: int,
    image_base_path: Path,
    run_dir: Path,
    coefficients_dir: Path,
    rand_seed: int = 42,
    rerun: bool = False,
) -> None:
    """
    Per-direction, per-layer ablation with ANLS scoring and Δlogit extraction.

    For each layer, direction, and baseline (per-category mean, global mean,
    zero, random): replaces the coefficient, reconstructs through remaining
    connector layers, generates, scores ANLS, and extracts first-token Δlogit
    artifacts.

    Saves per category in ``run_dir/{category}/``:

    ``anls_summary.json`` (schema: :data:`_ANLS_SUMMARY_SCHEMA`)
        Per-direction ANLS scores aggregated across images, one entry per
        layer per baseline.

    ``delta_logits.pt`` (schema: :data:`_DELTA_SCHEMA`)
        ``{layer: {dir_idx: {metric: tensor}}}``. Mean logit deltas,
        top-K/bottom-K shifts, KL divergence, and gold prob change.

    ``nls_original.npy``
        ``(n_samples,)`` per-image ANLS scores without ablation.

    ``nls_ablated_{baseline}.npy``
        ``{layer: [[per_image_anls_per_dir]]}`` raw ANLS scores.

    :param adapter: Model adapter instance
    :type adapter: ModelAdapter
    :param data_categorized: Map of category name to list of data dicts
    :type data_categorized: dict[str, list]
    :param per_category_a_star: Per-layer category means (CUDA),
        ``{cat: {layer: Tensor}}``
    :type per_category_a_star: dict[str, dict[str, torch.Tensor]]
    :param global_a_star: Per-layer global means (CUDA), ``{layer: Tensor}``
    :type global_a_star: dict[str, torch.Tensor]
    :param directions_to_ablate: Per-layer direction indices,
        ``{layer: [dir_indices]}``
    :type directions_to_ablate: dict[str, list[int]]
    :param batch_size: Number of images per forward pass
    :type batch_size: int
    :param K: Number of top/bottom tokens to keep per image
    :type K: int
    :param image_base_path: Root directory for image files
    :type image_base_path: Path
    :param run_dir: Output directory for this run
    :type run_dir: Path
    :param coefficients_dir: Directory containing per-category coefficient
        checkpoints (from :func:`compute_category_means`)
    :type coefficients_dir: Path
    :param rand_seed: Seed for the random baseline generator
    :type rand_seed: int
    """
    svd_layer_map = {l.name: l for l in adapter.svd_layers}

    tasks = [(ln, d) for ln, dirs in directions_to_ablate.items() for d in dirs]
    n_tasks = len(tasks)

    # Merge new directions with any existing ones in metadata
    merged_directions = dict(directions_to_ablate)
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            old_meta = json.load(f)
        for ln, dirs in old_meta.get("directions", {}).items():
            existing = set(merged_directions.get(ln, []))
            existing.update(dirs)
            merged_directions[ln] = sorted(existing)

    update_metadata(
        run_dir,
        dict(
            model=adapter.model_name,
            directions=merged_directions,
            batch_size=batch_size,
            K=K,
            rand_seed=rand_seed,
            n_categories=len(data_categorized),
            category_sizes={n: len(d) for n, d in data_categorized.items()},
        ),
    )

    for name, data in tqdm(data_categorized.items(), desc="Ablation + ANLS"):
        cat_dir = run_dir / fs_safe(name)
        cat_dir.mkdir(parents=True, exist_ok=True)

        existing_dirs_set = set()
        old_delta = None
        old_summary = None
        summary_path = cat_dir / "anls_summary.json"
        if not rerun and summary_path.exists():
            with open(summary_path) as f:
                old_summary = json.load(f)
            for ln, dirs in old_summary.get("directions", {}).items():
                for d in dirs:
                    existing_dirs_set.add((ln, d))

        skip_tasks = {i for i, (ln, d) in enumerate(tasks) if (ln, d) in existing_dirs_set}

        if len(skip_tasks) == n_tasks:
            print(f'  Skipping "{name}" (all directions already done)')
            continue

        if skip_tasks:
            n_new = n_tasks - len(skip_tasks)
            print(f'  Running {n_new} new directions for "{name}" ({len(skip_tasks)} already done)')
            old_delta = torch.load(cat_dir / "delta_logits.pt", weights_only=False)

        length = len(data)
        cat_a_star = per_category_a_star[name]
        cat_coefficients = load_category_coefficients(
            coefficients_dir, name, adapter.component_name
        )
        cat_std = {ln: cat_coefficients[ln].float().std(dim=(0, 1)) for ln in directions_to_ablate}
        rand_gen = torch.Generator().manual_seed(rand_seed)

        acc = {
            bl: _make_accumulators(n_tasks, adapter.vocab_size)
            for bl in _BASELINE_KEYS
        }
        nls_original = []

        for i in tqdm(
            range(0, length, batch_size),
            total=math.ceil(length / batch_size),
            desc=f'"{name}"',
        ):
            batch = data[i : i + batch_size]
            actual = len(batch)
            targets_batch = [datum["answers"] for datum in batch]

            inputs = adapter.preprocess(batch, image_base_path)
            attention_mask = inputs["attention_mask"]
            text_embeds = adapter.get_text_embeds(inputs)

            batch_coeffs = {}
            layer_outputs = {}
            for layer in adapter.svd_layers:
                if layer.name not in directions_to_ablate:
                    continue
                coeff = cat_coefficients[layer.name][i : i + actual].to(
                    dtype=adapter.compute_dtype, device="cuda"
                )
                batch_coeffs[layer.name] = coeff
                out = coeff @ layer.U.T.to(coeff.dtype)
                if layer.bias is not None:
                    out = out + layer.bias.to(coeff.dtype)
                layer_outputs[layer.name] = out

            with torch.no_grad():
                last_layer = adapter.svd_layers[-1]
                if last_layer.name in layer_outputs:
                    conn_out_orig = layer_outputs[last_layer.name]
                else:
                    last_coeff = cat_coefficients[last_layer.name][i : i + actual].to(
                        dtype=adapter.compute_dtype, device="cuda"
                    )
                    conn_out_orig = adapter.reconstruct(last_coeff)
                embeds_orig = adapter.merge_embeds(inputs, text_embeds, conn_out_orig)
                preds_orig, logits_orig = adapter.generate_with_logits(embeds_orig, attention_mask)

            for targets, pred in zip(targets_batch, preds_orig):
                nls_original.append(best_anls(pred, targets))

            probs_orig = F.softmax(logits_orig, dim=-1)
            log_probs_orig = F.log_softmax(logits_orig, dim=-1)
            pad_id = adapter.processor.tokenizer.pad_token_id or 0
            gold_tok = []
            for targets in targets_batch:
                ids = adapter.processor.tokenizer(targets[0], add_special_tokens=False)["input_ids"]
                gold_tok.append(ids[-1] if ids else pad_id)
            gold_idx = torch.tensor(gold_tok, device=logits_orig.device)
            lp_orig = log_probs_orig[range(actual), gold_idx]
            stacked_attn = attention_mask.repeat(4, 1)

            for t_idx, (layer_name, dir_idx) in enumerate(tasks):
                if t_idx in skip_tasks:
                    continue
                layer = svd_layer_map[layer_name]
                u_d = layer.U[:, dir_idx].to(batch_coeffs[layer_name].dtype)
                orig_d = batch_coeffs[layer_name][..., dir_idx]
                l_out = layer_outputs[layer_name]

                rand_vals = cat_a_star[layer_name][dir_idx] + torch.randn(
                    *batch_coeffs[layer_name].shape[:-1],
                    generator=rand_gen,
                ).to(
                    device=batch_coeffs[layer_name].device,
                    dtype=batch_coeffs[layer_name].dtype,
                ) * cat_std[
                    layer_name
                ][
                    dir_idx
                ].to(
                    batch_coeffs[layer_name].device,
                    batch_coeffs[layer_name].dtype,
                )

                with torch.no_grad():
                    stacked_l = torch.cat(
                        [
                            l_out + (cat_a_star[layer_name][dir_idx] - orig_d).unsqueeze(-1) * u_d,
                            l_out
                            + (global_a_star[layer_name][dir_idx] - orig_d).unsqueeze(-1) * u_d,
                            l_out + (-orig_d).unsqueeze(-1) * u_d,
                            l_out + (rand_vals - orig_d).unsqueeze(-1) * u_d,
                        ],
                        dim=0,
                    )
                    stacked_conn = adapter.forward_connector_from(layer_name, stacked_l)
                    conn_parts = stacked_conn.split(actual)

                    stacked_embeds = torch.cat(
                        [adapter.merge_embeds(inputs, text_embeds, c) for c in conn_parts], dim=0
                    )

                    all_preds, all_logits = adapter.generate_with_logits(
                        stacked_embeds, stacked_attn
                    )

                all_log_probs = F.log_softmax(all_logits, dim=-1)
                probs_orig_exp = probs_orig.repeat(4, 1)
                kl_all = F.kl_div(all_log_probs, probs_orig_exp, reduction="none").sum(-1)
                kl_parts = kl_all.split(actual)
                gold_idx_exp = gold_idx.repeat(4)
                all_lp = all_log_probs[range(4 * actual), gold_idx_exp]
                lp_parts = all_lp.split(actual)

                logits_parts = all_logits.split(actual)
                deltas = {
                    bl: (logits_orig - logits_parts[bi]).float()
                    for bi, bl in enumerate(_BASELINE_KEYS)
                }
                preds_parts = [
                    all_preds[:actual],
                    all_preds[actual : 2 * actual],
                    all_preds[2 * actual : 3 * actual],
                    all_preds[3 * actual :],
                ]

                for bi, bl in enumerate(_BASELINE_KEYS):
                    a = acc[bl]
                    delta = deltas[bl]
                    a["kl_div"][t_idx].extend(kl_parts[bi].cpu().tolist())
                    a["delta_gold_prob"][t_idx].extend(
                        (lp_orig - lp_parts[bi]).cpu().tolist()
                    )
                    a["delta_sum"][t_idx] += delta.sum(dim=0).cpu()
                    a["delta_abs_sum"][t_idx] += delta.abs().sum(dim=0).cpu()

                    probs_abl = F.softmax(logits_parts[bi], dim=-1)
                    top_v, top_i = delta.topk(K, dim=-1)
                    bot_v, bot_i = (-delta).topk(K, dim=-1)
                    top_p_orig = probs_orig.gather(-1, top_i.long())
                    bot_p_orig = probs_orig.gather(-1, bot_i.long())
                    top_p_abl = probs_abl.gather(-1, top_i.long())
                    bot_p_abl = probs_abl.gather(-1, bot_i.long())
                    a["topk"][t_idx].append(
                        torch.stack([top_i, top_v, top_p_orig, top_p_abl], dim=1).cpu()
                    )
                    a["botk"][t_idx].append(
                        torch.stack([bot_i, bot_v, bot_p_orig, bot_p_abl], dim=1).cpu()
                    )

                    for targets, pred in zip(targets_batch, preds_parts[bi]):
                        a["nls"][t_idx].append(best_anls(pred, targets))

        # --- Save per-category results (merge with existing if resuming) ---
        np.save(cat_dir / "nls_original.npy", np.array(nls_original))

        layer_delta = {}
        layer_nls = {}
        for t_idx, (ln, dir_idx) in enumerate(tasks):
            if t_idx in skip_tasks:
                continue
            if ln not in layer_delta:
                layer_delta[ln] = {}
                layer_nls[ln] = {
                    "dirs": [],
                    "cat": [],
                    "global": [],
                    "zero": [],
                    "rand": [],
                }
            d = {}
            for bl in _BASELINE_KEYS:
                a = acc[bl]
                d[f"signed_mean_{bl}"] = a["delta_sum"][t_idx] / length
                d[f"abs_mean_{bl}"] = a["delta_abs_sum"][t_idx] / length
                d[f"topk_{bl}"] = torch.cat(a["topk"][t_idx], dim=0)
                d[f"botk_{bl}"] = torch.cat(a["botk"][t_idx], dim=0)
                d[f"kl_div_{bl}"] = a["kl_div"][t_idx]
                d[f"delta_gold_prob_{bl}"] = a["delta_gold_prob"][t_idx]
            layer_delta[ln][dir_idx] = d
            layer_nls[ln]["dirs"].append(dir_idx)
            for bl in _BASELINE_KEYS:
                layer_nls[ln][bl].append(sum(acc[bl]["nls"][t_idx]) / length)

        # Merge old direction results into new
        _ANLS_BL_KEYS = {
            "cat": "anls_ablated_per_category_mean",
            "global": "anls_ablated_global_mean",
            "zero": "anls_ablated_zero_mean",
            "rand": "anls_ablated_random",
        }
        if old_delta is not None and old_summary is not None:
            for ln in old_delta:
                if ln == "_schema":
                    continue
                if ln not in layer_delta:
                    layer_delta[ln] = {}
                    layer_nls[ln] = {
                        "dirs": [],
                        "cat": [],
                        "global": [],
                        "zero": [],
                        "rand": [],
                    }
                old_dirs = old_summary["directions"].get(ln, [])
                for i, d_idx in enumerate(old_dirs):
                    layer_delta[ln][d_idx] = old_delta[ln][d_idx]
                    layer_nls[ln]["dirs"].append(d_idx)
                    for bl in _BASELINE_KEYS:
                        layer_nls[ln][bl].append(
                            old_summary[_ANLS_BL_KEYS[bl]][ln][i]
                        )

        for bl in _BASELINE_KEYS:
            np.save(
                cat_dir / f"nls_ablated_{bl}.npy",
                {
                    ln: [acc[bl]["nls"][t] for t, (l, _) in enumerate(tasks) if l == ln and t not in skip_tasks]
                    for ln in layer_nls
                },
                allow_pickle=True,
            )

        torch.save(
            {"_schema": _DELTA_SCHEMA, **layer_delta},
            cat_dir / "delta_logits.pt",
        )

        anls_orig = sum(nls_original) / length
        save_json(
            cat_dir / "anls_summary.json",
            dict(
                _schema=_ANLS_SUMMARY_SCHEMA,
                category=name,
                n_samples=length,
                directions={ln: layer_nls[ln]["dirs"] for ln in layer_nls},
                anls_original=anls_orig,
                anls_ablated_per_category_mean={ln: layer_nls[ln]["cat"] for ln in layer_nls},
                anls_ablated_global_mean={ln: layer_nls[ln]["global"] for ln in layer_nls},
                anls_ablated_zero_mean={ln: layer_nls[ln]["zero"] for ln in layer_nls},
                anls_ablated_random={ln: layer_nls[ln]["rand"] for ln in layer_nls},
            ),
        )

        for ln in layer_nls:
            print(
                f"\n{name}/{ln}: orig={anls_orig:.4f}, "
                f"cat={[f'{a:.4f}' for a in layer_nls[ln]['cat']]}, "
                f"global={[f'{a:.4f}' for a in layer_nls[ln]['global']]}, "
                f"zero={[f'{a:.4f}' for a in layer_nls[ln]['zero']]}, "
                f"rand={[f'{a:.4f}' for a in layer_nls[ln]['rand']]}"
            )


def run_joint_ablation(
    adapter: ModelAdapter,
    data_categorized: dict[str, list],
    per_category_a_star: dict[str, dict[str, torch.Tensor]],
    global_a_star: dict[str, torch.Tensor],
    *,
    direction_sets: dict[str, dict[str, dict[str, list[int]]]],
    batch_size: int,
    K: int,
    image_base_path: Path,
    run_dir: Path,
    coefficients_dir: Path,
    rand_seed: int = 42,
    rerun: bool = False,
) -> None:
    """
    Joint ablation of entire direction sets with ANLS scoring and Δlogit
    extraction.

    Ablates all directions in a set simultaneously (multi-rank correction)
    and measures the effect. Designed for comparing DM mask active sets
    against random controls.

    Saves per category in ``run_dir/{category}/``:

    ``joint_anls_summary.json`` (schema: :data:`_JOINT_ANLS_SUMMARY_SCHEMA`)
        Per-set, per-layer ANLS, KL, and gold prob change. KL and
        delta_gold_prob are per-image lists; ANLS is a scalar mean.

    ``joint_delta_logits.pt`` (schema: :data:`_JOINT_DELTA_SCHEMA`)
        ``{set_label: {layer: {metric: tensor}}}``. Mean logit deltas and
        top-K/bottom-K shifts per set per layer.

    :param adapter: Model adapter instance
    :type adapter: ModelAdapter
    :param data_categorized: Map of category name to list of data dicts
    :type data_categorized: dict[str, list]
    :param per_category_a_star: Per-layer category means (CUDA),
        ``{cat: {layer: Tensor}}``
    :type per_category_a_star: dict[str, dict[str, torch.Tensor]]
    :param global_a_star: Per-layer global means (CUDA), ``{layer: Tensor}``
    :type global_a_star: dict[str, torch.Tensor]
    :param direction_sets: Per-set, per-category, per-layer direction indices
        ``{set_label: {category: {layer: [dir_indices]}}}``
    :type direction_sets: dict[str, dict[str, dict[str, list[int]]]]
    :param batch_size: Number of images per forward pass
    :type batch_size: int
    :param K: Number of top/bottom tokens to keep per image
    :type K: int
    :param image_base_path: Root directory for image files
    :type image_base_path: Path
    :param run_dir: Output directory for this run
    :type run_dir: Path
    :param coefficients_dir: Directory containing per-category coefficient
        checkpoints (from :func:`compute_category_means`)
    :type coefficients_dir: Path
    :param rand_seed: Seed for the random baseline generator
    :type rand_seed: int
    """
    svd_layer_map = {l.name: l for l in adapter.svd_layers}
    set_labels = list(direction_sets.keys())

    update_metadata(
        run_dir,
        dict(
            model=adapter.model_name,
            experiment="joint_ablation",
            direction_sets={
                sl: {
                    cat: {ln: dirs for ln, dirs in layers.items()}
                    for cat, layers in cat_dirs.items()
                }
                for sl, cat_dirs in direction_sets.items()
            },
            batch_size=batch_size,
            rand_seed=rand_seed,
            n_categories=len(data_categorized),
            category_sizes={n: len(d) for n, d in data_categorized.items()},
        ),
    )

    completed = set()
    if not rerun:
        completed = {
            name
            for name in data_categorized
            if (run_dir / fs_safe(name) / "joint_anls_summary.json").exists()
        }
    if completed:
        print(f"Resuming joint ablation: skipping {len(completed)} completed categories")

    for name, data in tqdm(data_categorized.items(), desc="Joint ablation"):
        if name in completed:
            print(f'  Skipping "{name}" (results exist)')
            continue

        length = len(data)
        cat_a_star = per_category_a_star[name]
        cat_coefficients = load_category_coefficients(
            coefficients_dir, name, adapter.component_name
        )
        cat_std = {ln: cat_coefficients[ln].float().std(dim=(0, 1)) for ln in cat_coefficients}

        nls_original = []
        set_results: dict[str, dict[str, dict]] = {}
        for sl in set_labels:
            set_results[sl] = {}
            cat_layers = direction_sets[sl].get(name, {})
            for ln in cat_layers:
                set_results[sl][ln] = {
                    "nls_cat": [],
                    "nls_global": [],
                    "nls_zero": [],
                    "nls_rand": [],
                    "kl_cat": [],
                    "kl_global": [],
                    "kl_zero": [],
                    "kl_rand": [],
                    "dgp_cat": [],
                    "dgp_global": [],
                    "dgp_zero": [],
                    "dgp_rand": [],
                    "delta_sum_cat": torch.zeros(adapter.vocab_size),
                    "delta_abs_sum_cat": torch.zeros(adapter.vocab_size),
                    "delta_sum_global": torch.zeros(adapter.vocab_size),
                    "delta_abs_sum_global": torch.zeros(adapter.vocab_size),
                    "delta_sum_zero": torch.zeros(adapter.vocab_size),
                    "delta_abs_sum_zero": torch.zeros(adapter.vocab_size),
                    "delta_sum_rand": torch.zeros(adapter.vocab_size),
                    "delta_abs_sum_rand": torch.zeros(adapter.vocab_size),
                    "topk_cat": [],
                    "botk_cat": [],
                    "topk_global": [],
                    "botk_global": [],
                    "topk_zero": [],
                    "botk_zero": [],
                    "topk_rand": [],
                    "botk_rand": [],
                }

        rand_gen = torch.Generator().manual_seed(rand_seed)

        for i in tqdm(
            range(0, length, batch_size),
            total=math.ceil(length / batch_size),
            desc=f'"{name}"',
        ):
            batch = data[i : i + batch_size]
            actual = len(batch)
            targets_batch = [datum["answers"] for datum in batch]

            inputs = adapter.preprocess(batch, image_base_path)
            attention_mask = inputs["attention_mask"]
            text_embeds = adapter.get_text_embeds(inputs)

            batch_coeffs = {}
            layer_outputs = {}
            for layer in adapter.svd_layers:
                coeff = cat_coefficients[layer.name][i : i + actual].to(
                    dtype=adapter.compute_dtype, device="cuda"
                )
                batch_coeffs[layer.name] = coeff
                out = coeff @ layer.U.T.to(coeff.dtype)
                if layer.bias is not None:
                    out = out + layer.bias.to(coeff.dtype)
                layer_outputs[layer.name] = out

            with torch.no_grad():
                last_layer = adapter.svd_layers[-1]
                conn_out_orig = layer_outputs[last_layer.name]
                embeds_orig = adapter.merge_embeds(inputs, text_embeds, conn_out_orig)
                preds_orig, logits_orig = adapter.generate_with_logits(embeds_orig, attention_mask)

            for targets, pred in zip(targets_batch, preds_orig):
                nls_original.append(best_anls(pred, targets))

            probs_orig = F.softmax(logits_orig, dim=-1)
            log_probs_orig = F.log_softmax(logits_orig, dim=-1)
            pad_id = adapter.processor.tokenizer.pad_token_id or 0
            gold_tok = []
            for targets in targets_batch:
                ids = adapter.processor.tokenizer(targets[0], add_special_tokens=False)["input_ids"]
                gold_tok.append(ids[-1] if ids else pad_id)
            gold_idx = torch.tensor(gold_tok, device=logits_orig.device)
            lp_orig = log_probs_orig[range(actual), gold_idx]
            stacked_attn = attention_mask.repeat(4, 1)

            for sl in set_labels:
                cat_layers = direction_sets[sl].get(name, {})

                for layer_name, dir_list in cat_layers.items():
                    if not dir_list:
                        continue
                    layer = svd_layer_map[layer_name]
                    l_out = layer_outputs[layer_name]
                    coeff = batch_coeffs[layer_name]

                    U_sub = layer.U[:, dir_list].to(coeff.dtype)
                    orig_sub = coeff[..., dir_list]
                    cat_sub = cat_a_star[layer_name][dir_list]
                    global_sub = global_a_star[layer_name][dir_list]

                    std_sub = cat_std[layer_name][dir_list].to(coeff.device, coeff.dtype)
                    rand_sub = (
                        cat_sub
                        + torch.randn(
                            *orig_sub.shape,
                            generator=rand_gen,
                        ).to(device=coeff.device, dtype=coeff.dtype)
                        * std_sub
                    )

                    with torch.no_grad():
                        conn_delta_cat = (cat_sub - orig_sub) @ U_sub.T
                        conn_delta_global = (global_sub - orig_sub) @ U_sub.T
                        conn_delta_zero = (-orig_sub) @ U_sub.T
                        conn_delta_rand = (rand_sub - orig_sub) @ U_sub.T

                        stacked_l = torch.cat(
                            [
                                l_out + conn_delta_cat,
                                l_out + conn_delta_global,
                                l_out + conn_delta_zero,
                                l_out + conn_delta_rand,
                            ],
                            dim=0,
                        )

                        stacked_conn = adapter.forward_connector_from(layer_name, stacked_l)
                        conn_parts = stacked_conn.split(actual)

                        stacked_embeds = torch.cat(
                            [adapter.merge_embeds(inputs, text_embeds, c) for c in conn_parts],
                            dim=0,
                        )

                        all_preds, all_logits = adapter.generate_with_logits(
                            stacked_embeds, stacked_attn
                        )

                    preds_cat = all_preds[:actual]
                    preds_global = all_preds[actual : 2 * actual]
                    preds_zero = all_preds[2 * actual : 3 * actual]
                    preds_rand = all_preds[3 * actual :]

                    for targets, pred in zip(targets_batch, preds_cat):
                        set_results[sl][layer_name]["nls_cat"].append(best_anls(pred, targets))
                    for targets, pred in zip(targets_batch, preds_global):
                        set_results[sl][layer_name]["nls_global"].append(best_anls(pred, targets))
                    for targets, pred in zip(targets_batch, preds_zero):
                        set_results[sl][layer_name]["nls_zero"].append(best_anls(pred, targets))
                    for targets, pred in zip(targets_batch, preds_rand):
                        set_results[sl][layer_name]["nls_rand"].append(best_anls(pred, targets))

                    all_log_probs = F.log_softmax(all_logits, dim=-1)
                    probs_orig_exp = probs_orig.repeat(4, 1)
                    kl_all = F.kl_div(all_log_probs, probs_orig_exp, reduction="none").sum(-1)
                    kl_c, kl_g, kl_z, kl_r = kl_all.split(actual)
                    set_results[sl][layer_name]["kl_cat"].extend(kl_c.cpu().tolist())
                    set_results[sl][layer_name]["kl_global"].extend(kl_g.cpu().tolist())
                    set_results[sl][layer_name]["kl_zero"].extend(kl_z.cpu().tolist())
                    set_results[sl][layer_name]["kl_rand"].extend(kl_r.cpu().tolist())

                    gold_idx_exp = gold_idx.repeat(4)
                    all_lp = all_log_probs[range(4 * actual), gold_idx_exp]
                    lp_c, lp_g, lp_z, lp_r = all_lp.split(actual)
                    set_results[sl][layer_name]["dgp_cat"].extend((lp_orig - lp_c).cpu().tolist())
                    set_results[sl][layer_name]["dgp_global"].extend(
                        (lp_orig - lp_g).cpu().tolist()
                    )
                    set_results[sl][layer_name]["dgp_zero"].extend((lp_orig - lp_z).cpu().tolist())
                    set_results[sl][layer_name]["dgp_rand"].extend((lp_orig - lp_r).cpu().tolist())

                    logits_cat, logits_global, logits_zero, logits_rand = all_logits.split(actual)
                    delta_cat = (logits_orig - logits_cat).float()
                    delta_global = (logits_orig - logits_global).float()
                    delta_zero = (logits_orig - logits_zero).float()
                    delta_rand = (logits_orig - logits_rand).float()

                    r = set_results[sl][layer_name]
                    r["delta_sum_cat"] += delta_cat.sum(dim=0).cpu()
                    r["delta_abs_sum_cat"] += delta_cat.abs().sum(dim=0).cpu()
                    r["delta_sum_global"] += delta_global.sum(dim=0).cpu()
                    r["delta_abs_sum_global"] += delta_global.abs().sum(dim=0).cpu()
                    r["delta_sum_zero"] += delta_zero.sum(dim=0).cpu()
                    r["delta_abs_sum_zero"] += delta_zero.abs().sum(dim=0).cpu()
                    r["delta_sum_rand"] += delta_rand.sum(dim=0).cpu()
                    r["delta_abs_sum_rand"] += delta_rand.abs().sum(dim=0).cpu()

                    probs_cat = F.softmax(logits_cat, dim=-1)
                    probs_global = F.softmax(logits_global, dim=-1)
                    probs_zero = F.softmax(logits_zero, dim=-1)
                    probs_rand = F.softmax(logits_rand, dim=-1)

                    for delta, probs_abl, tk_key, bk_key in [
                        (delta_cat, probs_cat, "topk_cat", "botk_cat"),
                        (delta_global, probs_global, "topk_global", "botk_global"),
                        (delta_zero, probs_zero, "topk_zero", "botk_zero"),
                        (delta_rand, probs_rand, "topk_rand", "botk_rand"),
                    ]:
                        top_v, top_i = delta.topk(K, dim=-1)
                        bot_v, bot_i = (-delta).topk(K, dim=-1)
                        top_p_orig = probs_orig.gather(-1, top_i.long())
                        bot_p_orig = probs_orig.gather(-1, bot_i.long())
                        top_p_abl = probs_abl.gather(-1, top_i.long())
                        bot_p_abl = probs_abl.gather(-1, bot_i.long())
                        r[tk_key].append(
                            torch.stack([top_i, top_v, top_p_orig, top_p_abl], dim=1).cpu()
                        )
                        r[bk_key].append(
                            torch.stack([bot_i, bot_v, bot_p_orig, bot_p_abl], dim=1).cpu()
                        )

        cat_dir = run_dir / fs_safe(name)
        cat_dir.mkdir(parents=True, exist_ok=True)

        anls_orig = sum(nls_original) / length
        summary: dict = {
            "_schema": _JOINT_ANLS_SUMMARY_SCHEMA,
            "category": name,
            "n_samples": length,
            "anls_original": anls_orig,
            "sets": {},
        }
        for sl in set_labels:
            summary["sets"][sl] = {}
            for ln, res in set_results[sl].items():
                dirs = direction_sets[sl].get(name, {}).get(ln, [])
                summary["sets"][sl][ln] = {
                    "directions": dirs,
                    "n_directions": len(dirs),
                    "anls_cat": sum(res["nls_cat"]) / length if res["nls_cat"] else None,
                    "anls_global": sum(res["nls_global"]) / length if res["nls_global"] else None,
                    "anls_zero": sum(res["nls_zero"]) / length if res["nls_zero"] else None,
                    "anls_rand": sum(res["nls_rand"]) / length if res["nls_rand"] else None,
                    "kl_cat": res["kl_cat"],
                    "kl_global": res["kl_global"],
                    "kl_zero": res["kl_zero"],
                    "kl_rand": res["kl_rand"],
                    "delta_gold_prob_cat": res["dgp_cat"],
                    "delta_gold_prob_global": res["dgp_global"],
                    "delta_gold_prob_zero": res["dgp_zero"],
                    "delta_gold_prob_rand": res["dgp_rand"],
                }

        save_json(cat_dir / "joint_anls_summary.json", summary)

        delta_logits = {}
        for sl in set_labels:
            delta_logits[sl] = {}
            for ln, res in set_results[sl].items():
                delta_logits[sl][ln] = {
                    "signed_mean_cat": res["delta_sum_cat"] / length,
                    "abs_mean_cat": res["delta_abs_sum_cat"] / length,
                    "signed_mean_global": res["delta_sum_global"] / length,
                    "abs_mean_global": res["delta_abs_sum_global"] / length,
                    "signed_mean_zero": res["delta_sum_zero"] / length,
                    "abs_mean_zero": res["delta_abs_sum_zero"] / length,
                    "signed_mean_rand": res["delta_sum_rand"] / length,
                    "abs_mean_rand": res["delta_abs_sum_rand"] / length,
                    "topk_cat": (
                        torch.cat(res["topk_cat"], dim=0) if res["topk_cat"] else torch.empty(0)
                    ),
                    "botk_cat": (
                        torch.cat(res["botk_cat"], dim=0) if res["botk_cat"] else torch.empty(0)
                    ),
                    "topk_global": (
                        torch.cat(res["topk_global"], dim=0)
                        if res["topk_global"]
                        else torch.empty(0)
                    ),
                    "botk_global": (
                        torch.cat(res["botk_global"], dim=0)
                        if res["botk_global"]
                        else torch.empty(0)
                    ),
                    "topk_zero": (
                        torch.cat(res["topk_zero"], dim=0) if res["topk_zero"] else torch.empty(0)
                    ),
                    "botk_zero": (
                        torch.cat(res["botk_zero"], dim=0) if res["botk_zero"] else torch.empty(0)
                    ),
                    "topk_rand": (
                        torch.cat(res["topk_rand"], dim=0) if res["topk_rand"] else torch.empty(0)
                    ),
                    "botk_rand": (
                        torch.cat(res["botk_rand"], dim=0) if res["botk_rand"] else torch.empty(0)
                    ),
                }
        torch.save({"_schema": _JOINT_DELTA_SCHEMA, **delta_logits}, cat_dir / "joint_delta_logits.pt")

        print(f"\n{name}: orig={anls_orig:.4f}")
        for sl in set_labels:
            for ln, res in set_results[sl].items():
                n_dirs = len(direction_sets[sl].get(name, {}).get(ln, []))
                anls_z = sum(res["nls_zero"]) / length if res["nls_zero"] else 0
                mean_kl = sum(res["kl_zero"]) / length if res["kl_zero"] else 0
                print(
                    f"  {sl}/{ln} ({n_dirs} dirs): "
                    f"anls_zero={anls_z:.4f}, mean_kl_zero={mean_kl:.4f}"
                )


def run_total_ablation(
    adapter: ModelAdapter,
    data_categorized: dict[str, list],
    global_a_star: dict[str, torch.Tensor],
    *,
    batch_size: int,
    image_base_path: Path,
    run_dir: Path,
    coefficients_dir: Path,
    rerun: bool = False,
) -> None:
    """
    Zero and global-mean ablation of all directions to measure total KL budget.

    For each layer, replaces every coefficient with zero or the global mean
    and measures the resulting KL divergence. This gives the upper-bound KL
    that partial ablation results (active sets, random controls) should be
    interpreted against. Comparing zero vs global-mean KL reveals OOD
    spoofing from zero ablation.

    Saves per category in ``run_dir/{category}/``:

    ``total_ablation.json`` (schema: :data:`_TOTAL_ABLATION_SCHEMA`)
        Per-layer KL, ANLS, and gold prob change for zero and global-mean
        ablation of all directions.

    :param adapter: Model adapter instance
    :type adapter: ModelAdapter
    :param data_categorized: Map of category name to list of data dicts
    :type data_categorized: dict[str, list]
    :param global_a_star: Per-layer global means (CUDA), ``{layer: Tensor}``
    :type global_a_star: dict[str, torch.Tensor]
    :param batch_size: Number of images per forward pass
    :type batch_size: int
    :param image_base_path: Root directory for image files
    :type image_base_path: Path
    :param run_dir: Output directory for this run
    :type run_dir: Path
    :param coefficients_dir: Directory containing per-category coefficient
        checkpoints (from :func:`compute_category_means`)
    :type coefficients_dir: Path
    """
    update_metadata(
        run_dir,
        dict(
            model=adapter.model_name,
            experiment="total_ablation",
            batch_size=batch_size,
            n_categories=len(data_categorized),
            category_sizes={n: len(d) for n, d in data_categorized.items()},
        ),
    )

    completed = set()
    if not rerun:
        completed = {
            name
            for name in data_categorized
            if (run_dir / fs_safe(name) / "total_ablation.json").exists()
        }
    if completed:
        print(f"Resuming total ablation: skipping {len(completed)} completed categories")

    for name, data in tqdm(data_categorized.items(), desc="Total ablation"):
        if name in completed:
            print(f'  Skipping "{name}" (results exist)')
            continue

        length = len(data)
        cat_coefficients = load_category_coefficients(
            coefficients_dir, name, adapter.component_name
        )

        nls_original = []
        layer_results: dict[str, dict] = {}
        for layer in adapter.svd_layers:
            layer_results[layer.name] = {
                "kl_zero": [],
                "kl_global": [],
                "nls_zero": [],
                "nls_global": [],
                "dgp_zero": [],
                "dgp_global": [],
            }

        for i in tqdm(
            range(0, length, batch_size),
            total=math.ceil(length / batch_size),
            desc=f'"{name}"',
        ):
            batch = data[i : i + batch_size]
            actual = len(batch)
            targets_batch = [datum["answers"] for datum in batch]

            inputs = adapter.preprocess(batch, image_base_path)
            attention_mask = inputs["attention_mask"]
            text_embeds = adapter.get_text_embeds(inputs)

            batch_coeffs = {}
            layer_outputs = {}
            for layer in adapter.svd_layers:
                coeff = cat_coefficients[layer.name][i : i + actual].to(
                    dtype=adapter.compute_dtype, device="cuda"
                )
                batch_coeffs[layer.name] = coeff
                out = coeff @ layer.U.T.to(coeff.dtype)
                if layer.bias is not None:
                    out = out + layer.bias.to(coeff.dtype)
                layer_outputs[layer.name] = out

            with torch.no_grad():
                last_layer = adapter.svd_layers[-1]
                conn_out_orig = layer_outputs[last_layer.name]
                embeds_orig = adapter.merge_embeds(inputs, text_embeds, conn_out_orig)
                preds_orig, logits_orig = adapter.generate_with_logits(
                    embeds_orig, attention_mask
                )

            for targets, pred in zip(targets_batch, preds_orig):
                nls_original.append(best_anls(pred, targets))

            probs_orig = F.softmax(logits_orig, dim=-1)
            log_probs_orig = F.log_softmax(logits_orig, dim=-1)
            pad_id = adapter.processor.tokenizer.pad_token_id or 0
            gold_tok = []
            for targets in targets_batch:
                ids = adapter.processor.tokenizer(targets[0], add_special_tokens=False)["input_ids"]
                gold_tok.append(ids[-1] if ids else pad_id)
            gold_idx = torch.tensor(gold_tok, device=logits_orig.device)
            lp_orig = log_probs_orig[range(actual), gold_idx]
            stacked_attn = attention_mask.repeat(2, 1)

            for layer in adapter.svd_layers:
                ln = layer.name
                l_out = layer_outputs[ln]
                coeff = batch_coeffs[ln]

                global_coeff = global_a_star[ln].expand_as(coeff)
                conn_delta_zero = (-coeff) @ layer.U.T.to(coeff.dtype)
                conn_delta_global = (global_coeff - coeff) @ layer.U.T.to(coeff.dtype)

                with torch.no_grad():
                    stacked_l = torch.cat(
                        [l_out + conn_delta_zero, l_out + conn_delta_global], dim=0
                    )
                    stacked_conn = adapter.forward_connector_from(ln, stacked_l)
                    conn_z, conn_g = stacked_conn.split(actual)

                    stacked_embeds = torch.cat(
                        [
                            adapter.merge_embeds(inputs, text_embeds, conn_z),
                            adapter.merge_embeds(inputs, text_embeds, conn_g),
                        ],
                        dim=0,
                    )
                    all_preds, all_logits = adapter.generate_with_logits(
                        stacked_embeds, stacked_attn
                    )

                preds_zero = all_preds[:actual]
                preds_global = all_preds[actual:]

                for targets, pred in zip(targets_batch, preds_zero):
                    layer_results[ln]["nls_zero"].append(best_anls(pred, targets))
                for targets, pred in zip(targets_batch, preds_global):
                    layer_results[ln]["nls_global"].append(best_anls(pred, targets))

                all_log_probs = F.log_softmax(all_logits, dim=-1)
                probs_orig_exp = probs_orig.repeat(2, 1)
                kl_all = F.kl_div(all_log_probs, probs_orig_exp, reduction="none").sum(
                    -1
                )
                kl_z, kl_g = kl_all.split(actual)
                layer_results[ln]["kl_zero"].extend(kl_z.cpu().tolist())
                layer_results[ln]["kl_global"].extend(kl_g.cpu().tolist())

                gold_idx_exp = gold_idx.repeat(2)
                all_lp = all_log_probs[range(2 * actual), gold_idx_exp]
                lp_z, lp_g = all_lp.split(actual)
                layer_results[ln]["dgp_zero"].extend((lp_orig - lp_z).cpu().tolist())
                layer_results[ln]["dgp_global"].extend(
                    (lp_orig - lp_g).cpu().tolist()
                )

        cat_dir = run_dir / fs_safe(name)
        cat_dir.mkdir(parents=True, exist_ok=True)

        anls_orig = sum(nls_original) / length
        summary: dict = {
            "_schema": _TOTAL_ABLATION_SCHEMA,
            "category": name,
            "n_samples": length,
            "n_directions": {
                ln: cat_coefficients[ln].shape[-1] for ln in layer_results
            },
            "anls_original": anls_orig,
            "layers": {},
        }
        for ln, res in layer_results.items():
            summary["layers"][ln] = {
                "kl_zero": res["kl_zero"],
                "kl_global": res["kl_global"],
                "anls_zero": sum(res["nls_zero"]) / length,
                "anls_global": sum(res["nls_global"]) / length,
                "delta_gold_prob_zero": res["dgp_zero"],
                "delta_gold_prob_global": res["dgp_global"],
            }

        save_json(cat_dir / "total_ablation.json", summary)

        print(f"\n{name}: orig={anls_orig:.4f}")
        for ln, res in layer_results.items():
            mean_kl_z = sum(res["kl_zero"]) / length
            mean_kl_g = sum(res["kl_global"]) / length
            anls_z = sum(res["nls_zero"]) / length
            anls_g = sum(res["nls_global"]) / length
            print(
                f"  {ln}: kl_zero={mean_kl_z:.4f}, kl_global={mean_kl_g:.4f}, "
                f"anls_zero={anls_z:.4f}, anls_global={anls_g:.4f}"
            )
