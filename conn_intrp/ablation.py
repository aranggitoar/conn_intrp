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
    >>> coeffs, cat_means, global_mean = compute_category_means(
    ...     adapter, categorized, batch_size=4,
    ...     image_base_path=img_path, run_dir=run_dir)
    >>> run_ablation(adapter, categorized, coeffs, cat_means, global_mean,
    ...     directions_to_ablate={"linear_1": [3, 7], "linear_2": [23, 70]},
    ...     batch_size=4, K=15,
    ...     image_base_path=img_path, run_dir=run_dir)

Main Functions:
    compute_category_means: SVD coefficients + per-category/global means.
    run_ablation: Per-direction ablation with ANLS scoring and Δlogit extraction.
    run_joint_ablation: Multi-direction set ablation for validating DM masks.
"""

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


def _load_ckpt_compat(
    ckpt: dict, component_name: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    raw_coeff = ckpt["coefficients"]
    raw_astar = ckpt["a_star"]
    if isinstance(raw_coeff, torch.Tensor):
        raw_coeff = {component_name: raw_coeff}
        raw_astar = {component_name: raw_astar}
    raw_contrib = ckpt.get("global_mean_contrib")
    if isinstance(raw_contrib, torch.Tensor):
        raw_contrib = {component_name: raw_contrib}
    return raw_coeff, raw_astar, raw_contrib


def compute_category_means(
    adapter: ModelAdapter,
    data_categorized: dict[str, list],
    *,
    batch_size: int,
    image_base_path: Path,
    run_dir: Path,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, torch.Tensor]],
    dict[str, torch.Tensor],
]:
    """
    Compute per-layer SVD coefficients and per-category/global mean vectors.

    Images appearing in multiple categories are counted once for the
    global mean (tracked via a ``seen_images`` set). Supports resuming
    from checkpoints (including old single-layer format).

    :param adapter: Model adapter instance.
    :type adapter: ModelAdapter
    :param data_categorized: Map of category name to list of data dicts.
    :type data_categorized: dict[str, list]
    :param batch_size: Number of images per forward pass.
    :type batch_size: int
    :param image_base_path: Root directory for image files.
    :type image_base_path: Path
    :param run_dir: Output directory for this run.
    :type run_dir: Path
    :returns: ``(per_category_coefficients, per_category_a_star, global_a_star)``
        where coefficients are ``{cat: {layer: Tensor}}`` on CPU and
        means are ``{cat: {layer: Tensor}}`` / ``{layer: Tensor}`` on CUDA.
    :rtype: tuple[dict[str, dict[str, torch.Tensor]],
        dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    """
    layers = adapter.svd_layers
    layer_names = [l.name for l in layers]
    completed = get_completed_categories(run_dir)

    per_category_coefficients: dict[str, dict[str, torch.Tensor]] = {}
    per_category_a_star: dict[str, dict[str, torch.Tensor]] = {}

    global_mean_sum = {
        l.name: torch.zeros(l.n_dirs, dtype=torch.float32) for l in layers
    }
    n_unique_images = 0
    seen_images: set[str] = set()

    for name, data in tqdm(data_categorized.items(), desc="Computing category means"):
        if fs_safe(name) in completed:
            ckpt = load_checkpoint(run_dir, name)
            raw_coeff, raw_astar, raw_contrib = _load_ckpt_compat(
                ckpt, adapter.component_name
            )
            per_category_coefficients[name] = raw_coeff
            per_category_a_star[name] = {
                ln: raw_astar[ln].to(dtype=adapter.compute_dtype, device="cuda")
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

        length = len(data)
        cat_coefficients = {
            l.name: torch.empty(length, adapter.n_patches, l.n_dirs)
            for l in layers
        }
        cat_global_contrib = {
            l.name: torch.zeros(l.n_dirs, dtype=torch.float32) for l in layers
        }
        cat_new_images: list[str] = []

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

            for j, datum in enumerate(batch):
                img_id = datum["image"]
                if img_id not in seen_images:
                    seen_images.add(img_id)
                    cat_new_images.append(img_id)
                    for ln in layer_names:
                        cat_global_contrib[ln] += (
                            coefficients[ln][j].mean(dim=0).cpu().float()
                        )

        per_category_coefficients[name] = cat_coefficients
        per_category_a_star[name] = {
            ln: cat_coefficients[ln].mean(dim=(0, 1)).to(
                dtype=adapter.compute_dtype, device="cuda"
            )
            for ln in layer_names
        }
        for ln in layer_names:
            global_mean_sum[ln] += cat_global_contrib[ln]
        n_unique_images += len(cat_new_images)

        save_checkpoint(
            run_dir,
            name,
            {
                "coefficients": cat_coefficients,
                "a_star": {
                    ln: per_category_a_star[name][ln].cpu() for ln in layer_names
                },
                "new_image_ids": cat_new_images,
                "global_mean_contrib": cat_global_contrib,
                "n_new_images": len(cat_new_images),
            },
        )

    global_a_star = {
        ln: (global_mean_sum[ln] / n_unique_images).to(
            dtype=adapter.compute_dtype, device="cuda"
        )
        for ln in layer_names
    }

    save_json(
        run_dir / "means.json",
        {
            "n_unique_images": n_unique_images,
            "per_category_sizes": {n: len(d) for n, d in data_categorized.items()},
        },
    )

    return per_category_coefficients, per_category_a_star, global_a_star


def run_ablation(
    adapter: ModelAdapter,
    data_categorized: dict[str, list],
    per_category_coefficients: dict[str, dict[str, torch.Tensor]],
    per_category_a_star: dict[str, dict[str, torch.Tensor]],
    global_a_star: dict[str, torch.Tensor],
    *,
    directions_to_ablate: dict[str, list[int]],
    batch_size: int,
    K: int,
    image_base_path: Path,
    run_dir: Path,
    rand_seed: int = 42,
) -> None:
    """
    Per-direction, per-layer ablation with ANLS scoring and Δlogit extraction.

    For each layer, direction, and baseline (per-category mean, global mean,
    zero, random): replaces the coefficient, reconstructs through remaining
    connector layers, generates, scores ANLS, and extracts first-token Δlogit
    artifacts.

    :param adapter: Model adapter instance.
    :type adapter: ModelAdapter
    :param data_categorized: Map of category name to list of data dicts.
    :type data_categorized: dict[str, list]
    :param per_category_coefficients: Per-layer SVD coefficients per category
        (CPU), ``{cat: {layer: Tensor}}``.
    :type per_category_coefficients: dict[str, dict[str, torch.Tensor]]
    :param per_category_a_star: Per-layer category means (CUDA),
        ``{cat: {layer: Tensor}}``.
    :type per_category_a_star: dict[str, dict[str, torch.Tensor]]
    :param global_a_star: Per-layer global means (CUDA), ``{layer: Tensor}``.
    :type global_a_star: dict[str, torch.Tensor]
    :param directions_to_ablate: Per-layer direction indices,
        ``{layer: [dir_indices]}``.
    :type directions_to_ablate: dict[str, list[int]]
    :param batch_size: Number of images per forward pass.
    :type batch_size: int
    :param K: Number of top/bottom tokens to keep per image.
    :type K: int
    :param image_base_path: Root directory for image files.
    :type image_base_path: Path
    :param run_dir: Output directory for this run.
    :type run_dir: Path
    :param rand_seed: Seed for the random baseline generator.
    :type rand_seed: int
    """
    svd_layer_map = {l.name: l for l in adapter.svd_layers}

    tasks = [
        (ln, j, d)
        for ln, dirs in directions_to_ablate.items()
        for j, d in enumerate(dirs)
    ]
    n_tasks = len(tasks)

    update_metadata(
        run_dir,
        dict(
            model=adapter.model_name,
            directions=directions_to_ablate,
            batch_size=batch_size,
            K=K,
            rand_seed=rand_seed,
            n_categories=len(data_categorized),
            category_sizes={n: len(d) for n, d in data_categorized.items()},
        ),
    )

    completed_ablation = {
        name
        for name in data_categorized
        if (run_dir / fs_safe(name) / "anls_summary.json").exists()
    }
    if completed_ablation:
        print(f"Resuming ablation: skipping {len(completed_ablation)} completed categories")

    for name, data in tqdm(data_categorized.items(), desc="Ablation + ANLS"):
        if name in completed_ablation:
            print(f'  Skipping "{name}" (results exist)')
            continue

        length = len(data)
        cat_a_star = per_category_a_star[name]
        cat_std = {
            ln: per_category_coefficients[name][ln].float().std(dim=(0, 1))
            for ln in directions_to_ablate
        }
        rand_gen = torch.Generator().manual_seed(rand_seed)

        nls_ablated_cat = [[] for _ in range(n_tasks)]
        nls_ablated_global = [[] for _ in range(n_tasks)]
        nls_ablated_zero = [[] for _ in range(n_tasks)]
        nls_ablated_rand = [[] for _ in range(n_tasks)]
        nls_original = []

        delta_sum_cat = [torch.zeros(adapter.vocab_size) for _ in range(n_tasks)]
        delta_abs_sum_cat = [torch.zeros(adapter.vocab_size) for _ in range(n_tasks)]
        delta_sum_global = [torch.zeros(adapter.vocab_size) for _ in range(n_tasks)]
        delta_abs_sum_global = [torch.zeros(adapter.vocab_size) for _ in range(n_tasks)]
        delta_sum_zero = [torch.zeros(adapter.vocab_size) for _ in range(n_tasks)]
        delta_abs_sum_zero = [torch.zeros(adapter.vocab_size) for _ in range(n_tasks)]
        delta_sum_rand = [torch.zeros(adapter.vocab_size) for _ in range(n_tasks)]
        delta_abs_sum_rand = [torch.zeros(adapter.vocab_size) for _ in range(n_tasks)]

        topk_cat = [[] for _ in range(n_tasks)]
        botk_cat = [[] for _ in range(n_tasks)]
        topk_global = [[] for _ in range(n_tasks)]
        botk_global = [[] for _ in range(n_tasks)]
        topk_zero = [[] for _ in range(n_tasks)]
        botk_zero = [[] for _ in range(n_tasks)]
        topk_rand = [[] for _ in range(n_tasks)]
        botk_rand = [[] for _ in range(n_tasks)]

        kl_div_cat = [[] for _ in range(n_tasks)]
        kl_div_global = [[] for _ in range(n_tasks)]
        kl_div_zero = [[] for _ in range(n_tasks)]
        kl_div_rand = [[] for _ in range(n_tasks)]

        delta_gold_prob_cat = [[] for _ in range(n_tasks)]
        delta_gold_prob_global = [[] for _ in range(n_tasks)]
        delta_gold_prob_zero = [[] for _ in range(n_tasks)]
        delta_gold_prob_rand = [[] for _ in range(n_tasks)]

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
                coeff = per_category_coefficients[name][layer.name][
                    i : i + actual
                ].to(dtype=adapter.compute_dtype, device="cuda")
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
                    last_coeff = per_category_coefficients[name][
                        last_layer.name
                    ][i : i + actual].to(
                        dtype=adapter.compute_dtype, device="cuda"
                    )
                    conn_out_orig = adapter.reconstruct(last_coeff)
                embeds_orig = adapter.merge_embeds(
                    inputs, text_embeds, conn_out_orig
                )
                preds_orig, logits_orig = adapter.generate_with_logits(
                    embeds_orig, attention_mask
                )

            for targets, pred in zip(targets_batch, preds_orig):
                nls_original.append(best_anls(pred, targets))

            probs_orig = F.softmax(logits_orig, dim=-1)
            log_probs_orig = F.log_softmax(logits_orig, dim=-1)
            gold_tok = [
                adapter.processor.tokenizer(
                    targets[0], add_special_tokens=False
                )["input_ids"][-1]
                for targets in targets_batch
            ]
            gold_idx = torch.tensor(gold_tok, device=logits_orig.device)
            lp_orig = log_probs_orig[range(actual), gold_idx]
            stacked_attn = attention_mask.repeat(4, 1)

            for t_idx, (layer_name, j, dir_idx) in enumerate(tasks):
                layer = svd_layer_map[layer_name]
                u_d = layer.U[:, dir_idx].to(batch_coeffs[layer_name].dtype)
                orig_d = batch_coeffs[layer_name][..., dir_idx]
                l_out = layer_outputs[layer_name]

                rand_vals = (
                    cat_a_star[layer_name][dir_idx]
                    + torch.randn(
                        *batch_coeffs[layer_name].shape[:-1],
                        generator=rand_gen,
                    ).to(
                        device=batch_coeffs[layer_name].device,
                        dtype=batch_coeffs[layer_name].dtype,
                    )
                    * cat_std[layer_name][dir_idx].to(
                        batch_coeffs[layer_name].device,
                        batch_coeffs[layer_name].dtype,
                    )
                )

                with torch.no_grad():
                    stacked_l = torch.cat([
                        l_out + (cat_a_star[layer_name][dir_idx] - orig_d).unsqueeze(-1) * u_d,
                        l_out + (global_a_star[layer_name][dir_idx] - orig_d).unsqueeze(-1) * u_d,
                        l_out + (-orig_d).unsqueeze(-1) * u_d,
                        l_out + (rand_vals - orig_d).unsqueeze(-1) * u_d,
                    ], dim=0)
                    stacked_conn = adapter.forward_connector_from(
                        layer_name, stacked_l
                    )
                    conn_parts = stacked_conn.split(actual)

                    stacked_embeds = torch.cat([
                        adapter.merge_embeds(inputs, text_embeds, c)
                        for c in conn_parts
                    ], dim=0)

                    all_preds, all_logits = adapter.generate_with_logits(
                        stacked_embeds, stacked_attn
                    )

                preds_cat = all_preds[:actual]
                preds_global = all_preds[actual : 2 * actual]
                preds_zero = all_preds[2 * actual : 3 * actual]
                preds_rand = all_preds[3 * actual :]

                logits_cat, logits_global, logits_zero, logits_rand = (
                    all_logits.split(actual)
                )

                delta_cat = (logits_orig - logits_cat).float()
                delta_global = (logits_orig - logits_global).float()
                delta_zero = (logits_orig - logits_zero).float()
                delta_rand = (logits_orig - logits_rand).float()

                all_log_probs = F.log_softmax(all_logits, dim=-1)
                probs_orig_exp = probs_orig.repeat(4, 1)
                kl_all = F.kl_div(
                    all_log_probs, probs_orig_exp, reduction="none"
                ).sum(-1)
                kl_c, kl_g, kl_z, kl_r = kl_all.split(actual)
                kl_div_cat[t_idx].extend(kl_c.cpu().tolist())
                kl_div_global[t_idx].extend(kl_g.cpu().tolist())
                kl_div_zero[t_idx].extend(kl_z.cpu().tolist())
                kl_div_rand[t_idx].extend(kl_r.cpu().tolist())

                gold_idx_exp = gold_idx.repeat(4)
                all_lp = all_log_probs[range(4 * actual), gold_idx_exp]
                lp_c, lp_g, lp_z, lp_r = all_lp.split(actual)
                delta_gold_prob_cat[t_idx].extend((lp_orig - lp_c).cpu().tolist())
                delta_gold_prob_global[t_idx].extend((lp_orig - lp_g).cpu().tolist())
                delta_gold_prob_zero[t_idx].extend((lp_orig - lp_z).cpu().tolist())
                delta_gold_prob_rand[t_idx].extend((lp_orig - lp_r).cpu().tolist())

                delta_sum_cat[t_idx] += delta_cat.sum(dim=0).cpu()
                delta_abs_sum_cat[t_idx] += delta_cat.abs().sum(dim=0).cpu()
                delta_sum_global[t_idx] += delta_global.sum(dim=0).cpu()
                delta_abs_sum_global[t_idx] += delta_global.abs().sum(dim=0).cpu()
                delta_sum_zero[t_idx] += delta_zero.sum(dim=0).cpu()
                delta_abs_sum_zero[t_idx] += delta_zero.abs().sum(dim=0).cpu()
                delta_sum_rand[t_idx] += delta_rand.sum(dim=0).cpu()
                delta_abs_sum_rand[t_idx] += delta_rand.abs().sum(dim=0).cpu()

                probs_cat = F.softmax(logits_cat, dim=-1)
                probs_global = F.softmax(logits_global, dim=-1)
                probs_zero = F.softmax(logits_zero, dim=-1)
                probs_rand = F.softmax(logits_rand, dim=-1)

                top_v, top_i = delta_cat.topk(K, dim=-1)
                bot_v, bot_i = (-delta_cat).topk(K, dim=-1)
                top_p_orig = probs_orig.gather(-1, top_i.long())
                bot_p_orig = probs_orig.gather(-1, bot_i.long())
                top_p_cat = probs_cat.gather(-1, top_i.long())
                bot_p_cat = probs_cat.gather(-1, bot_i.long())
                topk_cat[t_idx].append(torch.stack([top_i, top_v, top_p_orig, top_p_cat], dim=1).cpu())
                botk_cat[t_idx].append(torch.stack([bot_i, bot_v, bot_p_orig, bot_p_cat], dim=1).cpu())

                top_v, top_i = delta_global.topk(K, dim=-1)
                bot_v, bot_i = (-delta_global).topk(K, dim=-1)
                top_p_orig = probs_orig.gather(-1, top_i.long())
                bot_p_orig = probs_orig.gather(-1, bot_i.long())
                top_p_global = probs_global.gather(-1, top_i.long())
                bot_p_global = probs_global.gather(-1, bot_i.long())
                topk_global[t_idx].append(torch.stack([top_i, top_v, top_p_orig, top_p_global], dim=1).cpu())
                botk_global[t_idx].append(torch.stack([bot_i, bot_v, bot_p_orig, bot_p_global], dim=1).cpu())

                top_v, top_i = delta_zero.topk(K, dim=-1)
                bot_v, bot_i = (-delta_zero).topk(K, dim=-1)
                top_p_orig = probs_orig.gather(-1, top_i.long())
                bot_p_orig = probs_orig.gather(-1, bot_i.long())
                top_p_zero = probs_zero.gather(-1, top_i.long())
                bot_p_zero = probs_zero.gather(-1, bot_i.long())
                topk_zero[t_idx].append(torch.stack([top_i, top_v, top_p_orig, top_p_zero], dim=1).cpu())
                botk_zero[t_idx].append(torch.stack([bot_i, bot_v, bot_p_orig, bot_p_zero], dim=1).cpu())

                top_v, top_i = delta_rand.topk(K, dim=-1)
                bot_v, bot_i = (-delta_rand).topk(K, dim=-1)
                top_p_orig = probs_orig.gather(-1, top_i.long())
                bot_p_orig = probs_orig.gather(-1, bot_i.long())
                top_p_rand = probs_rand.gather(-1, top_i.long())
                bot_p_rand = probs_rand.gather(-1, bot_i.long())
                topk_rand[t_idx].append(torch.stack([top_i, top_v, top_p_orig, top_p_rand], dim=1).cpu())
                botk_rand[t_idx].append(torch.stack([bot_i, bot_v, bot_p_orig, bot_p_rand], dim=1).cpu())

                for targets, pred in zip(targets_batch, preds_cat):
                    nls_ablated_cat[t_idx].append(best_anls(pred, targets))
                for targets, pred in zip(targets_batch, preds_global):
                    nls_ablated_global[t_idx].append(best_anls(pred, targets))
                for targets, pred in zip(targets_batch, preds_zero):
                    nls_ablated_zero[t_idx].append(best_anls(pred, targets))
                for targets, pred in zip(targets_batch, preds_rand):
                    nls_ablated_rand[t_idx].append(best_anls(pred, targets))

        # --- Save per-category results ---
        cat_dir = run_dir / fs_safe(name)
        cat_dir.mkdir(parents=True, exist_ok=True)

        np.save(cat_dir / "nls_original.npy", np.array(nls_original))

        layer_delta = {}
        layer_nls = {}
        for t_idx, (ln, j, dir_idx) in enumerate(tasks):
            if ln not in layer_delta:
                layer_delta[ln] = {}
                layer_nls[ln] = {
                    "dirs": [], "cat": [], "global": [], "zero": [], "rand": [],
                }
            layer_delta[ln][dir_idx] = {
                "signed_mean_cat": delta_sum_cat[t_idx] / length,
                "abs_mean_cat": delta_abs_sum_cat[t_idx] / length,
                "topk_cat": torch.cat(topk_cat[t_idx], dim=0),
                "botk_cat": torch.cat(botk_cat[t_idx], dim=0),
                "kl_div_cat": kl_div_cat[t_idx],
                "delta_gold_prob_cat": delta_gold_prob_cat[t_idx],
                "signed_mean_global": delta_sum_global[t_idx] / length,
                "abs_mean_global": delta_abs_sum_global[t_idx] / length,
                "topk_global": torch.cat(topk_global[t_idx], dim=0),
                "botk_global": torch.cat(botk_global[t_idx], dim=0),
                "kl_div_global": kl_div_global[t_idx],
                "delta_gold_prob_global": delta_gold_prob_global[t_idx],
                "signed_mean_zero": delta_sum_zero[t_idx] / length,
                "abs_mean_zero": delta_abs_sum_zero[t_idx] / length,
                "topk_zero": torch.cat(topk_zero[t_idx], dim=0),
                "botk_zero": torch.cat(botk_zero[t_idx], dim=0),
                "kl_div_zero": kl_div_zero[t_idx],
                "delta_gold_prob_zero": delta_gold_prob_zero[t_idx],
                "signed_mean_rand": delta_sum_rand[t_idx] / length,
                "abs_mean_rand": delta_abs_sum_rand[t_idx] / length,
                "topk_rand": torch.cat(topk_rand[t_idx], dim=0),
                "botk_rand": torch.cat(botk_rand[t_idx], dim=0),
                "kl_div_rand": kl_div_rand[t_idx],
                "delta_gold_prob_rand": delta_gold_prob_rand[t_idx],
            }
            layer_nls[ln]["dirs"].append(dir_idx)
            layer_nls[ln]["cat"].append(sum(nls_ablated_cat[t_idx]) / length)
            layer_nls[ln]["global"].append(sum(nls_ablated_global[t_idx]) / length)
            layer_nls[ln]["zero"].append(sum(nls_ablated_zero[t_idx]) / length)
            layer_nls[ln]["rand"].append(sum(nls_ablated_rand[t_idx]) / length)

        np.save(
            cat_dir / "nls_ablated_cat.npy",
            {ln: [nls_ablated_cat[t] for t, (l, _, _) in enumerate(tasks) if l == ln] for ln in layer_nls},
            allow_pickle=True,
        )
        np.save(
            cat_dir / "nls_ablated_global.npy",
            {ln: [nls_ablated_global[t] for t, (l, _, _) in enumerate(tasks) if l == ln] for ln in layer_nls},
            allow_pickle=True,
        )
        np.save(
            cat_dir / "nls_ablated_zero.npy",
            {ln: [nls_ablated_zero[t] for t, (l, _, _) in enumerate(tasks) if l == ln] for ln in layer_nls},
            allow_pickle=True,
        )
        np.save(
            cat_dir / "nls_ablated_rand.npy",
            {ln: [nls_ablated_rand[t] for t, (l, _, _) in enumerate(tasks) if l == ln] for ln in layer_nls},
            allow_pickle=True,
        )

        _DELTA_SCHEMA = {
            "signed_mean_*": "(vocab_size,) mean logit delta across images",
            "abs_mean_*": "(vocab_size,) mean |logit delta| across images",
            "topk_*": "(n_images, 4, K) top-K positive logit shifts; "
            "channels: [token_idx, logit_delta, prob_orig, prob_ablated]",
            "botk_*": "(n_images, 4, K) top-K negative logit shifts; "
            "channels: [token_idx, logit_delta, prob_orig, prob_ablated]",
            "kl_div_*": "(directions, n_images) KL divergence between original "
            "and ablated logit vectors",
            "delta_gold_prob_*": "(directions, n_images) difference between last "
            "token's log probability of original and ablated "
            "logit vectors",
        }
        torch.save(
            {"_schema": _DELTA_SCHEMA, **layer_delta},
            cat_dir / "delta_logits.pt",
        )

        anls_orig = sum(nls_original) / length
        save_json(
            cat_dir / "anls_summary.json",
            dict(
                category=name,
                n_samples=length,
                directions={ln: layer_nls[ln]["dirs"] for ln in layer_nls},
                anls_original=anls_orig,
                anls_ablated_per_category_mean={
                    ln: layer_nls[ln]["cat"] for ln in layer_nls
                },
                anls_ablated_global_mean={
                    ln: layer_nls[ln]["global"] for ln in layer_nls
                },
                anls_ablated_zero_mean={
                    ln: layer_nls[ln]["zero"] for ln in layer_nls
                },
                anls_ablated_random={
                    ln: layer_nls[ln]["rand"] for ln in layer_nls
                },
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
    per_category_coefficients: dict[str, dict[str, torch.Tensor]],
    per_category_a_star: dict[str, dict[str, torch.Tensor]],
    global_a_star: dict[str, torch.Tensor],
    *,
    direction_sets: dict[str, dict[str, dict[str, list[int]]]],
    batch_size: int,
    K: int,
    image_base_path: Path,
    run_dir: Path,
    rand_seed: int = 42,
) -> None:
    """
    Joint ablation of entire direction sets with ANLS scoring and Δlogit
    extraction.

    Ablates all directions in a set simultaneously (multi-rank correction)
    and measures the effect. Designed for comparing DM mask active sets
    against random controls.

    :param adapter: Model adapter instance.
    :type adapter: ModelAdapter
    :param data_categorized: Map of category name to list of data dicts.
    :type data_categorized: dict[str, list]
    :param per_category_coefficients: Per-layer SVD coefficients per category
        (CPU), ``{cat: {layer: Tensor}}``.
    :type per_category_coefficients: dict[str, dict[str, torch.Tensor]]
    :param per_category_a_star: Per-layer category means (CUDA),
        ``{cat: {layer: Tensor}}``.
    :type per_category_a_star: dict[str, dict[str, torch.Tensor]]
    :param global_a_star: Per-layer global means (CUDA), ``{layer: Tensor}``.
    :type global_a_star: dict[str, torch.Tensor]
    :param direction_sets: Per-set, per-category, per-layer direction indices.
        ``{set_label: {category: {layer: [dir_indices]}}}``.
    :type direction_sets: dict[str, dict[str, dict[str, list[int]]]]
    :param batch_size: Number of images per forward pass.
    :type batch_size: int
    :param K: Number of top/bottom tokens to keep per image.
    :type K: int
    :param image_base_path: Root directory for image files.
    :type image_base_path: Path
    :param run_dir: Output directory for this run.
    :type run_dir: Path
    :param rand_seed: Seed for the random baseline generator.
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
        cat_std = {
            ln: per_category_coefficients[name][ln].float().std(dim=(0, 1))
            for ln in per_category_coefficients[name]
        }

        nls_original = []
        set_results: dict[str, dict[str, dict]] = {}
        for sl in set_labels:
            set_results[sl] = {}
            cat_layers = direction_sets[sl].get(name, {})
            for ln in cat_layers:
                set_results[sl][ln] = {
                    "nls_cat": [], "nls_global": [], "nls_zero": [], "nls_rand": [],
                    "kl_cat": [], "kl_global": [], "kl_zero": [], "kl_rand": [],
                    "dgp_cat": [], "dgp_global": [], "dgp_zero": [], "dgp_rand": [],
                    "delta_sum_cat": torch.zeros(adapter.vocab_size),
                    "delta_abs_sum_cat": torch.zeros(adapter.vocab_size),
                    "delta_sum_global": torch.zeros(adapter.vocab_size),
                    "delta_abs_sum_global": torch.zeros(adapter.vocab_size),
                    "delta_sum_zero": torch.zeros(adapter.vocab_size),
                    "delta_abs_sum_zero": torch.zeros(adapter.vocab_size),
                    "delta_sum_rand": torch.zeros(adapter.vocab_size),
                    "delta_abs_sum_rand": torch.zeros(adapter.vocab_size),
                    "topk_cat": [], "botk_cat": [],
                    "topk_global": [], "botk_global": [],
                    "topk_zero": [], "botk_zero": [],
                    "topk_rand": [], "botk_rand": [],
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
                coeff = per_category_coefficients[name][layer.name][
                    i : i + actual
                ].to(dtype=adapter.compute_dtype, device="cuda")
                batch_coeffs[layer.name] = coeff
                out = coeff @ layer.U.T.to(coeff.dtype)
                if layer.bias is not None:
                    out = out + layer.bias.to(coeff.dtype)
                layer_outputs[layer.name] = out

            with torch.no_grad():
                last_layer = adapter.svd_layers[-1]
                conn_out_orig = layer_outputs[last_layer.name]
                embeds_orig = adapter.merge_embeds(
                    inputs, text_embeds, conn_out_orig
                )
                preds_orig, logits_orig = adapter.generate_with_logits(
                    embeds_orig, attention_mask
                )

            for targets, pred in zip(targets_batch, preds_orig):
                nls_original.append(best_anls(pred, targets))

            probs_orig = F.softmax(logits_orig, dim=-1)
            log_probs_orig = F.log_softmax(logits_orig, dim=-1)
            gold_tok = [
                adapter.processor.tokenizer(
                    targets[0], add_special_tokens=False
                )["input_ids"][-1]
                for targets in targets_batch
            ]
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

                    std_sub = cat_std[layer_name][dir_list].to(
                        coeff.device, coeff.dtype
                    )
                    rand_sub = cat_sub + torch.randn(
                        *orig_sub.shape, generator=rand_gen,
                    ).to(device=coeff.device, dtype=coeff.dtype) * std_sub

                    with torch.no_grad():
                        delta_cat = (cat_sub - orig_sub) @ U_sub.T
                        delta_global = (global_sub - orig_sub) @ U_sub.T
                        delta_zero = (-orig_sub) @ U_sub.T
                        delta_rand = (rand_sub - orig_sub) @ U_sub.T

                        stacked_l = torch.cat([
                            l_out + delta_cat,
                            l_out + delta_global,
                            l_out + delta_zero,
                            l_out + delta_rand,
                        ], dim=0)

                        stacked_conn = adapter.forward_connector_from(
                            layer_name, stacked_l
                        )
                        conn_parts = stacked_conn.split(actual)

                        stacked_embeds = torch.cat([
                            adapter.merge_embeds(inputs, text_embeds, c)
                            for c in conn_parts
                        ], dim=0)

                        all_preds, all_logits = adapter.generate_with_logits(
                            stacked_embeds, stacked_attn
                        )

                    preds_cat = all_preds[:actual]
                    preds_global = all_preds[actual : 2 * actual]
                    preds_zero = all_preds[2 * actual : 3 * actual]
                    preds_rand = all_preds[3 * actual :]

                    for targets, pred in zip(targets_batch, preds_cat):
                        set_results[sl][layer_name]["nls_cat"].append(
                            best_anls(pred, targets)
                        )
                    for targets, pred in zip(targets_batch, preds_global):
                        set_results[sl][layer_name]["nls_global"].append(
                            best_anls(pred, targets)
                        )
                    for targets, pred in zip(targets_batch, preds_zero):
                        set_results[sl][layer_name]["nls_zero"].append(
                            best_anls(pred, targets)
                        )
                    for targets, pred in zip(targets_batch, preds_rand):
                        set_results[sl][layer_name]["nls_rand"].append(
                            best_anls(pred, targets)
                        )

                    all_log_probs = F.log_softmax(all_logits, dim=-1)
                    probs_orig_exp = probs_orig.repeat(4, 1)
                    kl_all = F.kl_div(
                        all_log_probs, probs_orig_exp, reduction="none"
                    ).sum(-1)
                    kl_c, kl_g, kl_z, kl_r = kl_all.split(actual)
                    set_results[sl][layer_name]["kl_cat"].extend(
                        kl_c.cpu().tolist()
                    )
                    set_results[sl][layer_name]["kl_global"].extend(
                        kl_g.cpu().tolist()
                    )
                    set_results[sl][layer_name]["kl_zero"].extend(
                        kl_z.cpu().tolist()
                    )
                    set_results[sl][layer_name]["kl_rand"].extend(
                        kl_r.cpu().tolist()
                    )

                    gold_idx_exp = gold_idx.repeat(4)
                    all_lp = all_log_probs[range(4 * actual), gold_idx_exp]
                    lp_c, lp_g, lp_z, lp_r = all_lp.split(actual)
                    set_results[sl][layer_name]["dgp_cat"].extend(
                        (lp_orig - lp_c).cpu().tolist()
                    )
                    set_results[sl][layer_name]["dgp_global"].extend(
                        (lp_orig - lp_g).cpu().tolist()
                    )
                    set_results[sl][layer_name]["dgp_zero"].extend(
                        (lp_orig - lp_z).cpu().tolist()
                    )
                    set_results[sl][layer_name]["dgp_rand"].extend(
                        (lp_orig - lp_r).cpu().tolist()
                    )

                    logits_cat, logits_global, logits_zero, logits_rand = (
                        all_logits.split(actual)
                    )
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
                        r[tk_key].append(torch.stack([top_i, top_v, top_p_orig, top_p_abl], dim=1).cpu())
                        r[bk_key].append(torch.stack([bot_i, bot_v, bot_p_orig, bot_p_abl], dim=1).cpu())

        cat_dir = run_dir / fs_safe(name)
        cat_dir.mkdir(parents=True, exist_ok=True)

        anls_orig = sum(nls_original) / length
        summary: dict = {
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
                    "topk_cat": torch.cat(res["topk_cat"], dim=0) if res["topk_cat"] else torch.empty(0),
                    "botk_cat": torch.cat(res["botk_cat"], dim=0) if res["botk_cat"] else torch.empty(0),
                    "topk_global": torch.cat(res["topk_global"], dim=0) if res["topk_global"] else torch.empty(0),
                    "botk_global": torch.cat(res["botk_global"], dim=0) if res["botk_global"] else torch.empty(0),
                    "topk_zero": torch.cat(res["topk_zero"], dim=0) if res["topk_zero"] else torch.empty(0),
                    "botk_zero": torch.cat(res["botk_zero"], dim=0) if res["botk_zero"] else torch.empty(0),
                    "topk_rand": torch.cat(res["topk_rand"], dim=0) if res["topk_rand"] else torch.empty(0),
                    "botk_rand": torch.cat(res["botk_rand"], dim=0) if res["botk_rand"] else torch.empty(0),
                }
        torch.save(delta_logits, cat_dir / "joint_delta_logits.pt")

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
