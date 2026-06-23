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
    ...     directions_to_ablate=[23, 70], batch_size=4, K=15,
    ...     image_base_path=img_path, run_dir=run_dir)

Main Functions:
    compute_category_means: SVD coefficients + per-category/global means.
    run_ablation: Per-direction ablation with ANLS scoring and Δlogit extraction.
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


def compute_category_means(
    adapter: ModelAdapter,
    data_categorized: dict[str, list],
    *,
    batch_size: int,
    image_base_path: Path,
    run_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """
    Compute SVD coefficients and per-category/global mean vectors.

    Images appearing in multiple categories are counted once for the
    global mean (tracked via a ``seen_images`` set). Supports resuming
    from checkpoints.

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
        where coefficients are on CPU and means are on CUDA.
    :rtype: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]
    """
    completed = get_completed_categories(run_dir)

    per_category_coefficients = {}
    per_category_a_star = {}

    global_mean_sum = torch.zeros(adapter.n_dirs, dtype=torch.float32)
    n_unique_images = 0
    seen_images = set()

    for name, data in tqdm(data_categorized.items(), desc="Computing category means"):
        if fs_safe(name) in completed:
            ckpt = load_checkpoint(run_dir, name)
            per_category_coefficients[name] = ckpt["coefficients"]
            per_category_a_star[name] = ckpt["a_star"].to(
                dtype=adapter.compute_dtype, device="cuda"
            )
            for img_id in ckpt.get("new_image_ids", []):
                seen_images.add(img_id)
            global_mean_sum += ckpt.get("global_mean_contrib", torch.zeros(adapter.n_dirs))
            n_unique_images += ckpt.get("n_new_images", 0)
            print(f'  Loaded checkpoint for "{name}"')
            continue

        length = len(data)
        cat_coefficients = torch.empty(length, adapter.n_patches, adapter.n_dirs)
        cat_global_contrib = torch.zeros(adapter.n_dirs, dtype=torch.float32)
        cat_new_images = []

        for i in tqdm(
            range(0, length, batch_size),
            total=math.ceil(length / batch_size),
            desc=f'"{name}"',
        ):
            batch = data[i : i + batch_size]
            inputs = adapter.preprocess(batch, image_base_path)
            coefficients = adapter.compute_coefficients(inputs)
            actual = coefficients.shape[0]
            cat_coefficients[i : i + actual] = coefficients.cpu()

            for j, datum in enumerate(batch):
                img_id = datum["image"]
                if img_id not in seen_images:
                    seen_images.add(img_id)
                    cat_new_images.append(img_id)
                    cat_global_contrib += coefficients[j].mean(dim=0).cpu().float()

        per_category_coefficients[name] = cat_coefficients
        per_category_a_star[name] = cat_coefficients.mean(dim=(0, 1)).to(
            dtype=adapter.compute_dtype, device="cuda"
        )
        global_mean_sum += cat_global_contrib
        n_unique_images += len(cat_new_images)

        save_checkpoint(
            run_dir,
            name,
            {
                "coefficients": cat_coefficients,
                "a_star": per_category_a_star[name].cpu(),
                "new_image_ids": cat_new_images,
                "global_mean_contrib": cat_global_contrib,
                "n_new_images": len(cat_new_images),
            },
        )

    global_a_star = (global_mean_sum / n_unique_images).to(
        dtype=adapter.compute_dtype, device="cuda"
    )

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
    per_category_coefficients: dict[str, torch.Tensor],
    per_category_a_star: dict[str, torch.Tensor],
    global_a_star: torch.Tensor,
    *,
    directions_to_ablate: list[int],
    batch_size: int,
    K: int,
    image_base_path: Path,
    run_dir: Path,
    rand_seed: int = 42,
) -> None:
    """
    Per-direction ablation with ANLS scoring and Δlogit extraction.

    For each direction and baseline (per-category mean, global mean,
    zero, random):  replaces the coefficient, reconstructs, generates,
    scores ANLS, and extracts first-token Δlogit artifacts (signed mean,
    abs mean, top-K, bottom-K, KL divergence, gold-token probability shift).

    The random baseline samples from ``N(cat_mean, cat_std)`` for each
    direction, seeded by *rand_seed* for reproducibility.

    :param adapter: Model adapter instance.
    :type adapter: ModelAdapter
    :param data_categorized: Map of category name to list of data dicts.
    :type data_categorized: dict[str, list]
    :param per_category_coefficients: SVD coefficients per category (CPU).
    :type per_category_coefficients: dict[str, torch.Tensor]
    :param per_category_a_star: Per-category mean vectors (CUDA).
    :type per_category_a_star: dict[str, torch.Tensor]
    :param global_a_star: Global mean vector (CUDA).
    :type global_a_star: torch.Tensor
    :param directions_to_ablate: SVD direction indices to ablate.
    :type directions_to_ablate: list[int]
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
        cat_std = per_category_coefficients[name].float().std(dim=(0, 1))
        rand_gen = torch.Generator().manual_seed(rand_seed)

        nls_ablated_cat = [[] for _ in directions_to_ablate]
        nls_ablated_global = [[] for _ in directions_to_ablate]
        nls_ablated_zero = [[] for _ in directions_to_ablate]
        nls_ablated_rand = [[] for _ in directions_to_ablate]
        nls_original = []

        delta_sum_cat = [torch.zeros(adapter.vocab_size) for _ in directions_to_ablate]
        delta_abs_sum_cat = [torch.zeros(adapter.vocab_size) for _ in directions_to_ablate]
        delta_sum_global = [torch.zeros(adapter.vocab_size) for _ in directions_to_ablate]
        delta_abs_sum_global = [torch.zeros(adapter.vocab_size) for _ in directions_to_ablate]
        delta_sum_zero = [torch.zeros(adapter.vocab_size) for _ in directions_to_ablate]
        delta_abs_sum_zero = [torch.zeros(adapter.vocab_size) for _ in directions_to_ablate]
        delta_sum_rand = [torch.zeros(adapter.vocab_size) for _ in directions_to_ablate]
        delta_abs_sum_rand = [torch.zeros(adapter.vocab_size) for _ in directions_to_ablate]

        topk_cat = [[] for _ in directions_to_ablate]
        botk_cat = [[] for _ in directions_to_ablate]
        topk_global = [[] for _ in directions_to_ablate]
        botk_global = [[] for _ in directions_to_ablate]
        topk_zero = [[] for _ in directions_to_ablate]
        botk_zero = [[] for _ in directions_to_ablate]
        topk_rand = [[] for _ in directions_to_ablate]
        botk_rand = [[] for _ in directions_to_ablate]

        kl_div_cat = [[] for _ in directions_to_ablate]
        kl_div_global = [[] for _ in directions_to_ablate]
        kl_div_zero = [[] for _ in directions_to_ablate]
        kl_div_rand = [[] for _ in directions_to_ablate]

        delta_gold_prob_cat = [[] for _ in directions_to_ablate]
        delta_gold_prob_global = [[] for _ in directions_to_ablate]
        delta_gold_prob_zero = [[] for _ in directions_to_ablate]
        delta_gold_prob_rand = [[] for _ in directions_to_ablate]

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

            batch_coeff = per_category_coefficients[name][i : i + actual].to(
                dtype=adapter.compute_dtype, device="cuda"
            )
            text_embeds = adapter.get_text_embeds(inputs)

            with torch.no_grad():
                conn_out_orig = adapter.reconstruct(batch_coeff)
                embeds_orig = adapter.merge_embeds(
                    inputs, text_embeds, conn_out_orig
                )
                preds_orig = adapter.generate(embeds_orig, attention_mask)
                logits_orig = adapter.get_logits(embeds_orig, attention_mask)

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

            for j, dir_idx in enumerate(directions_to_ablate):
                u_d = adapter.U[:, dir_idx].to(batch_coeff.dtype)
                orig_d = batch_coeff[..., dir_idx]

                rand_vals = (
                    cat_a_star[dir_idx]
                    + torch.randn(
                        *batch_coeff.shape[:-1],
                        generator=rand_gen,
                    ).to(device=batch_coeff.device, dtype=batch_coeff.dtype)
                    * cat_std[dir_idx].to(batch_coeff.device, batch_coeff.dtype)
                )

                with torch.no_grad():
                    conn_cat = conn_out_orig + (cat_a_star[dir_idx] - orig_d).unsqueeze(-1) * u_d
                    conn_global = conn_out_orig + (global_a_star[dir_idx] - orig_d).unsqueeze(-1) * u_d
                    conn_zero = conn_out_orig + (-orig_d).unsqueeze(-1) * u_d
                    conn_rand = conn_out_orig + (rand_vals - orig_d).unsqueeze(-1) * u_d

                    stacked_embeds = torch.cat([
                        adapter.merge_embeds(inputs, text_embeds, conn_cat),
                        adapter.merge_embeds(inputs, text_embeds, conn_global),
                        adapter.merge_embeds(inputs, text_embeds, conn_zero),
                        adapter.merge_embeds(inputs, text_embeds, conn_rand),
                    ], dim=0)

                    all_preds = adapter.generate(stacked_embeds, stacked_attn)
                    all_logits = adapter.get_logits(stacked_embeds, stacked_attn)

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
                kl_div_cat[j].extend(kl_c.cpu().tolist())
                kl_div_global[j].extend(kl_g.cpu().tolist())
                kl_div_zero[j].extend(kl_z.cpu().tolist())
                kl_div_rand[j].extend(kl_r.cpu().tolist())

                gold_idx_exp = gold_idx.repeat(4)
                all_lp = all_log_probs[range(4 * actual), gold_idx_exp]
                lp_c, lp_g, lp_z, lp_r = all_lp.split(actual)
                delta_gold_prob_cat[j].extend((lp_orig - lp_c).cpu().tolist())
                delta_gold_prob_global[j].extend((lp_orig - lp_g).cpu().tolist())
                delta_gold_prob_zero[j].extend((lp_orig - lp_z).cpu().tolist())
                delta_gold_prob_rand[j].extend((lp_orig - lp_r).cpu().tolist())

                delta_sum_cat[j] += delta_cat.sum(dim=0).cpu()
                delta_abs_sum_cat[j] += delta_cat.abs().sum(dim=0).cpu()
                delta_sum_global[j] += delta_global.sum(dim=0).cpu()
                delta_abs_sum_global[j] += delta_global.abs().sum(dim=0).cpu()
                delta_sum_zero[j] += delta_zero.sum(dim=0).cpu()
                delta_abs_sum_zero[j] += delta_zero.abs().sum(dim=0).cpu()
                delta_sum_rand[j] += delta_rand.sum(dim=0).cpu()
                delta_abs_sum_rand[j] += delta_rand.abs().sum(dim=0).cpu()

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
                topk_cat[j].append(torch.stack([top_i, top_v, top_p_orig, top_p_cat], dim=1).cpu())
                botk_cat[j].append(torch.stack([bot_i, bot_v, bot_p_orig, bot_p_cat], dim=1).cpu())

                top_v, top_i = delta_global.topk(K, dim=-1)
                bot_v, bot_i = (-delta_global).topk(K, dim=-1)
                top_p_orig = probs_orig.gather(-1, top_i.long())
                bot_p_orig = probs_orig.gather(-1, bot_i.long())
                top_p_global = probs_global.gather(-1, top_i.long())
                bot_p_global = probs_global.gather(-1, bot_i.long())
                topk_global[j].append(torch.stack([top_i, top_v, top_p_orig, top_p_global], dim=1).cpu())
                botk_global[j].append(torch.stack([bot_i, bot_v, bot_p_orig, bot_p_global], dim=1).cpu())

                top_v, top_i = delta_zero.topk(K, dim=-1)
                bot_v, bot_i = (-delta_zero).topk(K, dim=-1)
                top_p_orig = probs_orig.gather(-1, top_i.long())
                bot_p_orig = probs_orig.gather(-1, bot_i.long())
                top_p_zero = probs_zero.gather(-1, top_i.long())
                bot_p_zero = probs_zero.gather(-1, bot_i.long())
                topk_zero[j].append(torch.stack([top_i, top_v, top_p_orig, top_p_zero], dim=1).cpu())
                botk_zero[j].append(torch.stack([bot_i, bot_v, bot_p_orig, bot_p_zero], dim=1).cpu())

                top_v, top_i = delta_rand.topk(K, dim=-1)
                bot_v, bot_i = (-delta_rand).topk(K, dim=-1)
                top_p_orig = probs_orig.gather(-1, top_i.long())
                bot_p_orig = probs_orig.gather(-1, bot_i.long())
                top_p_rand = probs_rand.gather(-1, top_i.long())
                bot_p_rand = probs_rand.gather(-1, bot_i.long())
                topk_rand[j].append(torch.stack([top_i, top_v, top_p_orig, top_p_rand], dim=1).cpu())
                botk_rand[j].append(torch.stack([bot_i, bot_v, bot_p_orig, bot_p_rand], dim=1).cpu())

                for targets, pred in zip(targets_batch, preds_cat):
                    nls_ablated_cat[j].append(best_anls(pred, targets))
                for targets, pred in zip(targets_batch, preds_global):
                    nls_ablated_global[j].append(best_anls(pred, targets))
                for targets, pred in zip(targets_batch, preds_zero):
                    nls_ablated_zero[j].append(best_anls(pred, targets))
                for targets, pred in zip(targets_batch, preds_rand):
                    nls_ablated_rand[j].append(best_anls(pred, targets))

        # --- Save per-category results ---
        cat_dir = run_dir / fs_safe(name)
        cat_dir.mkdir(parents=True, exist_ok=True)

        np.save(cat_dir / "nls_original.npy", np.array(nls_original))
        np.save(
            cat_dir / "nls_ablated_cat.npy",
            np.array(nls_ablated_cat, dtype=object),
        )
        np.save(
            cat_dir / "nls_ablated_global.npy",
            np.array(nls_ablated_global, dtype=object),
        )
        np.save(
            cat_dir / "nls_ablated_zero.npy",
            np.array(nls_ablated_zero, dtype=object),
        )
        np.save(
            cat_dir / "nls_ablated_rand.npy",
            np.array(nls_ablated_rand, dtype=object),
        )

        torch.save(
            {
                "_schema": {
                    "signed_mean_*": "(vocab_size,) mean logit delta across images",
                    "abs_mean_*": "(vocab_size,) mean |logit delta| across images",
                    "topk_*": "(n_images, 4, K) top-K positive logit shifts; "
                    "channels: [token_idx, logit_delta, prob_orig, prob_ablated]",
                    "botk_*": "(n_images, 4, K) top-K negative logit shifts; "
                    "channels: [token_idx, logit_delta, prob_orig, prob_ablated]",
                    "kl_div_*": "(directions, n_images) KL divergence between original "
                    "and ablated logit vectors",
                    "delta_gold_prob_*": "(directions, n_images) difference between last "
                    "token's log probabiliy of original and ablated "
                    "logit vectors",
                },
                **{
                    dir_idx: {
                        "signed_mean_cat": delta_sum_cat[j] / length,
                        "abs_mean_cat": delta_abs_sum_cat[j] / length,
                        "topk_cat": torch.cat(topk_cat[j], dim=0),
                        "botk_cat": torch.cat(botk_cat[j], dim=0),
                        "kl_div_cat": kl_div_cat[j],
                        "delta_gold_prob_cat": delta_gold_prob_cat[j],
                        "signed_mean_global": delta_sum_global[j] / length,
                        "abs_mean_global": delta_abs_sum_global[j] / length,
                        "topk_global": torch.cat(topk_global[j], dim=0),
                        "botk_global": torch.cat(botk_global[j], dim=0),
                        "kl_div_global": kl_div_global[j],
                        "delta_gold_prob_global": delta_gold_prob_global[j],
                        "signed_mean_zero": delta_sum_zero[j] / length,
                        "abs_mean_zero": delta_abs_sum_zero[j] / length,
                        "topk_zero": torch.cat(topk_zero[j], dim=0),
                        "botk_zero": torch.cat(botk_zero[j], dim=0),
                        "kl_div_zero": kl_div_zero[j],
                        "delta_gold_prob_zero": delta_gold_prob_zero[j],
                        "signed_mean_rand": delta_sum_rand[j] / length,
                        "abs_mean_rand": delta_abs_sum_rand[j] / length,
                        "topk_rand": torch.cat(topk_rand[j], dim=0),
                        "botk_rand": torch.cat(botk_rand[j], dim=0),
                        "kl_div_rand": kl_div_rand[j],
                        "delta_gold_prob_rand": delta_gold_prob_rand[j],
                    }
                    for j, dir_idx in enumerate(directions_to_ablate)
                },
            },
            cat_dir / "delta_logits.pt",
        )

        anls_orig = sum(nls_original) / length
        anls_cat = [sum(nls) / length for nls in nls_ablated_cat]
        anls_global = [sum(nls) / length for nls in nls_ablated_global]
        anls_zero = [sum(nls) / length for nls in nls_ablated_zero]
        anls_rand = [sum(nls) / length for nls in nls_ablated_rand]

        save_json(
            cat_dir / "anls_summary.json",
            dict(
                category=name,
                n_samples=length,
                directions=directions_to_ablate,
                anls_original=anls_orig,
                anls_ablated_per_category_mean=anls_cat,
                anls_ablated_global_mean=anls_global,
                anls_ablated_zero_mean=anls_zero,
                anls_ablated_random=anls_rand,
            ),
        )

        print(
            f"\n{name}: orig={anls_orig:.4f}, "
            f"cat={[f'{a:.4f}' for a in anls_cat]}, "
            f"global={[f'{a:.4f}' for a in anls_global]}, "
            f"zero={[f'{a:.4f}' for a in anls_zero]}, "
            f"rand={[f'{a:.4f}' for a in anls_rand]}"
        )
