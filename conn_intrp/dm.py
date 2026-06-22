"""
Directional masking (Phase 1).

Learns a ``[0, 1]`` mask weight per SVD direction per DocVQA category.
Loss is ``KL(masked || original) + sparsity * L1(mask)``, optimized
via projected SGD (gradient step then clamp to ``[0, 1]``).

Batch size is resolved per category so that all categories receive a
similar number of gradient updates per epoch (see ``_resolve_step``).
Training stops early when the ``<0.5`` mask count stabilises across
all layers for ``patience`` consecutive epochs.

Example::

    >>> from conn_intrp import run_dm, load_docvqa, make_run_dir
    >>> from conn_intrp.models import InternVLAdapter
    >>> adapter = InternVLAdapter("OpenGVLab/InternVL3_5-2B-HF")
    >>> _, categorized = load_docvqa("dataset/docVQA/train_v1.0_withQT.json")
    >>> run_dir = make_run_dir("outputs", "internvl3_5", "dm")
    >>> run_dm(adapter, categorized, sparsity_coef=5e-3, lr=1.0,
    ...        target_updates_per_epoch=300, max_step=50,
    ...        image_base_path=Path("dataset/docVQA"), run_dir=run_dir)

Main Functions:
    run_dm: Train per-category directional masks across all categories.
    evaluate_mask_kl: Measure KL divergence of a given mask without training.
    evaluate_dm_baselines: Optimized vs random mask KL across thresholds.
"""

import math
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .config import DirectionalMaskingConfig, save_dm_run
from .models.base import ModelAdapter
from .output import fs_safe, save_json, update_metadata


def _resolve_step(
    n_images: int,
    step: int | None,
    target_updates_per_epoch: int | None,
    max_step: int | None,
) -> int:
    if step is not None:
        return step
    if target_updates_per_epoch is not None:
        s = max(1, round(n_images / target_updates_per_epoch))
        if max_step is not None:
            s = min(s, max_step)
        return s
    return 1


def _check_converged(
    stats: dict,
    n_dirs: int,
    patience: int,
    conv_threshold: float,
) -> bool:
    """All layers' <0.5 and near-zero counts stable for *patience* epochs."""
    threshold = max(1, int(n_dirs * conv_threshold))
    for ln, s in stats.items():
        for key in ("below_half", "near_zero"):
            seq = s[key]
            if len(seq) < patience + 1:
                return False
            for i in range(-patience, 0):
                if abs(seq[i] - seq[i - 1]) > threshold:
                    return False
    return True


def evaluate_mask_kl(
    adapter: ModelAdapter,
    data: list,
    mask_weights: dict[str, torch.Tensor],
    *,
    step: int = 5,
    image_base_path: Path,
) -> dict[str, float]:
    """
    Measure average KL divergence for a given set of mask weights.

    Runs one pass over *data* (no gradient, no training) and returns
    the mean KL(masked || original) per layer.  Use to compare an
    optimized mask against a random baseline.

    :param adapter: Model adapter instance.
    :type adapter: ModelAdapter
    :param data: List of data dicts (single category).
    :type data: list
    :param mask_weights: ``{layer_name: mask_tensor}`` with values
        in ``[0, 1]``, shape ``(n_dirs,)``.
    :type mask_weights: dict[str, torch.Tensor]
    :param step: Batch size.
    :type step: int
    :param image_base_path: Root directory for image files.
    :type image_base_path: Path
    :returns: ``{layer_name: mean_kl}``
    :rtype: dict[str, float]
    """
    results = evaluate_masks_kl(
        adapter, data, [mask_weights],
        step=step, image_base_path=image_base_path,
    )
    return results[0]


def evaluate_masks_kl(
    adapter: ModelAdapter,
    data: list,
    mask_sets: list[dict[str, torch.Tensor]],
    *,
    step: int = 5,
    image_base_path: Path,
    desc: str = "Evaluating mask KL",
) -> list[dict[str, float]]:
    """
    Evaluate multiple mask sets in a single data pass.

    The expensive forward pass (vision encoder, original logits) is
    computed once per batch; each mask set's KL is then evaluated with
    only the cheap masked-connector forward.

    :param adapter: Model adapter instance.
    :type adapter: ModelAdapter
    :param data: List of data dicts (single category).
    :type data: list
    :param mask_sets: List of ``{layer_name: mask_tensor}`` dicts.
    :type mask_sets: list[dict[str, torch.Tensor]]
    :param step: Batch size.
    :type step: int
    :param image_base_path: Root directory for image files.
    :type image_base_path: Path
    :param desc: Progress bar description.
    :type desc: str
    :returns: One ``{layer_name: mean_kl}`` dict per mask set.
    :rtype: list[dict[str, float]]
    """
    layers = adapter.svd_layers
    length = len(data)
    n_sets = len(mask_sets)

    all_layer_names = set()
    for ms in mask_sets:
        all_layer_names.update(ms.keys())

    kl_sums = [{ln: 0.0 for ln in ms} for ms in mask_sets]
    n_steps = 0

    for i in tqdm(
        range(0, length, step),
        total=math.ceil(length / step),
        desc=desc,
        leave=False,
    ):
        batch = data[i : i + step]
        inputs = adapter.preprocess(batch, image_base_path)
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            vision_out = adapter.extract_vision(inputs)
            conn_out_orig = adapter.run_connector(vision_out)
            text_embeds = adapter.get_text_embeds(inputs)
            embeds_orig = adapter.merge_embeds(inputs, text_embeds, conn_out_orig)
            logits_orig = adapter.get_logits(embeds_orig, attention_mask)
            p_original = F.softmax(logits_orig, dim=-1)
            del conn_out_orig, embeds_orig

            for layer in layers:
                ln = layer.name
                if ln not in all_layer_names:
                    continue

                for si in range(n_sets):
                    if ln not in mask_sets[si]:
                        continue
                    mask = mask_sets[si][ln].to(device=layer.S.device)
                    S_masked = layer.S * mask
                    W_masked = layer.U @ torch.diag(S_masked) @ layer.Vt
                    conn_out_masked = adapter.run_connector_layer_masked(
                        vision_out, ln, W_masked
                    )

                    with torch.autocast(
                        device_type="cuda", dtype=adapter.compute_dtype
                    ):
                        embeds_masked = adapter.merge_embeds(
                            inputs, text_embeds, conn_out_masked
                        )
                        logits_masked = adapter.get_logits(
                            embeds_masked, attention_mask
                        )
                    p_masked = F.log_softmax(logits_masked.float(), dim=-1)

                    kl = F.kl_div(
                        p_masked, p_original.float(), reduction="batchmean"
                    )
                    kl_sums[si][ln] += kl.item()

        del vision_out
        torch.cuda.empty_cache()
        n_steps += 1

    return [
        {ln: kl_sums[si][ln] / max(n_steps, 1) for ln in mask_sets[si]}
        for si in range(n_sets)
    ]


def evaluate_dm_baselines(
    adapter: ModelAdapter,
    data_categorized: dict[str, list],
    *,
    step: int = 5,
    thresholds: list[float] | None = None,
    n_random_seeds: int = 3,
    image_base_path: Path,
    run_dir: Path,
) -> dict:
    """
    Evaluate optimized vs random mask KL across thresholds.

    For each category, runs a single pass over the data and evaluates
    all mask variants (continuous optimized, binarised at each threshold,
    and random baselines) using cached forward passes.  Results are
    saved incrementally to ``baseline_kl.json`` in *run_dir*.

    :param adapter: Model adapter instance.
    :type adapter: ModelAdapter
    :param data_categorized: Map of category name to list of data dicts.
    :type data_categorized: dict[str, list]
    :param step: Batch size for KL evaluation.
    :type step: int
    :param thresholds: Binarisation thresholds to sweep.  Defaults to
        ``[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]``.
    :type thresholds: list[float] | None
    :param n_random_seeds: Number of random seeds to average over.
    :type n_random_seeds: int
    :param image_base_path: Root directory for image files.
    :type image_base_path: Path
    :param run_dir: DM run output directory (must contain mask ``.pt`` files).
    :type run_dir: Path
    :returns: Full results dict (also saved to ``baseline_kl.json``).
    :rtype: dict
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    from .dm_analysis import load_dm_masks
    masks = load_dm_masks(run_dir)
    layer_names = list(masks.keys())

    results_path = run_dir / "baseline_kl.json"
    results = {}
    if results_path.exists():
        import json
        with open(results_path) as f:
            results = json.load(f)

    for name, data in tqdm(data_categorized.items(), desc="Baseline evaluation"):
        cat_key = fs_safe(name)
        if cat_key in results:
            print(f'  Skipping "{name}" (already evaluated)')
            continue

        opt_masks = {}
        for ln in layer_names:
            if cat_key in masks[ln]:
                opt_masks[ln] = masks[ln][cat_key]

        mask_sets = [opt_masks]
        layout = [("continuous", None, None)]

        for thr in thresholds:
            bin_masks = {}
            for ln, m in opt_masks.items():
                binary = torch.where(
                    m > thr, torch.ones_like(m), torch.zeros_like(m)
                )
                bin_masks[ln] = binary

            mask_sets.append(bin_masks)
            layout.append(("optimized", thr, None))

            for seed in range(n_random_seeds):
                rand_masks = {}
                for ln, m in opt_masks.items():
                    n_surv = int((m > thr).sum().item())
                    n_dirs = m.numel()
                    gen = torch.Generator().manual_seed(seed)
                    rm = torch.zeros(n_dirs)
                    if n_surv > 0:
                        idx = torch.randperm(n_dirs, generator=gen)[:n_surv]
                        rm[idx] = 1.0
                    rand_masks[ln] = rm

                mask_sets.append(rand_masks)
                layout.append(("random", thr, seed))

        n_mask_sets = len(mask_sets)
        print(
            f'  "{name}": {len(data)} images, '
            f"{n_mask_sets} mask variants ({len(thresholds)} thresholds × "
            f"{n_random_seeds} seeds + continuous)"
        )

        all_kls = evaluate_masks_kl(
            adapter, data, mask_sets,
            step=step, image_base_path=image_base_path,
            desc=f'  "{name}"',
        )

        cat_results = {
            "continuous": {"optimized_kl": all_kls[0]},
        }

        for idx, (kind, thr, seed) in enumerate(layout):
            if kind == "continuous":
                continue
            thr_key = f"{thr:.2f}"
            if thr_key not in cat_results:
                survivor_counts = {}
                for ln, m in opt_masks.items():
                    survivor_counts[ln] = int((m > thr).sum().item())
                cat_results[thr_key] = {
                    "survivors": survivor_counts,
                    "optimized_kl": None,
                    "random_kl_per_seed": {ln: [] for ln in opt_masks},
                }

            entry = cat_results[thr_key]
            if kind == "optimized":
                entry["optimized_kl"] = all_kls[idx]
            elif kind == "random":
                for ln in opt_masks:
                    entry["random_kl_per_seed"][ln].append(all_kls[idx][ln])

        for thr_key, entry in cat_results.items():
            if thr_key == "continuous":
                continue
            entry["random_kl_mean"] = {
                ln: sum(v) / len(v)
                for ln, v in entry["random_kl_per_seed"].items()
            }

        for thr in thresholds:
            thr_key = f"{thr:.2f}"
            entry = cat_results[thr_key]
            for ln in opt_masks:
                opt_v = entry["optimized_kl"][ln]
                rnd_v = entry["random_kl_mean"][ln]
                n_s = entry["survivors"][ln]
                print(
                    f'  {name}/{ln} thr={thr:.1f}: '
                    f'{n_s} dirs, '
                    f'opt_kl={opt_v:.6f}, '
                    f'rand_kl={rnd_v:.6f}'
                )

        results[cat_key] = cat_results
        save_json(results_path, results)

    return results


def run_dm(
    adapter: ModelAdapter,
    data_categorized: dict[str, list],
    *,
    sparsity_coef: float,
    lr: float,
    epochs: int = 10,
    step: int | None = None,
    target_updates_per_epoch: int | None = None,
    max_step: int | None = None,
    patience: int = 2,
    conv_threshold: float = 0.02,
    image_base_path: Path,
    run_dir: Path,
) -> None:
    """
    Train per-category directional masks for all question categories.

    For each category, initialises a fresh mask (all 0.99), runs projected
    SGD for up to *epochs* passes over the data (with early stopping),
    and saves the learned mask plus per-epoch statistics.

    Batch size per category is resolved as follows:

    - If *step* is given, it is used for all categories.
    - If *target_updates_per_epoch* is given, step is derived per category
      so each gets approximately that many gradient updates per epoch.
    - *max_step* caps the auto-computed step (GPU memory).

    :param adapter: Model adapter instance.
    :type adapter: ModelAdapter
    :param data_categorized: Map of category name to list of data dicts.
    :type data_categorized: dict[str, list]
    :param sparsity_coef: L1 regularization coefficient.
    :type sparsity_coef: float
    :param lr: SGD learning rate.
    :type lr: float
    :param epochs: Maximum training epochs per category.
    :type epochs: int
    :param step: Fixed batch size for all categories.  Mutually exclusive
        with *target_updates_per_epoch*; one of the two must be provided.
    :type step: int | None
    :param target_updates_per_epoch: Desired gradient updates per epoch;
        step is derived per category from ``round(n_images / target)``.
    :type target_updates_per_epoch: int | None
    :param max_step: Upper bound on the auto-computed step.
    :type max_step: int | None
    :param patience: Stop early when both the ``<0.5`` and near-zero
        (``<0.05``) counts change by less than *conv_threshold* of
        *n_dirs* for this many consecutive epochs.  Set to 0 to disable
        early stopping.
    :type patience: int
    :param conv_threshold: Fraction of *n_dirs* used as the convergence
        threshold for early stopping.
    :type conv_threshold: float
    :param image_base_path: Root directory for image files.
    :type image_base_path: Path
    :param run_dir: Output directory for this run.
    :type run_dir: Path
    """
    if step is None and target_updates_per_epoch is None:
        raise ValueError("Provide either step or target_updates_per_epoch")

    layers = adapter.svd_layers
    cache_vision = epochs > 1 or len(data_categorized) > 1

    update_metadata(
        run_dir,
        dict(
            model=adapter.model_name,
            layers=[layer.name for layer in layers],
            sparsity_coef=sparsity_coef,
            lr=lr,
            epochs=epochs,
            step=step,
            target_updates_per_epoch=target_updates_per_epoch,
            max_step=max_step,
            n_categories=len(data_categorized),
            category_sizes={n: len(d) for n, d in data_categorized.items()},
        ),
    )

    completed = {
        name
        for name in data_categorized
        if all((run_dir / f"mask_{layer.name}_{fs_safe(name)}.pt").exists() for layer in layers)
    }
    if completed:
        print(f"Resuming: skipping {len(completed)} completed categories")

    for name, data in tqdm(data_categorized.items(), desc="Learning masks"):
        if name in completed:
            print(f'  Skipping "{name}" (masks exist)')
            continue

        cat_step = _resolve_step(
            len(data),
            step,
            target_updates_per_epoch,
            max_step,
        )
        n_updates_ep = math.ceil(len(data) / cat_step)
        print(f'  "{name}": {len(data)} images, ' f"step={cat_step}, ~{n_updates_ep} updates/epoch")

        masks = {}
        optimizers = {}
        stats = {}
        for layer in layers:
            mask = torch.full(
                (layer.n_dirs,),
                0.99,
                device=layer.S.device,
                requires_grad=True,
            )
            masks[layer.name] = mask
            optimizers[layer.name] = torch.optim.SGD([mask], lr=lr)
            stats[layer.name] = {
                "kl": [],
                "l1": [],
                "below_half": [],
                "near_zero": [],
            }

        length = len(data)
        epoch_cache = [] if epochs > 1 else None

        for ep in tqdm(range(epochs), desc=f'"{name}"'):
            epoch_kl = {ln: 0.0 for ln in masks}
            epoch_l1 = {ln: 0.0 for ln in masks}
            n_steps = 0

            for batch_idx, i in enumerate(
                tqdm(
                    range(0, length, cat_step),
                    total=math.ceil(length / cat_step),
                    desc=f"  epoch {ep + 1}",
                    leave=False,
                )
            ):
                batch = data[i : i + cat_step]
                image_keys = [d["image"] for d in batch]
                all_vision_cached = cache_vision and all(
                    k in adapter._vision_cache for k in image_keys
                )

                if epoch_cache is not None and ep > 0 and all_vision_cached:
                    cached = epoch_cache[batch_idx]
                    inputs = {
                        "input_ids": cached["input_ids"].cuda(),
                        "attention_mask": cached["attention_mask"].cuda(),
                    }
                    attention_mask = inputs["attention_mask"]
                    p_original = cached["p_original"].cuda()
                    vision_out = torch.cat([adapter._vision_cache[k].cuda() for k in image_keys])
                    with torch.no_grad():
                        text_embeds = adapter.get_text_embeds(inputs)
                else:
                    inputs = adapter.preprocess(batch, image_base_path)
                    attention_mask = inputs["attention_mask"]

                    with torch.no_grad():
                        if all_vision_cached:
                            vision_out = torch.cat(
                                [adapter._vision_cache[k].cuda() for k in image_keys]
                            )
                        else:
                            vision_out = adapter.extract_vision(inputs)
                            if cache_vision:
                                for j, k in enumerate(image_keys):
                                    if k not in adapter._vision_cache:
                                        adapter._vision_cache[k] = vision_out[j : j + 1].cpu()

                        conn_out_orig = adapter.run_connector(vision_out)
                        text_embeds = adapter.get_text_embeds(inputs)
                        embeds_orig = adapter.merge_embeds(inputs, text_embeds, conn_out_orig)
                        logits_orig = adapter.get_logits(embeds_orig, attention_mask)
                        p_original = F.softmax(logits_orig, dim=-1)
                        del conn_out_orig, embeds_orig

                    if epoch_cache is not None and ep == 0:
                        epoch_cache.append(
                            {
                                "input_ids": inputs["input_ids"].cpu(),
                                "attention_mask": attention_mask.cpu(),
                                "p_original": p_original.cpu(),
                            }
                        )

                for layer in layers:
                    ln = layer.name
                    mask = masks[ln]
                    S_masked = layer.S * mask
                    W_masked = layer.U @ torch.diag(S_masked) @ layer.Vt
                    conn_out_masked = adapter.run_connector_layer_masked(vision_out, ln, W_masked)

                    with torch.autocast(device_type="cuda", dtype=adapter.compute_dtype):
                        embeds_masked = adapter.merge_embeds(inputs, text_embeds, conn_out_masked)
                        logits_masked = adapter.get_logits(embeds_masked, attention_mask)
                    p_masked = F.log_softmax(logits_masked.float(), dim=-1)

                    kl = F.kl_div(p_masked, p_original.float(), reduction="batchmean")
                    l1 = sparsity_coef * mask.sum()
                    loss = kl + l1

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([mask], max_norm=1.0)
                    optimizers[ln].step()
                    with torch.no_grad():
                        mask.clamp_(0, 1)
                    optimizers[ln].zero_grad()

                    epoch_kl[ln] += kl.item()
                    epoch_l1[ln] += l1.item()

                del vision_out
                torch.cuda.empty_cache()
                n_steps += 1

            for layer in layers:
                ln = layer.name
                m = masks[ln].data
                avg_kl = epoch_kl[ln] / max(n_steps, 1)
                avg_l1 = epoch_l1[ln] / max(n_steps, 1)
                stats[ln]["kl"].append(avg_kl)
                stats[ln]["l1"].append(avg_l1)
                stats[ln]["below_half"].append((m < 0.5).sum().item())
                stats[ln]["near_zero"].append((m < 0.05).sum().item())

                print(
                    f"  {name}/{ln} ep{ep + 1}: kl={avg_kl:.4f} "
                    f"l1={avg_l1:.4f} "
                    f"<0.5={stats[ln]['below_half'][-1]} "
                    f"<0.05={stats[ln]['near_zero'][-1]}"
                )

            n_dirs = layers[0].n_dirs
            if patience and _check_converged(stats, n_dirs, patience, conv_threshold):
                print(f'  "{name}" converged at epoch {ep + 1} ' f"(stable for {patience} epochs)")
                break

        actual_epochs = ep + 1
        del epoch_cache

        for layer in layers:
            ln = layer.name
            mask_path = run_dir / f"mask_{ln}_{fs_safe(name)}.pt"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(masks[ln].data, mask_path)

        for layer in layers:
            ln = layer.name
            s = stats[ln]
            config = DirectionalMaskingConfig(
                category=name,
                model=adapter.model_name,
                component=ln,
                optimizer="SGD",
                sparsity_coef=sparsity_coef,
                lr=lr,
                epochs=actual_epochs,
                step=cat_step,
                kl_per_epoch=s["kl"],
                l1_per_epoch=s["l1"],
                below_half_per_epoch=s["below_half"],
                near_zero_per_epoch=s["near_zero"],
            )
            save_dm_run(config, masks[ln])
