"""
Directional masking (Phase 1).

Learns a ``[0, 1]`` mask weight per SVD direction per DocVQA category.
Loss is ``KL(masked || original) + sparsity * L1(mask)``, optimized
via projected SGD (gradient step then clamp to ``[0, 1]``).

Example::

    >>> from conn_intrp import run_dm, load_docvqa, make_run_dir
    >>> from conn_intrp.models import SmolVLM2Adapter
    >>> adapter = SmolVLM2Adapter("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    >>> _, categorized = load_docvqa("dataset/docVQA/train_v1.0_withQT.json")
    >>> run_dir = make_run_dir("outputs", "smolvlm2", "dm")
    >>> run_dm(adapter, categorized, sparsity_coef=1.5e-3, lr=0.1,
    ...        epochs=3, step=5, image_base_path=Path("images"), run_dir=run_dir)

Main Functions:
    run_dm: Train per-category directional masks across all categories.
"""

import math
import torch
from pathlib import Path
from torch.nn import functional as F
from tqdm.auto import tqdm

from .config import DirectionalMaskingConfig, save_dm_run
from .models.base import ModelAdapter
from .output import fs_safe, save_json


def run_dm(
    adapter: ModelAdapter, data_categorized: dict[str, list], *,
    sparsity_coef: float, lr: float, epochs: int, step: int,
    image_base_path: Path, run_dir: Path,
) -> None:
    """
    Train per-category directional masks for all question categories.

    For each category, initialises a fresh mask (all 0.99), runs projected
    SGD for *epochs* passes over the data, and saves the learned mask
    plus per-epoch statistics.

    :param adapter: Model adapter instance.
    :type adapter: ModelAdapter
    :param data_categorized: Map of category name to list of data dicts.
    :type data_categorized: dict[str, list]
    :param sparsity_coef: L1 regularization coefficient.
    :type sparsity_coef: float
    :param lr: SGD learning rate.
    :type lr: float
    :param epochs: Number of training epochs per category.
    :type epochs: int
    :param step: Batch size (1 for InternVL, >1 for SmolVLM2).
    :type step: int
    :param image_base_path: Root directory for image files.
    :type image_base_path: Path
    :param run_dir: Output directory for this run.
    :type run_dir: Path
    """
    layers = adapter.svd_layers
    cache_vision = epochs > 1 or len(data_categorized) > 1

    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        save_json(meta_path, dict(
            model=adapter.model_name,
            layers=[l.name for l in layers],
            sparsity_coef=sparsity_coef,
            lr=lr,
            epochs=epochs,
            step=step,
            n_categories=len(data_categorized),
            category_sizes={n: len(d) for n, d in data_categorized.items()},
        ))

    completed = {
        name for name in data_categorized
        if all(
            (run_dir / f"mask_{l.name}_{fs_safe(name)}.pt").exists()
            for l in layers
        )
    }
    if completed:
        print(f"Resuming: skipping {len(completed)} completed categories")

    for name, data in tqdm(data_categorized.items(), desc="Learning masks"):
        if name in completed:
            print(f'  Skipping "{name}" (masks exist)')
            continue
        masks = {}
        optimizers = {}
        stats = {}
        for layer in layers:
            mask = torch.full(
                (layer.n_dirs,), 0.99, device=layer.S.device,
                requires_grad=True,
            )
            masks[layer.name] = mask
            optimizers[layer.name] = torch.optim.SGD([mask], lr=lr)
            stats[layer.name] = {
                "kl": [], "l1": [], "below_half": [], "near_zero": [],
            }

        length = len(data)
        epoch_cache = [] if epochs > 1 else None

        for ep in tqdm(range(epochs), desc=f'"{name}"'):
            epoch_kl = {ln: 0.0 for ln in masks}
            epoch_l1 = {ln: 0.0 for ln in masks}
            n_steps = 0

            for batch_idx, i in enumerate(tqdm(
                range(0, length, step),
                total=math.ceil(length / step),
                desc=f"  epoch {ep + 1}",
                leave=False,
            )):
                batch = data[i:i + step]
                image_keys = [d['image'] for d in batch]
                all_vision_cached = cache_vision and all(
                    k in adapter._vision_cache for k in image_keys
                )

                if (epoch_cache is not None
                        and ep > 0 and all_vision_cached):
                    cached = epoch_cache[batch_idx]
                    inputs = {
                        'input_ids': cached['input_ids'].cuda(),
                        'attention_mask': cached['attention_mask'].cuda(),
                    }
                    attention_mask = inputs['attention_mask']
                    p_original = cached['p_original'].cuda()
                    vision_out = torch.cat(
                        [adapter._vision_cache[k].cuda()
                         for k in image_keys]
                    )
                    with torch.no_grad():
                        text_embeds = adapter.get_text_embeds(inputs)
                else:
                    inputs = adapter.preprocess(batch, image_base_path)
                    attention_mask = inputs["attention_mask"]

                    with torch.no_grad():
                        if all_vision_cached:
                            vision_out = torch.cat(
                                [adapter._vision_cache[k].cuda()
                                 for k in image_keys]
                            )
                        else:
                            vision_out = adapter.extract_vision(inputs)
                            if cache_vision:
                                for j, k in enumerate(image_keys):
                                    if k not in adapter._vision_cache:
                                        adapter._vision_cache[k] = (
                                            vision_out[j:j + 1].cpu()
                                        )

                        conn_out_orig = adapter.run_connector(vision_out)
                        text_embeds = adapter.get_text_embeds(inputs)
                        embeds_orig = adapter.merge_embeds(
                            inputs, text_embeds, conn_out_orig
                        )
                        logits_orig = adapter.get_logits(
                            embeds_orig, attention_mask
                        )
                        p_original = F.softmax(logits_orig, dim=-1)
                        del conn_out_orig, embeds_orig

                    if epoch_cache is not None and ep == 0:
                        epoch_cache.append({
                            'input_ids': inputs['input_ids'].cpu(),
                            'attention_mask': attention_mask.cpu(),
                            'p_original': p_original.cpu(),
                        })

                for layer in layers:
                    ln = layer.name
                    mask = masks[ln]
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
                epochs=epochs,
                step=step,
                kl_per_epoch=s["kl"],
                l1_per_epoch=s["l1"],
                below_half_per_epoch=s["below_half"],
                near_zero_per_epoch=s["near_zero"],
            )
            save_dm_run(config, masks[ln])
