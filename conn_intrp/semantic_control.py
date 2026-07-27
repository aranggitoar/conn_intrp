"""
Semantic control experiment for the sub-semantic claim.

Injects known semantic perturbations (LLM input embeddings or embedding
differences) into the connector output at magnitude-matched scales, and
measures the delta-logit response using the same readout as ablation.

If semantic perturbations produce semantically coherent token changes
while ablation perturbations don't, the sub-semantic finding is a
property of the connector, not a limitation of the readout.

Example::

    >>> from conn_intrp.semantic_control import (
    ...     compute_reference_norms, run_semantic_control)
    >>> norms = compute_reference_norms(
    ...     adapter, coeff_dir, "Image_Photo")
    >>> run_semantic_control(
    ...     adapter, data, [("cat", "dog")], ["document"],
    ...     reference_norm=norms["median"], batch_size=4, K=15,
    ...     image_base_path=img_path, run_dir=out_dir)

Main Functions:
    compute_reference_norms: Typical per-direction ablation perturbation
        norms from stored SVD coefficients.
    run_semantic_control: Inject scaled semantic perturbations and
        measure delta-logit response.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .ablation import load_category_coefficients
from .data import best_anls
from .models.base import ModelAdapter
from .output import save_json, update_metadata


def compute_reference_norms(
    adapter: ModelAdapter,
    coefficients_dir: str | Path,
    category: str,
    directions: dict[str, list[int]] | None = None,
    max_dirs: int = 50,
) -> dict[str, float]:
    """
    Typical per-direction ablation perturbation norms from stored
    SVD coefficients.

    For direction *j*, the per-patch perturbation norm is
    ``|c_j(image, patch) - mean(c_j)|`` (since SVD left-singular
    vectors have unit norm).  Returns statistics across images,
    patches, and sampled directions.

    :param adapter: Model adapter instance.
    :param coefficients_dir: Directory with category coefficient
        checkpoints.
    :param category: Category name to load coefficients for.
    :param directions: Optional per-layer direction indices to sample.
        Defaults to the first *max_dirs* per layer.
    :param max_dirs: Number of directions to sample when *directions*
        is not given.
    :returns: Dict with keys ``median``, ``mean``, ``std``, ``q25``,
        ``q75``.
    """
    coefficients_dir = Path(coefficients_dir)
    cat_coefficients = load_category_coefficients(
        coefficients_dir, category, adapter.component_name
    )

    all_norms: list[float] = []
    for layer in adapter.svd_layers:
        coeffs = cat_coefficients[layer.name].float()
        mean_coeffs = coeffs.mean(dim=0)

        if directions and layer.name in directions:
            dirs = directions[layer.name]
        else:
            dirs = list(range(min(max_dirs, layer.n_dirs)))

        for d in dirs:
            dev = (coeffs[:, :, d] - mean_coeffs[:, d]).abs()
            all_norms.extend(dev.mean(dim=1).tolist())

    norms = np.array(all_norms)
    return {
        "median": float(np.median(norms)),
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "q25": float(np.percentile(norms, 25)),
        "q75": float(np.percentile(norms, 75)),
    }


def _get_semantic_vectors(
    adapter: ModelAdapter,
    word_pairs: list[tuple[str, str]],
    single_words: list[str] | None,
) -> dict[str, torch.Tensor]:
    """Build named semantic perturbation vectors from LLM embeddings."""
    embed_layer = adapter.model.model.text_model.get_input_embeddings()
    tokenizer = adapter.processor.tokenizer

    perturbations: dict[str, torch.Tensor] = {}

    if single_words:
        for word in single_words:
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            emb = embed_layer(
                torch.tensor([ids[-1]], device="cuda")
            ).squeeze(0).float()
            perturbations[f"word:{word}"] = emb

    if word_pairs:
        for w1, w2 in word_pairs:
            ids1 = tokenizer(w1, add_special_tokens=False)["input_ids"]
            ids2 = tokenizer(w2, add_special_tokens=False)["input_ids"]
            if not ids1 or not ids2:
                continue
            emb1 = embed_layer(
                torch.tensor([ids1[-1]], device="cuda")
            ).squeeze(0).float()
            emb2 = embed_layer(
                torch.tensor([ids2[-1]], device="cuda")
            ).squeeze(0).float()
            perturbations[f"diff:{w1}-{w2}"] = (emb1 - emb2)

    return perturbations


def run_semantic_control(
    adapter: ModelAdapter,
    data: list[dict],
    word_pairs: list[tuple[str, str]],
    single_words: list[str] | None = None,
    *,
    reference_norm: float,
    scales: list[float] | None = None,
    batch_size: int,
    K: int,
    image_base_path: Path,
    run_dir: Path,
) -> None:
    """
    Inject scaled semantic perturbations into the connector output
    and measure the delta-logit response.

    For each perturbation vector (word embeddings, embedding
    differences, and a random control), the vector is scaled so its
    norm equals ``reference_norm * scale`` and added uniformly to the
    connector output at all image-patch positions.  The same
    delta-logit readout as ablation (KL, delta gold log-prob,
    top-K/bottom-K token shifts) is then recorded.

    :param adapter: Model adapter instance.
    :param data: Flat list of data dicts with ``image``, ``question``,
        ``answers`` keys.
    :param word_pairs: Pairs of words whose embedding difference is
        used as a perturbation, e.g. ``[("cat", "dog")]``.
    :param single_words: Individual words whose embeddings are used
        as perturbations.
    :param reference_norm: Typical ablation perturbation norm (from
        :func:`compute_reference_norms`).
    :param scales: Multipliers on *reference_norm* to test.  Defaults
        to ``[1.0]``.
    :param batch_size: Images per forward pass.
    :param K: Number of top/bottom tokens to keep per image.
    :param image_base_path: Root directory for image files.
    :param run_dir: Output directory.
    """
    if scales is None:
        scales = [1.0]

    run_dir.mkdir(parents=True, exist_ok=True)

    raw_perturbations = _get_semantic_vectors(adapter, word_pairs, single_words)

    all_perturbations: dict[str, torch.Tensor] = {}
    for scale in scales:
        target_norm = reference_norm * scale
        suffix = f"@{scale}x"

        for name, vec in raw_perturbations.items():
            vec_norm = vec.norm().item()
            if vec_norm > 0:
                all_perturbations[name + suffix] = vec * (target_norm / vec_norm)

        rand_vec = torch.randn_like(next(iter(raw_perturbations.values())))
        rand_vec = rand_vec * (target_norm / rand_vec.norm().item())
        all_perturbations[f"random{suffix}"] = rand_vec

    labels = list(all_perturbations.keys())

    update_metadata(
        run_dir,
        dict(
            model=adapter.model_name,
            experiment="semantic_control",
            reference_norm=reference_norm,
            scales=scales,
            perturbations={
                name: {"norm": vec.norm().item()}
                for name, vec in all_perturbations.items()
            },
            batch_size=batch_size,
            n_samples=len(data),
        ),
    )

    acc: dict[str, dict] = {}
    for pl in labels:
        acc[pl] = {
            "kl": [],
            "delta_gold_prob": [],
            "delta_sum": torch.zeros(adapter.vocab_size),
            "delta_abs_sum": torch.zeros(adapter.vocab_size),
            "topk": [],
            "botk": [],
        }

    length = len(data)
    nls_original = []

    for i in tqdm(
        range(0, length, batch_size),
        total=math.ceil(length / batch_size),
        desc="Semantic control",
    ):
        batch = data[i : i + batch_size]
        actual = len(batch)
        targets_batch = [datum["answers"] for datum in batch]

        inputs = adapter.preprocess(batch, image_base_path)
        attention_mask = inputs["attention_mask"]
        text_embeds = adapter.get_text_embeds(inputs)

        with torch.no_grad():
            vision_out = adapter.extract_vision(inputs)
            conn_out_orig = adapter.run_connector(vision_out)
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
            ids = adapter.processor.tokenizer(
                targets[0], add_special_tokens=False
            )["input_ids"]
            gold_tok.append(ids[-1] if ids else pad_id)
        gold_idx = torch.tensor(gold_tok, device=logits_orig.device)
        lp_orig = log_probs_orig[range(actual), gold_idx]

        for pl in labels:
            pvec = all_perturbations[pl].to(
                dtype=conn_out_orig.dtype, device=conn_out_orig.device
            )

            with torch.no_grad():
                conn_out_pert = conn_out_orig + pvec
                embeds_pert = adapter.merge_embeds(
                    inputs, text_embeds, conn_out_pert
                )
                _, logits_pert = adapter.generate_with_logits(
                    embeds_pert, attention_mask
                )

            log_probs_pert = F.log_softmax(logits_pert, dim=-1)
            kl = F.kl_div(
                log_probs_pert, probs_orig, reduction="none"
            ).sum(-1)
            acc[pl]["kl"].extend(kl.cpu().tolist())

            lp_pert = log_probs_pert[range(actual), gold_idx]
            acc[pl]["delta_gold_prob"].extend(
                (lp_orig - lp_pert).cpu().tolist()
            )

            delta = (logits_orig - logits_pert).float()
            acc[pl]["delta_sum"] += delta.sum(dim=0).cpu()
            acc[pl]["delta_abs_sum"] += delta.abs().sum(dim=0).cpu()

            probs_pert = F.softmax(logits_pert, dim=-1)
            top_v, top_i = delta.topk(K, dim=-1)
            bot_v, bot_i = (-delta).topk(K, dim=-1)
            top_p_orig = probs_orig.gather(-1, top_i.long())
            bot_p_orig = probs_orig.gather(-1, bot_i.long())
            top_p_pert = probs_pert.gather(-1, top_i.long())
            bot_p_pert = probs_pert.gather(-1, bot_i.long())
            acc[pl]["topk"].append(
                torch.stack(
                    [top_i, top_v, top_p_orig, top_p_pert], dim=1
                ).cpu()
            )
            acc[pl]["botk"].append(
                torch.stack(
                    [bot_i, bot_v, bot_p_orig, bot_p_pert], dim=1
                ).cpu()
            )

    summary = {
        "n_samples": length,
        "anls_original": sum(nls_original) / length,
        "reference_norm": reference_norm,
        "scales": scales,
        "perturbations": {},
    }

    delta_logits = {}
    for pl in labels:
        a = acc[pl]
        summary["perturbations"][pl] = {
            "kl_mean": float(np.mean(a["kl"])),
            "kl_median": float(np.median(a["kl"])),
            "kl_values": a["kl"],
            "delta_gold_prob_mean": float(np.mean(a["delta_gold_prob"])),
            "delta_gold_prob_values": a["delta_gold_prob"],
        }
        delta_logits[pl] = {
            "signed_mean": a["delta_sum"] / length,
            "abs_mean": a["delta_abs_sum"] / length,
            "topk": torch.cat(a["topk"], dim=0),
            "botk": torch.cat(a["botk"], dim=0),
        }

    save_json(run_dir / "semantic_control_summary.json", summary)
    torch.save(delta_logits, run_dir / "semantic_control_delta_logits.pt")

    print(f"\nSemantic control (ref_norm={reference_norm:.4f}):")
    for pl in labels:
        s = summary["perturbations"][pl]
        print(
            f"  {pl}: KL={s['kl_mean']:.6f}, "
            f"delta_gold_lp={s['delta_gold_prob_mean']:.6f}"
        )
