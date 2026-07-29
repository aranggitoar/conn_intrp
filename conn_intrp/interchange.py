"""
Image interchange experiment for connector interpretability.

Swaps connector outputs between paired images that differ in one
semantic attribute (e.g. sitting vs standing) and measures whether the
model's response changes to match the swapped image.

Positive control for the sub-semantic claim: if swapping connector
outputs causes the model to answer according to the new image, the
connector encodes that semantic distinction at a scale the LLM reads.

Example::

    >>> from conn_intrp.interchange import (
    ...     load_image_pairs, prepare_interchange_items,
    ...     run_interchange)
    >>> pairs = download_pairs(...)  # from check_download.py
    >>> items = prepare_interchange_items(pairs, "verb")
    >>> run_interchange(adapter, items, batch_size=4,
    ...     image_base_path=Path("."), run_dir=Path("outputs/interchange"))

Main Functions:
    load_image_pairs: Load downloaded image pairs from disk.
    prepare_interchange_items: Generate interchange items with
        cloze prompts from SVO triplet pairs.
    run_interchange: Swap connector outputs and measure response.
    coefficient_diagnostic: Compare SVD coefficient magnitudes
        between interchange images and reference dataset images.
"""

import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .models.base import ModelAdapter
from .output import save_json, update_metadata


_GERUND = {
    "carry": "carrying",
    "catch": "catching",
    "cross": "crossing",
    "float": "floating",
    "hold": "holding",
    "jog": "jogging",
    "jump": "jumping",
    "lay": "laying",
    "lean": "leaning",
    "lie": "lying",
    "look": "looking",
    "pass": "passing",
    "perform": "performing",
    "play": "playing",
    "rest": "resting",
    "rise": "rising",
    "run": "running",
    "sit": "sitting",
    "stand": "standing",
    "sweep": "sweeping",
    "swim": "swimming",
    "take": "taking",
    "trek": "trekking",
    "walk": "walking",
}


def _to_gerund(verb: str) -> str:
    if verb in _GERUND:
        return _GERUND[verb]
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e"):
        return verb[:-1] + "ing"
    if (
        len(verb) >= 3
        and verb[-1] not in "aeiouwxy"
        and verb[-2] in "aeiou"
        and verb[-3] not in "aeiou"
    ):
        return verb + verb[-1] + "ing"
    return verb + "ing"


def _make_prompt(
    pos_triplet: list[str], neg_triplet: list[str], swap_type: str
) -> tuple[str, str, str]:
    """Generate a cloze prompt and expected completions.

    Neither target word appears in the prompt, so any preference
    between them must come from the visual representation.

    :returns: ``(prompt, pos_answer, neg_answer)``
    """
    s, v, o = [x.strip() for x in pos_triplet]
    s2, v2, o2 = [x.strip() for x in neg_triplet]

    if swap_type == "verb":
        return "In this image, the action being performed is", _to_gerund(v), _to_gerund(v2)
    elif swap_type == "subject":
        return "The main subject in this image is a", s, s2
    else:
        return "The setting of this image is", o, o2


def load_image_pairs(
    download_dir: str | Path,
) -> dict[str, list[tuple[str, str]]]:
    """Load image pairs from the download directory structure.

    Expects: ``download_dir/<swap_type>/<a>_vs_<b>/<id>_pos.jpg``

    :param download_dir: Root of the downloaded images.
    :returns: Dict mapping labels (e.g. ``"verb:sit_vs_stand"``) to
        lists of ``(pos_path, neg_path)`` tuples with absolute paths.
    """
    download_dir = Path(download_dir).resolve()
    pair_groups: dict[str, list[tuple[str, str]]] = {}

    for swap_type_dir in sorted(download_dir.iterdir()):
        if not swap_type_dir.is_dir():
            continue
        swap_type = swap_type_dir.name

        for pair_dir in sorted(swap_type_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            label = f"{swap_type}:{pair_dir.name}"

            pos_files = sorted(pair_dir.glob("*_pos.jpg"))
            pairs = []
            for pos_file in pos_files:
                pair_id = pos_file.stem.replace("_pos", "")
                neg_file = pair_dir / f"{pair_id}_neg.jpg"
                if neg_file.exists():
                    pairs.append((str(pos_file), str(neg_file)))

            if pairs:
                pair_groups[label] = pairs

    return pair_groups


def prepare_interchange_items(
    csv_path: str | Path,
    image_dir: str | Path,
    swap_type: str,
    pairs: list[tuple[str, str]],
) -> list[dict]:
    """Build interchange items from the SVO probes CSV and downloaded
    images.

    Reads triplet metadata from the CSV, filters for the given swap
    type and attribute pairs, and matches against images on disk.
    Each matched pair produces two items (forward and reverse swap).

    Image directory structure expected::

        image_dir/<swap_type>/<a>_vs_<b>/<pos_image_id>_pos.jpg
        image_dir/<swap_type>/<a>_vs_<b>/<pos_image_id>_neg.jpg

    :param csv_path: Path to ``svo_probes.csv``.
    :param image_dir: Root of the downloaded images.
    :param swap_type: One of ``"subject"``, ``"verb"``, ``"object"``.
    :param pairs: Attribute pairs to include, e.g.
        ``[("sit", "stand"), ("run", "walk")]``.
    :returns: List of interchange item dicts ready for
        :func:`run_interchange`.
    """
    import csv

    col = {"subject": "subj_neg", "verb": "verb_neg", "object": "obj_neg"}[
        swap_type
    ]
    idx = {"subject": 0, "verb": 1, "object": 2}[swap_type]
    pair_set = {tuple(sorted(p)) for p in pairs}
    image_dir = Path(image_dir).resolve()

    items: list[dict] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row[col] != "True":
                continue

            pos_trip = [x.strip() for x in row["pos_triplet"].split(",")]
            neg_raw = row["neg_triplet"].strip("[] '\"")
            neg_trip = [x.strip() for x in neg_raw.split(",")]
            if len(pos_trip) != 3 or len(neg_trip) != 3:
                continue

            pair = tuple(sorted([pos_trip[idx], neg_trip[idx]]))
            if pair not in pair_set:
                continue

            a, b = pair
            pair_id = row["pos_image_id"]
            pair_dir = image_dir / swap_type / f"{a}_vs_{b}"
            pos_path = pair_dir / f"{pair_id}_pos.jpg"
            neg_path = pair_dir / f"{pair_id}_neg.jpg"

            if not pos_path.exists() or not neg_path.exists():
                continue

            prompt, pos_answer, neg_answer = _make_prompt(
                pos_trip, neg_trip, swap_type
            )
            label = f"{swap_type}:{a}_vs_{b}"

            items.append(
                {
                    "base_image": str(pos_path),
                    "swap_image": str(neg_path),
                    "question": prompt,
                    "base_answer": pos_answer,
                    "swap_answer": neg_answer,
                    "direction": "pos_to_neg",
                    "label": label,
                    "pair_id": pair_id,
                }
            )

            items.append(
                {
                    "base_image": str(neg_path),
                    "swap_image": str(pos_path),
                    "question": prompt,
                    "base_answer": neg_answer,
                    "swap_answer": pos_answer,
                    "direction": "neg_to_pos",
                    "label": label,
                    "pair_id": pair_id,
                }
            )

    print(f"Loaded {len(items)} interchange items ({len(items)//2} pairs)")
    by_label = defaultdict(int)
    for item in items:
        by_label[item["label"]] += 1
    for label, count in sorted(by_label.items()):
        print(f"  {label}: {count} items ({count//2} pairs)")

    return items


def _score_condition(
    batch, tokenizer, preds, log_probs, probs_orig, arm,
):
    """Score one generation condition, return list of result dicts."""
    results = []
    kl = F.kl_div(
        log_probs, probs_orig, reduction="none"
    ).sum(-1)

    for j in range(len(batch)):
        item = batch[j]
        ids_base_ans = tokenizer(
            item["base_answer"], add_special_tokens=False
        )["input_ids"]
        ids_swap_ans = tokenizer(
            item["swap_answer"], add_special_tokens=False
        )["input_ids"]

        lp = {}
        if ids_base_ans and ids_swap_ans:
            ba, sa = ids_base_ans[0], ids_swap_ans[0]
            lp["base_under_swap"] = log_probs[j, ba].item()
            lp["swap_under_swap"] = log_probs[j, sa].item()

        swap_prefers_swap = (
            lp.get("swap_under_swap", 0)
            > lp.get("base_under_swap", 0)
        )

        pred_s = preds[j].lower()
        sa_re = re.compile(
            r"\b" + re.escape(item["swap_answer"].lower()) + r"\b"
        )
        swap_has_swap = bool(sa_re.search(pred_s))

        results.append({
            "arm": arm,
            "kl": kl[j].item(),
            "logprobs_swap": lp,
            "swap_prefers_swap": swap_prefers_swap,
            "pred_swap": preds[j],
            "swap_text_has_swap": swap_has_swap,
        })
    return results


def coefficient_diagnostic(
    adapter: ModelAdapter,
    interchange_items: list[dict],
    reference_data: list[dict],
    *,
    batch_size: int,
    image_base_path: str | Path,
    directions: list[int],
    n_samples: int = 200,
) -> dict:
    """Compare SVD coefficient magnitudes on interchange images vs
    reference dataset images.

    Projects both sets of images onto the last SVD layer's directions
    and reports per-direction mean absolute coefficient for each set.
    If the DM-identified directions have comparable magnitudes on
    interchange images and reference images, the partial-arm experiment
    is valid despite using a different image domain.

    :param adapter: Model adapter instance.
    :param interchange_items: Output of :func:`prepare_interchange_items`.
    :param reference_data: Data dicts from the reference dataset
        (DocVQA or OKVQA), same format as adapter.preprocess expects.
    :param batch_size: Images per forward pass.
    :param image_base_path: Root for image files.
    :param directions: DM-identified direction indices to focus on.
    :param n_samples: Max images to sample from each set.
    :returns: Dict with per-direction stats and summary.
    """
    image_base_path = Path(image_base_path)
    last_layer = adapter.svd_layers[-1]
    U = last_layer.U.float()
    bias = last_layer.bias

    def _collect_coefficients(data_items, desc):
        all_coeffs = []
        skipped = 0
        for i in tqdm(range(0, len(data_items), batch_size),
                      desc=desc):
            batch = data_items[i : i + batch_size]
            try:
                inputs = adapter.preprocess(batch, image_base_path)
                with torch.no_grad():
                    vision = adapter.extract_vision(inputs)
                    conn_out = adapter.run_connector(vision).float()
                if bias is not None:
                    conn_out = conn_out - bias.float()
                coeffs = conn_out @ U
                all_coeffs.append(coeffs.abs().mean(dim=1).cpu())
            except Exception:
                for item in batch:
                    try:
                        inputs = adapter.preprocess([item], image_base_path)
                        with torch.no_grad():
                            vision = adapter.extract_vision(inputs)
                            conn_out = adapter.run_connector(vision).float()
                        if bias is not None:
                            conn_out = conn_out - bias.float()
                        coeffs = conn_out @ U
                        all_coeffs.append(coeffs.abs().mean(dim=1).cpu())
                    except Exception:
                        skipped += 1
        if skipped:
            print(f"  Skipped {skipped} corrupt/unreadable images")
        return torch.cat(all_coeffs, dim=0)

    seen = set()
    ic_data = []
    for item in interchange_items:
        img = item["base_image"]
        if img not in seen and len(ic_data) < n_samples:
            seen.add(img)
            ic_data.append({
                "image": img,
                "question": item["question"],
                "answers": [item["base_answer"]],
            })

    import random
    ref_sample = random.sample(
        reference_data, min(n_samples, len(reference_data))
    )

    ic_coeffs = _collect_coefficients(ic_data, "Interchange images")
    ref_coeffs = _collect_coefficients(ref_sample, "Reference images")

    dirs = sorted(directions)
    all_dirs = list(range(U.shape[1]))
    other_dirs = sorted(set(all_dirs) - set(dirs))

    ic_dm = ic_coeffs[:, dirs].mean(dim=0).numpy()
    ref_dm = ref_coeffs[:, dirs].mean(dim=0).numpy()
    ic_other = ic_coeffs[:, other_dirs].mean(dim=0).numpy()
    ref_other = ref_coeffs[:, other_dirs].mean(dim=0).numpy()

    result = {
        "n_interchange": len(ic_data),
        "n_reference": len(ref_sample),
        "n_dm_directions": len(dirs),
        "dm_directions": {
            "interchange_mean": float(ic_dm.mean()),
            "interchange_std": float(ic_dm.std()),
            "reference_mean": float(ref_dm.mean()),
            "reference_std": float(ref_dm.std()),
            "ratio": float(ic_dm.mean() / ref_dm.mean())
            if ref_dm.mean() > 0
            else None,
        },
        "other_directions": {
            "interchange_mean": float(ic_other.mean()),
            "reference_mean": float(ref_other.mean()),
            "ratio": float(ic_other.mean() / ref_other.mean())
            if ref_other.mean() > 0
            else None,
        },
    }

    print(f"\nCoefficient diagnostic ({len(ic_data)} interchange, "
          f"{len(ref_sample)} reference images):")
    d = result["dm_directions"]
    o = result["other_directions"]
    print(f"  DM directions ({len(dirs)}):")
    print(f"    interchange: {d['interchange_mean']:.4f} ± {d['interchange_std']:.4f}")
    print(f"    reference:   {d['reference_mean']:.4f} ± {d['reference_std']:.4f}")
    print(f"    ratio:       {d['ratio']:.2f}")
    print(f"  Other directions ({len(other_dirs)}):")
    print(f"    interchange: {o['interchange_mean']:.4f}")
    print(f"    reference:   {o['reference_mean']:.4f}")
    print(f"    ratio:       {o['ratio']:.2f}")

    return result


def _partial_swap(conn_out_base, conn_out_swap, directions, svd_layer):
    """Swap specific SVD direction coefficients between two connector outputs."""
    U = svd_layer.U.float()
    bias = svd_layer.bias
    base_f = conn_out_base.float()
    swap_f = conn_out_swap.float()
    if bias is not None:
        base_f = base_f - bias.float()
        swap_f = swap_f - bias.float()
    coeff_base = base_f @ U
    coeff_swap = swap_f @ U

    coeff_mixed = coeff_base.clone()
    dirs = list(directions)
    coeff_mixed[..., dirs] = coeff_swap[..., dirs]

    out = coeff_mixed @ U.T
    if bias is not None:
        out = out + bias.float()
    return out.to(conn_out_base.dtype)


def _process_batch(adapter, batch, tokenizer, image_base_path,
                   directions=None, rand_gen=None):
    """Run interchange on a single batch, return list of result dicts.

    When *directions* is None, performs a full connector output swap.
    When *directions* is a list of direction indices, swaps only those
    directions (partial arm) and also swaps a random set of the same
    size (random arm).
    """
    actual = len(batch)

    base_data = [
        {
            "image": item["base_image"],
            "question": item["question"],
            "answers": [item["base_answer"]],
        }
        for item in batch
    ]
    swap_data = [
        {
            "image": item["swap_image"],
            "question": item["question"],
            "answers": [item["swap_answer"]],
        }
        for item in batch
    ]

    inputs_base = adapter.preprocess(base_data, image_base_path)
    inputs_swap = adapter.preprocess(swap_data, image_base_path)

    with torch.no_grad():
        vision_base = adapter.extract_vision(inputs_base)
        conn_out_base = adapter.run_connector(vision_base)

        vision_swap = adapter.extract_vision(inputs_swap)
        conn_out_swap = adapter.run_connector(vision_swap)

        text_embeds = adapter.get_text_embeds(inputs_base)
        attention_mask = inputs_base["attention_mask"]

        embeds_orig = adapter.merge_embeds(
            inputs_base, text_embeds, conn_out_base
        )
        preds_orig, logits_orig = adapter.generate_with_logits(
            embeds_orig, attention_mask
        )

    log_probs_orig = F.log_softmax(logits_orig, dim=-1)
    probs_orig = F.softmax(logits_orig, dim=-1)

    # Score the original (baseline) condition
    orig_scores = []
    for j in range(actual):
        item = batch[j]
        ids_base_ans = tokenizer(
            item["base_answer"], add_special_tokens=False
        )["input_ids"]
        ids_swap_ans = tokenizer(
            item["swap_answer"], add_special_tokens=False
        )["input_ids"]
        lp = {}
        if ids_base_ans and ids_swap_ans:
            ba, sa = ids_base_ans[0], ids_swap_ans[0]
            lp["base_under_orig"] = log_probs_orig[j, ba].item()
            lp["swap_under_orig"] = log_probs_orig[j, sa].item()
        orig_prefers_base = (
            lp.get("base_under_orig", 0) > lp.get("swap_under_orig", 0)
        )
        pred_o = preds_orig[j].lower()
        ba_re = re.compile(
            r"\b" + re.escape(item["base_answer"].lower()) + r"\b"
        )
        orig_has_base = bool(ba_re.search(pred_o))
        orig_scores.append({
            "logprobs_orig": lp,
            "orig_prefers_base": orig_prefers_base,
            "pred_orig": preds_orig[j],
            "orig_text_has_base": orig_has_base,
        })

    if directions is None:
        # Full swap
        with torch.no_grad():
            embeds_swapped = adapter.merge_embeds(
                inputs_base, text_embeds, conn_out_swap
            )
            preds_swap, logits_swap = adapter.generate_with_logits(
                embeds_swapped, attention_mask
            )

        log_probs_swap = F.log_softmax(logits_swap, dim=-1)
        swap_scores = _score_condition(
            batch, tokenizer, preds_swap, log_probs_swap, probs_orig, "full"
        )

        results = []
        for j in range(actual):
            item = batch[j]
            o = orig_scores[j]
            s = swap_scores[j]
            lp = {**o["logprobs_orig"], **s["logprobs_swap"]}
            results.append({
                "label": item["label"],
                "pair_id": item["pair_id"],
                "direction": item["direction"],
                "question": item["question"],
                "base_answer": item["base_answer"],
                "swap_answer": item["swap_answer"],
                "pred_orig": o["pred_orig"],
                "pred_swap": s["pred_swap"],
                "kl": s["kl"],
                "logprobs": lp,
                "orig_prefers_base": o["orig_prefers_base"],
                "swap_prefers_swap": s["swap_prefers_swap"],
                "interchange_success_logprob": (
                    o["orig_prefers_base"] and s["swap_prefers_swap"]
                ),
                "orig_text_has_base": o["orig_text_has_base"],
                "swap_text_has_swap": s["swap_text_has_swap"],
                "interchange_success_text": (
                    o["orig_text_has_base"] and s["swap_text_has_swap"]
                ),
            })
        return results

    # Partial + random arms
    last_layer = adapter.svd_layers[-1]
    n_dirs = last_layer.U.shape[1]

    conn_partial = _partial_swap(
        conn_out_base, conn_out_swap, directions, last_layer
    )

    rand_dirs = torch.randperm(n_dirs, generator=rand_gen)[
        : len(directions)
    ].tolist()
    conn_random = _partial_swap(
        conn_out_base, conn_out_swap, rand_dirs, last_layer
    )

    arms = [
        ("partial", conn_partial),
        ("random", conn_random),
    ]

    all_results = {}
    for arm_name, conn_swapped in arms:
        with torch.no_grad():
            embeds = adapter.merge_embeds(
                inputs_base, text_embeds, conn_swapped
            )
            preds, logits = adapter.generate_with_logits(
                embeds, attention_mask
            )
        log_probs = F.log_softmax(logits, dim=-1)
        swap_scores = _score_condition(
            batch, tokenizer, preds, log_probs, probs_orig, arm_name
        )

        arm_results = []
        for j in range(actual):
            item = batch[j]
            o = orig_scores[j]
            s = swap_scores[j]
            lp = {**o["logprobs_orig"], **s["logprobs_swap"]}
            arm_results.append({
                "label": item["label"],
                "pair_id": item["pair_id"],
                "direction": item["direction"],
                "question": item["question"],
                "base_answer": item["base_answer"],
                "swap_answer": item["swap_answer"],
                "arm": arm_name,
                "pred_orig": o["pred_orig"],
                "pred_swap": s["pred_swap"],
                "kl": s["kl"],
                "logprobs": lp,
                "orig_prefers_base": o["orig_prefers_base"],
                "swap_prefers_swap": s["swap_prefers_swap"],
                "interchange_success_logprob": (
                    o["orig_prefers_base"] and s["swap_prefers_swap"]
                ),
                "orig_text_has_base": o["orig_text_has_base"],
                "swap_text_has_swap": s["swap_text_has_swap"],
                "interchange_success_text": (
                    o["orig_text_has_base"] and s["swap_text_has_swap"]
                ),
            })
            if arm_name == "random":
                arm_results[-1]["random_directions"] = rand_dirs
        all_results[arm_name] = arm_results

    return all_results


def run_interchange(
    adapter: ModelAdapter,
    items: list[dict],
    *,
    batch_size: int,
    image_base_path: str | Path,
    run_dir: str | Path,
    directions: list[int] | None = None,
    seed: int = 42,
) -> None:
    """Interchange intervention on connector outputs.

    When *directions* is ``None`` (default), performs a full connector
    output swap — the existing full-arm experiment.

    When *directions* is a list of SVD direction indices, performs a
    **partial-arm** interchange: only the listed directions' coefficients
    are swapped (decomposed on the last SVD layer), while remaining
    directions keep the base image's values.  A **random arm** with the
    same number of randomly chosen directions is always run alongside
    as a baseline.

    Results are saved as:

    - Full arm: ``interchange_results.json`` / ``interchange_summary.json``
    - Partial arm: ``partial_interchange_results.json`` / ``partial_interchange_summary.json``
    - Random arm: ``random_interchange_results.json`` / ``random_interchange_summary.json``

    :param adapter: Model adapter instance.
    :param items: Output of :func:`prepare_interchange_items`.
    :param batch_size: Images per forward pass.
    :param image_base_path: Root directory for image files.
    :param run_dir: Output directory.
    :param directions: SVD direction indices to swap. ``None`` for full swap.
    :param seed: Random seed for the random arm baseline.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    image_base_path = Path(image_base_path)

    tokenizer = adapter.processor.tokenizer
    is_partial = directions is not None

    meta = dict(
        model=adapter.model_name,
        experiment="interchange",
        mode="partial" if is_partial else "full",
        n_items=len(items),
        batch_size=batch_size,
        labels=sorted(set(item["label"] for item in items)),
    )
    if is_partial:
        meta["n_directions"] = len(directions)
        meta["n_total_directions"] = adapter.svd_layers[-1].U.shape[1]
        meta["seed"] = seed
    update_metadata(run_dir, meta)

    rand_gen = torch.Generator().manual_seed(seed) if is_partial else None

    if is_partial:
        results_partial: list[dict] = []
        results_random: list[dict] = []
    else:
        results: list[dict] = []
    skipped: list[str] = []

    desc = "Partial interchange" if is_partial else "Interchange"
    for i in tqdm(
        range(0, len(items), batch_size),
        total=math.ceil(len(items) / batch_size),
        desc=desc,
    ):
        batch = items[i : i + batch_size]

        try:
            batch_out = _process_batch(
                adapter, batch, tokenizer, image_base_path,
                directions=directions, rand_gen=rand_gen,
            )
            if is_partial:
                results_partial.extend(batch_out["partial"])
                results_random.extend(batch_out["random"])
            else:
                results.extend(batch_out)
        except Exception:
            for item in batch:
                try:
                    single = _process_batch(
                        adapter, [item], tokenizer, image_base_path,
                        directions=directions, rand_gen=rand_gen,
                    )
                    if is_partial:
                        results_partial.extend(single["partial"])
                        results_random.extend(single["random"])
                    else:
                        results.extend(single)
                except Exception:
                    skipped.append(
                        f"{item['label']} pair={item['pair_id']} "
                        f"dir={item['direction']}"
                    )

    if skipped:
        print(f"\nSkipped {len(skipped)} items (corrupt/unreadable images):")
        for s in skipped[:20]:
            print(f"  {s}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    if is_partial:
        for arm_name, arm_results in [
            ("partial", results_partial),
            ("random", results_random),
        ]:
            _save_arm(run_dir, arm_name, arm_results)
    else:
        _save_arm(run_dir, "interchange", results)


def _save_arm(run_dir, prefix, results):
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_label[r["label"]].append(r)

    summary: dict = {
        "n_items": len(results),
        "overall": _aggregate(results),
        "by_label": {
            label: _aggregate(label_results)
            for label, label_results in sorted(by_label.items())
        },
    }

    save_json(run_dir / f"{prefix}_results.json", results)
    save_json(run_dir / f"{prefix}_summary.json", summary)

    arm_label = prefix.replace("_", " ").title()
    print(f"\n{arm_label} results ({len(results)} items):")
    o = summary["overall"]
    print(
        f"  Overall: "
        f"logprob_success={o['interchange_success_logprob']:.1%}, "
        f"text_success={o['interchange_success_text']:.1%}, "
        f"baseline_correct={o['orig_prefers_base']:.1%}, "
        f"KL={o['kl_mean']:.4f}"
    )
    print("  By pair:")
    for label, s in sorted(summary["by_label"].items()):
        print(
            f"    {label}: "
            f"logprob={s['interchange_success_logprob']:.1%}, "
            f"text={s['interchange_success_text']:.1%} "
            f"({s['n']} items, KL={s['kl_mean']:.4f})"
        )


def _aggregate(results: list[dict]) -> dict:
    return {
        "n": len(results),
        "interchange_success_logprob": float(
            np.mean([r["interchange_success_logprob"] for r in results])
        ),
        "interchange_success_text": float(
            np.mean([r["interchange_success_text"] for r in results])
        ),
        "orig_prefers_base": float(
            np.mean([r["orig_prefers_base"] for r in results])
        ),
        "swap_prefers_swap": float(
            np.mean([r["swap_prefers_swap"] for r in results])
        ),
        "orig_text_has_base": float(
            np.mean([r["orig_text_has_base"] for r in results])
        ),
        "swap_text_has_swap": float(
            np.mean([r["swap_text_has_swap"] for r in results])
        ),
        "kl_mean": float(np.mean([r["kl"] for r in results])),
        "kl_median": float(np.median([r["kl"] for r in results])),
    }
