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
    downloaded_pairs: list[dict],
    swap_type: str,
) -> list[dict]:
    """Convert downloaded pair data into interchange items.

    Each downloaded pair produces two items — forward (pos base, neg
    swap) and reverse (neg base, pos swap) — so both directions are
    tested.

    :param downloaded_pairs: List of dicts from
        ``check_download.download_pairs``, each with keys ``pair``,
        ``pair_id``, ``pos_path``, ``neg_path``, ``pos_triplet``,
        ``neg_triplet``.
    :param swap_type: One of ``"subject"``, ``"verb"``, ``"object"``.
    :returns: List of interchange item dicts ready for
        :func:`run_interchange`.
    """
    items: list[dict] = []
    for p in downloaded_pairs:
        pos_trip = [x.strip() for x in p["pos_triplet"].split(",")]
        neg_trip_raw = p["neg_triplet"].strip("[] '\"")
        neg_trip = [x.strip() for x in neg_trip_raw.split(",")]

        if len(pos_trip) != 3 or len(neg_trip) != 3:
            continue

        question, pos_answer, neg_answer = _make_prompt(
            pos_trip, neg_trip, swap_type
        )
        a, b = p["pair"]
        label = f"{swap_type}:{a}_vs_{b}"

        items.append(
            {
                "base_image": p["pos_path"],
                "swap_image": p["neg_path"],
                "question": question,
                "base_answer": pos_answer,
                "swap_answer": neg_answer,
                "direction": "pos_to_neg",
                "label": label,
                "pair_id": p["pair_id"],
            }
        )

        items.append(
            {
                "base_image": p["neg_path"],
                "swap_image": p["pos_path"],
                "question": question,
                "base_answer": neg_answer,
                "swap_answer": pos_answer,
                "direction": "neg_to_pos",
                "label": label,
                "pair_id": p["pair_id"],
            }
        )

    return items


def run_interchange(
    adapter: ModelAdapter,
    items: list[dict],
    *,
    batch_size: int,
    image_base_path: str | Path,
    run_dir: str | Path,
) -> None:
    """True interchange intervention: swap connector outputs between
    paired images and measure the effect on model output.

    For each item, runs the base image through vision encoder +
    connector to get the baseline, then replaces the connector output
    with the swap image's and generates again.  Records:

    - **Forced-choice log-probs**: probability of each answer option
      at the first generated token, under both conditions.
    - **Text match**: whether the generated text contains the expected
      answer word.
    - **KL divergence**: between baseline and swapped output
      distributions.
    - **Generated text**: full predictions for qualitative inspection.

    :param adapter: Model adapter instance.
    :param items: Output of :func:`prepare_interchange_items`.
    :param batch_size: Images per forward pass.
    :param image_base_path: Root directory for image files.
    :param run_dir: Output directory.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    image_base_path = Path(image_base_path)

    tokenizer = adapter.processor.tokenizer

    update_metadata(
        run_dir,
        dict(
            model=adapter.model_name,
            experiment="interchange",
            n_items=len(items),
            batch_size=batch_size,
            labels=sorted(set(item["label"] for item in items)),
        ),
    )

    results: list[dict] = []

    for i in tqdm(
        range(0, len(items), batch_size),
        total=math.ceil(len(items) / batch_size),
        desc="Interchange",
    ):
        batch = items[i : i + batch_size]
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

            embeds_swapped = adapter.merge_embeds(
                inputs_base, text_embeds, conn_out_swap
            )
            preds_swap, logits_swap = adapter.generate_with_logits(
                embeds_swapped, attention_mask
            )

        log_probs_orig = F.log_softmax(logits_orig, dim=-1)
        log_probs_swap = F.log_softmax(logits_swap, dim=-1)
        probs_orig = F.softmax(logits_orig, dim=-1)

        kl = F.kl_div(
            log_probs_swap, probs_orig, reduction="none"
        ).sum(-1)

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
                lp["base_under_swap"] = log_probs_swap[j, ba].item()
                lp["swap_under_swap"] = log_probs_swap[j, sa].item()

            orig_prefers_base = (
                lp.get("base_under_orig", 0)
                > lp.get("swap_under_orig", 0)
            )
            swap_prefers_swap = (
                lp.get("swap_under_swap", 0)
                > lp.get("base_under_swap", 0)
            )

            pred_o = preds_orig[j].lower()
            pred_s = preds_swap[j].lower()
            ba_re = re.compile(r"\b" + re.escape(item["base_answer"].lower()) + r"\b")
            sa_re = re.compile(r"\b" + re.escape(item["swap_answer"].lower()) + r"\b")
            orig_has_base = bool(ba_re.search(pred_o))
            swap_has_swap = bool(sa_re.search(pred_s))

            results.append(
                {
                    "label": item["label"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "question": item["question"],
                    "base_answer": item["base_answer"],
                    "swap_answer": item["swap_answer"],
                    "pred_orig": preds_orig[j],
                    "pred_swap": preds_swap[j],
                    "kl": kl[j].item(),
                    "logprobs": lp,
                    "orig_prefers_base": orig_prefers_base,
                    "swap_prefers_swap": swap_prefers_swap,
                    "interchange_success_logprob": (
                        orig_prefers_base and swap_prefers_swap
                    ),
                    "orig_text_has_base": orig_has_base,
                    "swap_text_has_swap": swap_has_swap,
                    "interchange_success_text": (
                        orig_has_base and swap_has_swap
                    ),
                }
            )

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

    save_json(run_dir / "interchange_summary.json", summary)
    save_json(run_dir / "interchange_results.json", results)

    print(f"\nInterchange results ({len(results)} items):")
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
