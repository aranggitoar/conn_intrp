"""
Analysis functions for semantic control experiment results.

Example::

    >>> from conn_intrp.semantic_control_analysis import (
    ...     load_semantic_control, topk_botk_summary, comparison_table)
    >>> data = load_semantic_control("outputs/semantic_control_run")
    >>> top, bot = topk_botk_summary(data, "word:cat@1.0x", tokenizer=tok)
    >>> print(top.head(10).to_string())
    >>> print(comparison_table(data).to_string())

Main Functions:
    load_semantic_control: Load results from a semantic control run.
    topk_botk_summary: Aggregated top-K/bottom-K token shifts for a
        perturbation.
    comparison_table: Side-by-side KL and delta gold log-prob for all
        perturbations.
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch


def load_semantic_control(run_dir: str | Path) -> dict:
    """
    Load semantic control results.

    :param run_dir: Directory containing
        ``semantic_control_summary.json`` and
        ``semantic_control_delta_logits.pt``.
    :returns: Dict with ``summary`` and ``delta_logits`` keys.
    """
    run_dir = Path(run_dir)
    with open(run_dir / "semantic_control_summary.json") as f:
        summary = json.load(f)
    delta_logits = torch.load(
        run_dir / "semantic_control_delta_logits.pt", weights_only=False
    )
    return {"summary": summary, "delta_logits": delta_logits}


def comparison_table(data: dict) -> pd.DataFrame:
    """
    Side-by-side KL and delta gold log-prob for all perturbations.

    :param data: Loaded results from :func:`load_semantic_control`.
    :returns: DataFrame with one row per perturbation.
    """
    summary = data["summary"]
    rows = []
    for name, p in summary["perturbations"].items():
        rows.append({
            "perturbation": name,
            "kl_mean": p["kl_mean"],
            "kl_median": p["kl_median"],
            "delta_gold_lp_mean": p["delta_gold_prob_mean"],
        })
    return pd.DataFrame(rows)


def topk_botk_summary(
    data: dict,
    perturbation: str,
    *,
    k: int = 10,
    tokenizer=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregated top-K and bottom-K token shifts for a perturbation.

    Counts how often each token appears in the top/bottom-K across
    images, with mean logit delta and mean probability change.

    :param data: Loaded results from :func:`load_semantic_control`.
    :param perturbation: Perturbation label (e.g. ``"word:cat@1.0x"``).
    :param k: Number of top/bottom tokens to aggregate over.
    :param tokenizer: HuggingFace tokenizer for decoding token IDs.
    :returns: ``(topk_df, botk_df)``.
    """
    dl = data["delta_logits"][perturbation]
    topk = dl["topk"]
    botk = dl["botk"]

    def _aggregate(tensor: torch.Tensor) -> pd.DataFrame:
        n_images = tensor.shape[0]
        K = min(k, tensor.shape[2])
        token_ids = tensor[:, 0, :K].long().reshape(-1).tolist()
        deltas = tensor[:, 1, :K].reshape(-1).tolist()
        prob_orig = tensor[:, 2, :K].reshape(-1).tolist()
        prob_pert = tensor[:, 3, :K].reshape(-1).tolist()

        acc = defaultdict(
            lambda: {"count": 0, "delta_sum": 0.0, "po_sum": 0.0, "pp_sum": 0.0}
        )
        for tid, d, po, pp in zip(token_ids, deltas, prob_orig, prob_pert):
            acc[tid]["count"] += 1
            acc[tid]["delta_sum"] += d
            acc[tid]["po_sum"] += po
            acc[tid]["pp_sum"] += pp

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
                "mean_prob_perturbed": v["pp_sum"] / v["count"],
            })
        return pd.DataFrame(rows)

    return _aggregate(topk), _aggregate(botk)


def all_topk_botk(
    data: dict,
    *,
    k: int = 10,
    n_tokens: int = 5,
    tokenizer=None,
) -> pd.DataFrame:
    """
    Compact comparison: top N most-shifted tokens per perturbation.

    :param data: Loaded results from :func:`load_semantic_control`.
    :param k: Top/bottom-K to aggregate over per image.
    :param n_tokens: Number of most-frequent tokens to show per
        perturbation.
    :param tokenizer: HuggingFace tokenizer for decoding.
    :returns: DataFrame with perturbation, direction (top/bot), and
        token columns.
    """
    rows = []
    for name in data["delta_logits"]:
        top_df, bot_df = topk_botk_summary(
            data, name, k=k, tokenizer=tokenizer
        )
        for _, r in top_df.head(n_tokens).iterrows():
            rows.append({
                "perturbation": name,
                "direction": "top",
                "token": r["token"],
                "count": r["count"],
                "frequency": r["frequency"],
                "mean_delta": r["mean_delta"],
            })
        for _, r in bot_df.head(n_tokens).iterrows():
            rows.append({
                "perturbation": name,
                "direction": "bot",
                "token": r["token"],
                "count": r["count"],
                "frequency": r["frequency"],
                "mean_delta": r["mean_delta"],
            })
    return pd.DataFrame(rows)
