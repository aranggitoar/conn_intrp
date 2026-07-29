"""
Analysis functions for interchange experiment results.

Example::

    >>> from conn_intrp.interchange_analysis import (
    ...     load_interchange, margins, failure_decomposition,
    ...     significance, direction_asymmetry, comparison_table)
    >>> data = load_interchange("outputs/smolvlm2_interchange_run")
    >>> m = margins(data)
    >>> f = failure_decomposition(data)
    >>> s = significance(data)
    >>> a = direction_asymmetry(data)
    >>> t = comparison_table(data)

Main Functions:
    load_interchange: Load results from an interchange run.
    margins: Per-item logprob margins with spread statistics.
    failure_decomposition: Categorize failures by root cause.
    significance: Binomial test per pair type.
    direction_asymmetry: pos_to_neg vs neg_to_pos success rates.
    comparison_table: One-row-per-pair summary with all metrics.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def load_interchange(run_dir: str | Path) -> list[dict]:
    run_dir = Path(run_dir)
    with open(run_dir / "interchange_results.json") as f:
        return json.load(f)


def margins(results: list[dict]) -> pd.DataFrame:
    """Per-item logprob margins and per-label spread.

    For each item, computes:
    - ``margin_orig``: base_under_orig − swap_under_orig
      (positive = baseline prefers correct answer)
    - ``margin_swap``: swap_under_swap − base_under_swap
      (positive = swapped prefers swapped answer)
    - ``margin_delta``: margin_swap − margin_orig
      (positive = swap flipped preference beyond baseline)

    :returns: DataFrame with one row per item, plus label/direction
        columns for grouping.
    """
    rows = []
    for r in results:
        lp = r.get("logprobs", {})
        if not lp:
            continue
        m_orig = lp["base_under_orig"] - lp["swap_under_orig"]
        m_swap = lp["swap_under_swap"] - lp["base_under_swap"]
        rows.append({
            "label": r["label"],
            "pair_id": r["pair_id"],
            "direction": r["direction"],
            "margin_orig": m_orig,
            "margin_swap": m_swap,
            "margin_delta": m_swap - m_orig,
            "kl": r["kl"],
            "logprob_success": r["interchange_success_logprob"],
            "text_success": r["interchange_success_text"],
        })
    return pd.DataFrame(rows)


def failure_decomposition(results: list[dict]) -> pd.DataFrame:
    """Categorize each item's outcome into one of four modes.

    - ``baseline_wrong``: model doesn't prefer the base answer
      even without swapping (noise in data, not connector).
    - ``no_flip``: baseline correct but swap didn't override
      (connector encodes it, but swap wasn't enough).
    - ``both_wrong``: neither condition works.
    - ``success``: baseline correct AND swap flipped.

    :returns: DataFrame with counts and rates per label.
    """
    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))

    for r in results:
        label = r["label"]
        base_ok = r["orig_prefers_base"]
        swap_ok = r["swap_prefers_swap"]

        if base_ok and swap_ok:
            counts[label]["success"] += 1
        elif base_ok and not swap_ok:
            counts[label]["no_flip"] += 1
        elif not base_ok and swap_ok:
            counts[label]["baseline_wrong"] += 1
        else:
            counts[label]["both_wrong"] += 1

    rows = []
    for label in sorted(counts):
        c = counts[label]
        n = sum(c.values())
        rows.append({
            "label": label,
            "n": n,
            "success": c["success"],
            "baseline_wrong": c["baseline_wrong"],
            "no_flip": c["no_flip"],
            "both_wrong": c["both_wrong"],
            "success_rate": c["success"] / n,
            "baseline_wrong_rate": c["baseline_wrong"] / n,
            "no_flip_rate": c["no_flip"] / n,
            "both_wrong_rate": c["both_wrong"] / n,
        })

    overall = {m: sum(r[m] for r in rows) for m in
               ["n", "success", "baseline_wrong", "no_flip", "both_wrong"]}
    overall["label"] = "_overall"
    for m in ["success", "baseline_wrong", "no_flip", "both_wrong"]:
        overall[f"{m}_rate"] = overall[m] / overall["n"]
    rows.append(overall)

    return pd.DataFrame(rows).set_index("label")


def significance(results: list[dict], metric: str = "logprob") -> pd.DataFrame:
    """Binomial test per pair type: is success rate above chance (50%)?

    :param metric: ``"logprob"`` or ``"text"`` — which success
        criterion to test.
    :returns: DataFrame with n, successes, rate, p-value, and
        significance flag per label.
    """
    key = (
        "interchange_success_logprob" if metric == "logprob"
        else "interchange_success_text"
    )

    from collections import defaultdict
    by_label = defaultdict(list)
    for r in results:
        by_label[r["label"]].append(r[key])

    rows = []
    for label in sorted(by_label):
        vals = by_label[label]
        n = len(vals)
        k = sum(vals)
        rate = k / n
        binom = sp_stats.binomtest(k, n, 0.5, alternative="greater")
        rows.append({
            "label": label,
            "n": n,
            "successes": k,
            "rate": rate,
            "p_value": binom.pvalue,
            "ci_low": binom.proportion_ci(confidence_level=0.95).low,
            "ci_high": binom.proportion_ci(confidence_level=0.95).high,
            "sig_01": binom.pvalue < 0.01,
            "sig_05": binom.pvalue < 0.05,
        })

    return pd.DataFrame(rows).set_index("label")


def direction_asymmetry(results: list[dict]) -> (pd.DataFrame, pd.DataFrame):
    """Success rate split by interchange direction (pos_to_neg vs neg_to_pos).

    :returns: DataFrame with one row per (label, direction) combination,
        plus a delta column showing asymmetry magnitude.
    """
    from collections import defaultdict
    by_ld = defaultdict(list)
    for r in results:
        by_ld[(r["label"], r["direction"])].append(r)

    rows = []
    for (label, direction), items in sorted(by_ld.items()):
        n = len(items)
        lp_succ = sum(r["interchange_success_logprob"] for r in items) / n
        txt_succ = sum(r["interchange_success_text"] for r in items) / n
        base_ok = sum(r["orig_prefers_base"] for r in items) / n
        rows.append({
            "label": label,
            "direction": direction,
            "n": n,
            "logprob_success": lp_succ,
            "text_success": txt_succ,
            "baseline_accuracy": base_ok,
        })

    df = pd.DataFrame(rows)

    deltas = []
    labels = df["label"].unique()
    for label in labels:
        sub = df[df["label"] == label]
        p2n = sub[sub["direction"] == "pos_to_neg"]["logprob_success"]
        n2p = sub[sub["direction"] == "neg_to_pos"]["logprob_success"]
        if len(p2n) and len(n2p):
            deltas.append({
                "label": label,
                "pos_to_neg": p2n.iloc[0],
                "neg_to_pos": n2p.iloc[0],
                "delta": p2n.iloc[0] - n2p.iloc[0],
                "abs_delta": abs(p2n.iloc[0] - n2p.iloc[0]),
            })

    return df, pd.DataFrame(deltas).set_index("label").sort_values(
        "abs_delta", ascending=False
    )


def comparison_table(results: list[dict]) -> pd.DataFrame:
    """One row per pair type with all key metrics side by side.

    Columns: n, baseline_acc, logprob_success, text_success,
    margin_orig (mean/std), margin_swap (mean/std), kl_mean,
    p_value, significant.

    :returns: DataFrame sorted by logprob_success descending.
    """
    m_df = margins(results)
    f_df = failure_decomposition(results)
    s_df = significance(results)

    rows = []
    for label in m_df["label"].unique():
        sub = m_df[m_df["label"] == label]
        n = len(sub)

        row = {
            "label": label,
            "swap_type": label.split(":")[0],
            "pair": label.split(":")[1],
            "n": n,
            "n_pairs": n // 2,
        }

        if label in f_df.index:
            row["baseline_acc"] = 1 - f_df.loc[label, "baseline_wrong_rate"] - f_df.loc[label, "both_wrong_rate"]
        row["logprob_success"] = sub["logprob_success"].mean()
        row["text_success"] = sub["text_success"].mean()

        row["margin_orig_mean"] = sub["margin_orig"].mean()
        row["margin_orig_std"] = sub["margin_orig"].std()
        row["margin_swap_mean"] = sub["margin_swap"].mean()
        row["margin_swap_std"] = sub["margin_swap"].std()
        row["margin_delta_mean"] = sub["margin_delta"].mean()

        row["kl_mean"] = sub["kl"].mean()
        row["kl_std"] = sub["kl"].std()

        if label in s_df.index:
            row["p_value"] = s_df.loc[label, "p_value"]
            row["sig_01"] = s_df.loc[label, "sig_01"]

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "logprob_success", ascending=False
    ).set_index("label")
