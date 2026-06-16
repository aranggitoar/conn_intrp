"""
Dataset loading and evaluation metrics.

Handles DocVQA dataset loading with per-category grouping, and provides
the ANLS (Averaged Normalized Levenshtein Similarity) scoring functions.

Example::

    >>> from conn_intrp.data import load_docvqa, best_anls
    >>> all_data, categorized = load_docvqa("dataset/docVQA/train_v1.0_withQT.json")
    >>> score = best_anls("hello world", ["hello world", "hi"])

Main Functions:
    load_docvqa: Load and categorize the DocVQA dataset.
    best_anls: Best ANLS score across multiple ground-truth targets.
    relaxed_anls: ANLS between a single prediction and ground truth.
"""

import json
import random


HARNESS_PROMPT = (
    "NO repeat the words in the question, NO 'the answer is', "
    "NO 'the document type is', NO 'the address is', and other "
    "similar opening or closing words that does not answer question. "
    "Only exactly words that answers question."
)


def load_docvqa(json_path: str, seed: int = 99) -> tuple[list, dict]:
    """
    Load and shuffle the DocVQA dataset, grouped by question category.

    :param json_path: Path to the JSON annotation file.
    :type json_path: str
    :param seed: Random seed for reproducible shuffling.
    :type seed: int
    :returns: ``(all_data, categorized)`` where *categorized* maps
        category name to a list of data dicts.
    :rtype: tuple[list, dict]
    """
    with open(json_path) as f:
        dataset = json.load(f)
    random.seed(seed)
    random.shuffle(dataset["data"])

    categories = {t for datum in dataset["data"] for t in datum["question_types"]}
    categorized = {cat: [] for cat in categories}
    for datum in dataset["data"]:
        for t in datum["question_types"]:
            categorized[t].append(datum)

    return dataset["data"], categorized


def relaxed_anls(pred: str, gt: str, tau: float = 0.5) -> float:
    """
    ANLS between a single prediction and ground truth.

    :param pred: Model prediction string.
    :type pred: str
    :param gt: Ground-truth answer string.
    :type gt: str
    :param tau: Threshold below which score is clamped to 0.
    :type tau: float
    :returns: Score in [0, 1].
    :rtype: float
    """
    import rapidfuzz.distance
    pred, gt = pred.strip().lower(), gt.strip().lower()
    if gt in pred:
        return 1.0
    nl = rapidfuzz.distance.Levenshtein.normalized_distance(pred, gt)
    return (1 - nl) if nl < tau else 0.0


def best_anls(prediction: str, targets: list[str], tau: float = 0.5) -> float:
    """
    Best ANLS score across multiple ground-truth targets.

    :param prediction: Model prediction string.
    :type prediction: str
    :param targets: List of acceptable ground-truth answers.
    :type targets: list[str]
    :param tau: Threshold passed to :func:`relaxed_anls`.
    :type tau: float
    :returns: Maximum score across all targets.
    :rtype: float
    """
    return max(relaxed_anls(prediction, t, tau) for t in targets)
