"""
Dataset loading and evaluation metrics.

Handles DocVQA and OK-VQA dataset loading with per-category grouping,
and provides the ANLS (Averaged Normalized Levenshtein Similarity) scoring
functions.

Example::

    >>> from conn_intrp.data import load_docvqa, load_okvqa, best_anls
    >>> all_data, categorized = load_docvqa("dataset/docVQA/train_v1.0_withQT.json")
    >>> all_data, categorized = load_okvqa("okvqa/questions.json", "okvqa/annotations.json")
    >>> score = best_anls("hello world", ["hello world", "hi"])

Main Functions:
    load_docvqa: Load and categorize the DocVQA dataset.
    load_okvqa: Load and categorize the OK-VQA dataset.
    best_anls: Best ANLS score across multiple ground-truth targets.
    relaxed_anls: ANLS between a single prediction and ground truth.
"""

import json
import random
from pathlib import Path

import rapidfuzz.distance

HARNESS_PROMPT = (
    "NO repeat the words in the question, NO 'the answer is', "
    "NO 'the document type is', NO 'the address is', and other "
    "similar opening or closing words that does not answer question. "
    "Only exactly words that answers question."
)

def load_docvqa(json_path: str, seed: int = 99) -> tuple[list, dict]:
    """
    Load and shuffle the DocVQA dataset, grouped by question category.

    :param json_path: Path to the JSON annotation file
    :type json_path: str
    :param seed: Random seed for reproducible shuffling
    :type seed: int
    :returns: ``(all_data, categorized)`` where *categorized* maps
        category name to a list of data dicts
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


def load_okvqa(
    questions_path: str,
    annotations_path: str,
    seed: int = 99,
) -> tuple[list, dict]:
    """
    Load and shuffle the OK-VQA dataset, grouped by knowledge category.

    Joins the separate questions and annotations files by ``question_id``,
    normalises each entry to the same dict format used by :func:`load_docvqa`
    (``image``, ``question``, ``answers`` keys).

    :param questions_path: Path to the questions JSON
        (``OpenEnded_mscoco_*_questions.json``)
    :type questions_path: str
    :param annotations_path: Path to the annotations JSON
        (``mscoco_*_annotations.json``)
    :type annotations_path: str
    :param seed: Random seed for reproducible shuffling
    :type seed: int
    :returns: ``(all_data, categorized)`` where *categorized* maps
        knowledge category name to a list of data dicts
    :rtype: tuple[list, dict]
    """
    with open(questions_path) as f:
        q_data = json.load(f)
    with open(annotations_path) as f:
        a_data = json.load(f)

    ann_by_qid = {a["question_id"]: a for a in a_data["annotations"]}
    qt_names = a_data.get("question_types", {})
    data_subtype = q_data.get("data_subtype", "train2014")

    all_data = []
    for q in q_data["questions"]:
        qid = q["question_id"]
        ann = ann_by_qid[qid]
        qt_code = ann["question_type"]
        datum = {
            "questionId": qid,
            "question": q["question"],
            "image": f"{data_subtype}/COCO_{data_subtype}_{q['image_id']:012d}.jpg",
            "answers": [a["answer"] for a in ann["answers"]],
            "question_type": qt_names.get(qt_code, qt_code),
        }
        all_data.append(datum)

    random.seed(seed)
    random.shuffle(all_data)

    categories = {d["question_type"] for d in all_data}
    categorized = {cat: [] for cat in categories}
    for datum in all_data:
        categorized[datum["question_type"]].append(datum)

    return all_data, categorized


def relaxed_anls(pred: str, gt: str, tau: float = 0.5) -> float:
    """
    ANLS between a single prediction and ground truth.

    :param pred: Model prediction string
    :type pred: str
    :param gt: Ground-truth answer string
    :type gt: str
    :param tau: Threshold below which score is clamped to 0
    :type tau: float
    :returns: Score in [0, 1]
    :rtype: float
    """
    pred, gt = pred.strip().lower(), gt.strip().lower()
    if gt in pred:
        return 1.0
    nl = rapidfuzz.distance.Levenshtein.normalized_distance(pred, gt)
    return (1 - nl) if nl < tau else 0.0


def best_anls(prediction: str, targets: list[str], tau: float = 0.5) -> float:
    """
    Best ANLS score across multiple ground-truth targets.

    :param prediction: Model prediction string
    :type prediction: str
    :param targets: List of acceptable ground-truth answers
    :type targets: list[str]
    :param tau: Threshold passed to :func:`relaxed_anls`
    :type tau: float
    :returns: Maximum score across all targets
    :rtype: float
    """
    return max(relaxed_anls(prediction, t, tau) for t in targets)


def filter_categories(
    data_categorized: dict[str, list],
    *,
    categories: list[str] | None = None,
    max_samples: int | None = None,
) -> dict[str, list]:
    """
    Filter and/or truncate a categorized dataset.

    :param data_categorized: Mapping of category name to list of data dicts
    :type data_categorized: dict[str, list]
    :param categories: Keep only these categories.  ``None`` keeps all
    :type categories: list[str] | None
    :param max_samples: Cap each category to at most this many samples
    :type max_samples: int | None
    :returns: Filtered copy of the dict (never mutates the original)
    :rtype: dict[str, list]
    """
    out = data_categorized
    if categories is not None:
        missing = set(categories) - data_categorized.keys()
        if missing:
            raise KeyError(f"Unknown categories: {missing}")
        out = {k: v for k, v in out.items() if k in categories}
    if max_samples is not None:
        out = {k: v[:max_samples] for k, v in out.items()}
    return out


DATASETS = {
    "docvqa": {
        "default_path": "dataset/docVQA",
        "train": lambda p: load_docvqa(p / "train_v1.0_withQT.json"),
        "val": lambda p: load_docvqa(p / "val_v1.0_withQT.json"),
    },
    "okvqa": {
        "default_path": "dataset/OK-VQA",
        "train": lambda p: load_okvqa(
            p / "OpenEnded_mscoco_train2014_questions.json",
            p / "mscoco_train2014_annotations.json",
        ),
        "val": lambda p: load_okvqa(
            p / "OpenEnded_mscoco_val2014_questions.json",
            p / "mscoco_val2014_annotations.json",
        ),
    },
}


def load_dataset(
    name: str,
    split: str = "train",
    dataset_path: str | None = None,
) -> tuple[list, dict]:
    """
    Load a dataset by name and split.

    :param name: Dataset key (one of :data:`DATASETS`)
    :type name: str
    :param split: ``"train"`` or ``"val"``
    :type split: str
    :param dataset_path: Override default path for this dataset
    :type dataset_path: str | None
    :returns: ``(all_data, categorized)``
    :rtype: tuple[list, dict]
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}', choose from: {list(DATASETS)}")
    ds = DATASETS[name]
    path = Path(dataset_path) if dataset_path else Path(ds["default_path"])
    return ds[split](path)
