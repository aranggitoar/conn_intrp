"""
Run mean ablation + ANLS + Δlogit for a specific model.

Means are computed on the training split; ANLS is scored on the
validation split (``val_v1.0_withQT.json``).

Usage:
    python scripts/run_ablation.py                          # SmolVLM2 (default)
    python scripts/run_ablation.py --internvl               # InternVL3.5
    python scripts/run_ablation.py --directions 23 70 255   # specific directions
    python scripts/run_ablation.py --resume                 # resume latest run
    python scripts/run_ablation.py --categories "table/list"
    python scripts/run_ablation.py --max-samples 20
"""

import argparse
from pathlib import Path

from conn_intrp.data import load_docvqa, filter_categories
from conn_intrp.output import make_run_dir
from conn_intrp.ablation import compute_category_means, run_ablation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internvl", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--directions", type=int, nargs="+", default=[23, 70, 255],
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--dataset", type=str, default="dataset/docVQA")
    parser.add_argument("--val-json", type=str, default="val_v1.0_withQT.json",
                        help="Validation JSON filename (relative to --dataset).")
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    image_base_path = Path(args.dataset)

    _, train_categorized = load_docvqa(
        image_base_path / "train_v1.0_withQT.json"
    )
    val_path = image_base_path / args.val_json
    _, val_categorized = load_docvqa(val_path)

    train_categorized = filter_categories(
        train_categorized,
        categories=args.categories,
        max_samples=args.max_samples,
    )
    val_categorized = filter_categories(
        val_categorized,
        categories=args.categories,
        max_samples=args.max_samples,
    )

    if args.internvl:
        from conn_intrp.models import InternVLAdapter
        adapter = InternVLAdapter("OpenGVLab/InternVL3_5-2B-HF")
        batch_size = args.batch_size or 1
    else:
        from conn_intrp.models import SmolVLM2Adapter
        adapter = SmolVLM2Adapter("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        batch_size = args.batch_size or 5

    run_dir = make_run_dir(
        Path(args.output), adapter.model_name, "ablation", resume=args.resume,
    )
    _, cat_means, global_mean = compute_category_means(
        adapter, train_categorized,
        batch_size=batch_size,
        image_base_path=image_base_path,
        run_dir=run_dir,
    )

    val_coefficients, _, _ = compute_category_means(
        adapter, val_categorized,
        batch_size=batch_size,
        image_base_path=image_base_path,
        run_dir=run_dir / "val_coefficients",
    )

    run_ablation(
        adapter, val_categorized,
        val_coefficients, cat_means, global_mean,
        directions_to_ablate=args.directions,
        batch_size=batch_size,
        K=args.K,
        image_base_path=image_base_path,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
