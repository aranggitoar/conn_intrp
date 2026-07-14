"""
Run directional masking for a specific model.

Usage:
    python scripts/run_dm.py              # SmolVLM2 + docvqa (default)
    python scripts/run_dm.py --internvl   # InternVL3.5
    python scripts/run_dm.py --dataset okvqa
    python scripts/run_dm.py --dataset okvqa --dataset-path /path/to/okvqa
    python scripts/run_dm.py --resume     # resume latest run
    python scripts/run_dm.py --categories "table/list" "figure/list"
    python scripts/run_dm.py --max-samples 20  # quick profiling run
"""

import argparse
from pathlib import Path

from conn_intrp.data import DATASETS, filter_categories, load_dataset
from conn_intrp.dm import run_dm
from conn_intrp.output import make_run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internvl", action="store_true")
    parser.add_argument(
        "--dataset", type=str, default="docvqa", choices=list(DATASETS),
        help="Dataset to use (default: docvqa)",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Override default dataset path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sparsity-coef", type=float, default=5e-3)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--step", type=int, default=None, help="Fixed batch size. Overrides --target-updates."
    )
    parser.add_argument(
        "--target-updates",
        type=int,
        default=None,
        help="Target gradient updates per epoch; step is derived per category from its size.",
    )
    parser.add_argument(
        "--max-step", type=int, default=None, help="Cap auto-computed step (GPU memory limit)."
    )
    parser.add_argument(
        "--patience", type=int, default=2, help="Early stop after this many stable epochs."
    )
    parser.add_argument(
        "--conv-threshold",
        type=float,
        default=0.02,
        help="Convergence threshold as fraction of n_dirs.",
    )
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    image_base_path = Path(args.dataset_path or DATASETS[args.dataset]["default_path"])
    _, data_categorized = load_dataset(args.dataset, "train", args.dataset_path)
    data_categorized = filter_categories(
        data_categorized,
        categories=args.categories,
        max_samples=args.max_samples,
    )

    if args.internvl:
        from conn_intrp.models import InternVLAdapter

        adapter = InternVLAdapter("OpenGVLab/InternVL3_5-2B-HF")
        default_step = 1
    else:
        from conn_intrp.models import SmolVLM2Adapter

        adapter = SmolVLM2Adapter("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        default_step = 5

    step = args.step
    target_updates = args.target_updates
    if step is None and target_updates is None:
        step = default_step

    run_dir = make_run_dir(
        Path(args.output),
        adapter.model_name,
        "dm",
        dataset=args.dataset,
        resume=args.resume,
    )

    run_dm(
        adapter,
        data_categorized,
        sparsity_coef=args.sparsity_coef,
        lr=args.lr,
        epochs=args.epochs,
        step=step,
        target_updates_per_epoch=target_updates,
        max_step=args.max_step,
        patience=args.patience,
        conv_threshold=args.conv_threshold,
        image_base_path=image_base_path,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
