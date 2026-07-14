"""
Run spatial probe for a specific model.

Usage:
    python scripts/run_spatial_probe.py --directions 23 70 255
    python scripts/run_spatial_probe.py --internvl --directions 23 70 255
    python scripts/run_spatial_probe.py --dataset okvqa --directions 23 70 255
    python scripts/run_spatial_probe.py --resume --directions 23 70 255
    python scripts/run_spatial_probe.py --categories "table/list" --directions 23 70
    python scripts/run_spatial_probe.py --max-samples 20 --directions 23 70
"""

import argparse
from pathlib import Path

from conn_intrp.data import DATASETS, filter_categories, load_dataset
from conn_intrp.output import make_run_dir
from conn_intrp.spatial_probe import run_spatial_probe


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
    parser.add_argument(
        "--directions",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=None)
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
        batch_size = args.batch_size or 1
    else:
        from conn_intrp.models import SmolVLM2Adapter

        adapter = SmolVLM2Adapter("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        batch_size = args.batch_size or 5

    run_dir = make_run_dir(
        Path(args.output),
        adapter.model_name,
        "probe",
        dataset=args.dataset,
        resume=args.resume,
    )

    run_spatial_probe(
        adapter,
        data_categorized,
        directions=args.directions,
        batch_size=batch_size,
        image_base_path=image_base_path,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
