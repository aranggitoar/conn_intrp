"""
Run spatial probe for a specific model.

Usage:
    python scripts/run_spatial_probe.py --directions 23 70 255
    python scripts/run_spatial_probe.py --internvl --directions 23 70 255
"""

import argparse
from pathlib import Path

from conn_intrp.data import load_docvqa
from conn_intrp.output import make_run_dir
from conn_intrp.spatial_probe import run_spatial_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internvl", action="store_true")
    parser.add_argument(
        "--directions", type=int, nargs="+", required=True,
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="dataset/docVQA")
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    image_base_path = Path(args.dataset)
    _, data_categorized = load_docvqa(
        image_base_path / "train_v1.0_withQT.json"
    )

    if args.internvl:
        from conn_intrp.models import InternVLAdapter
        adapter = InternVLAdapter("OpenGVLab/InternVL3_5-2B-HF")
        batch_size = args.batch_size or 1
    else:
        from conn_intrp.models import SmolVLM2Adapter
        adapter = SmolVLM2Adapter("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        batch_size = args.batch_size or 5

    run_dir = make_run_dir(Path(args.output), adapter.model_name, "probe")

    run_spatial_probe(
        adapter, data_categorized,
        directions=args.directions,
        batch_size=batch_size,
        image_base_path=image_base_path,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
