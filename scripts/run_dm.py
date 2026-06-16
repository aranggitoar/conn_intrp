"""
Run directional masking for a specific model.

Usage:
    python scripts/run_dm.py              # SmolVLM2 (default)
    python scripts/run_dm.py --internvl   # InternVL3.5
"""

import argparse
from pathlib import Path

from conn_intrp.data import load_docvqa
from conn_intrp.output import make_run_dir
from conn_intrp.dm import run_dm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internvl", action="store_true")
    parser.add_argument("--sparsity-coef", type=float, default=1.5e-3)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--step", type=int, default=None)
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
        step = args.step or 1
    else:
        from conn_intrp.models import SmolVLM2Adapter
        adapter = SmolVLM2Adapter("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        step = args.step or 5

    run_dir = make_run_dir(Path(args.output), adapter.model_name, "dm")

    run_dm(
        adapter, data_categorized,
        sparsity_coef=args.sparsity_coef,
        lr=args.lr,
        epochs=args.epochs,
        step=step,
        image_base_path=image_base_path,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
