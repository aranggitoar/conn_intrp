# conn-intrp

Mechanistic interpretability of cross-modal connectors in Vision-Language Models.

Analyses connector layers via SVD decomposition, directional masking, mean ablation, and delta-logit extraction. Currently supports **SmolVLM2** (single linear connector) and **InternVL3.5** (MLP connector), benchmarked on DocVQA.

## Pipeline

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `conn_intrp.dm` | Directional masking, learn per-direction importance masks via projected SGD |
| 2 | `conn_intrp.ablation` | Mean ablation, replace directions with mean, measure differences
| 3 | `conn_intrp.spatial_probe` | Spatial probe, per-patch cosine similarity with right singular vectors |

## Installation

```bash
pip install -e .
```

Requires Python >= 3.12, PyTorch >= 2.9, and `transformers >= 5.11`.

## Usage

```python
from pathlib import Path
from conn_intrp import load_docvqa, make_run_dir, run_dm
from conn_intrp.models import SmolVLM2Adapter

adapter = SmolVLM2Adapter("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
_, categorized = load_docvqa("dataset/docVQA/train_v1.0_withQT.json")
run_dir = make_run_dir("outputs", "smolvlm2", "dm", "docvqa")

run_dm(
    adapter, categorized,
    sparsity_coef=1.5e-3, lr=0.1, epochs=3, step=5,
    image_base_path=Path("dataset/docVQA"),
    run_dir=run_dir,
)
```

## Package structure

```
conn_intrp/
    __init__.py          # Public API re-exports
    config.py            # DirectionalMaskingConfig dataclass + DM experiment CSV logging
    data.py              # DocVQA loading + ANLS scoring
    output.py            # Run directories, JSON serialization, checkpointing
    dm.py                # directional masking loop
    dm_analysis.py       # directional masking analysis functions
    ablation.py          # mean ablation + ANLS + delta-logit
    ablation_analysis.py # mean ablation analysis functions
    spatial_probe.py     # spatial probe projections + heatmaps
    probe_analysis.py    # spatial probe analysis functions
    models/
        base.py          # ModelAdapter abstract base
        smolvlm2.py      # SmolVLM2Adapter (single linear)
        internvl.py       # InternVLAdapter (MLP: LN -> linear_1 -> GELU -> linear_2)
```
