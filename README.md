# conn-intrp

Mechanistic interpretability of cross-modal connectors in Vision-Language Models.

Analyses connector layers via SVD decomposition, directional masking, mean ablation, and delta-logit extraction. Currently supports **SmolVLM2** (single linear connector) and **InternVL3.5** (MLP connector), benchmarked on DocVQA with 9 question-type categories.

## Pipeline

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `conn_intrp.dm` | Directional masking — learn per-direction importance masks via projected SGD |
| 2 | `conn_intrp.ablation` | Mean ablation — replace directions with mean, score ANLS, extract delta-logits |
| 3 | `conn_intrp.spatial_probe` | Spatial probe — per-patch cosine similarity with right singular vectors |

## Installation

```bash
pip install -e .
```

Requires Python >= 3.10, PyTorch >= 2.0, and `transformers >= 4.40`.

## Usage

### CLI scripts

```bash
# Directional masking (Phase 1)
python scripts/run_dm.py                        # SmolVLM2 (default)
python scripts/run_dm.py --internvl             # InternVL3.5

# Mean ablation (Phase 2)
python scripts/run_ablation.py                          # SmolVLM2
python scripts/run_ablation.py --internvl               # InternVL3.5
python scripts/run_ablation.py --directions 23 70 255   # specific directions

# Spatial probe (Phase 3)
python scripts/run_spatial_probe.py --directions 23 70 255
python scripts/run_spatial_probe.py --internvl --directions 23 70 255
```

### As a library

```python
from pathlib import Path
from conn_intrp import load_docvqa, make_run_dir, run_dm
from conn_intrp.models import SmolVLM2Adapter

adapter = SmolVLM2Adapter("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
_, categorized = load_docvqa("dataset/docVQA/train_v1.0_withQT.json")
run_dir = make_run_dir("outputs", "smolvlm2", "dm")

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
    config.py            # DirectionalMaskingConfig dataclass + CSV logging
    data.py              # DocVQA loading + ANLS scoring
    output.py            # Run directories, JSON serialization, checkpointing
    dm.py                # Phase 1: directional masking loop
    ablation.py          # Phase 2: mean ablation + ANLS + delta-logit
    spatial_probe.py     # Phase 3: spatial probe projections + heatmaps
    models/
        base.py          # ModelAdapter abstract base
        smolvlm2.py      # SmolVLM2Adapter (single linear)
        internvl.py       # InternVLAdapter (MLP: LN -> linear_1 -> GELU -> linear_2)
scripts/
    run_dm.py            # CLI entry point for directional masking
    run_ablation.py      # CLI entry point for mean ablation
    run_spatial_probe.py # CLI entry point for spatial probe
notebooks/
    dm_smolvlm2_2b.ipynb
    dm_internvl3_5_2b.ipynb
    ablation_smolvlm2_2b.ipynb
    ablation_internvl3_5_2b.ipynb
```

## Adding a new model

Subclass `ModelAdapter` and implement the required methods:

- `preprocess` — tokenize inputs
- `extract_vision` — run vision encoder
- `pre_svd_forward` — layers before the SVD-decomposed layer (identity if none)
- `run_connector` — full connector forward pass
- `get_text_embeds` — text embedding lookup
- `merge_embeds` — insert connector output into text embeddings
- `generate` — decode predictions
- `get_logits` — first-token logits
- `svd_layers` — return list of `SVDLayer` descriptors (one per linear layer)
- `run_connector_layer_masked` — forward pass replacing one layer's weight
- `compute_probe_projections` — L2-normalised projections for spatial probe

The three SVD helpers (`run_connector_masked`, `compute_coefficients`, `reconstruct`) work out of the box for any connector whose last layer is decomposed as `U @ diag(S) @ Vt`.
