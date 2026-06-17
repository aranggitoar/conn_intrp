"""
Mechanistic interpretability of cross-modal connectors in VLMs.

Pipeline phases:
  1. Directional masking — learn per-direction importance masks
  2. Mean ablation — replace directions with mean, measure ANLS + Δlogit
  3. Spatial probe — visualize direction activation patterns
"""

from .config import DirectionalMaskingConfig
from .data import load_docvqa, HARNESS_PROMPT, best_anls, relaxed_anls, filter_categories
from .output import (
    make_run_dir, find_latest_run, save_json, save_checkpoint, load_checkpoint,
)
from .dm import run_dm
from .ablation import compute_category_means, run_ablation
from .spatial_probe import run_spatial_probe, plot_probe_heatmap, save_probe_heatmaps

__all__ = [
    "DirectionalMaskingConfig",
    "load_docvqa",
    "HARNESS_PROMPT",
    "best_anls",
    "relaxed_anls",
    "filter_categories",
    "make_run_dir",
    "find_latest_run",
    "save_json",
    "save_checkpoint",
    "load_checkpoint",
    "run_dm",
    "compute_category_means",
    "run_ablation",
    "run_spatial_probe",
    "plot_probe_heatmap",
    "save_probe_heatmaps",
]
