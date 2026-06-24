"""
Mechanistic interpretability of cross-modal connectors in VLMs.

Pipeline phases:
  1. Directional masking — learn per-direction importance masks
  2. Mean ablation — replace directions with mean, measure ANLS + Δlogit
  3. Spatial probe — visualize direction activation patterns
"""

from .ablation import (
    compute_category_means,
    load_all_coefficients,
    load_category_coefficients,
    run_ablation,
    run_joint_ablation,
)
from .cka import linear_cka
from .config import DirectionalMaskingConfig
from .data import HARNESS_PROMPT, best_anls, filter_categories, load_docvqa, relaxed_anls
from .dm import evaluate_dm_baselines, evaluate_mask_kl, evaluate_masks_kl, run_dm
from .dm_analysis import (
    compare_categories,
    direction_profile,
    distribution,
    jaccard_matrix,
    load_dm_masks,
    mask_agreement,
    overlap_matrix,
    random_mask,
    ranked_directions,
    summary_table,
    survivors,
)
from .output import (
    find_latest_run,
    load_checkpoint,
    make_run_dir,
    save_checkpoint,
    save_json,
    update_metadata,
)
from .spatial_probe import plot_probe_heatmap, run_spatial_probe, save_probe_heatmaps

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
    "update_metadata",
    "save_checkpoint",
    "load_checkpoint",
    "run_dm",
    "evaluate_mask_kl",
    "evaluate_masks_kl",
    "evaluate_dm_baselines",
    "compute_category_means",
    "load_all_coefficients",
    "load_category_coefficients",
    "run_ablation",
    "run_joint_ablation",
    "run_spatial_probe",
    "plot_probe_heatmap",
    "save_probe_heatmaps",
    "load_dm_masks",
    "summary_table",
    "survivors",
    "ranked_directions",
    "distribution",
    "overlap_matrix",
    "jaccard_matrix",
    "compare_categories",
    "direction_profile",
    "random_mask",
    "mask_agreement",
    "linear_cka",
]
