"""
Mechanistic interpretability of cross-modal connectors in VLMs.

Pipeline phases:
  1. Directional masking — learn per-direction importance masks
  2. Mean ablation — replace directions with mean, measure ANLS + delta-logit
  3. Spatial probe — visualize direction activation patterns

Example::

    >>> from conn_intrp import load_dm_masks, load_ablation, run_spatial_probe
    >>> masks = load_dm_masks("outputs/internvl3_5_dm_20260617_133647")
    >>> abl = load_ablation("outputs/internvl3_5_ablation_20260624_025712")

Main Functions:
    run_dm: Train directional masks via gradient optimization
    run_ablation: Single-direction mean ablation with delta-logit recording
    run_joint_ablation: Joint ablation of direction sets (active vs random)
    run_total_ablation: Zero/global-mean ablation of all directions (KL budget)
    run_spatial_probe: Per-direction spatial activation heatmaps
    load_dm_masks: Load mask tensors from a DM run directory
    load_ablation: Load ablation results from a run directory
    joint_kl_table: Active vs random KL summary per category
    baseline_comparison: Zero vs cat-mean vs global-mean vs random KL
    cumulative_kl: Per-direction KL sorted by DM weight
    super_additivity: Joint vs sum-of-individual KL ratios
    kl_budget: Active-set KL as fraction of total-layer KL budget
    load_probe: Load probe projections from a run directory
    probe_selectivity_table: Per-direction spatial selectivity summary
    probe_ablation_cross: Cross-reference spatial selectivity with ablation KL
    probe_direction_clusters: Pairwise spatial similarity and cluster assignment
"""

from .ablation import (
    compute_category_means,
    load_all_coefficients,
    load_category_coefficients,
    run_ablation,
    run_joint_ablation,
    run_total_ablation,
)
from .ablation_analysis import (
    anls_summary,
    baseline_comparison,
    cumulative_kl,
    delta_to_prob_change,
    gold_prob_summary,
    joint_kl_table,
    kl_budget,
    load_ablation,
    most_changed_directions,
    super_additivity,
    topk_botk_summary,
)
from .cka import linear_cka
from .config import DirectionalMaskingConfig
from .data import (
    DATASETS,
    HARNESS_PROMPT,
    best_anls,
    filter_categories,
    load_dataset,
    load_docvqa,
    load_okvqa,
    relaxed_anls,
)
from .dm import evaluate_dm_baselines, evaluate_mask_kl, evaluate_masks_kl, run_dm
from .dm_analysis import (
    compare_categories,
    cross_dataset_overlap,
    direction_profile,
    distribution,
    gap_survivors,
    jaccard_matrix,
    load_dm_masks,
    mask_agreement,
    mask_baseline_table,
    overlap_matrix,
    random_baseline_masks,
    random_continuous_masks,
    random_mask,
    ranked_directions,
    shared_direction_profile,
    summary_table,
    survivors,
    weight_correlation,
)
from .probe_analysis import (
    load_probe,
    plot_probe_direction,
    plot_probe_heatmap,
    probe_ablation_cross,
    probe_direction_clusters,
    probe_selectivity_table,
    save_probe_heatmaps,
)
from .output import (
    find_latest_run,
    load_checkpoint,
    make_run_dir,
    save_checkpoint,
    save_json,
    update_metadata,
)
from .spatial_probe import run_spatial_probe

__all__ = [
    # config
    "DirectionalMaskingConfig",
    # data
    "DATASETS",
    "load_dataset",
    "load_docvqa",
    "load_okvqa",
    "HARNESS_PROMPT",
    "best_anls",
    "relaxed_anls",
    "filter_categories",
    # output
    "make_run_dir",
    "find_latest_run",
    "save_json",
    "update_metadata",
    "save_checkpoint",
    "load_checkpoint",
    # directional masking
    "run_dm",
    "evaluate_mask_kl",
    "evaluate_masks_kl",
    "evaluate_dm_baselines",
    # dm analysis
    "load_dm_masks",
    "summary_table",
    "survivors",
    "gap_survivors",
    "ranked_directions",
    "distribution",
    "overlap_matrix",
    "jaccard_matrix",
    "compare_categories",
    "direction_profile",
    "random_baseline_masks",
    "random_continuous_masks",
    "random_mask",
    "mask_agreement",
    "mask_baseline_table",
    "cross_dataset_overlap",
    "weight_correlation",
    "shared_direction_profile",
    # ablation
    "compute_category_means",
    "load_all_coefficients",
    "load_category_coefficients",
    "run_ablation",
    "run_joint_ablation",
    "run_total_ablation",
    # ablation analysis
    "load_ablation",
    "delta_to_prob_change",
    "anls_summary",
    "joint_kl_table",
    "kl_budget",
    "baseline_comparison",
    "cumulative_kl",
    "gold_prob_summary",
    "topk_botk_summary",
    "super_additivity",
    "most_changed_directions",
    # spatial probe
    "run_spatial_probe",
    # probe analysis
    "load_probe",
    "plot_probe_direction",
    "plot_probe_heatmap",
    "save_probe_heatmaps",
    "probe_selectivity_table",
    "probe_ablation_cross",
    "probe_direction_clusters",
    # cka
    "linear_cka",
]
