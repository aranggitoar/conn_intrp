"""
Configuration and experiment logging for connector interpretability.

Provides the run-record dataclass for directional masking experiments
and a helper to persist masks alongside a CSV experiment log.

Example::

    >>> from conn_intrp.config import DirectionalMaskingConfig, save_dm_run
    >>> config = DirectionalMaskingConfig(
    ...     category="table/list", model="smolvlm2", component="proj",
    ...     optimizer="SGD", sparsity_coef=1.5e-3, lr=0.1, epochs=3, step=5,
    ...     kl_per_epoch=[0.01], l1_per_epoch=[0.5],
    ...     below_half_per_epoch=[120], near_zero_per_epoch=[80],
    ... )
    >>> save_dm_run(config, mask_logits)

Main Classes:
    DirectionalMaskingConfig: Run record for a single category's mask training.
"""

import csv
import datetime
import torch
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class DirectionalMaskingConfig:
    """Run record for a single category's directional masking training."""
    category: str
    model: str
    component: str
    optimizer: str
    sparsity_coef: float
    lr: float
    epochs: int
    step: int
    kl_per_epoch: list
    l1_per_epoch: list
    below_half_per_epoch: list
    near_zero_per_epoch: list


def save_dm_run(config: DirectionalMaskingConfig, mask: torch.Tensor) -> None:
    """
    Save mask weights and append a row to the experiment log CSV.

    :param config: Run record with hyperparameters and per-epoch stats.
    :type config: DirectionalMaskingConfig
    :param mask: Learned mask tensor to persist.
    :type mask: torch.Tensor
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("./weights").mkdir(parents=True, exist_ok=True)
    weight_file = f"./weights/{config.model}_{config.component}_{run_id}.pt"
    torch.save(mask.data, weight_file)

    row = asdict(config)
    row.update({
        "run_id": run_id,
        "weight_file": weight_file,
        "mask_min": mask.min().item(),
        "mask_max": mask.max().item(),
        "mask_mean": mask.mean().item(),
        "final_below_half": config.below_half_per_epoch[-1],
        "final_near_zero": config.near_zero_per_epoch[-1],
        "final_kl": config.kl_per_epoch[-1],
        "final_l1": config.l1_per_epoch[-1],
    })

    file_exists = Path("experiment_log.csv").exists()
    with open("experiment_log.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
