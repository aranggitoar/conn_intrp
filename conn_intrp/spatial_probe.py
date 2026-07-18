"""
Spatial probe.

Projects connector layer inputs onto right singular vectors to produce
per-patch activation maps, revealing which spatial regions each SVD
direction responds to. Projections are L2-normalised (cosine similarity),
yielding values in ``[-1, 1]``.

Example::

    >>> from conn_intrp import run_spatial_probe, save_probe_heatmaps
    >>> run_spatial_probe(adapter, categorized,
    ...     directions={"linear_1": [3, 7], "linear_2": [23, 70]},
    ...     batch_size=1, image_base_path=img_path, run_dir=run_dir)
    >>> # Later, generate heatmaps from saved projections:
    >>> probes = torch.load(run_dir / "table_list" / "probe_projections.pt")
    >>> save_probe_heatmaps(img_path / "image.png", probes["linear_2"][0],
    ...     directions=[23, 70], grid_size=8, out_dir=Path("heatmaps"))

Main Functions:
    run_spatial_probe: Compute and save per-category probe projections.
"""

import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .models.base import ModelAdapter
from .output import fs_safe, save_json


def run_spatial_probe(
    adapter: ModelAdapter,
    data_categorized: dict[str, list],
    *,
    directions: dict[str, list[int]] | list[int],
    batch_size: int,
    image_base_path: Path,
    run_dir: Path,
) -> None:
    """
    Compute and save spatial probe projections for specified directions.

    For each category, iterates over images, computes L2-normalised
    projections via :meth:`~ModelAdapter.compute_probe_projections`,
    indexes to the requested directions, and saves per-category ``.pt``
    files.

    :param adapter: Model adapter instance.
    :type adapter: ModelAdapter
    :param data_categorized: Map of category name to list of data dicts.
    :type data_categorized: dict[str, list]
    :param directions: Per-layer direction indices, e.g.
        ``{"linear_1": [3, 7], "linear_2": [23, 70]}``. A flat list
        is broadcast to all layers (indices beyond a layer's rank are
        dropped).
    :type directions: dict[str, list[int]] | list[int]
    :param batch_size: Number of images per forward pass (1 for InternVL).
    :type batch_size: int
    :param image_base_path: Root directory for image files.
    :type image_base_path: Path
    :param run_dir: Output directory for this run.
    :type run_dir: Path
    """
    svd_info = {sl.name: sl for sl in adapter.svd_layers}
    if isinstance(directions, list):
        directions = {
            name: [d for d in directions if d < layer.n_dirs] for name, layer in svd_info.items()
        }

    grid_size = int(math.sqrt(adapter.n_patches))

    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        save_json(
            meta_path,
            {
                "model": adapter.model_name,
                "n_patches": adapter.n_patches,
                "grid_size": grid_size,
                "directions": directions,
            },
        )

    completed_probe = {
        name
        for name in data_categorized
        if (run_dir / fs_safe(name) / "probe_projections.pt").exists()
    }
    if completed_probe:
        print(f"Resuming probe: skipping {len(completed_probe)} completed categories")

    for name, data in tqdm(data_categorized.items(), desc="Spatial probe"):
        if name in completed_probe:
            print(f'  Skipping "{name}" (projections exist)')
            continue

        n_images = len(data)
        projections: dict[str, list[torch.Tensor]] = {}
        image_files: list[str] = []

        for i in tqdm(
            range(0, n_images, batch_size),
            total=math.ceil(n_images / batch_size),
            desc=f'"{name}"',
        ):
            batch = data[i : i + batch_size]
            image_files.extend(datum["image"] for datum in batch)

            with torch.no_grad():
                inputs = adapter.preprocess(batch, image_base_path)
                probes = adapter.compute_probe_projections(inputs)

            for layer_name, proj in probes.items():
                dir_list = directions.get(layer_name, [])
                if not dir_list:
                    continue
                if layer_name not in projections:
                    projections[layer_name] = []
                projections[layer_name].append(proj[..., dir_list].cpu())

        cat_dir = run_dir / fs_safe(name)
        cat_dir.mkdir(exist_ok=True)

        result = {layer: torch.cat(chunks, dim=0) for layer, chunks in projections.items()}
        torch.save(result, cat_dir / "probe_projections.pt")

        save_json(
            cat_dir / "probe_meta.json",
            {
                "category": name,
                "n_images": n_images,
                "grid_size": grid_size,
                "layers": {
                    ln: {
                        "directions": directions[ln],
                        "n_dirs_total": svd_info[ln].n_dirs,
                    }
                    for ln in projections
                },
                "image_files": image_files,
            },
        )
