"""
Spatial probe (Phase 3).

Projects connector layer inputs onto right singular vectors to produce
per-patch activation maps, revealing which spatial regions each SVD
direction responds to. Projections are L2-normalised (cosine similarity),
yielding values in ``[-1, 1]``.

Example::

    >>> from conn_intrp import run_spatial_probe, save_probe_heatmaps
    >>> run_spatial_probe(adapter, categorized, directions=[23, 70],
    ...     batch_size=1, image_base_path=img_path, run_dir=run_dir)
    >>> # Later, generate heatmaps from saved projections:
    >>> probes = torch.load(run_dir / "table_list" / "probe_projections.pt")
    >>> save_probe_heatmaps(img_path / "image.png", probes["proj"][0],
    ...     directions=[23, 70], grid_size=8, out_dir=Path("heatmaps"))

Main Functions:
    run_spatial_probe: Compute and save per-category probe projections.
    plot_probe_heatmap: Overlay a single heatmap on an image.
    save_probe_heatmaps: Save heatmap overlays for one image across directions.
"""

import math
import torch
from pathlib import Path
from torch.nn import functional as F
from tqdm.auto import tqdm

from .models.base import ModelAdapter
from .output import fs_safe, save_json


def run_spatial_probe(
    adapter: ModelAdapter, data_categorized: dict[str, list], *,
    directions: list[int], batch_size: int,
    image_base_path: Path, run_dir: Path,
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
    :param directions: SVD direction indices to save projections for.
    :type directions: list[int]
    :param batch_size: Number of images per forward pass (1 for InternVL).
    :type batch_size: int
    :param image_base_path: Root directory for image files.
    :type image_base_path: Path
    :param run_dir: Output directory for this run.
    :type run_dir: Path
    """
    grid_size = int(math.sqrt(adapter.n_patches))

    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        save_json(meta_path, {
            "model": adapter.model_name,
            "n_patches": adapter.n_patches,
            "grid_size": grid_size,
            "directions": directions,
        })

    completed_probe = {
        name for name in data_categorized
        if (run_dir / fs_safe(name) / "probe_projections.pt").exists()
    }
    if completed_probe:
        print(
            f"Resuming probe: skipping "
            f"{len(completed_probe)} completed categories"
        )

    for name, data in tqdm(data_categorized.items(), desc="Spatial probe"):
        if name in completed_probe:
            print(f'  Skipping "{name}" (projections exist)')
            continue

        length = len(data)
        projections = {}
        image_files = []

        for i in tqdm(
            range(0, length, batch_size),
            total=math.ceil(length / batch_size),
            desc=f'"{name}"',
        ):
            batch = data[i:i + batch_size]
            image_files.extend(datum["image"] for datum in batch)

            with torch.no_grad():
                inputs = adapter.preprocess(batch, image_base_path)
                probes = adapter.compute_probe_projections(inputs)

            for layer_name, proj in probes.items():
                if layer_name not in projections:
                    projections[layer_name] = []
                n_dirs_layer = proj.shape[-1]
                valid = [d for d in directions if d < n_dirs_layer]
                projections[layer_name].append(proj[..., valid].cpu())

        cat_dir = run_dir / fs_safe(name)
        cat_dir.mkdir(exist_ok=True)

        result = {
            layer: torch.cat(chunks, dim=0)
            for layer, chunks in projections.items()
        }
        torch.save(result, cat_dir / "probe_projections.pt")

        layer_meta = {}
        for layer, tensor in result.items():
            n_dirs_layer_full = (
                adapter.n_dirs if layer == adapter.component_name
                else tensor.shape[-1]
            )
            valid = [d for d in directions if d < n_dirs_layer_full]
            layer_meta[layer] = {
                "valid_directions": valid,
                "n_dirs_total": n_dirs_layer_full,
            }

        save_json(cat_dir / "probe_meta.json", {
            "category": name,
            "n_images": length,
            "directions_requested": directions,
            "grid_size": grid_size,
            "layers": layer_meta,
            "image_files": image_files,
        })


def plot_probe_heatmap(
    image_path: str | Path, heatmap: torch.Tensor, *,
    mode: str = "signed", upsample_size: tuple[int, int] | None = None,
    ax=None,
):
    """
    Overlay a spatial probe heatmap on an image.

    :param image_path: Path to the original image.
    :type image_path: str | Path
    :param heatmap: 2-D tensor of shape ``(grid_h, grid_w)``.
    :type heatmap: torch.Tensor
    :param mode: ``"signed"`` (``RdBu_r``, range ``[-1, 1]``) or
        ``"abs"`` (``hot``, range ``[0, 1]``).
    :type mode: str
    :param upsample_size: ``(H, W)`` target for upsampling.
        Defaults to the image's native size.
    :type upsample_size: tuple[int, int] | None
    :param ax: Matplotlib axes. Created if ``None``.
    :returns: The axes with the overlay rendered.
    :rtype: matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    if upsample_size is None:
        upsample_size = (image.height, image.width)

    heatmap_up = F.interpolate(
        heatmap.float()[None, None],
        size=upsample_size,
        mode="nearest",
    ).squeeze().cpu()

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(image)
    if mode == "abs":
        ax.imshow(heatmap_up.abs(), cmap="hot", alpha=0.5, vmin=0, vmax=1)
    else:
        ax.imshow(heatmap_up, cmap="RdBu_r", alpha=0.5, vmin=-1, vmax=1)
    ax.axis("off")
    return ax


def save_probe_heatmaps(
    image_path: str | Path, projection: torch.Tensor, *,
    directions: list[int], grid_size: int, out_dir: str | Path,
    modes: tuple[str, ...] = ("signed", "abs"),
    layer_name: str = "",
) -> None:
    """
    Save heatmap overlay images for one image across specified directions.

    ``projection[:, i]`` must correspond to ``directions[i]``.

    :param image_path: Path to the original image.
    :type image_path: str | Path
    :param projection: Projection tensor of shape
        ``(n_patches, n_selected_dirs)``.
    :type projection: torch.Tensor
    :param directions: Original direction indices (used for filenames).
    :type directions: list[int]
    :param grid_size: Spatial grid side length (e.g. 8 for 64 patches).
    :type grid_size: int
    :param out_dir: Directory to write heatmap images.
    :type out_dir: str | Path
    :param modes: Rendering modes to save (``"signed"`` and/or ``"abs"``).
    :type modes: tuple[str, ...]
    :param layer_name: Optional prefix for filenames (e.g. ``"linear_2"``).
    :type layer_name: str
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(image_path).stem
    prefix = f"{layer_name}_" if layer_name else ""

    for i, dir_idx in enumerate(directions):
        heatmap = projection[:, i].reshape(grid_size, grid_size)
        for mode in modes:
            fig, ax = plt.subplots()
            plot_probe_heatmap(image_path, heatmap, mode=mode, ax=ax)
            fig.savefig(
                out_dir / f"{prefix}{image_stem}_dir{dir_idx}_{mode}.png",
                bbox_inches="tight", pad_inches=0, dpi=150,
            )
            plt.close(fig)
