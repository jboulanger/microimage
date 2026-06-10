"""Utilities from manipulating data."""

from typing import List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

# Register extra colormaps
for c in ["red", "green", "blue"]:
    if c not in mpl.colormaps:
        mpl.colormaps.register(cmap=LinearSegmentedColormap.from_list(c, ["black", c]))
if "gray" not in mpl.colormaps:
    mpl.colormaps.register(
        cmap=LinearSegmentedColormap.from_list("gray", ["black", "white"])
    )


def mask_to_yolo_contours(
    labels: np.ndarray, class_index: int = 0
) -> List[List[Union[int, float]]]:
    """Convert a label image or binary mask to YOLO contour lines using scikit-image.

    Parameters
    ----------
    labels : np.ndarray
        2D label image or binary mask.
    class_index : int
        The class index to assign to the contours. Defaults to 0.

    Returns
    -------
    List[List[Union[int, float]]]
    List of contours, where each contour is [class_index, x1, y1, x2, y2, ...].
    Coordinates are normalized to the image size [0, 1].
    """
    if labels.ndim != 2:
        return []

    h, w = labels.shape

    # Ensure disconnected components are treated as separate objects if input is binary
    unique_vals = np.unique(labels)
    if unique_vals.size == 2 and 0 in unique_vals:
        labeled_imgs = ski.measure.label(labels)
    else:
        labeled_imgs = labels

    props = ski.measure.regionprops(labeled_imgs)
    all_lines = []

    for prop in props:
        # Pad the object mask to prevent find_contours from clipping at edges
        mask = np.pad(prop.image_filled.astype(np.uint8), 2)

        # Extract contours (y, x coordinates)
        contours = ski.measure.find_contours(mask, level=0.5)
        if not contours:
            continue

        # Take the largest contour for this object (typically the outer boundary)
        largest_contour = max(contours, key=len)

        # Simplify the polygon to keep vertex count reasonable
        simplified = ski.measure.approximate_polygon(largest_contour, tolerance=1.0)

        min_y, min_x, _, _ = prop.bbox
        normalized_poly = []
        for y, x in simplified:
            # Adjust for padding and bounding box offset, then normalize
            global_x = (x - 2 + min_x) / w
            global_y = (y - 2 + min_y) / h
            # Round for storage efficiency and clamp to [0, 1]
            normalized_poly.append(float(np.clip(round(global_x, 6), 0, 1)))
            normalized_poly.append(float(np.clip(round(global_y, 6), 0, 1)))

        all_lines.append([class_index] + normalized_poly)

    return all_lines


def show_data(
    data: xr.DataArray,
    channels: List[str],
    cmap: List[str] = ["Reds", "Greens", "Blues"],
    ax=None,
):
    """Show the data as maximum intensity projection at frame 0.

    Parameters
    ----------
    data: xr.DataArray
        The 5D data
    channels: list[str]
        Channels name to use.
    cmap:list[str]
    ax: matplotlib.pyplot.axes | None
        Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.patches import Rectangle

    rgb = np.zeros([data.shape[-2], data.shape[-1], 4])
    for k in range(len(channels)):
        plane = 255 * np.array(data[0].sel(C=channels[k]).max("Z")).astype(float)
        a, b = np.percentile(plane, [0.5, 99])
        plane = (plane - a) / (b - a)
        cm = colormaps.get_cmap(cmap[k])
        rgb += cm(plane)
    rgb = rgb / len(channels)
    if ax is None:
        _, ax = plt.subplots(1, 1)
    extent = (
        data["X"][0].item(),
        data["X"][-1].item(),
        data["Y"][-1].item(),
        data["Y"][0].item(),
    )
    ax.imshow(rgb, extent=extent)
    ax.set_axis_off()
    pos_x = [0, *np.cumsum([len(x) for x in channels]).tolist()]
    for k in range(len(channels)):
        color = colormaps.get_cmap(cmap[k])(128)
        plt.text(
            0.03 * pos_x[k],
            0.97,
            channels[k],
            color=color,
            weight="bold",
            transform=ax.transAxes,
        )

    # Add a scale bar (approx 1/5 of the field of view, adjusted to a 'nice' round number)
    x_range = data["X"][-1].item() - data["X"][0].item()
    y_range = data["Y"][-1].item() - data["Y"][0].item()

    bar_width_raw = x_range / 5
    # Find a nice round number (1, 2, or 5 * 10^n)
    exp = np.floor(np.log10(bar_width_raw))
    unit = 10**exp
    bar_width = unit * (
        5 if bar_width_raw >= 5 * unit else 2 if bar_width_raw >= 2 * unit else 1
    )

    # Position in bottom right with 5% padding
    bar_x = data["X"][-1].item() - bar_width - (x_range * 0.05)
    bar_y = data["Y"][-1].item() - (y_range * 0.05)
    bar_h = y_range * 0.02

    scalebar = Rectangle(
        xy=(bar_x, bar_y),
        width=bar_width,
        height=bar_h,
        color="white",
    )
    ax.add_patch(scalebar)

    # Label the scale bar width
    ax.text(
        bar_x + bar_width / 2,
        bar_y - bar_h * 1.2,
        f"{bar_width:g}µm",
        color="white",
        ha="center",
        va="bottom",
        weight="bold",
        fontsize=10,
    )


def show_contours(shape, data, ax=None):
    """Plot the segmentation contours."""
    if ax is None:
        ax = plt.gca()
    for row in data:
        # parts[0] is the class_id, the rest are x, y pairs
        coords = np.array(row[1:], dtype=float).reshape(-1, 2)

        # 3. Denormalize coordinates
        # x_pixel = x_norm * width, y_pixel = y_norm * height
        x_pixels = coords[:, 0] * shape[1]
        y_pixels = coords[:, 1] * shape[0]

        # 4. Plot the polygon (close it by appending the first point at the end)
        ax.plot(
            np.append(x_pixels, x_pixels[0]),
            np.append(y_pixels, y_pixels[0]),
            linewidth=2,
            alpha=0.7,
        )
